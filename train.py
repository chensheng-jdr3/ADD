import os
import torch
import sys
import argparse
import logging
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader.loader import MultiClassPairDataset
from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from lib import resnet50, resnet50_w, embed_layer, PSR, refine_cams_with_bkg, SR_generation
from datetime import datetime


def compute_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def compute_metrics_from_confusion_matrix(confusion_matrix):
    row_sum = confusion_matrix.sum(axis=1)
    diagonal = np.diag(confusion_matrix)
    recall_per_class = np.divide(
        diagonal,
        row_sum,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=row_sum != 0,
    )

    col_sum = confusion_matrix.sum(axis=0)
    precision_per_class = np.divide(
        diagonal,
        col_sum,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=col_sum != 0,
    )

    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=(precision_per_class + recall_per_class) != 0,
    )
    macro_f1 = float(np.mean(f1_per_class))
    return recall_per_class, f1_per_class, macro_f1


def load_pretrained(teacher, student, scratch, teacher_modality='NBI', student_modality='WLI'):
    # prefer modality-specific pretrained files, fallback to legacy names
    stu_path_mod = f'./pretrained/{opt.fold}_student_{student_modality}.pth'
    tea_path_mod = f'./pretrained/{opt.fold}_teacher_{teacher_modality}.pth'
    stu_path_legacy = f'./pretrained/{opt.fold}_student.pth'
    tea_path_legacy = f'./pretrained/{opt.fold}_teacher.pth'

    if not scratch:
        # load student if exists
        if os.path.exists(stu_path_mod):
            student.load_state_dict(torch.load(stu_path_mod, map_location=opt.device), strict=False)
            print(f'load student from {stu_path_mod} completed')
        elif os.path.exists(stu_path_legacy):
            student.load_state_dict(torch.load(stu_path_legacy, map_location=opt.device), strict=False)
            print(f'load student from {stu_path_legacy} completed')

        # load teacher if exists
        if os.path.exists(tea_path_mod):
            teacher.load_state_dict(torch.load(tea_path_mod, map_location=opt.device), strict=False)
            print(f'load teacher from {tea_path_mod} completed')
        elif os.path.exists(tea_path_legacy):
            teacher.load_state_dict(torch.load(tea_path_legacy, map_location=opt.device), strict=False)
            print(f'load teacher from {tea_path_legacy} completed')
    return teacher, student


def get_ce_loss(img, label, network):
    pred,f_embed, f4= network(img)
    CEloss = ce_loss(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float()
    return CEloss, acc, pred, f4, f_embed


def save_model(epoch, student, train_save, student_modality=None):
    print('update model..')
    stu_path = train_save + '/weights/student_model-{}.pth'.format(epoch)
    torch.save(student.state_dict(), stu_path)
    # also save to ./pretrained with modality suffix if provided
    if student_modality is not None:
        pretrained_path = f'./pretrained/{opt.fold}_student_{student_modality}.pth'
        torch.save(student.state_dict(), pretrained_path)


def cam(fmaps,model_predict,cls_idx):
    score =torch.ones(size=(model_predict.shape[0],1))
    for pp in range(cls_idx.shape[0]):
        score[pp]=model_predict[pp, cls_idx[pp]]
    score = score.sum()
    weights = torch.autograd.grad(outputs=score, inputs=fmaps,create_graph=True)[0]
    weights = weights.mean(dim=(2, 3))
    grad_cam = (weights.view(*weights.shape, 1, 1) * fmaps.squeeze(0)).sum(1)

    def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        return cams
    
    B, H, W = grad_cam.shape
    grad_cam = _normalize(F.relu(grad_cam, inplace=True))
    grad_cam = grad_cam.view(B, 1, H, W)
    return grad_cam


def logit_standardization(x, tau=4.0, eps=1e-6, unbiased=False):
    """Per-sample Z-score standardization of logits then scale by base temperature tau.

    Args:
        x: Tensor of shape [B, C]
        tau: base temperature scalar to divide standardized logits
        eps: small value to avoid division by zero
        unbiased: whether to use unbiased estimator for std (default False)

    Returns:
        standardized logits of shape [B, C]
    """
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, unbiased=unbiased, keepdim=True)
    return (x - mu) / (sigma + eps) / tau


# The key component: Alignment-free Dense Distillation (ADD) module (in Eq. 2.1)
def ADD(pseudo_label1, pseudo_label2, q, kv):
    # Get Semantic Relation Map (in Eq. 2.2)
    aff_mask = SR_generation(pseudo_label1, pseudo_label2, ignore_index=opt.ignore_index, confuse_value=0)
    k = kv
    v = kv.transpose(2,1)
    matmul_qk = torch.matmul(q.transpose(2,1), k)
    matmul_qk_= matmul_qk * aff_mask
    matmul_qk_= matmul_qk_.masked_fill(aff_mask == 0, -1e9)
    attmap = F.softmax(matmul_qk_, dim=-1)
    f_cam_att = torch.matmul(attmap, v)            
    loss_align=sim_loss(f_cam_att.transpose(1,2).flatten(1), q.flatten(1), torch.ones(1).to(opt.device))
    return loss_align


def train(teacher, student, embed_layer_, class_names, epochs=1000, is_test=True, teacher_modality='NBI', student_modality='WLI'):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_embed_layer=torch.optim.Adam(embed_layer_.parameters(), lr=1e-4, weight_decay=1e-8)
    psr = PSR(num_iter=10, dilations=[1,2,4,8])
    psr.to(opt.device)
    
    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    val_acc_best = 0
    val_macro_f1_best = 0
    best_model_epoch = 0
    num_classes = len(class_names)
    space_loss = torch.zeros(1).to(opt.device)
    loss_ADD_1=torch.zeros(1).to(opt.device)
    loss_ADD_2=torch.zeros(1).to(opt.device)
    for epoch in range(1, epochs):
        for phase in iter(phases):
            if phase == 'train':
                teacher.eval()
                student.train()
                ldr = loader
                train_acc_num = 0
            else:
                teacher.eval()
                student.eval()
                ldr = val_loader
                val_acc_num = 0
                val_preds = []
                val_labels = []
            summary = []
            sample_num = 0
            for batch in tqdm(ldr, leave=False):
                wli_img, nbi_img, label = batch[0].to(opt.device).float(), batch[1].to(opt.device).float(), batch[2].to(opt.device).long()

                # select teacher / student inputs based on modality direction
                if teacher_modality.upper() == 'NBI' and student_modality.upper() == 'WLI':
                    teacher_input = nbi_img
                    student_input = wli_img
                elif teacher_modality.upper() == 'WLI' and student_modality.upper() == 'NBI':
                    teacher_input = wli_img
                    student_input = nbi_img
                else:
                    # fallback: teacher uses nbi, student uses wli
                    teacher_input = nbi_img
                    student_input = wli_img

                pred_tea, f4_tea = teacher(teacher_input)
                CE_loss, acc_wli, pred_stu, f4_stu, f_stu_att = get_ce_loss(student_input, label, student)
                if phase == 'train':
                    # train student
                    optimizer_stu.zero_grad()
                    optimizer_embed_layer.zero_grad()
                    f_tea_att = embed_layer_(f4_tea.detach())
                    # logit distillation: align teacher logits -> student logits
                    # apply per-sample Z-score standardization to both teacher and student logits
                    logits_t_z = logit_standardization(pred_tea.detach(), tau=opt.tau, eps=opt.eps)
                    logits_s_z = logit_standardization(pred_stu, tau=opt.tau, eps=opt.eps)
                    target = torch.ones(pred_stu.size(0)).to(opt.device)
                    logit_loss = sim_loss(logits_t_z, logits_s_z, target)

                    if epoch>0:
                        cam1 = cam(f4_stu, pred_stu, label)
                        cam2 = cam(f4_tea, pred_tea, label)

                        pseudo_label1 = refine_cams_with_bkg(psr, wli_img, cams=cam1, cfg=opt)
                        pseudo_label2 = refine_cams_with_bkg(psr, nbi_img, cams=cam2, cfg=opt)

                        loss_ADD_1 = ADD(pseudo_label1, pseudo_label2, f_tea_att, f_stu_att)
                        loss_ADD_2 = ADD(pseudo_label2, pseudo_label1, f_stu_att, f_tea_att)
                        space_loss = loss_ADD_1+loss_ADD_2

                    loss=CE_loss + space_loss +logit_loss
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss)
                        sys.exit(1)

                    loss.backward()
                    optimizer_embed_layer.step()
                    optimizer_stu.step()
                    summary.append((loss.item(), CE_loss.item(), logit_loss.item(), loss_ADD_1.item(),loss_ADD_2.item()))
                    # train_acc
                    train_acc_num = train_acc_num+acc_wli
                    sample_num += student_input.shape[0]
                else:
                    sample_num += student_input.shape[0]
                    val_acc_num = val_acc_num+acc_wli
                    pred_label = torch.argmax(pred_stu, dim=-1)
                    val_preds.extend(pred_label.detach().cpu().numpy().tolist())
                    val_labels.extend(label.detach().cpu().numpy().tolist())

            if len(summary) > 0:
                summary = np.array(summary).mean(axis=0)
            
            if phase == 'train':
                train_acc=train_acc_num / sample_num
                tags = ['acc_wli','train_total_loss','CE_loss', 'logit_loss', 'loss_ADD_1','loss_ADD_2']
                tb_writer.add_scalar(tags[0], train_acc, epoch)
                tb_writer.add_scalar(tags[1], summary[0], epoch)
                tb_writer.add_scalar(tags[2], summary[1], epoch)
                tb_writer.add_scalar(tags[3], summary[2], epoch)
                tb_writer.add_scalar(tags[4], summary[3], epoch)
                tb_writer.add_scalar(tags[5], summary[4], epoch)
                print('##############################################################################')
                print('[TRAIN] Epoch %d' % epoch,'acc: %0.2f, Total_loss: %0.2f, CE_loss: %0.2f, logit_loss: %0.2f, loss_ADD_1: %0.2f, loss_ADD_2: %0.2f' % (train_acc, summary[0], summary[1], summary[2], summary[3], summary[4]))#
                #save_model(epoch)
            else:
                val_acc = float(val_acc_num / sample_num)
                confusion_matrix = compute_confusion_matrix(val_labels, val_preds, num_classes)
                recall_per_class, f1_per_class, val_macro_f1 = compute_metrics_from_confusion_matrix(confusion_matrix)

                if val_macro_f1 > val_macro_f1_best:
                    val_macro_f1_best = val_macro_f1
                    val_acc_best = val_acc
                    best_model_epoch = epoch
                    save_model(epoch, student, opt.train_save, student_modality=student_modality)

                recall_msg = ', '.join([
                    '{}:{:.4f}'.format(class_names[idx], recall_per_class[idx])
                    for idx in range(num_classes)
                ])
                f1_msg = ', '.join([
                    '{}:{:.4f}'.format(class_names[idx], f1_per_class[idx])
                    for idx in range(num_classes)
                ])

                print(
                    '[EVAL] Epoch %d' % epoch,
                    'val_acc: %0.4f, val_macro_f1: %0.4f, best_macro_f1: %0.4f, best_epoch: %d'
                    % (val_acc, val_macro_f1, val_macro_f1_best, best_model_epoch)
                )
                print('[EVAL] recall_per_class -> {}'.format(recall_msg))
                print('[EVAL] confusion_matrix:\n{}'.format(confusion_matrix))

                logging.info(
                    '[EVAL] Epoch %d, val_acc: %0.4f, val_macro_f1: %0.4f, best_macro_f1: %0.4f, best_epoch: %d'
                    % (epoch, val_acc, val_macro_f1, val_macro_f1_best, best_model_epoch)
                )
                logging.info('[EVAL] recall_per_class -> {}'.format(recall_msg))
                logging.info('[EVAL] f1_per_class -> {}'.format(f1_msg))
                logging.info('[EVAL] confusion_matrix:\n{}'.format(confusion_matrix))

                tb_writer.add_scalar('val_acc', val_acc, epoch)
                tb_writer.add_scalar('val_macro_f1', val_macro_f1, epoch)
                for idx in range(num_classes):
                    tb_writer.add_scalar('val_recall/{}'.format(class_names[idx]), recall_per_class[idx], epoch)
        

if __name__ == '__main__':
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i in range(1):
        is_test = False

        parser = argparse.ArgumentParser()
        parser.add_argument('--train_save', type = str, default = f'./log/ADD/{timestamp}/{i}')
        parser.add_argument('--fold', type = int, default = i)
        parser.add_argument('--batch_size', type = int, default = 16)   
        parser.add_argument('--epochs', type = int, default = 200)      
        parser.add_argument('--device', default = 'cuda:0', help = 'device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--tau', type=float, default=4.0, help='base temperature for logit standardization')
        parser.add_argument('--eps', type=float, default=1e-6, help='epsilon to avoid zero std in logit standardization')
        parser.add_argument("--high_thre", default = 0.7, type = float, help = "high_bkg_score")
        parser.add_argument("--low_thre", default = 0.3, type = float, help = "low_bkg_score")
        parser.add_argument("--bkg_thre", default = 0.5, type = float, help = "bkg_score")
        parser.add_argument("--ignore_index", default = 255, type = int, help = "random index")
        opt = parser.parse_args()
        
        if os.path.exists(opt.train_save + '/run/') is False:
            os.makedirs(opt.train_save + '/run/')
        if os.path.exists(opt.train_save + '/weights/') is False:
            os.makedirs(opt.train_save + '/weights/')    
        logging.basicConfig(filename=opt.train_save + '/train_log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        dataset = MultiClassPairDataset(root_dir='./my_dataset', split='train', enable_aug=True, target_size=448)
        val_dataset = MultiClassPairDataset(root_dir='./my_dataset', split='val', enable_aug=False, target_size=448)
        class_names = [name for name, idx in sorted(val_dataset.class_map.items(), key=lambda x: x[1])]
        loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True)

        ce_loss = nn.CrossEntropyLoss()
        sim_loss = torch.nn.CosineEmbeddingLoss()
        
        # 两次独立蒸馏，分别写入子目录以便对比
        orig_train_save = opt.train_save

        # 第一次：NBI -> WLI
        mod_name = 'NBI2WLI'
        mod_save = os.path.join(orig_train_save, mod_name)
        os.makedirs(mod_save + '/run/', exist_ok=True)
        os.makedirs(mod_save + '/weights/', exist_ok=True)
        opt.train_save = mod_save
        logfilename = opt.train_save + '/train_log.log'
        logging.basicConfig(filename=logfilename,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
        tb_writer = SummaryWriter(opt.train_save + '/run/')
        student = resnet50_w(pretrained=True, num_classes=4).to(opt.device)
        teacher = resnet50(pretrained=False, num_classes=4).to(opt.device)
        embed_layer_ = embed_layer().to(opt.device)
        teacher, student = load_pretrained(teacher, student, scratch=False, teacher_modality='NBI', student_modality='WLI')
        train(teacher, student, embed_layer_, class_names=class_names, is_test=is_test, epochs=opt.epochs, teacher_modality='NBI', student_modality='WLI')

        # 第二次：WLI -> NBI
        mod_name = 'WLI2NBI'
        mod_save = os.path.join(orig_train_save, mod_name)
        os.makedirs(mod_save + '/run/', exist_ok=True)
        os.makedirs(mod_save + '/weights/', exist_ok=True)
        opt.train_save = mod_save
        logfilename = opt.train_save + '/train_log.log'
        logging.basicConfig(filename=logfilename,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
        tb_writer = SummaryWriter(opt.train_save + '/run/')
        student2 = resnet50_w(pretrained=True, num_classes=4).to(opt.device)
        teacher2 = resnet50(pretrained=False, num_classes=4).to(opt.device)
        embed_layer_2 = embed_layer().to(opt.device)
        teacher2, student2 = load_pretrained(teacher2, student2, scratch=False, teacher_modality='WLI', student_modality='NBI')
        train(teacher2, student2, embed_layer_2, class_names=class_names, is_test=is_test, epochs=opt.epochs, teacher_modality='WLI', student_modality='NBI')

        # 恢复原始路径
        opt.train_save = orig_train_save
