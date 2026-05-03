import os
import torch
import sys
import argparse
import logging
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader.loader import MultiClassPairDataset, CPCDataset
from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from lib import resnet50, resnet50_w, embed_layer, PSR, refine_cams_with_bkg, SR_generation
from datetime import datetime


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


def curriculum_lambda(epoch, total_epochs, max_lambda=1.0):
    # Cosine ramp from easy (0) to hard (max_lambda)
    if total_epochs <= 1:
        return float(max_lambda)
    progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
    return float(max_lambda) * 0.5 * (1.0 - np.cos(np.pi * progress))


class GlobalTemperature(nn.Module):
    def __init__(self, tau_min=1.0, tau_max=8.0, init_tau=4.0):
        super().__init__()
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        init_ratio = (float(init_tau) - self.tau_min) / max(self.tau_max - self.tau_min, 1e-6)
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        init_logit = np.log(init_ratio / (1.0 - init_ratio))
        self.raw = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))

    def forward(self, batch_size):
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.raw)
        return tau.expand(batch_size, 1)


class InstanceTemperature(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, tau_min=1.0, tau_max=8.0):
        super().__init__()
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        tau01 = torch.sigmoid(self.mlp(x))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau01
        return tau


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
    macro_recall = float(np.mean(recall_per_class))
    macro_precision = float(np.mean(precision_per_class))
    macro_f1 = float(np.mean(f1_per_class))
    return recall_per_class, precision_per_class, f1_per_class, macro_recall, macro_precision, macro_f1


def load_pretrained(teacher, student, scratch, teacher_modality='NBI', student_modality='WLI'):
    # prefer modality-specific pretrained files, fallback to legacy names
    stu_path_mod = f'./pretrained/{opt.dataset}_student_{student_modality}_{opt.fold}.pth'
    tea_path_mod = f'./pretrained/{opt.dataset}_teacher_{teacher_modality}_{opt.fold}.pth'
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
        pretrained_path = f'./pretrained/{opt.dataset}_student_{student_modality}_{opt.fold}.pth'
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


def register_backbone_feature_hooks(model, feature_cache, prefix='student'):
    handles = []
    for layer_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        layer = getattr(model, layer_name)

        def _hook(_module, _input, output, name=layer_name):
            feature_cache[f'{prefix}_{name}'] = output

        handles.append(layer.register_forward_hook(_hook))
    return handles


def parse_distill_mask(mask_str: str):
    """Parse a 4-char binary mask string (left->right = layer1..layer4).

    Examples:
        '0101' -> [False, True, False, True]
        '1111' -> [True, True, True, True]
    """
    s = mask_str.strip()
    if len(s) != 4 or any(c not in '01' for c in s):
        raise ValueError("--distill_layers must be a 4-character binary string like '0101'")
    return [c == '1' for c in s]


def train(teacher, student, embed_layers, class_names, epochs=1000, is_test=True, teacher_modality='NBI', student_modality='WLI', enabled_layers=None):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_embed_layer = torch.optim.Adam(embed_layers.parameters(), lr=1e-4, weight_decay=1e-8)

    temp_module = None
    optimizer_temp = None
    if opt.ctkd_enable:
        num_classes = len(class_names)
        if opt.ctkd_mode == 'global':
            temp_module = GlobalTemperature(
                tau_min=opt.ctkd_tau_min,
                tau_max=opt.ctkd_tau_max,
                init_tau=opt.tau,
            ).to(opt.device)
        else:
            in_dim = num_classes if opt.ctkd_instance_input == 'teacher' else (num_classes * 2)
            temp_module = InstanceTemperature(
                in_dim=in_dim,
                hidden_dim=opt.ctkd_hidden,
                tau_min=opt.ctkd_tau_min,
                tau_max=opt.ctkd_tau_max,
            ).to(opt.device)
        optimizer_temp = torch.optim.Adam(temp_module.parameters(), lr=opt.ctkd_lr, weight_decay=0.0)

    psr = PSR(num_iter=10, dilations=[1,2,4,8])
    psr.to(opt.device)
    feature_cache = {}
    teacher_hook_handles = register_backbone_feature_hooks(teacher, feature_cache, prefix='teacher')
    student_hook_handles = register_backbone_feature_hooks(student, feature_cache, prefix='student')
    layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
    if enabled_layers is None:
        enabled_layers = [True, True, True, True]
    else:
        if len(enabled_layers) != 4:
            raise ValueError('enabled_layers must be a list of 4 booleans')
    
    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    val_acc_best = 0
    val_macro_f1_best = 0
    best_model_epoch = 0
    num_classes = len(class_names)
    try:
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
                    wli_img = batch[0].to(opt.device).float()
                    nbi_img = batch[1].to(opt.device).float()
                    label = batch[2].to(opt.device).long()

                    if teacher_modality.upper() == 'NBI' and student_modality.upper() == 'WLI':
                        teacher_input = nbi_img
                        student_input = wli_img
                    elif teacher_modality.upper() == 'WLI' and student_modality.upper() == 'NBI':
                        teacher_input = wli_img
                        student_input = nbi_img
                    else:
                        teacher_input = nbi_img
                        student_input = wli_img

                    pred_tea, _ = teacher(teacher_input)
                    CE_loss, acc_wli, pred_stu, _, _ = get_ce_loss(student_input, label, student)
                    teacher_features = {name: feature_cache[f'teacher_{name}'] for name in layer_names}
                    student_features = {name: feature_cache[f'student_{name}'] for name in layer_names}

                    if phase == 'train':
                        optimizer_stu.zero_grad()
                        optimizer_embed_layer.zero_grad()
                        if optimizer_temp is not None:
                            optimizer_temp.zero_grad()

                        # ── Logit Standardization (independent of CTKD) ──
                        if opt.ls_enable:
                            logits_t = logit_standardization(pred_tea.detach(), tau=1.0, eps=opt.eps)
                            logits_s = logit_standardization(pred_stu, tau=1.0, eps=opt.eps)
                        else:
                            logits_t = pred_tea.detach()
                            logits_s = pred_stu

                        # ── Temperature strategy ──
                        if opt.ctkd_enable:
                            curr_lambd = curriculum_lambda(
                                epoch=epoch,
                                total_epochs=epochs,
                                max_lambda=opt.ctkd_lambda_max,
                            )
                            if opt.ctkd_mode == 'global':
                                tau = temp_module(batch_size=pred_stu.shape[0])
                            else:
                                tea_logits_raw = pred_tea.detach()
                                if opt.ctkd_instance_input == 'teacher':
                                    tau_input = tea_logits_raw
                                else:
                                    tau_input = torch.cat([tea_logits_raw, pred_stu.detach()], dim=1)
                                tau = temp_module(tau_input)

                            tau = torch.clamp(tau, min=opt.ctkd_tau_min, max=opt.ctkd_tau_max)
                            tau_adv = grad_reverse(tau, curr_lambd)
                            p_t = F.softmax(logits_t / tau_adv, dim=1)
                            log_p_s = F.log_softmax(logits_s / tau_adv, dim=1)
                            tau2 = (tau_adv * tau_adv).mean()
                            logit_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * tau2
                        else:
                            p_t = F.softmax(logits_t / opt.tau, dim=1)
                            log_p_s = F.log_softmax(logits_s / opt.tau, dim=1)
                            logit_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (opt.tau * opt.tau)

                        space_loss = torch.zeros(1).to(opt.device)
                        loss_add_by_scale = {name: torch.zeros(1).to(opt.device) for name in layer_names}
                        if epoch > 0 and torch.isfinite(pred_tea).all() and torch.isfinite(pred_stu).all():
                            for idx, layer_name in enumerate(layer_names):
                                if not enabled_layers[idx]:
                                    continue
                                s_feat = student_features[layer_name]
                                t_feat = teacher_features[layer_name]
                                if not (torch.isfinite(s_feat).all() and torch.isfinite(t_feat).all()):
                                    continue

                                try:
                                    cam_s = cam(s_feat, pred_stu, label)
                                    cam_t = cam(t_feat, pred_tea, label)
                                except RuntimeError:
                                    continue
                                pseudo_label_s = refine_cams_with_bkg(psr, student_input, cams=cam_s, cfg=opt)
                                pseudo_label_t = refine_cams_with_bkg(psr, teacher_input, cams=cam_t, cfg=opt)

                                q_t = embed_layers[layer_name](t_feat.detach())
                                q_s = embed_layers[layer_name](s_feat)
                                loss_scale_1 = ADD(pseudo_label_s, pseudo_label_t, q_t, q_s)
                                loss_scale_2 = ADD(pseudo_label_t, pseudo_label_s, q_s, q_t)
                                loss_add_by_scale[layer_name] = loss_scale_1 + loss_scale_2
                                space_loss = space_loss + loss_add_by_scale[layer_name]

                        loss = CE_loss + space_loss + logit_loss
                        if not torch.isfinite(loss):
                            print('WARNING: non-finite loss, ending training ', loss)
                            sys.exit(1)

                        loss.backward()
                        optimizer_embed_layer.step()
                        optimizer_stu.step()
                        if optimizer_temp is not None:
                            optimizer_temp.step()
                        summary.append((
                            loss.item(),
                            CE_loss.item(),
                            logit_loss.item(),
                            space_loss.item(),
                            loss_add_by_scale['layer1'].item(),
                            loss_add_by_scale['layer2'].item(),
                            loss_add_by_scale['layer3'].item(),
                            loss_add_by_scale['layer4'].item(),
                        ))
                        train_acc_num = train_acc_num + acc_wli
                        sample_num += student_input.shape[0]
                    else:
                        sample_num += student_input.shape[0]
                        val_acc_num = val_acc_num + acc_wli
                        pred_label = torch.argmax(pred_stu, dim=-1)
                        val_preds.extend(pred_label.detach().cpu().numpy().tolist())
                        val_labels.extend(label.detach().cpu().numpy().tolist())

                if len(summary) > 0:
                    summary = np.array(summary).mean(axis=0)

                if phase == 'train':
                    train_acc = train_acc_num / sample_num
                    tags = [
                        'acc_wli',
                        'train_total_loss',
                        'CE_loss',
                        'logit_loss',
                        'space_loss',
                        'loss_ADD_layer1',
                        'loss_ADD_layer2',
                        'loss_ADD_layer3',
                        'loss_ADD_layer4',
                    ]
                    tb_writer.add_scalar(tags[0], train_acc, epoch)
                    tb_writer.add_scalar(tags[1], summary[0], epoch)
                    tb_writer.add_scalar(tags[2], summary[1], epoch)
                    tb_writer.add_scalar(tags[3], summary[2], epoch)
                    tb_writer.add_scalar(tags[4], summary[3], epoch)
                    tb_writer.add_scalar(tags[5], summary[4], epoch)
                    tb_writer.add_scalar(tags[6], summary[5], epoch)
                    tb_writer.add_scalar(tags[7], summary[6], epoch)
                    tb_writer.add_scalar(tags[8], summary[7], epoch)
                    if opt.ctkd_enable:
                        tb_writer.add_scalar('ctkd/lambda', curr_lambd, epoch)
                        tb_writer.add_scalar('ctkd/tau_mean', tau.detach().mean().item(), epoch)
                        tb_writer.add_scalar('ctkd/tau_min', tau.detach().min().item(), epoch)
                        tb_writer.add_scalar('ctkd/tau_max', tau.detach().max().item(), epoch)
                    print('##############################################################################')
                    print('[TRAIN] Epoch %d' % epoch, 'acc: %0.2f, Total_loss: %0.2f, CE_loss: %0.2f, logit_loss: %0.2f, space_loss: %0.2f, ADD_l1: %0.2f, ADD_l2: %0.2f, ADD_l3: %0.2f, ADD_l4: %0.2f' % (train_acc, summary[0], summary[1], summary[2], summary[3], summary[4], summary[5], summary[6], summary[7]))
                else:
                    val_acc = float(val_acc_num / sample_num)
                    confusion_matrix = compute_confusion_matrix(val_labels, val_preds, num_classes)
                    recall_per_class, precision_per_class, f1_per_class, macro_recall, macro_precision, val_macro_f1 = compute_metrics_from_confusion_matrix(confusion_matrix)

                    is_new_best = val_macro_f1 > val_macro_f1_best
                    if is_new_best:
                        val_macro_f1_best = val_macro_f1
                        val_acc_best = val_acc
                        best_model_epoch = epoch
                        save_model(epoch, student, opt.train_save, student_modality=student_modality)

                    brief = (
                        f"[EVAL] Epoch {epoch:3d} | acc={val_acc:.4f} | macro_f1={val_macro_f1:.4f} | "
                        f"best_f1={val_macro_f1_best:.4f} (epoch {best_model_epoch})"
                    )
                    print(brief)
                    logging.info(brief)

                    if is_new_best:
                        header = f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}"
                        sep = "-" * 44
                        rows = []
                        for idx in range(num_classes):
                            rows.append(
                                f"{class_names[idx]:<12} {precision_per_class[idx]:>10.4f} "
                                f"{recall_per_class[idx]:>10.4f} {f1_per_class[idx]:>10.4f}"
                            )
                        macro_row = (
                            f"{'Macro Avg':<12} {macro_precision:>10.4f} "
                            f"{macro_recall:>10.4f} {val_macro_f1:>10.4f}"
                        )
                        detail = "\n".join([
                            f"[EVAL] New Best at Epoch {epoch}",
                            f"  Overall Accuracy = {val_acc:.4f}",
                            sep,
                            header,
                            sep,
                            *rows,
                            sep,
                            macro_row,
                            sep,
                            f"  Confusion Matrix:",
                            f"{confusion_matrix}",
                        ])
                        print(detail)
                        logging.info('\n' + detail)

                    tb_writer.add_scalar('val_acc', val_acc, epoch)
                    tb_writer.add_scalar('val_macro_f1', val_macro_f1, epoch)
                    tb_writer.add_scalar('val_macro_recall', macro_recall, epoch)
                    tb_writer.add_scalar('val_macro_precision', macro_precision, epoch)
                    for idx in range(num_classes):
                        tb_writer.add_scalar('val_recall/{}'.format(class_names[idx]), recall_per_class[idx], epoch)
                        tb_writer.add_scalar('val_precision/{}'.format(class_names[idx]), precision_per_class[idx], epoch)
    finally:
        for handle in teacher_hook_handles + student_hook_handles:
            handle.remove()


if __name__ == '__main__':
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i in range(1):
        is_test = False

        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='my_dataset', choices=['my_dataset', 'cpc_paired'],
                            help="'my_dataset' = MultiClassPairDataset, 'cpc_paired' = CPCDataset (binary)")
        parser.add_argument('--train_save', type=str, default='', help='override default save path (empty = auto)')
        parser.add_argument('--fold', type=int, default=i)
        parser.add_argument('--batch_size', type = int, default = 16)   
        parser.add_argument('--epochs', type = int, default = 200)      
        parser.add_argument('--device', default = 'cuda:0', help = 'device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument('--distill_layers', default='0001', help="4-bit mask (left->right = layer1..layer4), e.g. '0101' enables layer2 and layer4")
        parser.add_argument('--tau', type=float, default=1, help='base temperature for logit standardization')
        parser.add_argument('--eps', type=float, default=1e-6, help='epsilon to avoid zero std in logit standardization')
        parser.add_argument('--ls', default=True, action=argparse.BooleanOptionalAction, dest='ls_enable', help='logit standardization (Z-score)')
        parser.add_argument('--ctkd', default=True, action=argparse.BooleanOptionalAction, dest='ctkd_enable', help='curriculum temperature KD (GRL + adaptive tau)')
        parser.add_argument('--ctkd_mode', type=str, default='instatnce', choices=['global', 'instance'], help='temperature module type')
        parser.add_argument('--ctkd_tau_min', type=float, default=1.0, help='minimum adaptive temperature')
        parser.add_argument('--ctkd_tau_max', type=float, default=8.0, help='maximum adaptive temperature')
        parser.add_argument('--ctkd_lambda_max', type=float, default=1.0, help='maximum curriculum coefficient for adversarial temperature')
        parser.add_argument('--ctkd_hidden', type=int, default=64, help='hidden size for instance-level temperature MLP')
        parser.add_argument('--ctkd_lr', type=float, default=1e-4, help='learning rate for temperature module')
        parser.add_argument('--ctkd_instance_input', type=str, default='both', choices=['teacher', 'both'], help='input logits for instance-level temperature predictor')
        parser.add_argument("--high_thre", default = 0.7, type = float, help = "high_bkg_score")
        parser.add_argument("--low_thre", default = 0.3, type = float, help = "low_bkg_score")
        parser.add_argument("--bkg_thre", default = 0.5, type = float, help = "bkg_score")
        parser.add_argument("--ignore_index", default = 255, type = int, help = "random index")
        opt = parser.parse_args()
        if not opt.train_save:
            opt.train_save = f'./log/ADD/{opt.dataset}/{timestamp}/{i}'
        enabled_layers = parse_distill_mask(opt.distill_layers)

        if os.path.exists(opt.train_save + '/run/') is False:
            os.makedirs(opt.train_save + '/run/')
        if os.path.exists(opt.train_save + '/weights/') is False:
            os.makedirs(opt.train_save + '/weights/')
        logging.basicConfig(filename=opt.train_save + '/train_log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        if opt.dataset == 'cpc_paired':
            dataset = CPCDataset(is_train=True, split_id=opt.fold, enable_aug=True)
            val_dataset = CPCDataset(is_train=False, split_id=opt.fold, enable_aug=False)
            class_names = ['hyperplastic', 'adenomas']
            num_classes = 2
        else:
            dataset = MultiClassPairDataset(root_dir=f'./{opt.dataset}', split='train', enable_aug=True, target_size=448)
            val_dataset = MultiClassPairDataset(root_dir=f'./{opt.dataset}', split='val', enable_aug=False, target_size=448)
            class_names = val_dataset.class_names
            num_classes = val_dataset.num_classes
        loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True)

        ce_loss = nn.CrossEntropyLoss()
        sim_loss = torch.nn.CosineEmbeddingLoss()

        # NBI -> WLI 蒸馏
        mod_save = opt.train_save
        os.makedirs(mod_save + '/run/', exist_ok=True)
        os.makedirs(mod_save + '/weights/', exist_ok=True)
        logfilename = mod_save + '/train_log.log'
        logging.basicConfig(filename=logfilename,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
        tb_writer = SummaryWriter(mod_save + '/run/')
        student = resnet50_w(pretrained=True, num_classes=num_classes).to(opt.device)
        teacher = resnet50(pretrained=False, num_classes=num_classes).to(opt.device)
        embed_layer_ = nn.ModuleDict({
            'layer1': embed_layer(in_channels=256).to(opt.device),
            'layer2': embed_layer(in_channels=512).to(opt.device),
            'layer3': embed_layer(in_channels=1024).to(opt.device),
            'layer4': embed_layer(in_channels=2048).to(opt.device),
        }).to(opt.device)
        teacher, student = load_pretrained(teacher, student, scratch=False, teacher_modality='NBI', student_modality='WLI')
        train(teacher, student, embed_layer_, class_names=class_names, is_test=is_test, epochs=opt.epochs, teacher_modality='NBI', student_modality='WLI', enabled_layers=enabled_layers)
