import os
import torch
import argparse
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader.loader import MultiClassPairDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from lib import resnet50
from datetime import datetime


def compute_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def compute_metrics_from_confusion_matrix(confusion_matrix):
    # recall_i = TP_i / (TP_i + FN_i), row-wise over true labels
    row_sum = confusion_matrix.sum(axis=1)
    diagonal = np.diag(confusion_matrix)
    recall_per_class = np.divide(
        diagonal,
        row_sum,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=row_sum != 0
    )

    # precision_i = TP_i / (TP_i + FP_i), column-wise over predicted labels
    col_sum = confusion_matrix.sum(axis=0)
    precision_per_class = np.divide(
        diagonal,
        col_sum,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=col_sum != 0
    )

    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=(precision_per_class + recall_per_class) != 0
    )
    macro_recall = float(np.mean(recall_per_class))
    macro_precision = float(np.mean(precision_per_class))
    macro_f1 = float(np.mean(f1_per_class))
    return recall_per_class, precision_per_class, f1_per_class, macro_recall, macro_precision, macro_f1


def get_ce_loss(img, label, network):
    pred,_ = network(img)
    errCE = ce_loss(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float()
    return errCE, acc, pred_label


def save_model(epoch, teacher, train_save):
    print('update model..')
    tea_path = train_save + '/weights/teacher_model-{}.pth'.format(epoch)
    torch.save(teacher.state_dict(), tea_path)


def train(teacher, class_names, epochs=400, is_test=True, loader=None, val_loader=None, i=0, modality='NBI', train_save='./log/teacher', tb_writer=None):
    optimizer_tea = torch.optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-8)
    val_acc_best = 0
    val_macro_f1_best = 0
    best_model_epoch = 0
    best_model_path = './pretrained/{}_teacher_{}_{}.pth'.format(opt.dataset_name, modality, i)
    # 混淆矩阵不再另外保存为文件，改为写入日志
    num_classes = len(class_names)
    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    for epoch in range(1, epochs):
        for phase in iter(phases):
            if phase == 'train':
                teacher.train()
                ldr = loader
                train_acc_num=0
            else:
                teacher.eval()
                ldr = val_loader
                val_acc_num=0
                val_preds = []
                val_labels = []
            summary = []
            sample_num = 0
            for batch in tqdm(ldr, leave=False):
                # 根据模态选择输入：NBI 使用 batch[1], WLI 使用 batch[0]
                if modality.upper() == 'WLI':
                    img = batch[0].to(opt.device).float()
                    label = batch[2].to(opt.device).long()
                else:
                    img = batch[1].to(opt.device).float()
                    label = batch[2].to(opt.device).long()
                optimizer_tea.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    CE_loss, acc, pred_label = get_ce_loss(img, label, teacher)
                if phase == 'train':
                    CE_loss.backward()
                    optimizer_tea.step()
                    train_acc_num=train_acc_num+acc
                    sample_num += img.shape[0]
                else:
                    sample_num += img.shape[0]
                    val_acc_num=val_acc_num+acc
                    val_preds.extend(pred_label.detach().cpu().numpy().tolist())
                    val_labels.extend(label.detach().cpu().numpy().tolist())
                summary.append((CE_loss.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                train_acc=train_acc_num / sample_num
                print('##############################################################################')
                print('[TRAIN] Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f' % (summary, train_acc))
                tags = ['train_total_loss',  'acc']
                if tb_writer is not None:
                    tb_writer.add_scalar(tags[0], summary, epoch)
                    tb_writer.add_scalar(tags[1], train_acc, epoch)
                #save_model(epoch)
            else:
                val_acc=val_acc_num / sample_num
                val_acc = float(val_acc)

                confusion_matrix = compute_confusion_matrix(val_labels, val_preds, num_classes)
                recall_per_class, precision_per_class, f1_per_class, macro_recall, macro_precision, val_macro_f1 = compute_metrics_from_confusion_matrix(confusion_matrix)

                is_new_best = val_macro_f1 > val_macro_f1_best
                if is_new_best:
                    val_macro_f1_best = val_macro_f1
                    val_acc_best = val_acc
                    best_model_epoch = epoch
                    save_model(epoch, teacher, train_save)   #log中存储
                    torch.save(teacher.state_dict(), best_model_path)   #与train.py配合使用

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

                if tb_writer is not None:
                    tb_writer.add_scalar('val_acc', val_acc, epoch)
                    tb_writer.add_scalar('val_macro_f1', val_macro_f1, epoch)
                    tb_writer.add_scalar('val_macro_recall', macro_recall, epoch)
                    tb_writer.add_scalar('val_macro_precision', macro_precision, epoch)
                    for idx in range(num_classes):
                        tb_writer.add_scalar('val_recall/{}'.format(class_names[idx]), recall_per_class[idx], epoch)
                        tb_writer.add_scalar('val_precision/{}'.format(class_names[idx]), precision_per_class[idx], epoch)
                # logging.info('#############################################################################')
    
    



if __name__ == '__main__':
    
    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i in range(1):
        is_test = False  
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_name', type=str, default='my_dataset', help='dataset identifier for model paths')
        parser.add_argument('--train_save', type=str, default='', help='override default save path (empty = auto)')
        parser.add_argument('--fold', type=int, default=i)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
        opt = parser.parse_args()
        if not opt.train_save:
            opt.train_save = f'./log/teacher/{opt.dataset_name}/{timestamp}/{i}'
        
        dataset = MultiClassPairDataset(root_dir=f'./{opt.dataset_name}', split='train', enable_aug=True, target_size=448)
        val_dataset = MultiClassPairDataset(root_dir=f'./{opt.dataset_name}', split='val', enable_aug=False, target_size=448)
        class_names = val_dataset.class_names
        num_classes = val_dataset.num_classes
        loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=False, persistent_workers=True, pin_memory=True)

        ce_loss = nn.CrossEntropyLoss()

        # 训练 NBI 教师模型（用于 NBI→WLI 蒸馏）
        mod_save = os.path.join(opt.train_save, 'NBI')
        os.makedirs(mod_save + '/run/', exist_ok=True)
        os.makedirs(mod_save + '/weights/', exist_ok=True)

        logfilename = mod_save + '/train_log.log'
        logging.basicConfig(filename=logfilename,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True)
        tb_writer = SummaryWriter(mod_save + '/run/')

        teacher = resnet50(pretrained=True, num_classes=num_classes).to(opt.device)
        train(teacher, class_names=class_names, is_test=is_test, epochs=opt.epochs, loader=loader, val_loader=val_loader, i=i, modality='NBI', train_save=mod_save, tb_writer=tb_writer)

