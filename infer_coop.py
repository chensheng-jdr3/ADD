# 最简单的对称蒸馏联合推理测试，就只是加权融合了两个模型的logit层

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataloader.loader import MultiClassPairDataset
from lib import resnet50_w
from tqdm import tqdm


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


def load_model(path, device, num_classes=4):
    # `from lib import resnet50_w` imports the constructor function as `resnet50_w`
    model = resnet50_w(pretrained=False, num_classes=num_classes)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def infer(args):
    device = torch.device(args.device)
    dataset = MultiClassPairDataset(root_dir=args.root, split=args.split, enable_aug=False, target_size=448)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    class_names = [name for name, idx in sorted(dataset.class_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # load two student models (WLI / NBI)
    if not os.path.exists(args.wli_model):
        raise FileNotFoundError(f'WLI model not found: {args.wli_model}')
    if not os.path.exists(args.nbi_model):
        raise FileNotFoundError(f'NBI model not found: {args.nbi_model}')

    model_wli = load_model(args.wli_model, device, num_classes=num_classes)
    model_nbi = load_model(args.nbi_model, device, num_classes=num_classes)

    all_preds = []
    all_labels = []

    alpha = float(args.alpha)
    T = float(args.temperature)

    for batch in tqdm(loader, desc='Infer'):
        wli_tensor, nbi_tensor, label, wli_path, nbi_path, patient_name, set_prop = batch
        # inputs shape: (1, C, H, W) after dataset
        wli = wli_tensor.to(device).float()
        nbi = nbi_tensor.to(device).float()
        label = label.numpy().tolist()

        with torch.no_grad():
            logits_wli = model_wli(wli)[0]
            logits_nbi = model_nbi(nbi)[0]

            # temperature scaled softmax
            probs_wli = F.softmax(logits_wli / T, dim=1)
            probs_nbi = F.softmax(logits_nbi / T, dim=1)

            fused = alpha * probs_wli + (1.0 - alpha) * probs_nbi
            pred = torch.argmax(fused, dim=1).cpu().numpy().tolist()

        all_preds.extend(pred)
        all_labels.extend(label)

    # metrics
    cm = compute_confusion_matrix(all_labels, all_preds, num_classes)
    recall_per_class, f1_per_class, macro_f1 = compute_metrics_from_confusion_matrix(cm)
    acc = float(np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)) if len(all_labels) > 0 else 0.0

    print('Results:')
    print(f'  Samples: {len(all_labels)}')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Macro F1: {macro_f1:.4f}')
    for idx in range(num_classes):
        print(f'  {class_names[idx]} - Recall: {recall_per_class[idx]:.4f}, F1: {f1_per_class[idx]:.4f}')
    print('Confusion matrix:\n', cm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./my_dataset', help='dataset root')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='which split to run')
    parser.add_argument('--wli_model', default='./pretrained/0_student_WLI.pth', help='path to WLI student model')
    parser.add_argument('--nbi_model', default='./pretrained/0_student_NBI.pth', help='path to NBI student model')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for WLI probs in fusion')
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature')
    parser.add_argument('--device', default='cpu', help='device, e.g. cuda:0 or cpu')
    args = parser.parse_args()
    infer(args)
