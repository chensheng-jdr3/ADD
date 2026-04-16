"""
Route 1: ASF-based multimodal fusion on top of ADD-trained backbones.

Loads frozen NBI (teacher, ~80%) and WLI (student, ~60%) models,
then trains an Attention-based modality Shifting Fusion (ASF) module
to fuse their features for joint inference.

Usage examples:
    # basic: no adapter, no MHSA
    python train_fusion.py --fold 0 --nbi_ckpt ./pretrained/0_teacher.pth --wli_ckpt ./pretrained/0_student.pth

    # with projection adapter
    python train_fusion.py --fold 0 --nbi_ckpt ... --wli_ckpt ... --proj_adapter

    # with MHSA
    python train_fusion.py --fold 0 --nbi_ckpt ... --wli_ckpt ... --use_mhsa --mhsa_heads 3
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from dataloader.loader import MultiClassPairDataset
from lib import resnet50, resnet50_w


# ---------------------------------------------------------------------------
# ASF Fusion Module
# ---------------------------------------------------------------------------

class MultiHeadAttention2(nn.Module):
    """Minimal MHSA for sequence length 2.

    Takes [B, 2, d], applies multi-head self-attention, returns [B, d]
    (mean-pooled over the 2 tokens).
    """

    def __init__(self, dim=256, num_heads=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f'dim={dim} not divisible by heads={num_heads}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, 2, d]
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.proj(out)
        return out.mean(dim=1)  # [B, D]


class ASFFusion(nn.Module):
    """Attention-based modality Shifting Fusion.

    NBI (teacher) feature serves as the base; WLI (student) feature
    generates a displacement vector that shifts the teacher feature.

    Forward returns fused feature z: [B, 256].
    """

    def __init__(self, in_dim=2048, dim=256, gate_hidden=256, disp_hidden=512,
                 use_mhsa=False, mhsa_heads=3, use_adapter=False, adapter_hidden=256):
        super().__init__()
        self.use_mhsa = use_mhsa

        # Input projection: either bottleneck adapter or simple linear
        if use_adapter:
            # Bottleneck: in_dim -> adapter_hidden -> in_dim (residual) -> dim
            self.t_proj = nn.Sequential(
                nn.Linear(in_dim, adapter_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(adapter_hidden, in_dim),
                nn.Linear(in_dim, dim),
            )
            self.s_proj = nn.Sequential(
                nn.Linear(in_dim, adapter_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(adapter_hidden, in_dim),
                nn.Linear(in_dim, dim),
            )
        else:
            # Simple linear: in_dim -> dim
            self.t_proj = nn.Linear(in_dim, dim)
            self.s_proj = nn.Linear(in_dim, dim)

        if use_mhsa:
            self.mhsa = MultiHeadAttention2(dim=dim, num_heads=mhsa_heads)

        # Gate: concat(teacher_feat, student_feat) -> gate_hidden (sigmoid)
        # change gate to output a scalar per sample (alpha) to avoid per-dim destructive scaling
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid(),
        )

        # Displacement: student_feat -> disp_hidden -> dim
        self.disp_fc = nn.Sequential(
            nn.Linear(dim, disp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(disp_hidden, dim),
        )

    def forward(self, t_feat, s_feat, lambda_theta=0.1):
        """
        Args:
            t_feat: teacher (NBI) feature [B, in_dim]
            s_feat: student (WLI) feature [B, in_dim]
            lambda_theta: scaling factor threshold (TelME Eq.19)
        Returns:
            z: fused feature [B, dim]
        """
        t_feat = self.t_proj(t_feat)  # [B, dim]
        s_feat = self.s_proj(s_feat)  # [B, dim]

        if self.use_mhsa:
            # Stack teacher + student as length-2 sequence
            tokens = torch.stack([t_feat, s_feat], dim=1)  # [B, 2, dim]
            s_feat = self.mhsa(tokens)                       # [B, dim]

        gate = self.gate_fc(torch.cat([t_feat, s_feat], dim=1))  # [B, dim]

        # Displacement: normalize to avoid unbounded magnitude (prevents cheating by scaling)
        disp = self.disp_fc(s_feat)                               # [B, dim]
        disp_norm = torch.norm(disp, p=2, dim=1, keepdim=True)
        disp = disp / (disp_norm + 1e-8)

        H = gate * disp                                           # gated displacement (bounded)

        # Lambda scaling (TelME Eq.19): clamp displacement magnitude relative to teacher
        t_norm = torch.norm(t_feat, p=2, dim=1, keepdim=True)
        H_norm = torch.norm(H, p=2, dim=1, keepdim=True)
        lam = torch.clamp(t_norm / (H_norm + 1e-8) * lambda_theta, max=1.0)

        z = t_feat + lam * H                                      # [B, dim]

        # Also return diagnostics when requested via attribute (non-breaking):
        # store last forward stats for external logging
        # compute cosine mean between fused z and projected teacher t_feat
        z_n = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        t_proj_n = t_feat / (t_feat.norm(dim=1, keepdim=True) + 1e-8)
        cos_mean = (z_n * t_proj_n).sum(dim=1).mean().detach().cpu()

        self._last_stats = {
            'lam': lam.detach().cpu(),
            'gate_mean': gate.detach().cpu().mean(),
            'H_norm_mean': H_norm.detach().cpu().mean(),
            't_norm_mean': t_norm.detach().cpu().mean(),
            'cos_mean': cos_mean,
        }

        return z


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_global_feature(model, img, backbone_name='resnet50_w'):
    """Forward a backbone and return (logits, global_pooled_feature).

    Uses forward hooks to capture layer4 output, then applies avgpool.
    """
    feat_holder = {}

    def _hook(_m, _inp, out):
        feat_holder['f'] = out

    handle = getattr(model, 'layer4').register_forward_hook(_hook)
    try:
        if backbone_name == 'resnet50':
            # resnet50 forward returns (logits, conv_feat, layer4_feat)
            outs = model(img)
            logits = outs[0]
        else:
            # resnet50_w forward returns (logits, layer4_feat)
            outs = model(img)
            logits = outs[0]
        f = feat_holder['f']
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)  # [B, 2048]
    finally:
        handle.remove()
    return logits, pooled


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def compute_metrics_from_confusion_matrix(confusion_matrix):
    row_sum = confusion_matrix.sum(axis=1)
    diagonal = np.diag(confusion_matrix)
    recall_per_class = np.divide(
        diagonal, row_sum,
        out=np.zeros_like(diagonal, dtype=np.float64), where=row_sum != 0,
    )
    col_sum = confusion_matrix.sum(axis=0)
    precision_per_class = np.divide(
        diagonal, col_sum,
        out=np.zeros_like(diagonal, dtype=np.float64), where=col_sum != 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(diagonal, dtype=np.float64),
        where=(precision_per_class + recall_per_class) != 0,
    )
    macro_f1 = float(np.mean(f1_per_class))
    return recall_per_class, f1_per_class, macro_f1


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_single_model(model, loader, device, class_names, backbone_name='resnet50_w'):
    """Evaluate a frozen backbone on the val set."""
    model.eval()
    num_classes = len(class_names)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Eval {backbone_name}', leave=False):
            wli_img, nbi_img, label = batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device).long()
            if backbone_name == 'resnet50':
                inp = nbi_img
            else:
                inp = wli_img
            logits, _ = extract_global_feature(model, inp, backbone_name)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(pred)
            all_labels.extend(label.cpu().numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    cm = compute_confusion_matrix(all_labels, all_preds, num_classes)
    recall, f1, macro_f1 = compute_metrics_from_confusion_matrix(cm)
    return acc, macro_f1, recall, f1, cm


def eval_fusion(model_nbi, model_wli, fusion_net, classifier, loader, device, class_names,
                lambda_theta=0.1):
    """Evaluate the full fusion pipeline."""
    model_nbi.eval()
    model_wli.eval()
    fusion_net.eval()
    classifier.eval()

    num_classes = len(class_names)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval fused', leave=False):
            wli_img = batch[0].to(device).float()
            nbi_img = batch[1].to(device).float()
            label = batch[2].to(device).long()

            _, nbi_feat = extract_global_feature(model_nbi, nbi_img, 'resnet50')
            _, wli_feat = extract_global_feature(model_wli, wli_img, 'resnet50_w')

            z = fusion_net(nbi_feat, wli_feat, lambda_theta=lambda_theta)
            logits = classifier(z)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(pred)
            all_labels.extend(label.cpu().numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    cm = compute_confusion_matrix(all_labels, all_preds, num_classes)
    recall, f1, macro_f1 = compute_metrics_from_confusion_matrix(cm)
    return acc, macro_f1, recall, f1, cm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fusion(model_nbi, model_wli, fusion_net, classifier,
                 train_loader, val_loader, class_names, device,
                 epochs=100, lr=1e-4, lambda_theta=0.1, w_cos=0.01,
                 train_save='./log/fusion'):
    # Collect all trainable parameters
    trainable_params = list(fusion_net.parameters()) + list(classifier.parameters())

    # compute class weights from training dataset to mitigate imbalance
    num_classes = len(class_names)
    labels = [s[2] for s in train_loader.dataset.samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # inverse frequency, normalized
    inv_freq = 1.0 / (counts + 1e-8)
    class_weights = inv_freq / (inv_freq.sum() + 1e-8) * float(num_classes)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

    # Freeze backbones
    model_nbi.eval()
    model_wli.eval()
    for p in model_nbi.parameters():
        p.requires_grad = False
    for p in model_wli.parameters():
        p.requires_grad = False

    best_macro_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        fusion_net.train()
        classifier.train()

        epoch_loss = 0.0
        epoch_acc_num = 0
        sample_num = 0

        # diagnostics accumulators
        lam_means = []
        gate_means = []
        H_norm_means = []
        t_norm_means = []
        cos_z_t = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            wli_img = batch[0].to(device).float()
            nbi_img = batch[1].to(device).float()
            label = batch[2].to(device).long()

            with torch.no_grad():
                _, nbi_feat = extract_global_feature(model_nbi, nbi_img, 'resnet50')
                _, wli_feat = extract_global_feature(model_wli, wli_img, 'resnet50_w')

            z = fusion_net(nbi_feat, wli_feat, lambda_theta=lambda_theta)
            logits = classifier(z)
            # collect diagnostics if present on fusion_net
            if hasattr(fusion_net, '_last_stats'):
                s = fusion_net._last_stats
                try:
                    lam_means.append(float(s['lam'].mean().item()))
                except Exception:
                    pass
                try:
                    gate_means.append(float(s['gate_mean'].item()))
                    H_norm_means.append(float(s['H_norm_mean'].item()))
                    t_norm_means.append(float(s['t_norm_mean'].item()))
                except Exception:
                    pass
                # cos similarity between z and t_feat
                try:
                    z_cpu = z.detach().cpu()
                    t_cpu = nbi_feat.detach().cpu()
                    # reduce to batch-level mean cosine
                    z_n = z_cpu / (z_cpu.norm(dim=1, keepdim=True) + 1e-8)
                    t_n = t_cpu / (t_cpu.norm(dim=1, keepdim=True) + 1e-8)
                    cos = (z_n * t_n).sum(dim=1).mean().item()
                    cos_z_t.append(cos)
                except Exception:
                    pass
            base_loss = ce_loss(logits, label)

            # cosine regularization: use fusion_net._last_stats['cos_mean'] (computed in forward)
            if hasattr(fusion_net, '_last_stats') and 'cos_mean' in fusion_net._last_stats:
                cos_mean = float(fusion_net._last_stats['cos_mean'].item())
            else:
                # fallback: project raw nbi_feat via fusion_net.t_proj to compute cosine
                try:
                    t_proj = fusion_net.t_proj(nbi_feat)
                    z_n = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                    t_n = t_proj / (t_proj.norm(dim=1, keepdim=True) + 1e-8)
                    cos_mean = (z_n * t_n).sum(dim=1).mean().item()
                except Exception:
                    cos_mean = 0.0
            cos_reg = (1.0 - cos_mean)

            total_loss = base_loss + w_cos * cos_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            pred_label = torch.argmax(logits, dim=1)
            epoch_acc_num += (pred_label == label).sum().item()
            sample_num += label.shape[0]

        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_acc_num / sample_num

        # ---- Eval ----
        # --- diagnostics logging ---
        if len(lam_means) > 0:
            lam_epoch_mean = float(np.mean(lam_means))
        else:
            lam_epoch_mean = 0.0
        gate_epoch_mean = float(np.mean(gate_means)) if len(gate_means) > 0 else 0.0
        H_norm_epoch_mean = float(np.mean(H_norm_means)) if len(H_norm_means) > 0 else 0.0
        t_norm_epoch_mean = float(np.mean(t_norm_means)) if len(t_norm_means) > 0 else 0.0
        cos_z_t_epoch_mean = float(np.mean(cos_z_t)) if len(cos_z_t) > 0 else 0.0
        tb_writer.add_scalar('diagnostics/lam_mean', lam_epoch_mean, epoch)
        tb_writer.add_scalar('diagnostics/gate_mean', gate_epoch_mean, epoch)
        tb_writer.add_scalar('diagnostics/H_norm_mean', H_norm_epoch_mean, epoch)
        tb_writer.add_scalar('diagnostics/t_norm_mean', t_norm_epoch_mean, epoch)
        tb_writer.add_scalar('diagnostics/cos_z_t_mean', cos_z_t_epoch_mean, epoch)

        acc, macro_f1, recall, f1, cm = eval_fusion(
            model_nbi, model_wli, fusion_net, classifier,
            val_loader, device, class_names, lambda_theta,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            # Save best model
            save_path = os.path.join(train_save, 'weights', 'best_fusion.pth')
            torch.save({
                'fusion_net': fusion_net.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'macro_f1': macro_f1,
            }, save_path)

        # TensorBoard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/acc', train_acc, epoch)
        tb_writer.add_scalar('val/acc', acc, epoch)
        tb_writer.add_scalar('val/macro_f1', macro_f1, epoch)
        for idx in range(len(class_names)):
            tb_writer.add_scalar(f'val/recall/{class_names[idx]}', recall[idx], epoch)

        print(f'[{epoch:3d}/{epochs}] '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
              f'val_acc={acc:.4f} val_macro_f1={macro_f1:.4f} '
              f'best_f1={best_macro_f1:.4f}@epoch{best_epoch}')
        logging.info(f'[{epoch:3d}/{epochs}] '
                     f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
                     f'val_acc={acc:.4f} val_macro_f1={macro_f1:.4f} '
                     f'best_f1={best_macro_f1:.4f}@epoch{best_epoch}')

    # Final comparison report
    print('\n===== Final Comparison =====')
    # NBI only
    nbi_acc, nbi_f1, nbi_rec, nbi_f1s, _ = eval_single_model(
        model_nbi, val_loader, device, class_names, 'resnet50')
    # WLI only
    wli_acc, wli_f1, wli_rec, wli_f1s, _ = eval_single_model(
        model_wli, val_loader, device, class_names, 'resnet50_w')
    # Fused (best checkpoint)
    ckpt = torch.load(os.path.join(train_save, 'weights', 'best_fusion.pth'), map_location=device)
    fusion_net.load_state_dict(ckpt['fusion_net'])
    classifier.load_state_dict(ckpt['classifier'])

    fus_acc, fus_f1, fus_rec, fus_f1s, fus_cm = eval_fusion(
        model_nbi, model_wli, fusion_net, classifier,
        val_loader, device, class_names, lambda_theta,
    )

    print(f'NBI only  -> acc={nbi_acc:.4f} macro_f1={nbi_f1:.4f}')
    print(f'WLI only  -> acc={wli_acc:.4f} macro_f1={wli_f1:.4f}')
    print(f'Fused     -> acc={fus_acc:.4f} macro_f1={fus_f1:.4f}')

    recall_msg = ', '.join([f'{class_names[idx]}:{fus_rec[idx]:.4f}' for idx in range(len(class_names))])
    f1_msg = ', '.join([f'{class_names[idx]}:{fus_f1s[idx]:.4f}' for idx in range(len(class_names))])
    print(f'Fused recall -> {recall_msg}')
    print(f'Fused f1     -> {f1_msg}')
    print(f'Fused CM:\n{fus_cm}')

    logging.info(f'NBI only -> acc={nbi_acc:.4f} macro_f1={nbi_f1:.4f}')
    logging.info(f'WLI only -> acc={wli_acc:.4f} macro_f1={wli_f1:.4f}')
    logging.info(f'Fused    -> acc={fus_acc:.4f} macro_f1={fus_f1:.4f}')
    logging.info(f'Fused recall -> {recall_msg}')
    logging.info(f'Fused f1     -> {f1_msg}')
    logging.info(f'Fused CM:\n{fus_cm}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0', help='e.g. cuda:0 or cpu')
    parser.add_argument('--root', default='./my_dataset', help='dataset root')

    # Model checkpoints
    parser.add_argument('--nbi_ckpt', type=str, default='/root/autodl-tmp/ADD/pretrained/0_teacher_NBI.pth',
                        help='path to frozen NBI teacher checkpoint')
    parser.add_argument('--wli_ckpt', type=str, default='/root/autodl-tmp/ADD/pretrained/0_student_WLI.pth',
                        help='path to frozen WLI student checkpoint')

    # ASF config
    parser.add_argument('--proj_adapter', action='store_true', default=True,
                        help='enable projection adapters before ASF')
    parser.add_argument('--use_mhsa', action='store_true', default=False,
                        help='enable multi-head self-attention over [NBI, WLI] tokens')
    parser.add_argument('--mhsa_heads', type=int, default=2,
                        help='number of attention heads (sequence length=2)')
    parser.add_argument('--lambda_theta', type=float, default=0.1,
                        help='scaling factor threshold (fixed, TelME uses 0.1 for MELD)')
    parser.add_argument('--w_cos', type=float, default=0,
                        help='weight for cosine regularization term (1 - cos(z, t))')
    parser.add_argument('--dim', type=int, default=128,
                        help='latent feature dimension for ASF')
    parser.add_argument('--gate_hidden', type=int, default=128,
                        help='gate MLP hidden dimension')
    parser.add_argument('--disp_hidden', type=int, default=256,
                        help='displacement MLP hidden dimension (strengthened)')

    opt = parser.parse_args()

    # Resolve default checkpoint paths
    if opt.nbi_ckpt is None:
        opt.nbi_ckpt = f'./pretrained/{opt.fold}_teacher.pth'
    if opt.wli_ckpt is None:
        opt.wli_ckpt = f'./pretrained/{opt.fold}_student.pth'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_tag = ''
    if opt.proj_adapter:
        mode_tag += '_adapter'
    if opt.use_mhsa:
        mode_tag += f'_mhsa{opt.mhsa_heads}'
    if not mode_tag:
        mode_tag = '_basic'

    opt.train_save = f'./log/fusion/{timestamp}_fold{opt.fold}{mode_tag}'
    os.makedirs(os.path.join(opt.train_save, 'run'), exist_ok=True)
    os.makedirs(os.path.join(opt.train_save, 'weights'), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.train_save, 'train_log.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p', force=True,
    )
    tb_writer = SummaryWriter(os.path.join(opt.train_save, 'run'))

    device = torch.device(opt.device)
    num_classes = 4

    # Load frozen backbones
    model_nbi = resnet50(pretrained=False, num_classes=num_classes).to(device)
    model_wli = resnet50_w(pretrained=False, num_classes=num_classes).to(device)

    # Load checkpoints
    nbi_state = torch.load(opt.nbi_ckpt, map_location=device)
    model_nbi.load_state_dict(nbi_state, strict=False)
    print(f'Loaded NBI teacher from {opt.nbi_ckpt}')

    wli_state = torch.load(opt.wli_ckpt, map_location=device)
    model_wli.load_state_dict(wli_state, strict=False)
    print(f'Loaded WLI student from {opt.wli_ckpt}')

    # Data
    train_dataset = MultiClassPairDataset(root_dir=opt.root, split='train', enable_aug=True, target_size=448)
    val_dataset = MultiClassPairDataset(root_dir=opt.root, split='val', enable_aug=False, target_size=448)
    class_names = [name for name, idx in sorted(val_dataset.class_map.items(), key=lambda x: x[1])]
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=8,
                              shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8,
                            shuffle=False, pin_memory=True, persistent_workers=True)

    # Single-model baselines on val
    print('\n===== Single-Model Baselines =====')
    nbi_acc, nbi_f1, _, _, _ = eval_single_model(model_nbi, val_loader, device, class_names, 'resnet50')
    wli_acc, wli_f1, _, _, _ = eval_single_model(model_wli, val_loader, device, class_names, 'resnet50_w')
    print(f'NBI only -> acc={nbi_acc:.4f} macro_f1={nbi_f1:.4f}')
    print(f'WLI only -> acc={wli_acc:.4f} macro_f1={wli_f1:.4f}')
    print(f'===============================\n')

    logging.info(f'NBI baseline -> acc={nbi_acc:.4f} macro_f1={nbi_f1:.4f}')
    logging.info(f'WLI baseline -> acc={wli_acc:.4f} macro_f1={wli_f1:.4f}')

    # Build fusion components
    fusion_net = ASFFusion(
        in_dim=2048,
        dim=opt.dim,
        gate_hidden=opt.gate_hidden,
        disp_hidden=opt.disp_hidden,
        use_mhsa=opt.use_mhsa,
        mhsa_heads=opt.mhsa_heads,
        use_adapter=opt.proj_adapter,
        adapter_hidden=256,
    ).to(device)

    classifier = nn.Linear(opt.dim, num_classes).to(device)

    # Print parameter counts
    total_params = sum(p.numel() for p in fusion_net.parameters())
    total_params += sum(p.numel() for p in classifier.parameters())
    print(f'Trainable parameters: {total_params:,}')
    logging.info(f'Trainable parameters: {total_params:,}')

    train_fusion(
        model_nbi, model_wli, fusion_net, classifier,
        train_loader, val_loader, class_names, device,
        epochs=opt.epochs, lr=opt.lr,
        lambda_theta=opt.lambda_theta,
        w_cos=opt.w_cos,
        train_save=opt.train_save,
    )
