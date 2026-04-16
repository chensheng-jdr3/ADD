"""
极简版基于 TelME 官方逻辑的 ASF 多模态融合训练代码
强制遵循 KISS 原则与清晰的状态可视化
"""
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from dataloader.loader import MultiClassPairDataset
from lib import resnet50, resnet50_w

# ==========================================
# 核心网络：原生极简版 ASF (TelME 架构适配)
# ==========================================
class ASF(nn.Module):
    def __init__(self, in_dim=2048, cls_num=4, beta_shift=0.1, dropout_prob=0.2):
        super(ASF, self).__init__()
        self.beta_shift = beta_shift
        
        # 门控与位移特征计算权重映射
        self.W_hav = nn.Linear(in_dim * 2, in_dim)
        self.W_av = nn.Linear(in_dim, in_dim)
        
        # 归一化与分类器
        self.LayerNorm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(in_dim, cls_num)

    def forward(self, t_feat, s_feat):
        # 1. 独立映射
        t_embed = t_feat
        s_embed = s_feat
        cos_st = F.cosine_similarity(s_embed, t_embed, dim=-1)  # 计算原始学生与教师表征的余弦相似度
        
        # 2. Gate 门控生成联合权重
        weight_av = F.relu(self.W_hav(torch.cat((s_embed, t_embed), dim=-1)))
        h_m = weight_av * self.W_av(s_embed)  # 基于学生表征计算基础位移
        cos_ht = F.cosine_similarity(h_m, t_embed, dim=-1)  # 计算位移特征与教师表征的余弦相似度
        
        # 3. 模态转移控制 (Modality Shifting - 对应 TelME 原文方程)
        eps = 1e-6
        t_norm = t_embed.norm(p=2, dim=-1)
        h_norm = h_m.norm(p=2, dim=-1)
        
        hc_norm_safe = torch.where(h_norm == 0, torch.ones_like(h_norm), h_norm)
        thresh_hold = (t_norm / (hc_norm_safe + eps)) * self.beta_shift
        alpha = torch.clamp(thresh_hold, max=1.0).unsqueeze(-1)  # 严格截断至 1.0 (等效于原代码 min 操作)
        
        # 4. 位移修正并参与恒等映射融合
        # shifted_embed = alpha * h_m
        # z_embed = shifted_embed + t_embed
        # cos_zt = F.cosine_similarity(z_embed, t_embed, dim=-1)
        # fused_output = self.dropout(self.LayerNorm(z_embed))
        
        shifted_norm = alpha * h_norm
        z_norm = shifted_norm + t_norm
        cos_zt_norm = F.cosine_similarity(z_norm.unsqueeze(1), t_norm.unsqueeze(1), dim=-1)
        fused_output = self.dropout(self.LayerNorm(z_norm))
        
        
        return self.classifier(fused_output), cos_st, cos_ht, cos_zt_norm

# ==========================================
# 工具函数：通用特征提取与模型保存
# ==========================================
def extract_pooled_feature(model, img, backbone_type):
    """通用层挂钩特征提取器（KISS设计，避免破坏模型固有返回值）"""
    features = {}
    def hook(m, i, o): features['out'] = o
    
    handle = model.layer4.register_forward_hook(hook)
    _ = model(img)
    handle.remove()
    
    return F.adaptive_avg_pool2d(features['out'], (1, 1)).flatten(1)

def evaluate_model(model_nbi, model_wli, fusion_net, dataloader, device):
    """全量指标评估，状态高度透明化处理"""
    fusion_net.eval()
    all_preds, all_labels = [], []
    cos_st, cos_ht, cos_zt = [], [], []
    
    with torch.no_grad():
        for wli_img, nbi_img, label, *rest in tqdm(dataloader, desc="[Evaluate]", leave=False):
            t_feat = extract_pooled_feature(model_nbi, nbi_img.to(device).float(), 'nbi')
            s_feat = extract_pooled_feature(model_wli, wli_img.to(device).float(), 'wli')
                    
            logits, cos_st_in, cos_ht_in, cos_zt_in = fusion_net(t_feat, s_feat)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            cos_st.extend(cos_st_in.cpu().numpy())
            cos_ht.extend(cos_ht_in.cpu().numpy())
            cos_zt.extend(cos_zt_in.cpu().numpy())

    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    cos_st_mean = sum(cos_st) / len(cos_st) if cos_st else 0.0
    cos_ht_mean = sum(cos_ht) / len(cos_ht) if cos_ht else 0.0
    cos_zt_mean = sum(cos_zt) / len(cos_zt) if cos_zt else 0.0

    return acc, macro_pre, macro_rec, macro_f1, cos_st_mean, cos_ht_mean, cos_zt_mean

# ==========================================
# 主训练循环流水线
# ==========================================
def main(opt):
    device = torch.device(opt.device)
    
    # 1. 可视化/监控环境初始化
    os.makedirs(opt.save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s | %(message)s', level=logging.INFO,
        handlers=[logging.FileHandler(f"{opt.save_dir}/train.log"), logging.StreamHandler()]
    )
    writer = SummaryWriter(os.path.join(opt.save_dir, 'tb_logs'))
    
    # 2. 数据流初始化 (强制抛出关键显式监控)
    train_ds = MultiClassPairDataset(root_dir=opt.root, split='train', enable_aug=True, target_size=448)
    val_ds = MultiClassPairDataset(root_dir=opt.root, split='val', enable_aug=False, target_size=448)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    logging.info(f"Dataset Loaded: {len(train_ds)} train samples, {len(val_ds)} val samples, {len(val_ds.class_map)} classes.")

    # 3. 冻结加载师生骨干网络
    model_nbi = resnet50(pretrained=False, num_classes=len(val_ds.class_map)).to(device).eval()
    model_wli = resnet50_w(pretrained=False, num_classes=len(val_ds.class_map)).to(device).eval()
    model_nbi.load_state_dict(torch.load(opt.nbi_ckpt, map_location=device), strict=False)
    model_wli.load_state_dict(torch.load(opt.wli_ckpt, map_location=device), strict=False)
    for p in model_nbi.parameters(): p.requires_grad = False
    for p in model_wli.parameters(): p.requires_grad = False
    
    # 4. 构建 ASF 融合模块并初始化优化器
    fusion_net = ASF(in_dim=2048, cls_num=len(val_ds.class_map), beta_shift=opt.beta_shift).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fusion_net.parameters(), lr=opt.lr, weight_decay=5e-4)
    
    # 5. 训练流水线
    best_f1 = 0.0
    for epoch in range(1, opt.epochs + 1):
        fusion_net.train()
        train_loss, train_acc, samples = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{opt.epochs}", leave=False)
        for wli_img, nbi_img, label,_,_,_,_ in pbar:
            label = label.to(device).long()
            
            with torch.no_grad():
                t_feat = extract_pooled_feature(model_nbi, nbi_img.to(device).float(), 'nbi')
                s_feat = extract_pooled_feature(model_wli, wli_img.to(device).float(), 'wli')
                
            logits, _, _, _ = fusion_net(t_feat, s_feat)
            loss = criterion(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * label.size(0)
            train_acc += (logits.argmax(dim=1) == label).sum().item()
            samples += label.size(0)
            pbar.set_postfix({'Loss': loss.item()})
            
        epoch_loss, epoch_acc = train_loss / samples, train_acc / samples
        val_acc, val_pre, val_rec, val_f1, cos_st_mean, cos_ht_mean, cos_zt_mean = evaluate_model(model_nbi, model_wli, fusion_net, val_loader, device)
        
        # 彻底透明化的指标导出
        writer.add_scalars('Loss', {'Train': epoch_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': epoch_acc, 'Val': val_acc}, epoch)
        writer.add_scalar('Metrics/Val_Macro_F1', val_f1, epoch)
        writer.add_scalar('Metrics/Val_Cosine_Similarity_ST', cos_st_mean, epoch)
        writer.add_scalar('Metrics/Val_Cosine_Similarity_HT', cos_ht_mean, epoch)
        writer.add_scalar('Metrics/Val_Cosine_Similarity_ZT', cos_zt_mean, epoch)
        
        logging_msg = f"Epoch {epoch:02d} | Train L: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Cosine Similarity: {cos_st_mean:.4f}, {cos_ht_mean:.4f}, {cos_zt_mean:.4f}"
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(fusion_net.state_dict(), os.path.join(opt.save_dir, 'best_fusion.bin'))
            logging_msg += " --> [New Best Model Saved]"
            
        logging.info(logging_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./my_dataset', help='Dataset root path')
    parser.add_argument('--nbi_ckpt', default='./pretrained/0_teacher_NBI.pth', help='NBI Teacher weights')
    parser.add_argument('--wli_ckpt', default='./pretrained/0_student_WLI.pth', help='WLI Student weights')
    parser.add_argument('--save_dir', default=f'./log/fusion/asf_{datetime.now().strftime("%Y%m%d_%H%M")}')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4) # AdamW default tuning rate
    parser.add_argument('--beta_shift', type=float, default=0.1) # Default set from paper equivalent
    main(parser.parse_args())

