# eval_teacher_train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.loader import MultiClassPairDataset
from lib import resnet50

# 可选：从 train_fusion 中复用特征提取（若需要）
try:
    from train_fusion import extract_pooled_feature
except Exception:
    extract_pooled_feature = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = './pretrained/0_teacher.pth'   # 修改为你的教师权重路径
dataset_root = './my_dataset'         # 修改为你的数据根目录

# 数据集/加载器（不做增广）
train_ds = MultiClassPairDataset(root_dir=dataset_root, split='train', enable_aug=False, target_size=448)
loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)

# 加载模型
num_classes = len(train_ds.class_map)
model = resnet50(pretrained=False, num_classes=num_classes).to(device)
state = torch.load(ckpt, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for wli_img, nbi_img, label in tqdm(loader, desc='Eval teacher on train'):
        # 使用 NBI 图像作为输入
        inp = nbi_img.to(device).float()
        # 兼容不同 forward 签名：若返回 tuple/list 则取第0项当 logits
        outs = model(inp)
        logits = outs[0] if isinstance(outs, (tuple, list)) else outs
        preds = logits.argmax(dim=1)
        correct += (preds.cpu() == label).sum().item()
        total += label.size(0)

acc = correct / total if total > 0 else 0.0
print(f'Teacher accuracy on train set: {acc:.4f} ({correct}/{total})')