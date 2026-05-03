# -*- coding: utf-8 -*-
"""
掩码可视化脚本
==============
对每张输入图像，生成 2×3 的可视化面板：

  行1: 原图 | 黑边掩码 | 高光掩码
  行2: 暗区掩码 | 三者叠加 | 最终权重热力图

  所有掩码以原图为背景（半透明叠加），图像间用白边隔开。

用法：
  python test/visualize_masks.py --input_dir path/to/images --output_dir path/to/out
"""
import os, sys, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preproc_dark import compute_hard_mask, compute_dark_region


def safe_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ==================== 颜色方案 ====================
# 每种掩码使用不同的叠加颜色 (B, G, R)，alpha=0.5
COLOR_BLACK_EDGE = (0, 0, 255)       # 红色 → 黑边区
COLOR_HIGHLIGHT  = (255, 0, 0)        # 蓝色 → 高光区
COLOR_DARK       = (0, 255, 255)      # 黄色 → 暗区
ALPHA = 0.55


def overlay_mask(bg_bgr, mask_uint8, color_bgr):
    """
    在 bg_bgr 原图上叠加彩色掩码。
    mask_uint8: 0=无效，255=有效（该区域涂色）
    color_bgr: (B, G, R)
    """
    vis = bg_bgr.copy()
    overlay = np.zeros_like(bg_bgr)
    overlay[:] = color_bgr
    m = (mask_uint8 > 0)
    vis[m] = cv2.addWeighted(bg_bgr[m], 1 - ALPHA, overlay[m], ALPHA, 0)
    return vis


def weight_to_heatmap(weight, bg_bgr):
    """权重热力图，以原图为背景叠加"""
    w_norm = weight.astype(np.float32)
    if w_norm.max() > 1.01:
        w_norm = w_norm / 255.0
    w_u8 = (np.clip(w_norm, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(w_u8, cv2.COLORMAP_JET)
    vis = cv2.addWeighted(bg_bgr, 0.45, heat, 0.55, 0)
    return vis


def add_border(img, thickness=3, color=(255, 255, 255)):
    """四周加白边"""
    return cv2.copyMakeBorder(img, thickness, thickness, thickness, thickness,
                              cv2.BORDER_CONSTANT, value=color)


def process_and_visualize(img_path, out_path, resize=256):
    """
    处理单张图像，生成 2×3 面板并保存。
    """
    img = safe_imread(img_path)
    if img is None:
        print(f"  无法读取: {img_path}")
        return

    # 统一尺寸
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 1) 黑边掩码 ----
    # compute_hard_mask 内部检测黑边+高光，返回 M_hard（255=有效，0=无效）
    # 我们需要单独提取黑边区
    dark_inv = cv2.bitwise_not(cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV)[1])
    contours, _ = cv2.findContours(dark_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_mask = np.zeros_like(gray, dtype=np.uint8)
    if contours:
        cv2.drawContours(black_mask, [max(contours, key=cv2.contourArea)], 0, 255, -1)
    else:
        black_mask[:] = 255
    # 黑边区 = 原图中被标记为"非黑边"之外 (black_mask==0 即为黑边)
    black_edge_mask = (black_mask == 0).astype(np.uint8) * 255

    # ---- 2) 高光掩码 ----
    highlight_mask = (gray > 240).astype(np.uint8) * 255
    if highlight_mask.any():
        highlight_mask = cv2.dilate(highlight_mask,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # ---- 3) 暗区掩码 ----
    hard_mask = compute_hard_mask(gray)
    dark_mask = compute_dark_region(gray, range_percent=15, hard_mask=hard_mask,
                                     min_size=2000, min_count_ratio=0.04, start_gray=10)

    # ---- 4) 最终权重 ----
    # 使用与 evaluate_pairing 一致的权重计算（无 SIFT 简化版）
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    det_mask = (hard_mask > 0).astype(np.uint8) * 255
    det_mask[dark_mask > 0] = 0
    try:
        detector = cv2.SIFT_create(nfeatures=800)
        kp = detector.detect(gray, mask=det_mask)
    except Exception:
        kp = []

    kp_map = np.ones_like(gray, dtype=np.uint8)
    for k in kp:
        x, y = int(round(k.pt[0])), int(round(k.pt[1]))
        if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
            kp_map[y, x] = 0
    dist = cv2.distanceTransform(kp_map, cv2.DIST_L2, 3).astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-6)
    w_dist = np.exp(-(dist_norm ** 2) / (2 * (0.3 ** 2)))
    weight = w_dist * (0.3 + 0.7 * grad)
    weight[dark_mask > 0] *= 0.2
    weight[hard_mask == 0] = 0
    weight = np.clip(weight, 0, 1).astype(np.float32)

    # ---- 生成 6 张可视化图 ----
    panels = []

    # (a) 原图
    panels.append(img.copy())

    # (b) 黑边掩码叠加
    panels.append(overlay_mask(img, black_edge_mask, COLOR_BLACK_EDGE))

    # (c) 高光掩码叠加
    panels.append(overlay_mask(img, highlight_mask, COLOR_HIGHLIGHT))

    # (d) 暗区掩码叠加
    panels.append(overlay_mask(img, dark_mask, COLOR_DARK))

    # (e) 三者叠加：黑边 + 高光 + 暗区（按优先级：黑边>高光>暗区）
    triple = img.copy()
    # 先画暗区（底层）
    triple = overlay_mask(triple, dark_mask, COLOR_DARK)
    # 再画高光（中层）
    triple = overlay_mask(triple, highlight_mask, COLOR_HIGHLIGHT)
    # 再画黑边（顶层）
    triple = overlay_mask(triple, black_edge_mask, COLOR_BLACK_EDGE)
    panels.append(triple)

    # (f) 最终权重热力图
    panels.append(weight_to_heatmap(weight, img))

    # ---- 拼成 2×3，每格之间加白边 ----
    h, w = resize, resize
    border = 4
    for i in range(6):
        panels[i] = cv2.resize(panels[i], (w, h))
        panels[i] = add_border(panels[i], thickness=border, color=(255, 255, 255))

    row1 = np.concatenate([panels[0], panels[1], panels[2]], axis=1)
    row2 = np.concatenate([panels[3], panels[4], panels[5]], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)

    # 在每格上方加文字标签
    labels = ["Original", "Black Edge", "Highlight", "Dark Region", "All Masks", "Weight Heatmap"]
    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    cell_w = w + 2 * border
    cell_h = h + 2 * border
    label_y = 18
    for i, label in enumerate(labels):
        col = i % 3
        row_idx = i // 3
        x = col * cell_w + 6
        y = row_idx * cell_h + label_y
        cv2.putText(canvas, label, (x, y), font, fs, (255, 255, 255), thick,
                    cv2.LINE_AA)
        # 轻微阴影让文字更清晰
        cv2.putText(canvas, label, (x + 1, y + 1), font, fs, (0, 0, 0), thick,
                    cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description="Visualize preprocessing masks")
    ap.add_argument("--input_dir", default='dataset\White_light\darkarea', help="图像目录")
    ap.add_argument("--output_dir", default=None, help="输出目录（默认 input_dir/../mask_vis）")
    ap.add_argument("--resize", type=int, default=256, help="统一缩放尺寸")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_dir), "mask_vis")

    os.makedirs(args.output_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)])

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"共 {len(files)} 张图像")

    done, failed = 0, 0
    for f in files:
        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(args.output_dir, f"{os.path.splitext(f)[0]}_masks.jpg")
        try:
            if process_and_visualize(in_path, out_path, resize=args.resize):
                done += 1
        except Exception as e:
            failed += 1
            print(f"  出错 [{f}]: {e}")
        if (done + failed) % 20 == 0:
            print(f"  进度: {done + failed}/{len(files)}")

    print(f"完成: {done} 成功, {failed} 失败")


if __name__ == "__main__":
    main()
