# -*- coding: utf-8 -*-
"""
无训练：同一患者 NBI ↔ WLI 静态图像配对（简化版）
"""

import os, cv2, csv, json, math, argparse
import numpy as np
from datetime import datetime

# 从提取的模块导入暗区计算与掩码函数
from preproc_dark import compute_hard_mask, compute_dark_region

# 默认配置
CFG = {
    "nmi_bins": 64, "ambiguity_thresh": 0.1,
    "no_match_score_thresh": 0.06, "resize": 256,
    # 默认类别阈值：'正常' 更严格，其余保持 0.6
    "class_thresh": {
        "正常": 0.1,
        "鳞状细胞癌": 0.6,
        "低瘤": 0.6,
        "高瘤": 0.6
    },
    "percentile_p_nbi": 10, "percentile_p_wli": 10,
    "dark_range_min_ratio": 0.04,  # 灰度范围边界最小像素占比
    "dark_range_start_gray": 5,   # 从该灰度开始寻找范围边界
    "dark_soft_weight": 1,  # 暗区软权重（0=完全忽略，1=不降权）
    "similarity_method": "nmi",  # 相似性度量方法（仅支持 "nmi"）
    "crop_box": None,  # 可选: (top, bottom, left, right) 用于裁剪输入图像
    "ransac_ratio": 0.75,
    "ransac_thresh": 3.0,
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder):
    """列出文件夹内图像文件"""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) 
            if f.lower().endswith(IMG_EXTS)] if os.path.isdir(folder) else []


def safe_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ==================== 掩码与权重计算 ====================
# 注意：`compute_hard_mask` 和 `compute_dark_region` 已提取到模块 `preproc_dark.py` 并在文件顶部导入，
# 本处直接使用外部实现以保证代码一致性与可维护性。

def compute_weight_sift_distance(
    gray,
    hard_mask=None,
    dark_region=None,
    dark_soft_weight=0.2,
    max_features=800,
    dist_sigma=0.3,
    other_kp=None,
    other_des=None,
    ransac_ratio=None,
    ransac_thresh=None,
):
    """
    权重计算方式3：SIFT关键点 + 到最近关键点的距离
    1) 用SIFT在有效区域检测关键点
    2) 计算每个像素到最近关键点的距离（距离越近权重越高）
    3) 结合梯度幅值，增强结构显著区域
    """
    if gray is None:
        return None

    # 梯度幅值
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.normalize(grad, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # 构建检测掩码：以硬掩码（255=有效）为基础，去掉暗区（dark_region>0）
    if hard_mask is not None:
        det_mask = (hard_mask > 0).astype(np.uint8) * 255
    else:
        det_mask = np.ones_like(gray, dtype=np.uint8) * 255
    if dark_region is not None:
        det_mask[dark_region > 0] = 0

    # 检测器（固定使用 SIFT）
    detector = cv2.SIFT_create(nfeatures=max_features)

    # 如果传入了其它图像的 keypoints/descriptors，则尝试使用 RANSAC 过滤内点（默认尝试）
    use_inlier_coords = None
    # 直接一次性检测并计算描述子
    try:
        kp1, des1 = detector.detectAndCompute(gray, mask=det_mask)
    except Exception:
        kp1, des1 = [], None

    # 如果外部没有提供 ransac 参数，则使用全局 CFG 中的默认值
    if ransac_ratio is None:
        ransac_ratio = CFG.get("ransac_ratio", 0.75)
    if ransac_thresh is None:
        ransac_thresh = CFG.get("ransac_thresh", 3.0)

    if des1 is not None and other_des is not None and len(des1) > 0 and len(other_des) > 0:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, other_des, k=2)

        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n[0], m_n[1]
            if m.distance < ransac_ratio * n.distance:
                good.append(m)

        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([other_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
            if mask is not None:
                inlier_idx = np.where(mask.ravel() != 0)[0]
                if inlier_idx.size:
                    coords = np.array([kp1[good[i].queryIdx].pt for i in inlier_idx])
                    coords = np.round(coords).astype(np.int32)
                    use_inlier_coords = [tuple(c) for c in coords]

    # 如果没有使用 RANSAC 内点过滤，则按原逻辑检测 keypoints 并生成 kp_map
    if use_inlier_coords is None:
        # SIFT/ORB 检测并不计算描述子（若需要描述子，上层可另行调用）
        keypoints = detector.detect(gray, mask=det_mask)
        if not keypoints:
            weight = 0.3 + 0.7 * grad
        else:
            kp_map = np.ones_like(gray, dtype=np.uint8)
            for kp in keypoints:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
                    kp_map[y, x] = 0

            # 距离变换：每个像素到最近关键点的距离
            dist = cv2.distanceTransform(kp_map, cv2.DIST_L2, 3).astype(np.float32)
            dist_norm = dist / (dist.max() + 1e-6)

            # 距离权重（越近越大），并结合梯度
            w_dist = np.exp(-(dist_norm ** 2) / (2 * (dist_sigma ** 2)))
            weight = w_dist * (0.3 + 0.7 * grad)
    else:
        # 使用 RANSAC 内点坐标生成 kp_map
        kp_map = np.ones_like(gray, dtype=np.uint8)
        for (x, y) in use_inlier_coords:
            if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
                kp_map[y, x] = 0

        dist = cv2.distanceTransform(kp_map, cv2.DIST_L2, 3).astype(np.float32)
        dist_norm = dist / (dist.max() + 1e-6)
        w_dist = np.exp(-(dist_norm ** 2) / (2 * (dist_sigma ** 2)))
        weight = w_dist * (0.3 + 0.7 * grad)

    # 暗区软权重
    if dark_region is not None:
        weight[dark_region > 0] *= dark_soft_weight

    # 硬掩码：黑边和高光区域权重为0
    if hard_mask is not None:
        weight[hard_mask == 0] = 0

    return np.clip(weight, 0, 1).astype(np.float32)

compute_weight = compute_weight_sift_distance


# ==================== 预处理 ====================

def preprocess(
    img,
    crop_box=None,
    resize=256,
    range_percent=15,
    dark_soft_weight=0.2,
    dark_range_min_ratio=0.01,
    dark_range_start_gray=10,
):
    """
    预处理：灰度 -> 硬掩码 -> 暗区 -> 权重 -> CLAHE -> 梯度
    返回：(显示用BGR图, 梯度图, 权重图)
    """
    if img is None:
        return None, None, None
    
    # 可选裁剪：crop_box 格式 (top, bottom, left, right)
    if crop_box is not None:
        try:
            t, b, l, r = map(int, crop_box)
            h0, w0 = img.shape[:2]
            t = max(0, min(h0, t))
            b = max(0, min(h0, b))
            l = max(0, min(w0, l))
            r = max(0, min(w0, r))
            if b > t and r > l:
                img = img[t:b, l:r]
        except Exception:
            pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
    
    # 计算硬掩码和暗区
    hard_mask = compute_hard_mask(gray)
    dark_region = compute_dark_region(
        gray,
        range_percent,
        hard_mask,
        min_size=2000,
        min_count_ratio=dark_range_min_ratio,
        start_gray=dark_range_start_gray,
    )
    
    # 计算权重
    weight = compute_weight(gray, hard_mask, dark_region, dark_soft_weight)
    
    # CLAHE
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    
    # 梯度图
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img, grad, weight


# ==================== 加权NMI ====================

def compute_weighted_nmi(img1, img2, weight1, weight2, bins=64):
    """
    加权归一化互信息（Weighted NMI）
    
    参数：
        img1, img2: 梯度图（uint8）
        weight1, weight2: 权重图（float32，范围[0,1]）
        bins: 直方图bin数
    
    返回：NMI值（0~1）
    """
    if img1 is None or img2 is None:
        return 0.0
    
    # 合并权重：取两者权重的乘积（几何平均的思想）
    if weight1 is not None and weight2 is not None:
        w = weight1 * weight2
    elif weight1 is not None:
        w = weight1
    elif weight2 is not None:
        w = weight2
    else:
        w = np.ones_like(img1, dtype=np.float32)
    
    # 有效像素（权重>0）
    valid = w > 0
    if valid.sum() < 100:
        return 0.0
    
    v1 = img1[valid].astype(np.float32)
    v2 = img2[valid].astype(np.float32)
    wv = w[valid].astype(np.float32)
    
    # 加权联合直方图
    h2 = np.zeros((bins, bins), dtype=np.float64)
    bin_edges = np.linspace(0, 256, bins + 1)
    idx1 = np.clip(np.digitize(v1, bin_edges) - 1, 0, bins - 1)
    idx2 = np.clip(np.digitize(v2, bin_edges) - 1, 0, bins - 1)
    
    # 使用 np.add.at 进行加权累加
    np.add.at(h2, (idx1, idx2), wv)
    
    total = h2.sum()
    if total < 1e-12:
        return 0.0
    
    pxy = h2 / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    
    # 熵计算
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0] + 1e-12)).sum())
    hx, hy = H(px), H(py)
    
    if hx <= 0 or hy <= 0:
        return 0.0
    
    # 互信息
    nz = pxy > 0
    denom = (px @ py)
    mi = float((pxy[nz] * np.log((pxy[nz] + 1e-12) / (denom[nz] + 1e-12))).sum())
    
    # NMI
    nmi = mi / (math.sqrt(hx * hy) + 1e-12)
    return max(0.0, min(1.0, nmi))


# ==================== 相似性度量选择器 ====================
def greedy_assignment(scores, thresh):
    """贪心匹配"""
    N, M = scores.shape
    assign, matched = [-1] * N, set()
    pairs = sorted([(scores[i, j], i, j) for i in range(N) for j in range(M)], reverse=True)
    for s, i, j in pairs:
        if s < thresh: break
        if assign[i] == -1 and j not in matched:
            assign[i], _ = j, matched.add(j)
    return assign


def draw_triplet(nbi, wli, nw, ww, s, path):
    """保存可视化：只保存配对的原图与对应权重热力图"""
    h = 256
    to_panel = lambda img: cv2.resize(img, (h, h)) if img is not None else np.zeros((h, h, 3), dtype=np.uint8)

    def weight_to_heatmap(w, orig=None, alpha=0.6):
        orig_img = to_panel(orig) if orig is not None else np.zeros((h, h, 3), dtype=np.uint8)
        if w is None:
            return orig_img
        w_resized = cv2.resize(w, (h, h)) if w.shape[:2] != (h, h) else w
        w_f = w_resized.astype(np.float32)
        if w_f.max() <= 1.1:
            w_u8 = (np.clip(w_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            w_u8 = np.clip(w_f, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(w_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(color_map, alpha, orig_img, 1.0 - alpha, 0)
        return blended

    row1 = np.concatenate([to_panel(nbi), to_panel(wli)], axis=1)
    row2 = np.concatenate([weight_to_heatmap(nw, nbi), weight_to_heatmap(ww, wli)], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    cv2.putText(canvas, f"score={s:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", canvas)
    if ok:
        buf.tofile(path)


def match_patient(pid, nbi_paths, wli_paths, cfg, out_dir, patient_dir=None, label=None):
    """对单个患者做匹配"""
    # 预处理（返回：显示图, 梯度图, 权重图）
    load = lambda paths, key: [(path, *preprocess(
        safe_imread(path),
        cfg.get("crop_box"),
        cfg["resize"],
        cfg[f"percentile_p_{key}"],
        cfg["dark_soft_weight"],
        cfg["dark_range_min_ratio"],
        cfg["dark_range_start_gray"],
    )) 
                               for path in paths]
    nbi_data = [(p, d, g, w) for p, d, g, w in load(nbi_paths, "nbi") if g is not None]
    wli_data = [(p, d, g, w) for p, d, g, w in load(wli_paths, "wli") if g is not None]
    
    N, M = len(nbi_data), len(wli_data)
    if N == 0:
        return [], {"patient_id": pid, "note": "no NBI"}, 0, 0
    if M == 0:
        return [], {"patient_id": pid, "note": "no WLI"}, N, 0
    
    # 为每张图预计算灰度、硬掩码、暗区、以及关键点/描述子（用于可选的 SIFT+RANSAC 过滤）
    # 这样在计算权重时可以将另一幅图的描述子传入以进行 RANSAC 内点筛选。
    # 预先创建检测器（SIFT 优先）
    try:
        detector_global = cv2.SIFT_create(nfeatures=800)
    except Exception:
        detector_global = cv2.ORB_create(nfeatures=800)
    
    def prepare_item(item):
        path, disp, grad, weight = item
        img = safe_imread(path)
        if img is None:
            return {"path": path, "disp": disp, "grad": grad, "weight": weight,
                    "gray": None, "hard": None, "dark": None, "kp": [], "des": None}

        # 可选裁剪：与 preprocess 保持一致，cfg 中 crop_box 格式 (top, bottom, left, right)
        crop_box = cfg.get("crop_box")
        if crop_box is not None:
            try:
                t, b, l, r = map(int, crop_box)
                h0, w0 = img.shape[:2]
                t = max(0, min(h0, t))
                b = max(0, min(h0, b))
                l = max(0, min(w0, l))
                r = max(0, min(w0, r))
                if b > t and r > l:
                    img = img[t:b, l:r]
            except Exception:
                pass

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cfg.get("resize"):
            gray = cv2.resize(gray, (cfg["resize"], cfg["resize"]), interpolation=cv2.INTER_AREA)

        hard = compute_hard_mask(gray)
        # 使用 nbi 配置或通用配置来计算 dark region
        dark = compute_dark_region(gray, range_percent=cfg.get("percentile_p_nbi", 15),
                                  hard_mask=hard, min_size=2000,
                                  min_count_ratio=cfg.get("dark_range_min_ratio", 0.01),
                                  start_gray=cfg.get("dark_range_start_gray", 10))

        det_mask = (hard > 0).astype(np.uint8) * 255 if hard is not None else None
        if dark is not None and det_mask is not None:
            det_mask[dark > 0] = 0

        try:
            kp, des = detector_global.detectAndCompute(gray, mask=det_mask)
        except Exception:
            kp, des = [], None

        return {"path": path, "disp": disp, "grad": grad, "weight": weight,
                "gray": gray, "hard": hard, "dark": dark, "kp": kp or [], "des": des}

    nbi_items = [prepare_item((p, d, g, w)) for p, d, g, w in nbi_data]
    wli_items = [prepare_item((p, d, g, w)) for p, d, g, w in wli_data]

    # 恢复完整相似度矩阵：计算 N x M 的相似度，并将所有超过阈值的配对纳入结果
    vis_dir = os.path.join(out_dir, "vis", pid)
    os.makedirs(vis_dir, exist_ok=True)
    rows, debug = [], {"patient_id": pid, "items": []}

    sim_mat = np.zeros((N, M), dtype=np.float32)

    # 计算完整相似度矩阵
    for i in range(N):
        for j in range(M):
            g1 = nbi_items[i]["grad"]
            g2 = wli_items[j]["grad"]
            if compute_weight == compute_weight_sift_distance:
                w1 = compute_weight_sift_distance(nbi_items[i]["gray"], hard_mask=nbi_items[i]["hard"], dark_region=nbi_items[i]["dark"],
                                                   dark_soft_weight=cfg.get("dark_soft_weight", 0.2), max_features=800, dist_sigma=0.3,
                                                   other_kp=wli_items[j]["kp"], other_des=wli_items[j]["des"])
                w2 = compute_weight_sift_distance(wli_items[j]["gray"], hard_mask=wli_items[j]["hard"], dark_region=wli_items[j]["dark"],
                                                   dark_soft_weight=cfg.get("dark_soft_weight", 0.2), max_features=800, dist_sigma=0.3,
                                                   other_kp=nbi_items[i]["kp"], other_des=nbi_items[i]["des"])
            else:
                w1 = nbi_items[i]["weight"]
                w2 = wli_items[j]["weight"]

            s = compute_weighted_nmi(g1, g2, w1, w2, bins=cfg["nmi_bins"])
            sim_mat[i, j] = s

    # 计算每行的 top1（用于可视化参考），但不再考虑第二分数或模糊判定
    row_best = np.max(sim_mat, axis=1)
    # 支持基于类别的阈值映射：cfg["class_thresh"] = {"label": thresh}
    if label and isinstance(cfg.get("class_thresh"), dict) and label in cfg.get("class_thresh"):
        thresh = cfg["class_thresh"][label]
    else:
        thresh = cfg.get("no_match_score_thresh", 0.6)

    # 选择所有超过阈值的配对（先收集，再做可能的裁剪）；延迟可视化，裁剪后仅为保留项生成图像
    candidates = []
    inds = np.argwhere(sim_mat >= thresh)
    for (i, j) in inds:
        s = float(sim_mat[i, j])
        nbi_path, nbi_disp, _, nbi_weight = nbi_data[i]
        wli_path, wli_disp, _, wli_weight = wli_data[j]
        base = os.path.basename(nbi_path)
        wli_name = os.path.basename(wli_path)
        candidates.append({
            "i": i, "j": j, "nbi_path": nbi_path, "wli_path": wli_path,
            "nbi_disp": nbi_disp, "wli_disp": wli_disp,
            "nbi_weight": nbi_weight, "wli_weight": wli_weight,
            "base": base, "wli_name": wli_name, "s": s
        })

    # 如果配对过多（超过总可能配对数的25%），按得分排序只保留前25%
    total_possible = int(N * M)
    original_matched = len(candidates)
    max_allowed = int(math.floor(0.25 * total_possible)) if total_possible > 0 else 0
    if max_allowed > 0 and original_matched > max_allowed:
        candidates_sorted = sorted(candidates, key=lambda x: x["s"], reverse=True)
        kept = candidates_sorted[:max_allowed]
        pruned = True
    else:
        kept = candidates
        pruned = False

    # 为被保留的配对生成 rows/debug 与可视化
    rows = []
    debug = {"patient_id": pid, "items": []}
    for c in kept:
        rows.append([pid, c["base"], c["wli_name"], c["s"]])
        # 可视化：只为保留的配对保存图像
        draw_triplet(c["nbi_disp"], c["wli_disp"], c["nbi_weight"], c["wli_weight"], c["s"],
                     os.path.join(vis_dir, f"{os.path.splitext(c['base'])[0]}_match_{c['j']}.jpg"))
        debug["items"].append({
            "nbi": c["base"],
            "assigned_wli": c["wli_name"],
            "assigned_score": c["s"]
        })

    debug["pruned"] = pruned
    debug["original_matched"] = original_matched
    debug["kept_pairs"] = len(kept)
    debug["total_possible_pairs"] = total_possible
    
    # 额外输出：在原始患者目录下写入配对 CSV（如果提供了 patient_dir）
    if patient_dir is not None and rows:
        try:
            out_csv = os.path.join(patient_dir, "paired_matches.csv")
            with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["nbi_filename", "wli_filename", "score"])
                for r in rows:
                    # r 格式: [patient_id, nbi_filename, wli_filename, final_score]
                    w.writerow([r[1], r[2], r[3]])
        except Exception:
            pass

    return rows, debug, N, M


def main():
    ap = argparse.ArgumentParser(description="NBI-WLI pairing")
    ap.add_argument("--dataset_root", default="./my_dataset")
    ap.add_argument("--out_dir", default="./output/pair_images_results")
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    
    args = ap.parse_args()
    cfg = {**CFG, **vars(args)}

    dataset_root = args.dataset_root

    # 输出根目录（按时间戳）
    ts_root = os.path.join(cfg["out_dir"], f"{datetime.now():%Y%m%d%H%M}")
    os.makedirs(ts_root, exist_ok=True)
    print("输出目录:", ts_root)

    # 运行统计收集
    stats = {}

    # 固定任务与标签集合（根据用户确认）
    TASKS = ["train", "val", "test"]
    LABELS = ["鳞状细胞癌", "低瘤", "高瘤", "正常"]

    for task in TASKS:
        for label in LABELS:
            label_dir = os.path.join(dataset_root, task, label)
            if not os.path.isdir(label_dir):
                print(f"跳过不存在目录: {label_dir}")
                continue

            out_dir = os.path.join(ts_root, task, label)
            os.makedirs(out_dir, exist_ok=True)

            all_rows, all_debug = [], []
            patients = [d for d in sorted(os.listdir(label_dir)) if os.path.isdir(os.path.join(label_dir, d))]
            if not patients:
                print(f"[{task}/{label}] 未发现患者目录")
                continue

            # 初始化统计
            stats.setdefault(task, {})
            stats[task].setdefault(label, {"total_patients": 0, "no_pairs": 0, "pruned_count": 0})

            for pid in patients:
                pdir = os.path.join(label_dir, pid)
                nbi_dir = os.path.join(pdir, "NBI")
                wli_dir = os.path.join(pdir, "WLI")
                if not (os.path.isdir(nbi_dir) and os.path.isdir(wli_dir)):
                    continue
                stats[task][label]["total_patients"] += 1
                rows, dbg, nbi_count, wli_count = match_patient(pid, list_images(nbi_dir), list_images(wli_dir), cfg, out_dir, pdir, label=label)
                all_rows.extend(rows)
                if dbg.get("items"):
                    all_debug.append(dbg)
                prod = nbi_count * wli_count
                if dbg.get("pruned"):
                    kept = dbg.get("kept_pairs", len(rows))
                    orig = dbg.get("original_matched", "?")
                    stats[task][label]["pruned_count"] += 1
                    print(f"[{task}/{label}/{pid}] matched={len(rows)}/{prod} (NBI={nbi_count}, WLI={wli_count}) PRUNED {kept}/{orig} kept")
                else:
                    if len(rows) == 0:
                        stats[task][label]["no_pairs"] += 1
                    print(f"[{task}/{label}/{pid}] matched={len(rows)}/{prod} (NBI={nbi_count}, WLI={wli_count})")

            # 保存该 task/label 的结果（仅成功配对记录）
            if all_rows:
                with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["patient_id", "nbi_filename", "wli_filename", "final_score"])
                    w.writerows(all_rows)

            with open(os.path.join(out_dir, "debug.json"), "w", encoding="utf-8") as f:
                json.dump(all_debug, f, ensure_ascii=False, indent=2)

    # 保存运行日志与统计到 ts_root/run_log.json
    run_log = {
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "cfg": cfg,
        "stats": stats,
    }
    try:
        with open(os.path.join(ts_root, "run_log.json"), "w", encoding="utf-8") as f:
            json.dump(run_log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("完成！")


if __name__ == "__main__":
    
    main()
