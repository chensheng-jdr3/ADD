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
    "no_match_score_thresh": 0.6, "use_clahe": 1, "resize": 256,
    "percentile_p_nbi": 10, "percentile_p_wli": 10,
    "dark_range_min_ratio": 0.04,  # 灰度范围边界最小像素占比
    "dark_range_start_gray": 5,   # 从该灰度开始寻找范围边界
    "dark_soft_weight": 1,  # 暗区软权重（0=完全忽略，1=不降权）
    "similarity_method": "nmi",  # 相似性度量方法（仅支持 "nmi"）
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_patients(root_dir):
    """遍历根目录下的患者文件夹"""
    patients = []
    for name in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir) else []:
        pdir = os.path.join(root_dir, name)
        nbi, wli = os.path.join(pdir, "NBI"), os.path.join(pdir, "WLI")
        if os.path.isdir(nbi) and os.path.isdir(wli):
            patients.append((name, nbi, wli))
    return patients


def list_images(folder):
    """列出文件夹内图像文件"""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) 
            if f.lower().endswith(IMG_EXTS)] if os.path.isdir(folder) else []


def safe_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ==================== 掩码与权重计算 ====================
# 注意：`compute_hard_mask` 和 `compute_dark_region` 已提取到模块 `preproc_dark.py` 并在文件顶部导入，
# 本处直接使用外部实现以保证代码一致性与可维护性。


def compute_weight_gradient_energy(gray, hard_mask=None, dark_region=None, dark_soft_weight=0.2):
    """
    权重计算方式1：基于梯度能量 + 暗区软权重
    高梯度区域（边缘/结构）权重大，暗区降权但不归零
    
    参数：
        gray: 灰度图
        hard_mask: 硬掩码（黑边+高光），0的位置权重为0
        dark_region: 暗区区域，255的位置降权
        dark_soft_weight: 暗区的软权重值（0~1）
    
    返回：float32 权重图，范围 [0, 1]
    """
    # 梯度能量
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.magnitude(gx, gy)
    
    # 归一化到 [0, 1]
    energy = cv2.normalize(energy, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    
    # 加一个小的基础权重，避免低梯度区域完全被忽略
    weight = 0.3 + 0.7 * energy
    
    # 暗区软权重
    if dark_region is not None:
        weight[dark_region > 0] *= dark_soft_weight
    
    # 硬掩码：黑边和高光区域权重为0
    if hard_mask is not None:
        weight[hard_mask == 0] = 0
    
    return weight


def compute_weight_uniform(gray, hard_mask=None, dark_region=None, dark_soft_weight=0.2):
    """
    权重计算方式2：均匀权重 + 暗区软权重
    所有有效像素权重相同，仅暗区降权
    """
    weight = np.ones_like(gray, dtype=np.float32)
    
    if dark_region is not None:
        weight[dark_region > 0] = dark_soft_weight
    
    if hard_mask is not None:
        weight[hard_mask == 0] = 0
    
    return weight


def compute_weight_sift_distance(
    gray,
    hard_mask=None,
    dark_region=None,
    dark_soft_weight=0.2,
    max_features=800,
    dist_sigma=0.3,
    other_kp=None,
    other_des=None,
    ransac_ratio=0.75,
    ransac_thresh=3.0,
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

    # 检测器（SIFT 优先，失败回退 ORB）
    is_sift = True
    try:
        detector = cv2.SIFT_create(nfeatures=max_features)
    except Exception:
        detector = cv2.ORB_create(nfeatures=max_features)
        is_sift = False

    # 如果传入了其它图像的 keypoints/descriptors，则尝试使用 RANSAC 过滤内点
    use_inlier_coords = None
    if other_des is not None:
        # 先检测并计算当前图的 keypoints + descriptors
        kp1 = detector.detect(gray, mask=det_mask)
        if kp1:
            try:
                kp1, des1 = detector.compute(gray, kp1)
            except Exception:
                des1 = None
        else:
            des1 = None

        # 若没有描述子或对端没有描述子，退化为原方法
        if des1 is not None and other_des is not None and len(des1) > 0 and len(other_des) > 0:
            # 选择合适的匹配器
            if is_sift:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            # KNN+ratio test
            try:
                matches = bf.knnMatch(des1, other_des, k=2)
            except Exception:
                matches = []

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
                    inlier_idx = [i for i, v in enumerate(mask.ravel()) if v]
                    if inlier_idx:
                        use_inlier_coords = [tuple(map(int, map(round, kp1[good[i].queryIdx].pt))) for i in inlier_idx]

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


# 默认权重计算函数（可替换）
# compute_weight = compute_weight_gradient_energy
# compute_weight = compute_weight_uniform
compute_weight = compute_weight_sift_distance


# ==================== 预处理 ====================

def preprocess(
    img,
    use_clahe=0,
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
    if use_clahe:
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

def compute_similarity(img1, img2, weight1, weight2, method="zncc", **kwargs):
    """
    统一的相似性计算接口
    
    参数：
        method: 仅支持 "nmi"
    """
    if method != "nmi":
        raise ValueError(f"Unknown similarity method: {method}")
    return compute_weighted_nmi(img1, img2, weight1, weight2, **kwargs)


def hungarian_assignment(scores, thresh):
    """匈牙利算法"""
    try:
        from scipy.optimize import linear_sum_assignment
        N, M = scores.shape
        ext = np.full((N, M + N), thresh, dtype=np.float32)
        ext[:, :M] = scores
        row, col = linear_sum_assignment(-ext)
        return [int(c) if c < M else -1 for c in col]
    except:
        return greedy_assignment(scores, thresh)


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


def draw_triplet(nbi, wli1, wli2, nw, ww1, ww2, s1, s2, path):
    """保存可视化：第一行原图，第二行权重热力图"""
    h = 256
    to_panel = lambda img: cv2.resize(img, (h, h)) if img is not None else np.zeros((h, h, 3), dtype=np.uint8)
    
    def weight_to_heatmap(w):
        """将权重图转换为热力图"""
        # 如果没有权重或原图均为 None，返回黑图
        return np.zeros((h, h, 3), dtype=np.uint8)

    def weight_to_heatmap(w, orig=None, alpha=0.6):
        """将权重图转换为与原图叠加的热力图（orig 位于下层）

        参数：
            w: 权重图，float (0-1) 或 uint8 (0-255)
            orig: 对应的原图，用作下层（若 None 则使用黑底）
            alpha: 热力图的透明度（0-1），越大热力图越显眼
        返回：BGR 三通道图像，尺寸 (h,h,3)
        """
        orig_img = to_panel(orig) if orig is not None else np.zeros((h, h, 3), dtype=np.uint8)

        if w is None:
            return orig_img

        # 调整权重到 uint8 0-255
        w_resized = cv2.resize(w, (h, h)) if w.shape[:2] != (h, h) else w
        w_f = w_resized.astype(np.float32)
        if w_f.max() <= 1.1:
            w_u8 = (np.clip(w_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            w_u8 = np.clip(w_f, 0, 255).astype(np.uint8)

        color_map = cv2.applyColorMap(w_u8, cv2.COLORMAP_JET)

        # 整体混合：无论权重是否为0，都将 colormap 与原图按 alpha 混合，
        # 这样 0 对应 colormap(0)（通常为最暗色），不会出现“透明”区域。
        blended = cv2.addWeighted(color_map, alpha, orig_img, 1.0 - alpha, 0)
        return blended
    
    row1 = np.concatenate([to_panel(nbi), to_panel(wli1), to_panel(wli2)], axis=1)
    row2 = np.concatenate([
        weight_to_heatmap(nw, nbi),
        weight_to_heatmap(ww1, wli1),
        weight_to_heatmap(ww2, wli2)
    ], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    
    cv2.putText(canvas, f"best={s1:.3f}", (h + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"2nd ={s2:.3f}", (2*h + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", canvas)
    if ok: buf.tofile(path)


def match_patient(pid, nbi_paths, wli_paths, cfg, out_dir):
    """对单个患者做匹配"""
    # 预处理（返回：显示图, 梯度图, 权重图）
    load = lambda paths, key: [(path, *preprocess(
        safe_imread(path),
        cfg["use_clahe"],
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
        return [], {"patient_id": pid, "note": "no NBI"}
    if M == 0:
        return [[pid, os.path.basename(d[0]), "", 0.0, 1] for d in nbi_data], {"patient_id": pid, "note": "no WLI"}
    
    # 为每张图预计算灰度、硬掩码、暗区、以及关键点/描述子（用于可选的 SIFT+RANSAC 过滤）
    # 这样在计算权重时可以将另一幅图的描述子传入以进行 RANSAC 内点筛选。
    # 预先创建检测器（SIFT 优先）
    try:
        detector_global = cv2.SIFT_create(nfeatures=800)
        use_sift = True
    except Exception:
        detector_global = cv2.ORB_create(nfeatures=800)
        use_sift = False

    def prepare_item(item):
        path, disp, grad, weight = item
        img = safe_imread(path)
        if img is None:
            return {"path": path, "disp": disp, "grad": grad, "weight": weight,
                    "gray": None, "hard": None, "dark": None, "kp": [], "des": None}

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

    # 计算相似性矩阵（支持NMI或ZNCC），在使用 SIFT 权重函数时传入对端的 kp/des 用于 RANSAC 过滤
    sim_mat = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            g1 = nbi_items[i]["grad"]
            g2 = wli_items[j]["grad"]
            # 计算权重：如果当前全局 compute_weight 指向 compute_weight_sift_distance，使用 RANSAC 参数
            if compute_weight == compute_weight_sift_distance:
                w1 = compute_weight_sift_distance(nbi_items[i]["gray"], hard_mask=nbi_items[i]["hard"], dark_region=nbi_items[i]["dark"],
                                                   dark_soft_weight=cfg.get("dark_soft_weight", 0.2), max_features=800, dist_sigma=0.3,
                                                   other_kp=wli_items[j]["kp"], other_des=wli_items[j]["des"],
                                                   ransac_ratio=0.75, ransac_thresh=3.0)
                w2 = compute_weight_sift_distance(wli_items[j]["gray"], hard_mask=wli_items[j]["hard"], dark_region=wli_items[j]["dark"],
                                                   dark_soft_weight=cfg.get("dark_soft_weight", 0.2), max_features=800, dist_sigma=0.3,
                                                   other_kp=nbi_items[i]["kp"], other_des=nbi_items[i]["des"],
                                                   ransac_ratio=0.75, ransac_thresh=3.0)
            else:
                w1 = nbi_items[i]["weight"]
                w2 = wli_items[j]["weight"]

            sim_mat[i, j] = compute_similarity(g1, g2, w1, w2, method=cfg["similarity_method"], bins=cfg["nmi_bins"])
    
    # 归一化到 0-1（min-max）
    mn, mx = sim_mat.min(), sim_mat.max()
    scores = (sim_mat - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(sim_mat)
    
    # 匹配
    assign = (hungarian_assignment if cfg["global_assignment"] else 
              lambda s, t: [int(np.argmax(s[i])) if s[i].max() >= t else -1 for i in range(N)])(scores, cfg["no_match_score_thresh"])
    
    # 输出
    vis_dir = os.path.join(out_dir, "vis", pid)
    os.makedirs(vis_dir, exist_ok=True)
    rows, debug = [], {"patient_id": pid, "items": []}
    
    for i in range(N):
        nbi_path, nbi_disp, _, nbi_weight = nbi_data[i]
        base = os.path.basename(nbi_path)
        order = np.argsort(-scores[i])
        j1, j2 = (int(order[0]) if M >= 1 else -1), (int(order[1]) if M >= 2 else -1)
        s1, s2 = (float(scores[i, j1]) if j1 >= 0 else 0.0), (float(scores[i, j2]) if j2 >= 0 else 0.0)
        is_amb = 1 if s1 <= 1e-12 or (M >= 2 and (s1 - s2) / (s1 + 1e-12) < cfg["ambiguity_thresh"]) else 0
        
        mj = assign[i] if assign[i] is not None and assign[i] >= 0 and scores[i, assign[i]] >= cfg["no_match_score_thresh"] else -1
        final = float(scores[i, mj]) if mj >= 0 else 0.0
        wli_name = os.path.basename(wli_data[mj][0]) if mj >= 0 else ""
        
        # 仅当最终匹配分数 >= 阈值时，视为成功配对并保存可视化与记录
        if final >= cfg.get("no_match_score_thresh", 0.6) and mj >= 0:
            rows.append([pid, base, wli_name, final, is_amb])

            # 可视化（显示权重热力图）并保存
            draw_triplet(nbi_disp, wli_data[j1][1] if j1 >= 0 else None, wli_data[j2][1] if j2 >= 0 else None,
                         nbi_weight, wli_data[j1][3] if j1 >= 0 else None, wli_data[j2][3] if j2 >= 0 else None,
                         s1, s2, os.path.join(vis_dir, f"{os.path.splitext(base)[0]}_match.jpg"))

            debug["items"].append({"nbi": base, "assigned_wli": wli_name, "assigned_score": final,
                                   "top1_wli": os.path.basename(wli_data[j1][0]) if j1 >= 0 else "", "top1_score": s1,
                                   "top2_wli": os.path.basename(wli_data[j2][0]) if j2 >= 0 else "", "top2_score": s2,
                                   "is_ambiguous": is_amb})
    
    return rows, debug


def main():
    ap = argparse.ArgumentParser(description="NBI-WLI pairing")
    ap.add_argument("--dataset_root", default="./标框")
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--global_assignment", type=int, default=1)
    for k, v in CFG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    
    args = ap.parse_args()
    cfg = {**CFG, **vars(args), "global_assignment": args.global_assignment == 1}

    dataset_root = args.dataset_root

    # 输出根目录（按时间戳）
    ts_root = os.path.join(cfg["out_dir"], f"{datetime.now():%Y%m%d%H%M}")
    os.makedirs(ts_root, exist_ok=True)
    print("输出目录:", ts_root)

    # 固定任务与标签集合（根据用户确认）
    TASKS = ["train", "val", "test"]
    LABELS = ["鳞状细胞癌", "低瘤", "高瘤"]

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

            for pid in patients:
                pdir = os.path.join(label_dir, pid)
                nbi_dir = os.path.join(pdir, "NBI")
                wli_dir = os.path.join(pdir, "WLI")
                if not (os.path.isdir(nbi_dir) and os.path.isdir(wli_dir)):
                    continue

                rows, dbg = match_patient(pid, list_images(nbi_dir), list_images(wli_dir), cfg, out_dir)
                all_rows.extend(rows)
                if dbg.get("items"):
                    all_debug.append(dbg)
                print(f"[{task}/{label}/{pid}] matched={len(rows)}")

            # 保存该 task/label 的结果（仅成功配对记录）
            if all_rows:
                with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(["patient_id", "nbi_filename", "wli_filename", "final_score", "is_ambiguous"])
                    w.writerows(all_rows)

            with open(os.path.join(out_dir, "debug.json"), "w", encoding="utf-8") as f:
                json.dump(all_debug, f, ensure_ascii=False, indent=2)

    print("完成！")


if __name__ == "__main__":
    main()
