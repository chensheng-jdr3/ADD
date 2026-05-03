# -*- coding: utf-8 -*-
"""
MaskWeightNMI 配对算法评估脚本
================================
在 CPC-Paired 数据集上评估配对准确性，与多种基线方法对比。

数据集结构：
  dataset/NBI/{adenomas,hyperplastic_lesions}/*.png
  dataset/White_light/{adenomas,hyperplastic_lesions}/*.png
  同名文件 = 真实配对 (ground truth)

用法：
  python test/evaluate_pairing.py

输出：
  - test/results/ 下的指标表格 (CSV) 与评估报告 (JSON)
  - 终端打印的汇总表
"""
import os, sys, csv, json, time, math, argparse, re
import numpy as np
import cv2
from collections import defaultdict
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preproc_dark import compute_hard_mask, compute_dark_region
from present_paired import compute_weighted_nmi, compute_weight_sift_distance


# ======================== 配置 ========================

CFG = {
    "resize": 256,
    "nmi_bins": 64,
    "dark_range_min_ratio": 0.04,
    "dark_range_start_gray": 5,
    "dark_soft_weight": 0.2,
    "ransac_ratio": 0.85,
    "ransac_thresh": 7.0,
    "max_features": 50,
    "dist_sigma": 0.7,
    "percentile_p": 15,
}

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
# CLASSES = ["adenomas", "hyperplastic_lesions"]
# CLASSES = ['hyperplastic_lesions']  # 仅评估一个类别，避免过慢
# CLASSES = ['no_information']  # 包含无信息类
CLASSES = ['darkarea']  # 包含暗区类



# ======================== 工具函数 ========================

def safe_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def preprocess_image(img_path, cfg):
    """预处理单张图像：灰度 → 硬掩码 → 暗区 → CLAHE → 梯度。同时检测 SIFT 关键点。"""
    img = safe_imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg["resize"]:
        gray = cv2.resize(gray, (cfg["resize"], cfg["resize"]), interpolation=cv2.INTER_AREA)

    hard_mask = compute_hard_mask(gray)
    dark_region = compute_dark_region(
        gray, cfg["percentile_p"], hard_mask, min_size=2000,
        min_count_ratio=cfg["dark_range_min_ratio"],
        start_gray=cfg["dark_range_start_gray"],
    )

    # SIFT 关键点与描述子（在有效检测区域内）
    det_mask = (hard_mask > 0).astype(np.uint8) * 255
    if dark_region is not None:
        det_mask[dark_region > 0] = 0
    try:
        detector = cv2.SIFT_create(nfeatures=cfg["max_features"])
        kp, des = detector.detectAndCompute(gray, mask=det_mask)
    except Exception:
        kp, des = [], None

    # CLAHE 增强 → 梯度图
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "path": img_path,
        "gray": gray,
        "grad": grad,
        "hard_mask": hard_mask,
        "dark_region": dark_region,
        "kp": kp if kp else [],
        "des": des,
    }


# ======================== 权重计算（使用预计算 KP，避免重复 SIFT） ========================

def compute_weight_from_precomputed(gray, hard_mask, dark_region, kp_self, des_self,
                                     kp_other=None, des_other=None,
                                     dark_soft_weight=0.2, dist_sigma=0.3,
                                     ransac_ratio=0.75, ransac_thresh=3.0):
    """
    与 compute_weight_sift_distance 逻辑相同，但不重新跑 SIFT 检测，
    直接使用预计算的关键点 kp_self/des_self，大幅加速。
    """
    # 梯度幅值
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    use_inlier_coords = None
    if kp_other is not None and des_other is not None and len(kp_self) > 0 and len(kp_other) > 0 \
            and des_self is not None and des_other is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_self, des_other, k=2)
        good = [m for m_n in matches if len(m_n) == 2 for m, n in [m_n] if m.distance < ransac_ratio * n.distance]
        if len(good) >= 4:
            src_pts = np.float32([kp_self[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_other[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
            if mask is not None:
                inlier_idx = np.where(mask.ravel() != 0)[0]
                if inlier_idx.size:
                    coords = np.array([kp_self[good[i].queryIdx].pt for i in inlier_idx])
                    use_inlier_coords = [tuple(np.round(c).astype(int)) for c in coords]

    # 关键点→距离图→距离权重
    kp_map = np.ones_like(gray, dtype=np.uint8)
    if use_inlier_coords is not None:
        for x, y in use_inlier_coords:
            if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
                kp_map[y, x] = 0
    else:
        for kp in kp_self:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
                kp_map[y, x] = 0

    dist = cv2.distanceTransform(kp_map, cv2.DIST_L2, 3).astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-6)
    w_dist = np.exp(-(dist_norm ** 2) / (2 * (dist_sigma ** 2)))
    weight = w_dist * (0.3 + 0.7 * grad)

    if dark_region is not None:
        weight[dark_region > 0] *= dark_soft_weight
    if hard_mask is not None:
        weight[hard_mask == 0] = 0

    return np.clip(weight, 0, 1).astype(np.float32)


# ======================== 权重预计算与缓存 ========================

def precompute_item_weight(item, cfg):
    """计算 no-RANSAC 权重并缓存到 item['_w']（仅依赖于图像自身，与配对方无关）。"""
    gx = cv2.Sobel(item["gray"], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(item["gray"], cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    kp_map = np.ones_like(item["gray"], dtype=np.uint8)
    for kp in item["kp"]:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < kp_map.shape[1] and 0 <= y < kp_map.shape[0]:
            kp_map[y, x] = 0

    dist = cv2.distanceTransform(kp_map, cv2.DIST_L2, 3).astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-6)
    w_dist = np.exp(-(dist_norm ** 2) / (2 * (cfg["dist_sigma"] ** 2)))
    weight = w_dist * (0.3 + 0.7 * grad)

    if item["dark_region"] is not None:
        weight[item["dark_region"] > 0] *= cfg["dark_soft_weight"]
    if item["hard_mask"] is not None:
        weight[item["hard_mask"] == 0] = 0

    item["_w"] = np.clip(weight, 0, 1).astype(np.float32)


# ======================== 相似度计算方法 ========================

def sim_masked_nmi(item1, item2, cfg, cross_ransac=True):
    """MaskWeightNMI：使用预计算 KP，避免重复 SIFT"""
    if cross_ransac:
        w1 = compute_weight_from_precomputed(
            item1["gray"], item1["hard_mask"], item1["dark_region"],
            item1["kp"], item1["des"],
            kp_other=item2["kp"], des_other=item2["des"],
            dark_soft_weight=cfg["dark_soft_weight"], dist_sigma=cfg["dist_sigma"],
            ransac_ratio=cfg["ransac_ratio"], ransac_thresh=cfg["ransac_thresh"],
        )
        w2 = compute_weight_from_precomputed(
            item2["gray"], item2["hard_mask"], item2["dark_region"],
            item2["kp"], item2["des"],
            kp_other=item1["kp"], des_other=item1["des"],
            dark_soft_weight=cfg["dark_soft_weight"], dist_sigma=cfg["dist_sigma"],
            ransac_ratio=cfg["ransac_ratio"], ransac_thresh=cfg["ransac_thresh"],
        )
    else:
        w1 = item1.get("_w")
        w2 = item2.get("_w")
        if w1 is None:
            w1 = compute_weight_from_precomputed(
                item1["gray"], item1["hard_mask"], item1["dark_region"],
                item1["kp"], item1["des"],
                dark_soft_weight=cfg["dark_soft_weight"], dist_sigma=cfg["dist_sigma"],
            )
        if w2 is None:
            w2 = compute_weight_from_precomputed(
                item2["gray"], item2["hard_mask"], item2["dark_region"],
                item2["kp"], item2["des"],
                dark_soft_weight=cfg["dark_soft_weight"], dist_sigma=cfg["dist_sigma"],
            )
    return compute_weighted_nmi(item1["grad"], item2["grad"], w1, w2, bins=cfg["nmi_bins"])


def sim_nmi(item1, item2, cfg):
    """标准 NMI（无权重）"""
    w = np.ones_like(item1["grad"], dtype=np.float32)
    return compute_weighted_nmi(item1["grad"], item2["grad"], w, w, bins=cfg["nmi_bins"])


def sim_mi(item1, item2, cfg):
    """互信息 MI（不含 NMI 分母归一化）"""
    g1, g2 = item1["grad"], item2["grad"]
    bins = cfg["nmi_bins"]
    h2 = np.zeros((bins, bins), dtype=np.float64)
    bin_edges = np.linspace(0, 256, bins + 1)
    idx1 = np.clip(np.digitize(g1.ravel(), bin_edges) - 1, 0, bins - 1)
    idx2 = np.clip(np.digitize(g2.ravel(), bin_edges) - 1, 0, bins - 1)
    np.add.at(h2, (idx1, idx2), 1.0)
    total = h2.sum()
    if total < 1e-12:
        return 0.0
    pxy = h2 / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    denom = px @ py
    mi = float((pxy[nz] * np.log((pxy[nz] + 1e-12) / (denom[nz] + 1e-12))).sum())
    return mi


def sim_ssim(item1, item2, cfg=None):
    """结构相似性 SSIM（灰度图上计算）"""
    g1 = item1["gray"].astype(np.float32)
    g2 = item2["gray"].astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / (den + 1e-8)))


def sim_hist_correlation(item1, item2, cfg=None):
    """直方图相关性（梯度图上计算）"""
    g1, g2 = item1["grad"].ravel(), item2["grad"].ravel()
    h1 = np.histogram(g1, bins=64, range=(0, 256))[0].astype(np.float64)
    h2 = np.histogram(g2, bins=64, range=(0, 256))[0].astype(np.float64)
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL))


def sim_sift_match_ratio(item1, item2, cfg=None):
    """SIFT 匹配比例（Lowe's ratio test 后 good matches / min(kp1, kp2)）"""
    des1, des2 = item1["des"], item2["des"]
    kp1_len = len(item1["kp"])
    kp2_len = len(item2["kp"])
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return 0.0
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    return len(good) / max(1, min(kp1_len, kp2_len))


# ======================== 评估核心 ========================

def evaluate_class(nbi_items, wli_items, nbi_names, wli_names, methods, cfg):
    """
    对一个类别的 N×M 配对进行评估。
    返回:
      metrics: {method_name: {metric_name: value}}
      details: {method_name: {nbi_name: (best_wli_name, score, rank, is_correct, topK_correct)}}
    """
    N, M = len(nbi_items), len(wli_items)
    gt_map = {n: n for n in nbi_names}  # 同名配对为 ground truth

    results = {}
    for method_name, sim_fn in methods.items():
        print(f"  评估 {method_name}  ({N}×{M}={N*M} pairs)...", end=" ", flush=True)
        t0 = time.time()
        sim_mat = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            for j in range(M):
                sim_mat[i, j] = sim_fn(nbi_items[i], wli_items[j], cfg)

        # 计算排名指标
        ranks = np.zeros(N, dtype=int)
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        mrr = 0.0
        detail = {}

        for i in range(N):
            gt_name = nbi_names[i]
            gt_j = wli_names.index(gt_name) if gt_name in wli_names else -1
            sorted_j = np.argsort(-sim_mat[i])  # 降序
            if gt_j >= 0:
                rank = int(np.where(sorted_j == gt_j)[0][0]) + 1
            else:
                rank = M + 1  # 不在列表中
            ranks[i] = rank
            mrr += 1.0 / max(1, rank)

            if rank == 1:
                top1_correct += 1
            if rank <= 3:
                top3_correct += 1
            if rank <= 5:
                top5_correct += 1

            best_j = sorted_j[0]
            best_score = float(sim_mat[i, best_j])
            is_correct = (rank == 1)

            detail[nbi_names[i]] = {
                "best_wli": wli_names[best_j],
                "score": best_score,
                "rank": rank,
                "correct": is_correct,
                "top3": rank <= 3,
                "top5": rank <= 5,
                "gt_wli": gt_name if gt_j >= 0 else None,
            }

        # 聚合指标
        top1 = top1_correct / N if N > 0 else 0
        top3 = top3_correct / N if N > 0 else 0
        top5 = top5_correct / N if N > 0 else 0
        mrr_val = mrr / N if N > 0 else 0

        # Cohen's d：用完整 N×M 相似度矩阵中所有 true-pair 与所有 false-pair 的分数
        all_true_scores = []
        all_false_scores = []
        for i in range(N):
            gt_name = nbi_names[i]
            for j in range(M):
                score = float(sim_mat[i, j])
                if wli_names[j] == gt_name:
                    all_true_scores.append(score)
                else:
                    all_false_scores.append(score)

        mean_correct = np.mean(all_true_scores) if all_true_scores else 0.0
        mean_incorrect = np.mean(all_false_scores) if all_false_scores else 0.0
        score_gap = mean_correct - mean_incorrect
        if all_true_scores and all_false_scores:
            pooled_std = math.sqrt(
                (np.std(all_true_scores) ** 2 + np.std(all_false_scores) ** 2) / 2
            )
            cohens_d = score_gap / (pooled_std + 1e-8)
        else:
            cohens_d = 0.0

        elapsed = time.time() - t0
        print(f"完成 ({elapsed:.1f}s)")

        results[method_name] = {
            "top1_acc": top1,
            "top3_acc": top3,
            "top5_acc": top5,
            "mrr": mrr_val,
            "mean_correct_score": mean_correct,
            "mean_incorrect_score": mean_incorrect,
            "score_gap": score_gap,
            "cohens_d": cohens_d,
            "time_seconds": elapsed,
        }
        results[method_name]["_detail"] = detail

    return results


# ======================== 按患者分组评估 ========================

def evaluate_per_patient(nbi_items, wli_items, nbi_names, wli_names, methods, cfg,
                         include_single=True):
    """
    按患者分组评估：从文件名中提取患者 ID（首个 '-' 前的数字），
    仅在同一个患者内部进行 NBI-WLI 匹配，跨患者不比较。

    include_single=False 时排除仅含 1 对图像的患者（无歧义，自动正确）。
    准确率仍在每张 NBI 图像层面汇总。
    """
    def get_pid(name):
        m = re.match(r'^(\d+)', os.path.basename(name))
        return m.group(1) if m else '__unknown__'

    # 按患者分组
    nbi_groups = defaultdict(list)  # pid -> [(item, name)]
    for i, name in enumerate(nbi_names):
        nbi_groups[get_pid(name)].append((nbi_items[i], name))

    wli_groups = defaultdict(list)
    for j, name in enumerate(wli_names):
        wli_groups[get_pid(name)].append((wli_items[j], name))

    all_pids = sorted(set(nbi_groups.keys()) | set(wli_groups.keys()))

    # 统计患者信息
    single_pair_pids = set()
    multi_pair_pids = set()
    for pid in all_pids:
        n_list = nbi_groups.get(pid, [])
        w_list = wli_groups.get(pid, [])
        if len(n_list) == 1 and len(w_list) == 1:
            single_pair_pids.add(pid)
        else:
            multi_pair_pids.add(pid)

    # 根据 include_single 决定评估范围
    if include_single:
        eval_pids = all_pids
    else:
        eval_pids = [pid for pid in all_pids if pid in multi_pair_pids]

    print(f"  患者总数: {len(all_pids)}  (仅含1对: {len(single_pair_pids)}, "
          f"多对: {len(multi_pair_pids)})")
    if not include_single:
        print(f"  排除单对患者，实际评估: {len(eval_pids)} 个患者")

    results = {}
    for method_name, sim_fn in methods.items():
        print(f"  评估 {method_name}...", end=" ", flush=True)
        t0 = time.time()

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        mrr_sum = 0.0
        total_ranked = 0

        all_true_scores = []
        all_false_scores = []
        detail = {}

        for pid in eval_pids:
            n_list = nbi_groups.get(pid, [])
            w_list = wli_groups.get(pid, [])

            if len(n_list) == 0 or len(w_list) == 0:
                continue

            n_local = len(n_list)
            m_local = len(w_list)

            # 构建患者内相似度矩阵
            sim_mat = np.zeros((n_local, m_local), dtype=np.float32)
            for i in range(n_local):
                for j in range(m_local):
                    sim_mat[i, j] = sim_fn(n_list[i][0], w_list[j][0], cfg)

            local_nbi_names = [t[1] for t in n_list]
            local_wli_names = [t[1] for t in w_list]

            for i in range(n_local):
                gt_name = local_nbi_names[i]
                try:
                    gt_j = local_wli_names.index(gt_name)
                except ValueError:
                    continue  # 该 NBI 无对应 ground truth

                sorted_j = np.argsort(-sim_mat[i])
                rank = int(np.where(sorted_j == gt_j)[0][0]) + 1 if m_local > 1 else 1

                total_ranked += 1
                mrr_sum += 1.0 / max(1, rank)

                if rank == 1:
                    top1_correct += 1
                if rank <= 3:
                    top3_correct += 1
                if rank <= 5:
                    top5_correct += 1

                # Cohen's d 分数收集
                for j in range(m_local):
                    score = float(sim_mat[i, j])
                    if local_wli_names[j] == gt_name:
                        all_true_scores.append(score)
                    else:
                        all_false_scores.append(score)

                best_j = sorted_j[0]
                detail[f"{pid}/{local_nbi_names[i]}"] = {
                    "best_wli": local_wli_names[best_j],
                    "score": float(sim_mat[i, best_j]),
                    "rank": rank,
                    "correct": rank == 1,
                    "top3": rank <= 3,
                    "top5": rank <= 5,
                    "gt_wli": gt_name,
                    "patient": pid,
                    "n_wli_in_patient": m_local,
                }

        # 汇总指标
        top1 = top1_correct / total_ranked if total_ranked > 0 else 0.0
        top3 = top3_correct / total_ranked if total_ranked > 0 else 0.0
        top5 = top5_correct / total_ranked if total_ranked > 0 else 0.0
        mrr = mrr_sum / total_ranked if total_ranked > 0 else 0.0

        mean_correct = np.mean(all_true_scores) if all_true_scores else 0.0
        mean_incorrect = np.mean(all_false_scores) if all_false_scores else 0.0
        score_gap = mean_correct - mean_incorrect
        if all_true_scores and all_false_scores:
            pooled_std = math.sqrt(
                (np.std(all_true_scores) ** 2 + np.std(all_false_scores) ** 2) / 2)
            cohens_d = score_gap / (pooled_std + 1e-8)
        else:
            cohens_d = 0.0

        elapsed = time.time() - t0
        print(f"完成 ({elapsed:.1f}s)")

        results[method_name] = {
            "top1_acc": top1,
            "top3_acc": top3,
            "top5_acc": top5,
            "mrr": mrr,
            "mean_correct_score": mean_correct,
            "mean_incorrect_score": mean_incorrect,
            "score_gap": score_gap,
            "cohens_d": cohens_d,
            "time_seconds": elapsed,
            "_detail": detail,
            "_total_ranked": total_ranked,
            "_n_patients": len(eval_pids),
            "_include_single": include_single,
        }

    return results


# ======================== 参数扫描 ========================

def _truncate_kps(items, max_features):
    """截断预计算 KP 到前 max_features 个（SIFT 按响应强度降序），返回恢复函数。"""
    saved = {}
    for idx, item in enumerate(items):
        saved[idx] = (item["kp"], item["des"])
        if max_features < len(item["kp"]):
            item["kp"] = item["kp"][:max_features]
            if item["des"] is not None:
                item["des"] = item["des"][:max_features]
    def restore():
        for idx, (kp, des) in saved.items():
            items[idx]["kp"] = kp
            items[idx]["des"] = des
    return restore


def sweep_base_params(nbi_items, wli_items, nbi_names, wli_names, cfg, ts_dir):
    """扫描 max_features × dist_sigma（MaskWeightNMI no RANSAC）。"""
    max_features_list = [25, 50, 100, 200, 400, 800]
    dist_sigma_list = [0.1, 0.3, 0.5, 0.7, 1.0]

    N, M = len(nbi_items), len(wli_items)
    orig_features = cfg["max_features"]
    orig_sigma = cfg["dist_sigma"]

    # 清除缓存权重（参数变动后缓存失效）
    for item in nbi_items + wli_items:
        item.pop("_w", None)

    print(f"\n{'='*70}")
    print(f"Phase 1 — 扫描: max_features × dist_sigma  (MaskWeightNMI no RANSAC)")
    print(f"  max_features: {max_features_list}")
    print(f"  dist_sigma:   {dist_sigma_list}")
    print(f"  N={N}, M={M}  共 {len(max_features_list) * len(dist_sigma_list)} 组")
    print(f"{'='*70}")

    header = f"{'max_feat':<12} {'dist_sigma':<12} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'MRR':>8} {'ScoreGap':>10} {'Cohen d':>8} {'Time(s)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    best_top1 = -1.0
    best_params = None
    all_rows = []

    methods = {
        "MaskWeightNMI (no RANSAC)": lambda i1, i2, c: sim_masked_nmi(i1, i2, c, cross_ransac=False),
    }

    for mf in max_features_list:
        for ds in dist_sigma_list:
            cfg["max_features"] = mf
            cfg["dist_sigma"] = ds

            restore_nbi = _truncate_kps(nbi_items, mf)
            restore_wli = _truncate_kps(wli_items, mf)

            t0 = time.time()
            results = evaluate_class(nbi_items, wli_items, nbi_names, wli_names, methods, cfg)
            elapsed = time.time() - t0
            m = results["MaskWeightNMI (no RANSAC)"]

            row = [mf, ds, m["top1_acc"], m["top3_acc"], m["top5_acc"],
                   m["mrr"], m["score_gap"], m["cohens_d"], elapsed]
            all_rows.append(row)

            print(f"{mf:<12} {ds:<12} {m['top1_acc']:8.2%} {m['top3_acc']:8.2%} "
                  f"{m['top5_acc']:8.2%} {m['mrr']:8.3f} "
                  f"{m['score_gap']:10.4f} {m['cohens_d']:8.3f} {elapsed:8.1f}")

            if m["top1_acc"] > best_top1:
                best_top1 = m["top1_acc"]
                best_params = (mf, ds)

            restore_nbi()
            restore_wli()

    cfg["max_features"] = orig_features
    cfg["dist_sigma"] = orig_sigma

    print(f"\n最佳参数: max_features={best_params[0]}, dist_sigma={best_params[1]}  (Top-1={best_top1:.2%})")

    sweep_csv = os.path.join(ts_dir, "sweep_base.csv")
    with open(sweep_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["max_features", "dist_sigma", "Top-1", "Top-3", "Top-5", "MRR", "ScoreGap", "Cohen_d", "Time_s"])
        w.writerows(all_rows)
    print(f"结果已保存至: {sweep_csv}")

    return all_rows, best_params


def sweep_ransac_params(nbi_items, wli_items, nbi_names, wli_names, cfg, ts_dir):
    """扫描 ransac_ratio × ransac_thresh（MaskWeightNMI full，含 RANSAC 交叉过滤）。"""
    ransac_ratio_list = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    ransac_thresh_list = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    N, M = len(nbi_items), len(wli_items)
    orig_ratio = cfg["ransac_ratio"]
    orig_thresh = cfg["ransac_thresh"]

    print(f"\n{'='*70}")
    print(f"Phase 2 — 扫描: ransac_ratio × ransac_thresh  (MaskWeightNMI full)")
    print(f"  max_features={cfg['max_features']}, dist_sigma={cfg['dist_sigma']}")
    print(f"  ransac_ratio:  {ransac_ratio_list}")
    print(f"  ransac_thresh: {ransac_thresh_list}")
    print(f"  N={N}, M={M}  共 {len(ransac_ratio_list) * len(ransac_thresh_list)} 组")
    print(f"{'='*70}")

    header = f"{'ransac_ratio':<14} {'ransac_thresh':<15} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'MRR':>8} {'ScoreGap':>10} {'Cohen d':>8} {'Time(s)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    best_top1 = -1.0
    best_params = None
    all_rows = []

    methods = {
        "MaskWeightNMI (full)": lambda i1, i2, c: sim_masked_nmi(i1, i2, c, cross_ransac=True),
    }

    for rr in ransac_ratio_list:
        for rt in ransac_thresh_list:
            cfg["ransac_ratio"] = rr
            cfg["ransac_thresh"] = rt

            t0 = time.time()
            results = evaluate_class(nbi_items, wli_items, nbi_names, wli_names, methods, cfg)
            elapsed = time.time() - t0
            m = results["MaskWeightNMI (full)"]

            row = [rr, rt, m["top1_acc"], m["top3_acc"], m["top5_acc"],
                   m["mrr"], m["score_gap"], m["cohens_d"], elapsed]
            all_rows.append(row)

            print(f"{rr:<14} {rt:<15} {m['top1_acc']:8.2%} {m['top3_acc']:8.2%} "
                  f"{m['top5_acc']:8.2%} {m['mrr']:8.3f} "
                  f"{m['score_gap']:10.4f} {m['cohens_d']:8.3f} {elapsed:8.1f}")

            if m["top1_acc"] > best_top1:
                best_top1 = m["top1_acc"]
                best_params = (rr, rt)

    cfg["ransac_ratio"] = orig_ratio
    cfg["ransac_thresh"] = orig_thresh

    print(f"\n最佳参数: ransac_ratio={best_params[0]}, ransac_thresh={best_params[1]}  (Top-1={best_top1:.2%})")

    sweep_csv = os.path.join(ts_dir, "sweep_ransac.csv")
    with open(sweep_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ransac_ratio", "ransac_thresh", "Top-1", "Top-3", "Top-5", "MRR", "ScoreGap", "Cohen_d", "Time_s"])
        w.writerows(all_rows)
    print(f"结果已保存至: {sweep_csv}")

    return all_rows, best_params


# ======================== 主函数 ========================

def main():
    ap = argparse.ArgumentParser(description="Evaluate pairing algorithms on CPC-Paired dataset")
    ap.add_argument("--sweep_base", action="store_true",
                    help="扫描 max_features × dist_sigma（MaskWeightNMI no RANSAC）")
    ap.add_argument("--sweep_ransac", action="store_true",
                    help="扫描 ransac_ratio × ransac_thresh（MaskWeightNMI full，含 RANSAC）")
    ap.add_argument("--per_patient", action="store_true",
                    help="按患者分组评估（文件名首个 '-' 前的数字为患者 ID）")
    ap.add_argument("--exclude_single", action="store_true",
                    help="配合 --per_patient，排除仅含 1 对图像的患者")
    args = ap.parse_args()

    methods = {
        "NMI (baseline)":           sim_nmi,
        "MaskWeightNMI (no RANSAC)": lambda i1, i2, c: sim_masked_nmi(i1, i2, c, cross_ransac=False),
        "MaskWeightNMI (full)":     lambda i1, i2, c: sim_masked_nmi(i1, i2, c, cross_ransac=True),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    ts_dir = os.path.join(out_dir, f"pairing_eval_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(ts_dir, exist_ok=True)
    print(f"输出目录: {ts_dir}")

    all_class_results = {}

    for cls in CLASSES:
        nbi_dir = os.path.join(DATASET_ROOT, "NBI", cls)
        wli_dir = os.path.join(DATASET_ROOT, "White_light", cls)
        if not os.path.isdir(nbi_dir) or not os.path.isdir(wli_dir):
            print(f"跳过 {cls}: 目录不存在")
            continue

        nbi_paths = sorted([os.path.join(nbi_dir, f) for f in os.listdir(nbi_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        wli_paths = sorted([os.path.join(wli_dir, f) for f in os.listdir(wli_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        nbi_names = [os.path.basename(p) for p in nbi_paths]
        wli_names = [os.path.basename(p) for p in wli_paths]

        print(f"\n{'='*60}")
        print(f"类别: {cls}  |  NBI: {len(nbi_paths)}   WLI: {len(wli_paths)}")
        print(f"{'='*60}")

        # 预处理所有图像（sweep_base 用 max_features=800 以保证可截断）
        if args.sweep_base:
            CFG["max_features"] = max(CFG["max_features"], 800)
        print("预处理中...")
        nbi_items = []
        for p in nbi_paths:
            item = preprocess_image(p, CFG)
            if item:
                nbi_items.append(item)
        wli_items = []
        for p in wli_paths:
            item = preprocess_image(p, CFG)
            if item:
                wli_items.append(item)
        print(f"  NBI 有效: {len(nbi_items)}, WLI 有效: {len(wli_items)}")

        # 预计算 no-RANSAC 权重（避免 evaluate_class 内重复算）
        if not args.sweep_base:
            print("预计算权重缓存...", end=" ", flush=True)
            for item in nbi_items + wli_items:
                precompute_item_weight(item, CFG)
            print("完成")

        if args.sweep_base:
            sweep_base_params(nbi_items, wli_items, nbi_names, wli_names, CFG, ts_dir)
        elif args.sweep_ransac:
            sweep_ransac_params(nbi_items, wli_items, nbi_names, wli_names, CFG, ts_dir)
        else:
            # 正常评估模式
            if args.per_patient:
                results = evaluate_per_patient(nbi_items, wli_items, nbi_names, wli_names,
                                               methods, CFG,
                                               include_single=not args.exclude_single)
            else:
                results = evaluate_class(nbi_items, wli_items, nbi_names, wli_names, methods, CFG)
            all_class_results[cls] = results

            # 打印该类别的结果
            print(f"\n{'方法':<30} {'Top-1':>6} {'MRR':>6} {'ScoreGap':>9} {'Time(s)':>8}")
            print("-" * 65)
            for method_name, metrics in results.items():
                print(f"{method_name:<30} {metrics['top1_acc']:6.2%} "
                      f"{metrics['mrr']:6.3f} "
                      f"{metrics['score_gap']:9.4f} {metrics['time_seconds']:8.1f}")
            if args.per_patient:
                first_method = next(iter(results.values()))
                n_pat = first_method.get("_n_patients", "?")
                n_ranked = first_method.get("_total_ranked", "?")
                print(f"  [按患者: {n_pat} 个患者, 共评估 {n_ranked} 张 NBI 图像]")
            print()

    if args.sweep_base or args.sweep_ransac:
        print(f"结果已保存至: {ts_dir}")
        return

    # 保存结果
    summary_csv = os.path.join(ts_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Top-1 Acc", "MRR", "Score Gap"])
        for cls, results in all_class_results.items():
            for method_name, metrics in results.items():
                w.writerow([f"{cls}/{method_name}",
                            f"{metrics['top1_acc']:.4f}", f"{metrics['mrr']:.4f}",
                            f"{metrics['score_gap']:.4f}"])

    json_out = {}
    for cls, results in all_class_results.items():
        json_out[cls] = {}
        for method_name, metrics in results.items():
            json_out[cls][method_name] = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(os.path.join(ts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    for cls, results in all_class_results.items():
        cls_dir = os.path.join(ts_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for method_name, metrics in results.items():
            if "_detail" in metrics:
                detail_path = os.path.join(cls_dir, f"{method_name.replace(' ', '_')}_detail.csv")
                with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    if args.per_patient:
                        w.writerow(["nbi_key", "patient", "best_wli", "score", "rank",
                                    "correct", "top3", "top5", "gt_wli", "n_wli_in_patient"])
                        for nbi_name, d in metrics["_detail"].items():
                            w.writerow([nbi_name, d.get("patient", ""), d["best_wli"], d["score"],
                                        d["rank"], d["correct"], d["top3"], d["top5"],
                                        d["gt_wli"], d.get("n_wli_in_patient", "")])
                    else:
                        w.writerow(["nbi_name", "best_wli", "score", "rank",
                                    "correct", "top3", "top5", "gt_wli"])
                        for nbi_name, d in metrics["_detail"].items():
                            w.writerow([nbi_name, d["best_wli"], d["score"], d["rank"],
                                        d["correct"], d["top3"], d["top5"], d["gt_wli"]])

    print(f"结果已保存至: {ts_dir}")


if __name__ == "__main__":
    main()
