import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema


def compute_hard_mask(gray: np.ndarray):
    """
    计算硬掩码：黑边 + 高光（这些区域直接排除，不参与计算）
    返回：uint8，255=有效，0=无效
    """
    # 黑边掩码：移除与边界连通的暗区域
    dark_inv = cv2.bitwise_not(cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY_INV)[1])
    contours, _ = cv2.findContours(dark_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_mask = np.zeros_like(gray, dtype=np.uint8)
    if contours:
        cv2.drawContours(black_mask, [max(contours, key=cv2.contourArea)], 0, 255, -1)
    else:
        black_mask[:] = 255

    # 高光掩码：灰度>240 认为是反光区域
    highlight = (gray > 240).astype(np.uint8) * 255
    if highlight.any():
        highlight = cv2.dilate(highlight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return ((black_mask > 0) & (highlight == 0)).astype(np.uint8) * 255


# ----------------- 峰谷阈值法 -----------------

def compute_threshold_by_hist_peaks(gray: np.ndarray, black_edge_mask: np.ndarray = None,
                                     smooth_sigma: float = 2.0, fallback_ratio: float = 0.4,
                                     start_gray: int = 10, min_peak_height_ratio: float = 0.03,
                                     min_dark_pixels: float = 0.04):
    """
    峰谷法计算阈值（首选方法）。

    参数说明：
      - `min_dark_pixels` 可为整数（表示绝对像素数）或小数（0-1，表示相对于有效像素数的比例）。

    返回： (threshold or None, peak_info dict)
    """
    valid_mask = (black_edge_mask > 0) if black_edge_mask is not None else np.ones_like(gray, dtype=bool)
    valid_pixels = gray[valid_mask]

    peak_info = {
        'smoothed_hist': None,
        'peaks': [],
        'valleys': [],
        'tried_pairs': [],
        'selected_peak': None,
        'selected_valley': None,
        'dark_fraction': None,
        'fallback_reason': None
    }

    if valid_pixels.size == 0:
        peak_info['fallback_reason'] = '无有效像素'
        return None, peak_info

    hist = np.bincount(valid_pixels.ravel(), minlength=256).astype(np.float64)
    smoothed_hist = gaussian_filter1d(hist, sigma=smooth_sigma)
    peak_info['smoothed_hist'] = smoothed_hist

    peaks_idx = argrelextrema(smoothed_hist, np.greater, order=2)[0]
    valleys_idx = argrelextrema(smoothed_hist, np.less, order=2)[0]
    peak_info['peaks'] = peaks_idx.tolist()
    peak_info['valleys'] = valleys_idx.tolist()

    peaks_idx = peaks_idx[peaks_idx >= start_gray]
    if len(peaks_idx) == 0:
        peak_info['fallback_reason'] = f'在灰度>{start_gray}范围内未找到峰'
        return None, peak_info

    min_peak_height = smoothed_hist.max() * min_peak_height_ratio

    if 0 < min_dark_pixels < 1:
        computed_min_dark = max(1, int(round(valid_pixels.size * float(min_dark_pixels))))
    else:
        computed_min_dark = int(round(float(min_dark_pixels)))

    for peak in peaks_idx:
        if smoothed_hist[peak] < min_peak_height:
            continue
        valleys_after = valleys_idx[valleys_idx > peak]
        if len(valleys_after) == 0:
            continue
        valley = valleys_after[0]

        dark_count = int(np.sum(valid_pixels <= valley))
        dark_fraction = dark_count / valid_pixels.size
        peak_info['tried_pairs'].append((int(peak), int(valley), int(dark_count), float(dark_fraction)))

        if dark_count < computed_min_dark:
            continue

        if dark_fraction > fallback_ratio:
            peak_info['fallback_reason'] = f'候选暗像素占比({dark_fraction:.1%})>{fallback_ratio:.1%}，退化到范围法'
            return None, peak_info

        peak_info['selected_peak'] = int(peak)
        peak_info['selected_valley'] = int(valley)
        peak_info['dark_fraction'] = float(dark_fraction)
        return float(valley), peak_info

    peak_info['fallback_reason'] = '所有候选峰的谷均不满足最小暗像素数或未找到谷，退化到范围法'
    return None, peak_info


def compute_threshold_by_range(gray: np.ndarray, black_edge_mask: np.ndarray = None,
                               range_percent: int = 15, min_count_ratio: float = 0.01,
                               start_gray: int = 10,
                               use_peak_valley: bool = True, smooth_sigma: float = 2.0,
                               fallback_ratio: float = 0.4, min_dark_pixels: float = 0.15):
    """
    优先尝试峰谷法，失败则退化到范围法
    返回 (threshold, method_info)
    """
    method_info = {'method': 'range', 'peak_info': None}

    if use_peak_valley:
        threshold, peak_info = compute_threshold_by_hist_peaks(
            gray, black_edge_mask,
            smooth_sigma=smooth_sigma,
            fallback_ratio=fallback_ratio,
            start_gray=start_gray,
            min_dark_pixels=min_dark_pixels,
        )
        method_info['peak_info'] = peak_info
        if threshold is not None:
            method_info['method'] = 'peak_valley'
            return threshold, method_info

    valid_mask = (black_edge_mask > 0) if black_edge_mask is not None else np.ones_like(gray, dtype=bool)
    valid_pixels = gray[valid_mask]
    if valid_pixels.size == 0:
        return 50, method_info

    hist = np.bincount(valid_pixels.ravel(), minlength=256)
    min_count = max(1, int(round(valid_pixels.size * min_count_ratio)))
    start = int(np.clip(start_gray, 0, 255))

    g_min = None
    for g in range(start, 256):
        if hist[g] >= min_count:
            g_min = g
            break
    g_max = None
    for g in range(255, start - 1, -1):
        if hist[g] >= min_count:
            g_max = g
            break

    if g_min is not None and g_max is not None and g_max > g_min:
        p = np.clip(range_percent, 0, 100) / 100.0
        threshold = g_min + p * (g_max - g_min)
    else:
        threshold = np.percentile(valid_pixels, range_percent)

    return threshold, method_info


def compute_dark_area_mask_with_morphology(gray: np.ndarray, threshold: float,
                                           black_edge_mask: np.ndarray = None,
                                           min_component_size: int = 1000,
                                           morph_open_kernel: int = 3,
                                           morph_close_kernel: int = 7,
                                           morph_fill_holes: bool = True,
                                           morph_convex_hull: bool = False,
                                           return_steps: bool = False,
                                           max_dark_area_ratio: float = 0.3):
    """
    计算暗区掩码：最大连通域 → 形态学优化 → 再取最大连通域
    """
    total_pixels = gray.size
    max_allowed_dark_pixels = int(total_pixels * max_dark_area_ratio)

    # 1. 二值化：提取暗区域
    _, dark_binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # 2. 屏蔽黑边
    if black_edge_mask is not None:
        dark_binary[black_edge_mask == 0] = 0

    # 3. 第一次找最大连通域
    num_labels, labels = cv2.connectedComponents(dark_binary, connectivity=8)

    if num_labels <= 1:
        if return_steps:
            return np.zeros_like(gray, dtype=np.uint8), {}
        return np.zeros_like(gray, dtype=np.uint8)

    # 找最大分量
    max_area = 0
    max_label = 0
    for label_id in range(1, num_labels):
        area = np.sum(labels == label_id)
        if area > max_area:
            max_area = area
            max_label = label_id

    if max_area < min_component_size:
        if return_steps:
            return np.zeros_like(gray, dtype=np.uint8), {}
        return np.zeros_like(gray, dtype=np.uint8)

    dark_mask = (labels == max_label).astype(np.uint8) * 255

    steps = {}
    if return_steps:
        steps['step_0_first_cc'] = dark_mask.copy()

    # Ensure kernels are ints and odd (if >0)
    morph_open_kernel = int(morph_open_kernel) if morph_open_kernel is not None else 0
    morph_close_kernel = int(morph_close_kernel) if morph_close_kernel is not None else 0
    morph_open_kernel = (morph_open_kernel | 1) if morph_open_kernel > 0 else 0
    morph_close_kernel = (morph_close_kernel | 1) if morph_close_kernel > 0 else 0

    h, w = gray.shape
    max_kernel = min(h, w) if min(h, w) > 0 else 1

    # cap kernels to image size
    if morph_open_kernel > max_kernel:
        morph_open_kernel = max_kernel | 1
    if morph_close_kernel > max_kernel:
        morph_close_kernel = max_kernel | 1

    # 4. 形态学优化
    # 4a. 开运算（去噪点）
    if morph_open_kernel > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (morph_open_kernel, morph_open_kernel))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_open)
        if return_steps:
            steps['step_1_after_open'] = dark_mask.copy()

    # 4b. 闭运算（填缝隙、平滑边缘）
    dark_mask_before_close = dark_mask.copy()  # 备份闭运算前状态

    if morph_close_kernel > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (morph_close_kernel, morph_close_kernel))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_close)
        if return_steps:
            steps['step_2_after_close'] = dark_mask.copy()

    # 4c1. 第二次开运算（second_open_kernel = first close kernel）
    morph_second_open_kernel = morph_close_kernel
    morph_second_open_kernel = int(morph_second_open_kernel) if morph_second_open_kernel is not None else 0
    morph_second_open_kernel = (morph_second_open_kernel | 1) if morph_second_open_kernel > 0 else 0

    # cap second open
    if morph_second_open_kernel > max_kernel:
        morph_second_open_kernel = max_kernel | 1

    if morph_second_open_kernel > 0:
        k_open2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (morph_second_open_kernel, morph_second_open_kernel))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_open2)
        if return_steps:
            steps['step_3_after_second_open'] = dark_mask.copy()

    # 4c2. 第二次闭运算（second_close_kernel = second_open_kernel * 2，转为奇数）
    morph_second_close_kernel = morph_second_open_kernel * 2
    morph_second_close_kernel = int(morph_second_close_kernel)
    morph_second_close_kernel = (morph_second_close_kernel | 1) if morph_second_close_kernel > 0 else 0

    # cap second close to image size
    if morph_second_close_kernel > max_kernel:
        morph_second_close_kernel = max_kernel | 1

    # 备份第二次闭运算前的状态以便回退
    dark_mask_before_second_close = dark_mask.copy()

    if morph_second_close_kernel > 0:
        k_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (morph_second_close_kernel, morph_second_close_kernel))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_close2)
        if return_steps:
            steps['step_4_after_second_close'] = dark_mask.copy()

    # 4d. 填充孔洞（使用轮廓层级）
    if morph_fill_holes:
        contours, hierarchy = cv2.findContours(dark_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours and hierarchy is not None:
            filled_mask = dark_mask.copy()
            for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
                if hier[3] != -1:
                    cv2.drawContours(filled_mask, [contour], -1, 255, -1)
            dark_mask = filled_mask
        if return_steps:
            steps['step_4_after_fill_holes'] = dark_mask.copy()

    # 4e. 凸包（可选，默认关闭）
    if morph_convex_hull:
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            dark_mask = np.zeros_like(dark_mask)
            cv2.drawContours(dark_mask, [hull], -1, 255, -1)
        if return_steps:
            steps['step_5_after_convex_hull'] = dark_mask.copy()

    # 5. 第二次找最大连通域
    num_labels2, labels2 = cv2.connectedComponents(dark_mask, connectivity=8)
    if num_labels2 <= 1:
        if return_steps:
            return np.zeros_like(gray, dtype=np.uint8), steps
        return np.zeros_like(gray, dtype=np.uint8)

    max_area2 = 0
    max_label2 = 0
    for label_id in range(1, num_labels2):
        area = np.sum(labels2 == label_id)
        if area > max_area2:
            max_area2 = area
            max_label2 = label_id

    if max_area2 < min_component_size:
        if return_steps:
            return np.zeros_like(gray, dtype=np.uint8), steps
        return np.zeros_like(gray, dtype=np.uint8)

    # 构建最终掩码：255 表示暗区，0 表示非暗区
    final_mask = np.zeros_like(gray, dtype=np.uint8)
    final_mask[labels2 == max_label2] = 255

    if return_steps:
        steps['step_6_final_second_cc'] = final_mask.copy()
        return final_mask, steps

    return final_mask


def compute_dark_region(gray, range_percent=15, hard_mask=None, min_size=2000,
                        min_count_ratio=0.01, start_gray=10, use_peak_valley=True,
                        smooth_sigma=2.0, fallback_ratio=0.4, min_dark_pixels=0.04,
                        morph_open_kernel=3, morph_close_kernel=7, morph_fill_holes=True):
    """
    基于提取阈值 + 形态学的暗区计算流程，返回 255=暗区, 0=非暗区
    """
    if gray is None or gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    # 1) 硬掩码
    hard = hard_mask if hard_mask is not None else compute_hard_mask(gray)

    # 2) 计算阈值（峰谷法优先）
    thresh, method_info = compute_threshold_by_range(
        gray, black_edge_mask=hard, range_percent=range_percent,
        min_count_ratio=min_count_ratio, start_gray=start_gray,
        use_peak_valley=use_peak_valley, smooth_sigma=smooth_sigma,
        fallback_ratio=fallback_ratio, min_dark_pixels=min_dark_pixels
    )

    # 3) 用形态学计算暗区掩码并返回
    dark_mask = compute_dark_area_mask_with_morphology(
        gray, thresh, black_edge_mask=hard, min_component_size=min_size,
        morph_open_kernel=morph_open_kernel, morph_close_kernel=morph_close_kernel,
        morph_fill_holes=morph_fill_holes, morph_convex_hull=False, return_steps=False
    )

    # 调整返回格式：保持与旧代码兼容（255 表示暗区）
    return dark_mask
