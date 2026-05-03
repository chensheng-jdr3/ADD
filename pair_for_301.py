# -*- coding: utf-8 -*-
"""
轻量脚本：对“扁平化”患者目录（每个患者目录下含 NBI/ 和 WLI/ 子目录）运行配对。
脚本名：pair_for_301.py
用途：最小改动复用 `present_paired.match_patient`，把结果写到输出目录下的时间戳子目录。
"""

import os
import csv
import json
import argparse
from datetime import datetime

from present_paired import match_patient, list_images, CFG


def run_flat(dataset_root, out_root):
    ts_dir = os.path.join(out_root, f"pair_301_{datetime.now():%Y%m%d%H%M}")
    os.makedirs(ts_dir, exist_ok=True)
    entries = [d for d in sorted(os.listdir(dataset_root)) if os.path.isdir(os.path.join(dataset_root, d))]

    # 固定裁剪框：top, bottom, left, right
    CFG["crop_box"] = (40, 1040, 700, 1855)
    CFG["no_match_score_thresh"] = 0.08
    # 本脚本默认的类别阈值（按轻度、中度、重度、癌 递减）：
    CFG["class_thresh"] = {
        "轻度": 0.06,
        "中度": 0.06,
        "重度": 0.06,
        "癌": 0.06,
    }

    all_rows = []
    all_debug = []
    stats = {}

    # 假定 dataset_root 的第一层为 label，遍历每个 label 下的患者目录
    for label in entries:
        label_path = os.path.join(dataset_root, label)
        patients = [d for d in sorted(os.listdir(label_path)) if os.path.isdir(os.path.join(label_path, d))]

        # 初始化统计
        stats.setdefault(label, {"total_patients": 0, "no_pairs": 0, "pruned_count": 0})

        for sub in patients:
            pdir = os.path.join(label_path, sub)
            nbi_dir = os.path.join(pdir, "NBI")
            wli_dir = os.path.join(pdir, "WLI")
            if not (os.path.isdir(nbi_dir) and os.path.isdir(wli_dir)):
                print(f"跳过 (缺少 NBI 或 WLI 子目录): {label}/{sub}")
                continue

            pid = f"{label}/{sub}"
            stats[label]["total_patients"] += 1
            rows, dbg, nbi_count, wli_count = match_patient(pid, list_images(nbi_dir), list_images(wli_dir), CFG, ts_dir, pdir, label=label)
            all_rows.extend(rows)
            if dbg.get("items"):
                all_debug.append(dbg)
            prod = nbi_count * wli_count
            if dbg.get("pruned"):
                stats[label]["pruned_count"] += 1
                kept = dbg.get("kept_pairs", len(rows))
                orig = dbg.get("original_matched", "?")
                print(f"{pid}: matched={len(rows)}/{prod} (NBI={nbi_count}, WLI={wli_count}) PRUNED {kept}/{orig} kept")
            else:
                if len(rows) == 0:
                    stats[label]["no_pairs"] += 1
                print(f"{pid}: matched={len(rows)}/{prod} (NBI={nbi_count}, WLI={wli_count})")

    # 保存结果
    if all_rows:
        out_csv = os.path.join(ts_dir, "results.csv")
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "nbi_filename", "wli_filename", "final_score"])
            w.writerows(all_rows)

    with open(os.path.join(ts_dir, "debug.json"), "w", encoding="utf-8") as f:
        json.dump(all_debug, f, ensure_ascii=False, indent=2)

    # 保存运行日志和统计
    run_log = {
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "cfg": CFG,
        "stats": stats,
    }
    try:
        with open(os.path.join(ts_dir, "run_log.json"), "w", encoding="utf-8") as f:
            json.dump(run_log, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print("输出目录:", ts_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Pair NBI-WLI for flat patient folders")
    ap.add_argument("--dataset_root", default="./食管", help="根目录，包含多个患者子目录，每个患者目录下应有 NBI/ 和 WLI/")
    ap.add_argument("--out_root", default="./output/301_pair_images_results", help="结果输出根目录")
    args = ap.parse_args()
    run_flat(args.dataset_root, args.out_root)
