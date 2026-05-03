# -*- coding: utf-8 -*-
"""
从 results.csv 收集配对图片到新的输出目录，保留 label 和 patient 目录结构。
用法:
    python collect_pairs.py --results path/to/results.csv --dataset_root path/to/dataset --out_root path/to/out

脚本会：
 - 读取 CSV（列: patient_id, nbi_filename, wli_filename, final_score）
 - 根据 patient_id 推断 label 和 patient（支持 "label/patient" 格式，或只给 patient 名时会在 dataset_root 下搜索）
 - 将源图像复制到 out_root/label/patient/NBI/ 和 out_root/label/patient/WLI/
 - 若目标文件已存在则跳过复制
 - 打印并保存统计信息到 out_root/collect_log.json
"""

import os
import csv
import argparse
import shutil
import json


def find_patient_dir(dataset_root, patient_id, label=None):
    # 如果为 label/patient 格式
    if "/" in patient_id or "\\\\" in patient_id:
        parts = patient_id.replace('\\', '/').split('/')
        if len(parts) >= 2:
            lab = parts[0]
            pat = '/'.join(parts[1:])
            cand = os.path.join(dataset_root, lab, pat)
            if os.path.isdir(cand):
                return lab, pat, cand
    # 如果 label 已知
    if label:
        cand = os.path.join(dataset_root, label, patient_id)
        if os.path.isdir(cand):
            return label, patient_id, cand
    # 尝试在 dataset_root 的第二层作为 label -> patient
    for lab in sorted(os.listdir(dataset_root)):
        labp = os.path.join(dataset_root, lab)
        if not os.path.isdir(labp):
            continue
        cand = os.path.join(labp, patient_id)
        if os.path.isdir(cand):
            return lab, patient_id, cand
    # 最后回退到递归搜索（限制深度）
    for root, dirs, files in os.walk(dataset_root):
        if patient_id in dirs:
            cand = os.path.join(root, patient_id)
            # try to infer label as the directory name above
            parent = os.path.dirname(cand)
            label_name = os.path.basename(parent)
            return label_name, patient_id, cand
    return None, None, None


def safe_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return 'skipped'
    try:
        shutil.copy2(src, dst)
        return 'copied'
    except Exception:
        return 'missing'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='output\\301_pair_images_results\\pair_301_202604261553\\results.csv', help='results.csv 路径')
    ap.add_argument('--dataset_root', default='食管', help='原始数据根路径')
    ap.add_argument('--out_root', default='output\\301_pair_images', help='目标输出根路径')
    ap.add_argument('--patient_col', type=int, default=0, help='CSV 中 patient_id 列索引（0-based）')
    ap.add_argument('--nbi_col', type=int, default=1, help='CSV 中 nbi_filename 列索引')
    ap.add_argument('--wli_col', type=int, default=2, help='CSV 中 wli_filename 列索引')
    args = ap.parse_args()

    stats = {'total_pairs': 0, 'copied': 0, 'skipped': 0, 'missing_src': 0, 'missing_patient': 0}
    per_patient = {}

    with open(args.results, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            stats['total_pairs'] += 1
            patient_id = row[args.patient_col]
            nbi_name = row[args.nbi_col]
            wli_name = row[args.wli_col]

            label, patient, patient_dir = find_patient_dir(args.dataset_root, patient_id)
            if patient_dir is None:
                stats['missing_patient'] += 1
                print(f"未找到患者目录: {patient_id}")
                continue

            # 源路径
            src_nbi = os.path.join(patient_dir, 'NBI', nbi_name)
            src_wli = os.path.join(patient_dir, 'WLI', wli_name)

            # 目标路径
            out_nbi_dir = os.path.join(args.out_root, label, patient, 'NBI')
            out_wli_dir = os.path.join(args.out_root, label, patient, 'WLI')
            dst_nbi = os.path.join(out_nbi_dir, nbi_name)
            dst_wli = os.path.join(out_wli_dir, wli_name)

            r1 = safe_copy(src_nbi, dst_nbi)
            r2 = safe_copy(src_wli, dst_wli)

            for r in (r1, r2):
                if r == 'copied':
                    stats['copied'] += 1
                elif r == 'skipped':
                    stats['skipped'] += 1
                else:
                    stats['missing_src'] += 1

            key = f"{label}/{patient}"
            per_patient.setdefault(key, {'pairs': 0, 'copied': 0, 'missing': 0})
            per_patient[key]['pairs'] += 1
            if r1 == 'copied' or r2 == 'copied':
                per_patient[key]['copied'] += 1
            if r1 == 'missing' or r2 == 'missing':
                per_patient[key]['missing'] += 1

    # 保存日志
    os.makedirs(args.out_root, exist_ok=True)
    log = {'results_file': args.results, 'dataset_root': args.dataset_root, 'out_root': args.out_root,
           'stats': stats, 'per_patient': per_patient}
    with open(os.path.join(args.out_root, 'collect_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print('完成。统计：', stats)


if __name__ == '__main__':
    main()
