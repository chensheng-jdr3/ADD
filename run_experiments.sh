#!/bin/bash
# ============================================================
# ADD 蒸馏消融实验
# 用法: bash run_experiments.sh [my_dataset|cpc_paired]
# 实验 A: 原始 ADD (固定温度 KD, 无 LS, 无 CTKD)
# 实验 B: 改进 ADD (LS + CTKD)
# ============================================================
set -e

DATASET="${1:-my_dataset}"

echo "数据集: ${DATASET}"
echo ""

echo "=============================================="
echo " 实验 A: 原始 ADD (--no-ls --no-ctkd --tau 4)"
echo "=============================================="
python train.py \
    --dataset ${DATASET} \
    --no-ls --no-ctkd \
    --tau 4 \
    --epochs 200

echo ""
echo "=============================================="
echo " 实验 B: 改进 ADD (--ls --ctkd)"
echo "=============================================="
python train.py \
    --dataset ${DATASET} \
    --ls --ctkd \
    --epochs 200

echo ""
echo "全部实验完成！"
echo "日志路径: ./log/ADD/${DATASET}/"
