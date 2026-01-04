#!/bin/bash
# 清理7#冷冻站项目的中间文件
# 只保留必要的模型、数据和报告文件

PROJECT_ROOT="/home/long/energy_baseline_training/projects/shiyan7"

echo "================================================================================"
echo "                        7#冷冻站项目文件清理"
echo "================================================================================"

echo ""
echo "[1/5] 清理旧版本模型文件..."

# 删除旧版本的模型文件
rm -f "${PROJECT_ROOT}/models/advanced/best_model.pkl"
rm -f "${PROJECT_ROOT}/models/advanced/best_model_advanced.pkl"
rm -f "${PROJECT_ROOT}/models/advanced/best_model_no_month.pkl"
rm -f "${PROJECT_ROOT}/models/advanced/best_model_with_transition.pkl"

echo "✓ 已删除旧版本模型文件"
echo "  保留: best_model_final.pkl (v3.0最终版本）"

echo ""
echo "[2/5] 清理旧版本特征文件..."

# 删除旧版本的特征文件
rm -f "${PROJECT_ROOT}/models/metadata/best_feature_indices.pkl"
rm -f "${PROJECT_ROOT}/models/metadata/advanced_feature_columns.pkl"
rm -f "${PROJECT_ROOT}/models/metadata/no_month_feature_columns.pkl"
rm -f "${PROJECT_ROOT}/models/metadata/transition_feature_columns.pkl"

echo "✓ 已删除旧版本特征文件"
echo "  保留: final_feature_columns.pkl (v3.0特征）"

echo ""
echo "[3/5] 清理中间数据和测试文件..."

# 删除中间数据文件
rm -f "${PROJECT_ROOT}/processed_data/test_predictions.csv"
rm -f "${PROJECT_ROOT}/processed_data/test_advanced_data.csv"
rm -f "${PROJECT_ROOT}/processed_data/test_final_data.csv"
rm -f "${PROJECT_ROOT}/processed_data/test_transition_data.csv"
rm -f "${PROJECT_ROOT}/processed_data/dec2025_prediction_v31.csv"
rm -f "${PROJECT_ROOT}/processed_data/shiyan7_clean_data.csv"
rm -f "${PROJECT_ROOT}/processed_data/shiyan7_with_chillers.csv"

# 删除特征重要性CSV文件（保留final）
rm -f "${PROJECT_ROOT}/artifacts/feature_importance.csv"
rm -f "${PROJECT_ROOT}/artifacts/feature_importance_transition.csv"

echo "✓ 已删除中间数据和测试文件"
echo "  保留: shiyan7_features.csv, shiyan7_merged_data.csv"
echo "  保留: dec2025_prediction_only.csv, dec2025_accuracy_detail.csv"
echo "  保留: feature_importance_final.csv"

echo ""
echo "[4/5] 清理旧版本报告文件..."

# 删除旧版本的报告文件
rm -f "${PROJECT_ROOT}/reports/FINAL_REPORT.txt"
rm -f "${PROJECT_ROOT}/reports/FINAL_REPORT_ENHANCED.txt"
rm -f "${PROJECT_ROOT}/reports/model_training_report.txt"
rm -f "${PROJECT_ROOT}/reports/model_training_report_updated.txt"
rm -f "${PROJECT_ROOT}/reports/model_training_report_transition.txt"

echo "✓ 已删除旧版本报告文件"

echo ""
echo "[5/5] 显示清理后的文件结构..."

echo ""
echo "模型文件:"
ls -lh "${PROJECT_ROOT}/models/advanced/" | grep -E "\.pkl$"
echo ""
echo "特征文件:"
ls -lh "${PROJECT_ROOT}/models/metadata/" | grep -E "\.pkl$"
echo ""
echo "数据文件 (主要):"
ls -lh "${PROJECT_ROOT}/processed_data/" | grep -E "shiyan7_features|dec2025"
echo ""
echo "报告文件:"
ls -lh "${PROJECT_ROOT}/reports/" | tail -10

echo ""
echo "================================================================================"
echo "                        清理完成!"
echo "================================================================================"

echo ""
echo "保留的重要文件:"
echo "  模型: models/advanced/best_model_final.pkl"
echo "  特征: models/metadata/final_feature_columns.pkl"
echo "  训练数据: processed_data/shiyan7_features.csv"
echo "  12月预测: processed_data/dec2025_prediction_only.csv"
echo "  12月精度: processed_data/dec2025_accuracy_detail.csv"
echo ""
echo "重要报告:"
echo "  README.md - 使用指南"
echo "  reports/model_training_report_final.txt - 模型训练报告"
echo "  reports/MODEL_FINAL_VERSION_SUMMARY.txt - 模型版本总结"
echo "  reports/dec2025_accuracy_report.txt - 12月精度评估"
echo "  reports/TRANSITION_FEATURE_ANALYSIS.txt - 加减机特征分析"
echo "  reports/MODEL_UPDATE_SUMMARY.txt - 模型更新摘要"
echo ""
echo "已删除的文件:"
echo "  - 旧版本模型文件 (4个)"
echo "  - 旧版本特征文件 (4个)"
echo "  - 中间数据文件 (6个)"
echo "  - 旧版本报告文件 (4个)"
echo ""
echo "总删除: 18个文件"
