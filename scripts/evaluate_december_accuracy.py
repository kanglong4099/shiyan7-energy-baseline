#!/usr/bin/env python3
"""
12月预测精度评估（小时、天、月三个维度）
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = '/home/long/energy_baseline_training/projects/shiyan7'
PREDICTION_FILE = os.path.join(PROJECT_ROOT, 'processed_data/dec2025_prediction_only.csv')
ACTUAL_FILE = os.path.join(PROJECT_ROOT, 'raw_data/energy/SHIYAN7_energy_20251201_20251230.csv')
OUTPUT_REPORT = os.path.join(PROJECT_ROOT, 'reports/dec2025_accuracy_report.txt')
OUTPUT_DETAIL = os.path.join(PROJECT_ROOT, 'processed_data/dec2025_accuracy_detail.csv')

print("="*80)
print("12月预测精度评估（小时、天、月三个维度）")
print("="*80)

# 1. 读取数据
print("\n[1/6] 读取数据...")

# 读取预测结果
df_pred = pd.read_csv(PREDICTION_FILE)
df_pred['time'] = pd.to_datetime(df_pred['time'])
print(f"预测数据: {df_pred.shape[0]} 条")
print(f"  时间范围: {df_pred['time'].min()} ~ {df_pred['time'].max()}")

# 读取实际能耗
df_actual = pd.read_csv(ACTUAL_FILE)
# 尝试不同的时间格式
try:
    df_actual['time'] = pd.to_datetime(df_actual['time'], format='%Y/%m/%d %H:%M')
except:
    df_actual['time'] = pd.to_datetime(df_actual['time'])
print(f"实际能耗数据: {df_actual.shape[0]} 条")
print(f"  时间范围: {df_actual['time'].min()} ~ {df_actual['time'].max()}")

# 2. 合并数据
print("\n[2/6] 合并数据...")

# 重命名实际能耗列
df_actual = df_actual.rename(columns={'CHPL_cal_power_plant': 'actual_energy'})

# 合并（取交集）
df_merge = df_pred[['time', 'predicted_energy']].merge(
    df_actual[['time', 'actual_energy']], 
    on='time', 
    how='inner'
)

print(f"合并后数据: {df_merge.shape[0]} 条")
print(f"  时间范围: {df_merge['time'].min()} ~ {df_merge['time'].max()}")

# 计算误差
df_merge['abs_error'] = abs(df_merge['actual_energy'] - df_merge['predicted_energy'])
df_merge['rel_error'] = (df_merge['abs_error'] / df_merge['actual_energy']) * 100
df_merge['savings'] = df_merge['actual_energy'] - df_merge['predicted_energy']
df_merge['savings_rate'] = (df_merge['savings'] / df_merge['predicted_energy']) * 100

# 3. 小时级别精度
print("\n[3/6] 小时级别精度评估...")

y_true_hourly = df_merge['actual_energy']
y_pred_hourly = df_merge['predicted_energy']

r2_hourly = r2_score(y_true_hourly, y_pred_hourly)
rmse_hourly = np.sqrt(mean_squared_error(y_true_hourly, y_pred_hourly))
mae_hourly = mean_absolute_error(y_true_hourly, y_pred_hourly)
mape_hourly = np.mean(np.abs((y_true_hourly - y_pred_hourly) / y_true_hourly)) * 100

print(f"  R²: {r2_hourly:.4f}")
print(f"  RMSE: {rmse_hourly:.2f} kW")
print(f"  MAE: {mae_hourly:.2f} kW")
print(f"  MAPE: {mape_hourly:.2f}%")

# 4. 天级别精度
print("\n[4/6] 天级别精度评估...")

df_merge['date'] = df_merge['time'].dt.date

daily_summary = df_merge.groupby('date').agg({
    'actual_energy': 'sum',
    'predicted_energy': 'sum'
}).reset_index()

y_true_daily = daily_summary['actual_energy']
y_pred_daily = daily_summary['predicted_energy']

r2_daily = r2_score(y_true_daily, y_pred_daily)
rmse_daily = np.sqrt(mean_squared_error(y_true_daily, y_pred_daily))
mae_daily = mean_absolute_error(y_true_daily, y_pred_daily)
mape_daily = np.mean(np.abs((y_true_daily - y_pred_daily) / y_true_daily)) * 100

print(f"  天数: {len(daily_summary)} 天")
print(f"  R²: {r2_daily:.4f}")
print(f"  RMSE: {rmse_daily:.2f} kW")
print(f"  MAE: {mae_daily:.2f} kW")
print(f"  MAPE: {mape_daily:.2f}%")

# 5. 月级别精度
print("\n[5/6] 月级别精度评估...")

total_actual = df_merge['actual_energy'].sum()
total_pred = df_merge['predicted_energy'].sum()

mape_monthly = abs(total_pred - total_actual) / total_actual * 100
bias_monthly = (total_pred - total_actual) / total_actual * 100

print(f"  总实际能耗: {total_actual:.2f} kW")
print(f"  总预测能耗: {total_pred:.2f} kW")
print(f"  偏差: {total_pred - total_actual:.2f} kW ({bias_monthly:+.2f}%)")
print(f"  MAPE: {mape_monthly:.2f}%")

# 6. 与测试集对比
print("\n[6/6] 与测试集对比...")

TEST_MAPE = 2.55
TEST_R2 = 0.9897

print(f"  测试集 MAPE: {TEST_MAPE:.2f}%")
print(f"  12月 MAPE:   {mape_hourly:.2f}%")
print(f"  差异:       {mape_hourly - TEST_MAPE:+.2f}%")
print(f"  测试集 R²:  {TEST_R2:.4f}")
print(f"  12月 R²:    {r2_hourly:.4f}")
print(f"  差异:       {r2_hourly - TEST_R2:+.4f}")

# 7. 保存详细对比数据
print("\n保存详细对比数据...")
df_merge.to_csv(OUTPUT_DETAIL, index=False)
print(f"✓ 详细对比数据已保存到: {OUTPUT_DETAIL}")

# 8. 生成报告
print("\n生成精度评估报告...")

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("           12月预测精度评估报告\n")
    f.write("="*80 + "\n\n")
    
    f.write("一、数据信息\n")
    f.write("-" * 40 + "\n")
    f.write(f"预测数据: {df_pred.shape[0]} 条\n")
    f.write(f"  时间范围: {df_pred['time'].min()} ~ {df_pred['time'].max()}\n")
    f.write(f"实际能耗数据: {df_actual.shape[0]} 条\n")
    f.write(f"  时间范围: {df_actual['time'].min()} ~ {df_actual['time'].max()}\n")
    f.write(f"对比数据（合并后）: {df_merge.shape[0]} 条\n")
    f.write(f"  时间范围: {df_merge['time'].min()} ~ {df_merge['time'].max()}\n\n")
    
    f.write("二、小时级别精度\n")
    f.write("-" * 40 + "\n")
    f.write(f"数据量: {len(df_merge)} 条\n")
    f.write(f"  R² (决定系数): {r2_hourly:.4f}\n")
    f.write(f"  RMSE (均方根误差): {rmse_hourly:.2f} kW\n")
    f.write(f"  MAE (平均绝对误差): {mae_hourly:.2f} kW\n")
    f.write(f"  MAPE (平均百分比误差): {mape_hourly:.2f}%\n\n")
    
    f.write("误差分布:\n")
    f.write(f"  最大误差: {df_merge['abs_error'].max():.2f} kW\n")
    f.write(f"  最小误差: {df_merge['abs_error'].min():.2f} kW\n")
    f.write(f"  误差中位数: {df_merge['abs_error'].median():.2f} kW\n")
    f.write(f"  误差>10%: {(df_merge['rel_error'] > 10).sum()} 条 ({(df_merge['rel_error'] > 10).sum()/len(df_merge)*100:.1f}%)\n")
    f.write(f"  误差>20%: {(df_merge['rel_error'] > 20).sum()} 条 ({(df_merge['rel_error'] > 20).sum()/len(df_merge)*100:.1f}%)\n\n")
    
    f.write("三、天级别精度\n")
    f.write("-" * 40 + "\n")
    f.write(f"数据量: {len(daily_summary)} 天\n")
    f.write(f"  R² (决定系数): {r2_daily:.4f}\n")
    f.write(f"  RMSE (均方根误差): {rmse_daily:.2f} kW\n")
    f.write(f"  MAE (平均绝对误差): {mae_daily:.2f} kW\n")
    f.write(f"  MAPE (平均百分比误差): {mape_daily:.2f}%\n\n")
    
    f.write("每日能耗对比（前10天）:\n")
    f.write(f"{'日期':<12} {'实际能耗':<12} {'预测能耗':<12} {'误差':<10} {'误差率':<10}\n")
    f.write("-"*60 + "\n")
    for idx, row in daily_summary.head(10).iterrows():
        error = row['actual_energy'] - row['predicted_energy']
        error_rate = error / row['actual_energy'] * 100
        f.write(f"{str(row['date']):<12} {row['actual_energy']:<12.2f} {row['predicted_energy']:<12.2f} {error:<10.2f} {error_rate:<10.2f}%\n")
    f.write("\n")
    
    f.write("四、月级别精度\n")
    f.write("-" * 40 + "\n")
    f.write(f"总实际能耗: {total_actual:.2f} kW\n")
    f.write(f"总预测能耗: {total_pred:.2f} kW\n")
    f.write(f"偏差: {total_pred - total_actual:.2f} kW ({bias_monthly:+.2f}%)\n")
    f.write(f"MAPE: {mape_monthly:.2f}%\n\n")
    
    f.write("五、与测试集对比\n")
    f.write("-" * 40 + "\n")
    f.write(f"  测试集 MAPE: {TEST_MAPE:.2f}%\n")
    f.write(f"  12月 MAPE (小时): {mape_hourly:.2f}%\n")
    f.write(f"  差异: {mape_hourly - TEST_MAPE:+.2f}%\n\n")
    f.write(f"  测试集 R²: {TEST_R2:.4f}\n")
    f.write(f"  12月 R² (小时): {r2_hourly:.4f}\n")
    f.write(f"  差异: {r2_hourly - TEST_R2:+.4f}\n\n")
    
    f.write("六、精度等级评价\n")
    f.write("-" * 40 + "\n")
    if mape_hourly < 3.0:
        f.write("小时级别: 优秀 (MAPE < 3%)\n")
    elif mape_hourly < 5.0:
        f.write("小时级别: 良好 (MAPE < 5%)\n")
    elif mape_hourly < 10.0:
        f.write("小时级别: 可接受 (MAPE < 10%)\n")
    else:
        f.write("小时级别: 需改进 (MAPE >= 10%)\n")
    
    if mape_daily < 3.0:
        f.write("天级别: 优秀 (MAPE < 3%)\n")
    elif mape_daily < 5.0:
        f.write("天级别: 良好 (MAPE < 5%)\n")
    elif mape_daily < 10.0:
        f.write("天级别: 可接受 (MAPE < 10%)\n")
    else:
        f.write("天级别: 需改进 (MAPE >= 10%)\n")
    
    if mape_monthly < 3.0:
        f.write("月级别: 优秀 (MAPE < 3%)\n")
    elif mape_monthly < 5.0:
        f.write("月级别: 良好 (MAPE < 5%)\n")
    elif mape_monthly < 10.0:
        f.write("月级别: 可接受 (MAPE < 10%)\n")
    else:
        f.write("月级别: 需改进 (MAPE >= 10%)\n")
    f.write("\n")
    
    f.write("七、结论\n")
    f.write("-" * 40 + "\n")
    
    if mape_hourly < 3.0:
        f.write("✓ 12月预测精度优秀\n")
        f.write("✓ 模型在12月表现稳定\n")
        f.write("✓ 可以全年使用v3.0模型进行预测\n")
    elif mape_hourly < 5.0:
        f.write("✓ 12月预测精度良好\n")
        f.write("✓ 模型在12月表现可接受\n")
        f.write("ℹ 建议收集更多数据进一步验证\n")
    else:
        f.write("⚠ 12月预测精度需关注\n")
        f.write("⚠ 建议收集更多数据重新训练模型\n")
    f.write("\n")
    
    f.write("="*80 + "\n")

print(f"✓ 精度评估报告已保存到: {OUTPUT_REPORT}")

print("\n" + "="*80)
print("12月预测精度评估完成!")
print("="*80)

print(f"\n三个维度精度汇总:")
print(f"  {'维度':<10} {'数据量':<10} {'R²':<10} {'RMSE(kW)':<12} {'MAE(kW)':<10} {'MAPE(%)':<10}")
print("-"*65)
print(f"  {'小时':<10} {len(df_merge):<10} {r2_hourly:<10.4f} {rmse_hourly:<12.2f} {mae_hourly:<10.2f} {mape_hourly:<10.2f}")
print(f"  {'天':<10} {len(daily_summary):<10} {r2_daily:<10.4f} {rmse_daily:<12.2f} {mae_daily:<10.2f} {mape_daily:<10.2f}")
print(f"  {'月':<10} {'-':<10} {'-':<10} {'-':<12} {'-':<10} {mape_monthly:<10.2f}")

print(f"\n与测试集对比:")
print(f"  测试集 MAPE: {TEST_MAPE:.2f}%")
print(f"  12月 MAPE: {mape_hourly:.2f}%")
print(f"  差异: {mape_hourly - TEST_MAPE:+.2f}%")

print(f"\n输出文件:")
print(f"  1. 详细对比数据: {OUTPUT_DETAIL}")
print(f"  2. 精度评估报告: {OUTPUT_REPORT}")
