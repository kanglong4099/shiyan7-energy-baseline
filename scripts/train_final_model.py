#!/usr/bin/env python3
"""
7#冷冻站能耗基线模型最终版本 - 去除所有可优化特征
- 移除 low_temp_load (低温水温差)
- 移除 medium_temp_load (中温水温差)
- 移除 delta_temp_header (冷却水温差)
- 只保留不可优化或气象/负荷特征
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = '/home/long/energy_baseline_training/projects/shiyan7'
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'processed_data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ADVANCED_MODEL_DIR = os.path.join(MODELS_DIR, 'advanced')
METADATA_DIR = os.path.join(MODELS_DIR, 'metadata')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

print("="*80)
print("7#冷冻站能耗基线模型 - 最终版本（去除所有可优化特征）")
print("="*80)

# 1. 读取数据
print("\n[1/5] 读取数据...")
features_file = os.path.join(PROCESSED_DATA_DIR, 'shiyan7_features.csv')
df = pd.read_csv(features_file)
print(f"原始数据形状: {df.shape}")

# 2. 计算总冷机负荷和负荷分组
print("\n[2/5] 计算总冷机负荷和负荷分组...")

# 冷机额定制冷量
chiller_capacities = {
    'CH1': 9845.0, 'CH2': 9845.0, 'CH3': 9845.0,
    'CH4': 9845.0, 'CH5': 3517.0
}

# 计算总冷机负荷
df['total_chiller_load'] = (
    chiller_capacities['CH1'] * df['CH1-percentage_current'] / 100 +
    chiller_capacities['CH2'] * df['CH2-percentage_current'] / 100 +
    chiller_capacities['CH3'] * df['CH3-percentage_current'] / 100 +
    chiller_capacities['CH4'] * df['CH4-percentage_current'] / 100 +
    chiller_capacities['CH5'] * df['CH5-percentage_current'] / 100
)

print(f"总冷机负荷统计:")
print(f"  均值: {df['total_chiller_load'].mean():.2f} kW")
print(f"  范围: {df['total_chiller_load'].min():.2f} ~ {df['total_chiller_load'].max():.2f} kW")

# 计算负荷分组
q25 = df['total_chiller_load'].quantile(0.25)
q50 = df['total_chiller_load'].quantile(0.50)
q75 = df['total_chiller_load'].quantile(0.75)

def assign_load_group(load):
    if load <= q25:
        return 1
    elif load <= q50:
        return 2
    elif load <= q75:
        return 3
    else:
        return 4

df['load_group'] = df['total_chiller_load'].apply(assign_load_group)

# 计算衍生特征
df['temp_x_rh'] = df['temperature'] * df['RH'] / 100

print(f"\n负荷分组统计:")
print(f"  分组1 (≤ {q25:.0f} kW): {(df['load_group'] == 1).sum()} 条")
print(f"  分组2 ({q25:.0f} ~ {q50:.0f} kW): {(df['load_group'] == 2).sum()} 条")
print(f"  分组3 ({q50:.0f} ~ {q75:.0f} kW): {(df['load_group'] == 3).sum()} 条")
print(f"  分组4 (> {q75:.0f} kW): {(df['load_group'] == 4).sum()} 条")

# 3. 准备特征（去除所有可优化的温差特征）
print("\n[3/5] 准备特征...")

# 最终特征列表：只保留不可优化和气象/负荷特征
FEATURE_COLUMNS_FINAL = [
    # 时间特征
    'hour',
    
    # 气象特征
    'temperature', 'RH', 'dew_point', 'wet_bulb', 'ssrd',
    
    # 负荷特征
    'total_chiller_load', 'load_group',
    
    # 衍生特征
    'temp_ma3', 'temp_x_rh',
]

TARGET_COLUMN = 'total_plant_power'

print(f"\n最终特征列表 ({len(FEATURE_COLUMNS_FINAL)}个):")
for i, feat in enumerate(FEATURE_COLUMNS_FINAL, 1):
    print(f"  {i}. {feat}")

print(f"\n已移除的可优化特征:")
print("  ✗ low_temp_load (低温水温差)")
print("  ✗ medium_temp_load (中温水温差)")
print("  ✗ delta_temp_header (冷却水温差)")
print("  ✗ month (月份，为了全年预测)")

# 4. 清洗和准备数据
df_clean = df[FEATURE_COLUMNS_FINAL + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
print(f"\n清洗后数据量: {df_clean.shape[0]} 条")

X = df_clean[FEATURE_COLUMNS_FINAL]
y = df_clean[TARGET_COLUMN]

# 5. 训练模型
print("\n[4/5] 训练模型...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")

model_final = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model_final.fit(X_train, y_train)
print("模型训练完成!")

# 6. 评估模型
print("\n[5/5] 评估模型...")

y_pred = model_final.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n测试集性能:")
print(f"  R² (决定系数): {r2:.4f}")
print(f"  RMSE (均方根误差): {rmse:.2f} kW")
print(f"  MAE (平均绝对误差): {mae:.2f} kW")
print(f"  MAPE (平均百分比误差): {mape:.2f}%")

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS_FINAL,
    'importance': model_final.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n特征重要性排序:")
for idx, row in feature_importance.iterrows():
    print(f"  {idx + 1}. {row['feature']:30s} {row['importance']*100:6.2f}%")

# 保存模型
model_file_final = os.path.join(ADVANCED_MODEL_DIR, 'best_model_final.pkl')
feature_columns_file_final = os.path.join(METADATA_DIR, 'final_feature_columns.pkl')

with open(model_file_final, 'wb') as f:
    pickle.dump(model_final, f)

with open(feature_columns_file_final, 'wb') as f:
    pickle.dump(FEATURE_COLUMNS_FINAL, f)

print(f"\n最终模型已保存到: {model_file_final}")
print(f"特征列表已保存到: {feature_columns_file_final}")

# 保存特征重要性
feature_importance_file = os.path.join(ARTIFACTS_DIR, 'feature_importance_final.csv')
feature_importance.to_csv(feature_importance_file, index=False)
print(f"特征重要性已保存到: {feature_importance_file}")

# 保存测试数据
test_data_file = os.path.join(PROCESSED_DATA_DIR, 'test_final_data.csv')
test_df = X_test.copy()
test_df[TARGET_COLUMN] = y_test
test_df['predicted'] = y_pred
test_df.to_csv(test_data_file, index=False)
print(f"测试数据已保存到: {test_data_file}")

# 生成训练报告
report_file = os.path.join(REPORTS_DIR, 'model_training_report_final.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("           7#冷冻站能耗基线模型 - 最终版本训练报告\n")
    f.write("="*80 + "\n\n")
    
    f.write("一、项目信息\n")
    f.write("-" * 40 + "\n")
    f.write(f"项目名称: 7#冷冻站能耗基线模型（shiyan7）\n")
    f.write(f"版本: v3.0 (最终版本)\n")
    f.write(f"训练日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据量: {df_clean.shape[0]} 条（逐小时数据）\n\n")
    
    f.write("二、特征变更\n")
    f.write("-" * 40 + "\n")
    f.write("已移除的可优化特征:\n")
    f.write("  1. low_temp_load (低温水温差) - 供回水温差优化\n")
    f.write("  2. medium_temp_load (中温水温差) - 供回水温差优化\n")
    f.write("  3. delta_temp_header (冷却水温差) - 冷却水温差优化\n")
    f.write("  4. month (月份) - 仅1-7月数据，需全年预测\n\n")
    
    f.write("保留特征 (不可优化):\n")
    f.write("  - hour (小时): 时间特征\n")
    f.write("  - temperature, RH, dew_point, wet_bulb, ssrd: 气象特征\n")
    f.write("  - total_chiller_load, load_group: 负荷特征\n")
    f.write("  - temp_ma3, temp_x_rh: 衍生特征\n\n")
    
    f.write(f"三、模型配置\n")
    f.write("-" * 40 + "\n")
    f.write(f"模型类型: 极端随机森林 (ExtraTreesRegressor)\n")
    f.write(f"树的数量: {model_final.n_estimators}\n")
    f.write(f"特征数量: {len(FEATURE_COLUMNS_FINAL)}\n\n")
    
    f.write("四、模型性能\n")
    f.write("-" * 40 + "\n")
    f.write(f"测试集性能:\n")
    f.write(f"  R²: {r2:.4f}\n")
    f.write(f"  RMSE: {rmse:.2f} kW\n")
    f.write(f"  MAE: {mae:.2f} kW\n")
    f.write(f"  MAPE: {mape:.2f}%\n\n")
    
    f.write("五、特征重要性\n")
    f.write("-" * 40 + "\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{idx + 1}. {row['feature']:30s} {row['importance']*100:6.2f}%\n")
    f.write("\n")
    
    f.write("="*80 + "\n")
    f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")

print(f"\n训练报告已保存到: {report_file}")

print("\n" + "="*80)
print("最终模型训练完成!")
print("="*80)
print("\n模型特点:")
print("  ✓ 只包含不可优化的特征")
print("  ✓ 可全年预测（不含month）")
print("  ✓ 适用于基线模型")
print("  ✓ 不影响优化程序的设计")
