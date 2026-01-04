#!/usr/bin/env python3
"""
12月能耗预测脚本
不进行精度计算，只进行预测
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = '/home/long/energy_baseline_training/projects/shiyan7'
BMS_FILE = os.path.join(PROJECT_ROOT, 'raw_data/bms/AESC_SHIYAN7_history_data_202512.xlsx')
WEATHER_FILE = os.path.join(PROJECT_ROOT, 'raw_data/weather/weather_SHIYAN_251201-251230.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'processed_data/dec2025_prediction_only.csv')

# 冷机额定制冷量
CHILLER_CAPACITIES = {
    'CH1': 9845.0, 'CH2': 9845.0, 'CH3': 9845.0,
    'CH4': 9845.0, 'CH5': 3517.0
}

# 负荷分组阈值（从训练数据获取）
LOAD_GROUP_THRESHOLDS = {
    'q25': 6919.0,
    'q50': 10512.0,
    'q75': 13614.0
}

print("="*80)
print("12月能耗预测")
print("="*80)

# 1. 读取数据
print("\n[1/6] 读取数据...")
df_bms = pd.read_excel(BMS_FILE)
df_weather = pd.read_csv(WEATHER_FILE)

print(f"BMS数据: {df_bms.shape[0]} 行, {df_bms.shape[1]} 列")
print(f"天气数据: {df_weather.shape[0]} 行, {df_weather.shape[1]} 列")

# 标准化列名
df_bms = df_bms.rename(columns={'normalized_time': 'time'})

# 转换时间格式
df_bms['time'] = pd.to_datetime(df_bms['time'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

print(f"\nBMS时间范围: {df_bms['time'].min()} ~ {df_bms['time'].max()}")
print(f"天气时间范围: {df_weather['time'].min()} ~ {df_weather['time'].max()}")

# 2. 合并数据
print("\n[2/6] 合并数据...")
df = df_bms.merge(df_weather[['time', 'temperature', 'RH', 'ssrd']], on='time', how='inner')
print(f"合并后数据: {df.shape[0]} 行")

# 3. 计算特征
print("\n[3/6] 计算特征...")

# 3.1 hour
df['hour'] = df['time'].dt.hour

# 3.2 dew_point
df['dew_point'] = df['temperature'] - ((100 - df['RH']) / 5)

# 3.3 wet_bulb
df['wet_bulb'] = df['temperature'] * 0.6 + df['dew_point'] * 0.4

# 3.4 temp_x_rh
df['temp_x_rh'] = df['temperature'] * df['RH'] / 100

# 3.5 total_chiller_load (使用fla电流百分比)
df['total_chiller_load'] = (
    CHILLER_CAPACITIES['CH1'] * df['CH1_CH_raw_fla'] / 100 +
    CHILLER_CAPACITIES['CH2'] * df['CH2_CH_raw_fla'] / 100 +
    CHILLER_CAPACITIES['CH3'] * df['CH3_CH_raw_fla'] / 100 +
    CHILLER_CAPACITIES['CH4'] * df['CH4_CH_raw_fla'] / 100 +
    CHILLER_CAPACITIES['CH5'] * df['CH5_CH_raw_fla'] / 100
)

print(f"总冷机负荷范围: {df['total_chiller_load'].min():.2f} ~ {df['total_chiller_load'].max():.2f} kW")
print(f"总冷机负荷均值: {df['total_chiller_load'].mean():.2f} kW")

# 3.6 load_group
def assign_load_group(load):
    if pd.isna(load):
        return np.nan
    if load <= LOAD_GROUP_THRESHOLDS['q25']:
        return 1
    elif load <= LOAD_GROUP_THRESHOLDS['q50']:
        return 2
    elif load <= LOAD_GROUP_THRESHOLDS['q75']:
        return 3
    else:
        return 4

df['load_group'] = df['total_chiller_load'].apply(assign_load_group)

print(f"负荷分组分布:")
for i in range(1, 5):
    count = (df['load_group'] == i).sum()
    print(f"  分组{i}: {count} 条")

# 3.7 temp_ma3 (温度3小时移动平均)
df = df.sort_values('time')
df['temp_ma3'] = df['temperature'].rolling(window=3, min_periods=1).mean()

# 4. 准备模型输入
print("\n[4/6] 准备模型输入...")

FEATURE_COLUMNS = [
    'hour', 'temperature', 'RH', 'dew_point', 'wet_bulb', 'ssrd',
    'total_chiller_load', 'load_group', 'temp_ma3', 'temp_x_rh'
]

# 清洗数据
df_clean = df[FEATURE_COLUMNS + ['time']].replace([np.inf, -np.inf], np.nan).dropna()
print(f"清洗后数据: {df_clean.shape[0]} 行")

# 5. 加载模型
print("\n[5/6] 加载模型...")
model_file = os.path.join(PROJECT_ROOT, 'models/advanced/best_model_final.pkl')

with open(model_file, 'rb') as f:
    model = pickle.load(f)

print(f"✓ 模型已加载: {model_file}")

# 6. 预测
print("\n[6/6] 进行预测...")

X = df_clean[FEATURE_COLUMNS]
df_clean['predicted_energy'] = model.predict(X)

print(f"预测完成！")
print(f"预测能耗范围: {df_clean['predicted_energy'].min():.2f} ~ {df_clean['predicted_energy'].max():.2f} kW")
print(f"预测能耗均值: {df_clean['predicted_energy'].mean():.2f} kW")

# 7. 保存结果
print("\n保存结果...")

# 保存预测结果
output_columns = ['time'] + FEATURE_COLUMNS + ['predicted_energy']
df_clean[output_columns].to_csv(OUTPUT_FILE, index=False)
print(f"✓ 预测结果已保存到: {OUTPUT_FILE}")

# 生成统计报告
print("\n预测统计:")
print(f"  数据量: {df_clean.shape[0]} 条")
print(f"  时间范围: {df_clean['time'].min()} ~ {df_clean['time'].max()}")
print(f"\n预测能耗统计:")
print(f"  最小值: {df_clean['predicted_energy'].min():.2f} kW")
print(f"  最大值: {df_clean['predicted_energy'].max():.2f} kW")
print(f"  均值: {df_clean['predicted_energy'].mean():.2f} kW")
print(f"  中位数: {df_clean['predicted_energy'].median():.2f} kW")
print(f"  标准差: {df_clean['predicted_energy'].std():.2f} kW")

print(f"\n总预测能耗: {df_clean['predicted_energy'].sum():.2f} kW")
print(f"日平均能耗: {df_clean['predicted_energy'].sum() / df_clean.shape[0]:.2f} kW")

# 按日期汇总
df_clean['date'] = df_clean['time'].dt.date
daily_summary = df_clean.groupby('date').agg({
    'predicted_energy': ['sum', 'mean', 'min', 'max']
}).reset_index()
daily_summary.columns = ['date', 'daily_sum', 'daily_mean', 'daily_min', 'daily_max']

print(f"\n每日能耗统计（前10天）:")
print(daily_summary.head(10))

print("\n" + "="*80)
print("12月能耗预测完成!")
print("="*80)
print(f"\n结果文件: {OUTPUT_FILE}")
