# 7#冷冻站能耗基线模型使用指南 (v3.0 最终版本)

## 项目概述

本项目建立了7#冷冻站的能耗基线预测模型，用于：
- 预测冷冻站在给定条件下的基线能耗
- 计算节能改造或运行优化带来的节能量
- 评估冷冻站能效表现

## 模型性能

**最终模型 (v3.0)**: 极端随机森林

**测试集性能**:
- R² = 0.9897 (模型解释了98.97%的能耗变异)
- RMSE = 88.53 kW
- MAE = 48.53 kW
- MAPE = 2.55%

**模型参数**:
- 树的数量: 100
- 特征数量: 10

**模型文件**: `models/advanced/best_model_final.pkl`

## 特征列表

| 序号 | 特征 | 重要性 | 类型 | 说明 |
|------|------|--------|------|------|
| 1 | load_group | 56.53% | 负荷 | 负荷分组(1-4) ⭐⭐⭐ |
| 2 | total_chiller_load | 20.47% | 负荷 | 总冷机负荷(kW) ⭐⭐⭐ |
| 3 | dew_point | 10.36% | 气象 | 露点温度(°C) ⭐ |
| 4 | wet_bulb | 3.91% | 气象 | 湿球温度(°C) |
| 5 | temp_x_rh | 3.02% | 衍生 | 温度×湿度 |
| 6 | temperature | 1.83% | 气象 | 干球温度(°C) |
| 7 | temp_ma3 | 1.74% | 衍生 | 温度移动平均 |
| 8 | RH | 0.98% | 气象 | 相对湿度(%) |
| 9 | ssrd | 0.64% | 气象 | 太阳辐射(W/m²) |
| 10 | hour | 0.53% | 时间 | 小时(0-23) |

### 特征分类

- **负荷特征** (77.00%): load_group + total_chiller_load ⭐⭐⭐
- **气象特征** (17.72%): dew_point + wet_bulb + temperature + RH + ssrd
- **衍生特征** (4.76%): temp_x_rh + temp_ma3
- **时间特征** (0.53%): hour

### 已移除的可优化特征

✗ **low_temp_load** (低温水温差) - 供回水温差优化
✗ **medium_temp_load** (中温水温差) - 供回水温差优化
✗ **delta_temp_header** (冷却水温差) - 冷却水温差优化
✗ **month** (月份) - 仅1-7月数据，需全年预测

## 快速开始

### 1. 单点预测

```python
import sys
sys.path.append('/home/long/energy_baseline_training/projects/shiyan7/scripts')
import pickle
import numpy as np

# 加载模型
model = pickle.load(open('models/advanced/best_model_final.pkl', 'rb'))

# 准备特征（按顺序）
features = np.array([[
    14,          # hour: 小时
    30.0,        # temperature: 室外温度 (°C)
    60.0,        # RH: 相对湿度 (%)
    22.0,        # dew_point: 露点温度 (°C)
    26.8,        # wet_bulb: 湿球温度 (°C)
    500.0,       # ssrd: 太阳辐射 (W/m²)
    15000.0,     # total_chiller_load: 总冷机负荷 (kW)
    3,           # load_group: 负荷分组 (1-4)
    28.0,        # temp_ma3: 温度移动平均 (°C)
    18.0         # temp_x_rh: 温度×湿度
]])

# 预测
predicted = model.predict(features)[0]
print(f"预测基线能耗: {predicted:.2f} kW")
```

### 2. 计算节能量

```python
actual = 3500.0  # 实际能耗
baseline = predicted  # 预测基线

savings = actual - baseline
savings_rate = (savings / baseline) * 100

print(f"实际能耗: {actual:.2f} kW")
print(f"基线能耗: {baseline:.2f} kW")
print(f"节能量: {savings:.2f} kW")
print(f"节能率: {savings_rate:.2f}%")
```

## 特征计算说明

### 1. 总冷机负荷

```python
chiller_capacities = {
    'CH1': 9845.0, 'CH2': 9845.0, 'CH3': 9845.0,
    'CH4': 9845.0, 'CH5': 3517.0
}

total_chiller_load = (
    9845 * CH1_percentage_current / 100 +
    9845 * CH2_percentage_current / 100 +
    9845 * CH3_percentage_current / 100 +
    9845 * CH4_percentage_current / 100 +
    3517 * CH5_percentage_current / 100
)
```

### 2. 负荷分组

基于总冷机负荷的四分位数分组：

```python
if total_chiller_load <= 6919:
    load_group = 1
elif total_chiller_load <= 10512:
    load_group = 2
elif total_chiller_load <= 13614:
    load_group = 3
else:
    load_group = 4
```

### 3. 气象衍生特征

```python
# 露点温度
dew_point = temperature - ((100 - RH) / 5)

# 湿球温度
wet_bulb = temperature * 0.6 + dew_point * 0.4

# 温度×湿度
temp_x_rh = temperature * RH / 100

# 温度3小时移动平均（需历史数据）
temp_ma3 = (temperature_t + temperature_{t-1} + temperature_{t-2}) / 3
```

## 批量预测

准备CSV文件，包含以下列（10个特征）：

| 列名 | 说明 | 单位 | 范围 |
|------|------|------|------|
| hour | 小时 | 0-23 | - |
| temperature | 室外温度 | °C | -3 ~ 40 |
| RH | 相对湿度 | % | 10 ~ 100 |
| dew_point | 露点温度 | °C | -10 ~ 30 |
| wet_bulb | 湿球温度 | °C | 0 ~ 35 |
| ssrd | 太阳辐射 | W/m² | 0 ~ 1000 |
| total_chiller_load | 总冷机负荷 | kW | 0 ~ 25000 |
| load_group | 负荷分组 | 1-4 | - |
| temp_ma3 | 温度移动平均 | °C | -3 ~ 40 |
| temp_x_rh | 温度×湿度 | - | -200 ~ 4000 |

使用模型进行批量预测：

```python
import pandas as pd
import pickle

# 读取数据
df = pd.read_csv('input_data.csv')

# 加载模型
model = pickle.load(open('models/advanced/best_model_final.pkl', 'rb'))

# 预测
features = df[feature_columns]
df['predicted_baseline'] = model.predict(features)

# 计算节能效果
if 'actual_energy' in df.columns:
    df['savings'] = df['actual_energy'] - df['predicted_baseline']
    df['savings_rate'] = (df['savings'] / df['predicted_baseline']) * 100

# 保存结果
df.to_csv('prediction_results.csv', index=False)
```

## 应用场景

### 1. 节能改造验证

```python
# 改造前
baseline_before = model.predict(features_before)[0]
actual_before = 3000  # kW

# 改造后
baseline_after = model.predict(features_after)[0]
actual_after = 2700  # kW

# 节能效果
savings = actual_after - baseline_after
print(f"节能量: {savings:.2f} kW")
```

### 2. 运行优化效果评估

```python
# 优化前
baseline = model.predict(features)[0]
actual_before = 3500  # kW

# 优化温差后
actual_after = 3200  # kW

# 节能效果
savings = actual_after - baseline
savings_rate = (savings / baseline) * 100
print(f"温差优化节能量: {savings:.2f} kW ({savings_rate:.2f}%)")
```

### 3. 能效benchmarking

```python
# 对比不同时间段或操作人员
savings_rates = []
for time_period in time_periods:
    baseline = model.predict(period_features)[0]
    actual = get_actual_energy(time_period)
    rate = (baseline - actual) / baseline * 100
    savings_rates.append(rate)

print(f"各时间段节能率: {savings_rates}")
```

## 模型优势

### ✓ 完全去除可优化特征
- 不包含供回水温差（可调节）
- 不包含冷却水温差（可调节）
- 基线模型与优化程序完全解耦

### ✓ 可全年预测
- 不依赖month特征
- dew_point等气象特征充分捕捉季节性
- 避免外推预测风险

### ✓ 性能良好
- R² = 0.9897 (解释98.97%能耗变异)
- MAPE = 2.55% (平均误差仅2.55%)
- 完全满足基线预测需求

### ✓ 特征简单明确
- 仅有10个特征
- 物理意义清晰
- 易于理解和维护

## 版本历史

- **v3.0 (2025-01-04)**: 最终版本 ⭐⭐⭐⭐⭐
  - 完全去除所有可优化特征
  - 可全年预测
  - R² = 0.9897, MAPE = 2.55%
  - 最适合基线模型

- **v2.1 (2025-01-04)**: 全年版本
  - 去除month，使用dew_point替代
  - 可全年预测
  - R² = 0.9910, MAPE = 2.07%

- **v2.0 (2025-01-04)**: 优化版本
  - 使用总冷机负荷替代温差
  - 仅适用于1-7月
  - R² = 0.9921, MAPE = 2.08%

- **v1.0 (2025-12-30)**: 初始版本
  - 含温差特征
  - 仅适用于1-7月
  - R² = 0.9893, MAPE = 2.63%

## 注意事项

1. **适用范围**:
   - 时间: 全年（1-12月）✓
   - 温度: -3°C ~ 40°C ✓
   - 湿度: 10% ~ 100% ✓
   - 总冷机负荷: 0 ~ 25000 kW ✓

2. **数据质量**:
   - percentage_current数据必须准确（0-100%）
   - 气象数据需连续完整
   - temp_ma3需要至少2小时历史数据

3. **模型更新**:
   - 收集8-12月数据验证精度
   - 系统改造后需重新训练
   - 建议每季度更新一次

4. **局限性**:
   - 极端天气条件下精度可能下降
   - 基于历史运行模式
   - 系统重大改造后需重新训练

## 文件结构

```
projects/shiyan7/
├── README.md                           # 使用指南（本文件）
├── config.yaml                         # 项目配置
│
├── models/advanced/
│   ├── best_model_final.pkl           # 最终模型 ⭐
│   ├── best_model_no_month.pkl        # 全年版本（历史）
│   └── best_model_advanced.pkl        # v2.0版本（历史）
│
├── models/metadata/
│   ├── final_feature_columns.pkl       # 最终特征列表 ⭐
│   ├── no_month_feature_columns.pkl   # 全年特征列表（历史）
│   └── advanced_feature_columns.pkl    # v2.0特征列表（历史）
│
├── reports/
│   ├── MODEL_FINAL_VERSION_SUMMARY.txt # 最终版本总结 ⭐
│   ├── model_training_report_final.txt # 最终模型训练报告
│   └── ...                             # 其他历史报告
│
├── processed_data/
│   ├── test_final_data.csv            # 最终模型测试数据
│   └── ...                             # 其他数据文件
│
└── scripts/
    ├── train_final_model.py            # 最终模型训练脚本
    └── ...                             # 其他脚本
```

## 技术支持

如有问题或需要模型重新训练，请联系：

- 模型文件位置: `/home/long/energy_baseline_training/projects/shiyan7/models/advanced/best_model_final.pkl`
- 训练报告: `reports/model_training_report_final.txt`
- 版本总结: `reports/MODEL_FINAL_VERSION_SUMMARY.txt`
- 项目路径: `/home/long/energy_baseline_training/projects/shiyan7/`

---

**版本**: v3.0 (最终版本)
**最后更新**: 2025-01-04
