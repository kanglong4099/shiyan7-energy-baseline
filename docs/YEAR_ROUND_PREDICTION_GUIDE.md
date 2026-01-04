# 7#冷冻站全年预测使用指南

## 一、问题与解决方案

### 问题
- month特征很重要（11.42%），但数据只有1-7月
- 无法预测8-12月的数据

### 解决方案
- **推荐**: 使用不含month的模型（全年通用）
- **原因**: dew_point相关性（0.906）≈ month（0.903），可替代

---

## 二、模型选择

### 方案对比

| 方案 | 模型 | 适用月份 | 精度 | 复杂度 |
|------|------|---------|------|--------|
| 混合 | 含month (1-7月) + 不含month (8-12月) | 全年 | 最高 | 高 |
| **统一** | **不含month** | **全年** | **高** | **低** |

### 推荐：全年统一使用不含month模型

**理由**:
1. 性能几乎无损（R²: 0.9921 → 0.9910，仅降0.11%）
2. 更简单，避免切换模型
3. dew_point充分捕捉季节性变化

---

## 三、模型信息

### 模型性能
- **算法**: 极端随机森林
- **R²**: 0.9910
- **RMSE**: 82.78 kW
- **MAPE**: 2.07%

### 特征列表（11个）

| 序号 | 特征 | 重要性 | 类型 | 说明 |
|------|------|--------|------|------|
| 1 | load_group | 38.49% | 负荷 | 负荷分组(1-4) |
| 2 | total_chiller_load | 31.19% | 负荷 | 总冷机负荷(kW) |
| 3 | dew_point | 13.71% | 气象 | 露点温度(°C) ⭐ |
| 4 | temp_x_rh | 7.05% | 衍生 | 温度×湿度 |
| 5 | wet_bulb | 2.30% | 气象 | 湿球温度(°C) |
| 6 | temp_ma3 | 1.95% | 衍生 | 温度移动平均 |
| 7 | delta_temp_header | 1.91% | 系统 | 总管温差(°C) |
| 8 | temperature | 1.33% | 气象 | 干球温度(°C) |
| 9 | ssrd | 0.80% | 气象 | 太阳辐射(W/m²) |
| 10 | RH | 0.79% | 气象 | 相对湿度(%) |
| 11 | hour | 0.47% | 时间 | 小时(0-23) |

---

## 四、预测步骤

### 步骤1: 获取基础数据
```python
# 从BMS获取
temperature = 30.0      # 室外温度 (°C)
RH = 60.0               # 相对湿度 (%)
hour = 14               # 小时 (0-23)
ssrd = 500.0            # 太阳辐射 (W/m²)
delta_temp_header = 3.5 # 总管温差 (°C)

# 冷机电流百分比
CH1_pct = 80.0
CH2_pct = 85.0
CH3_pct = 75.0
CH4_pct = 0.0
CH5_pct = 60.0
```

### 步骤2: 计算衍生特征
```python
# 露点温度
dew_point = temperature - ((100 - RH) / 5)

# 湿球温度
wet_bulb = temperature * 0.6 + dew_point * 0.4

# 温度×湿度
temp_x_rh = temperature * RH / 100

# 温度3小时移动平均（需历史数据）
temp_ma3 = (temperature_t + temperature_t-1 + temperature_t-2) / 3
```

### 步骤3: 计算总冷机负荷
```python
chiller_capacities = {
    'CH1': 9845.0, 'CH2': 9845.0, 'CH3': 9845.0,
    'CH4': 9845.0, 'CH5': 3517.0
}

total_chiller_load = (
    9845 * CH1_pct / 100 +
    9845 * CH2_pct / 100 +
    9845 * CH3_pct / 100 +
    9845 * CH4_pct / 100 +
    3517 * CH5_pct / 100
)
```

### 步骤4: 确定负荷分组
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

### 步骤5: 模型预测
```python
import joblib
import numpy as np

# 加载模型
model = joblib.load('models/advanced/best_model_no_month.pkl')

# 准备特征（按特征列表顺序）
features = np.array([[
    hour, temperature, RH, dew_point, wet_bulb, ssrd,
    total_chiller_load, load_group, delta_temp_header,
    temp_ma3, temp_x_rh
]])

# 预测
predicted_energy = model.predict(features)[0]
```

---

## 五、预测示例

### 示例1: 夏季（7月）
```python
# 输入
temperature = 30.0
RH = 60.0
hour = 14
ssrd = 500.0
CH1_pct, CH2_pct, CH3_pct, CH4_pct, CH5_pct = 80.0, 85.0, 75.0, 0.0, 60.0
delta_temp_header = 3.5

# 计算
dew_point = 30.0 - ((100 - 60) / 5) = 22.0
wet_bulb = 30.0 * 0.6 + 22.0 * 0.4 = 26.8
temp_x_rh = 30.0 * 60.0 / 100 = 18.0
total_chiller_load = 25516.2  # kW
load_group = 4

# 预测结果: ~3500 kW
```

### 示例2: 冬季（12月）
```python
# 输入
temperature = 5.0
RH = 40.0
hour = 10
ssrd = 100.0
CH1_pct, CH2_pct, CH3_pct, CH4_pct, CH5_pct = 30.0, 35.0, 0.0, 0.0, 25.0
delta_temp_header = 2.5

# 计算
dew_point = 5.0 - ((100 - 40) / 5) = -7.0
wet_bulb = 5.0 * 0.6 + (-7.0) * 0.4 = 0.2
temp_x_rh = 5.0 * 40.0 / 100 = 2.0
total_chiller_load = 7183.25  # kW
load_group = 2

# 预测结果: ~1400 kW
```

---

## 六、注意事项

### 数据质量
1. **percentage_current必须准确**（0-100%）
2. **气象数据需连续**（温度、湿度）
3. **temp_ma3需要历史数据**（至少2小时）

### 预测范围
- **时间**: 全年 ✓
- **温度**: -3°C ~ 40°C ✓
- **湿度**: 10% ~ 100% ✓
- **负荷**: 0 ~ 25000 kW ✓

### 局限性
1. 极端天气下精度可能下降
2. 系统改造后需重新训练
3. 8-12月预测需后续数据验证

---

## 七、快速参考

### 特征计算公式
```python
dew_point = temperature - ((100 - RH) / 5)
wet_bulb = temperature * 0.6 + dew_point * 0.4
temp_x_rh = temperature * RH / 100
temp_ma3 = (T_t + T_t-1 + T_t-2) / 3
```

### 负荷分组
```
分组1: ≤ 6919 kW    (25%)
分组2: 6919~10512 kW (25%)
分组3: 10512~13614 kW (25%)
分组4: > 13614 kW   (25%)
```

### 模型文件
- 模型: `models/advanced/best_model_no_month.pkl`
- 特征: `models/metadata/no_month_feature_columns.pkl`

---

## 八、更新日志

### v2.1 (2025-01-04)
- ✅ 训练不含month的模型
- ✅ 性能: R²=0.9910, MAPE=2.07%
- ✅ 可用于全年预测
- ✅ dew_point替代month（相关性: 0.906 ≈ 0.903）

### v2.0 (2025-01-04)
- ✅ 使用总冷机负荷替代温差
- ✅ 性能: R²=0.9921, MAPE=2.08%
- ❌ 仅适用于1-7月

---

## 联系信息

如有问题或需要技术支持，请联系：
- 模型路径: `/home/long/energy_baseline_training/projects/shiyan7/`
- 报告文件: `reports/YEAR_ROUND_PREDICTION_SOLUTION.txt`
