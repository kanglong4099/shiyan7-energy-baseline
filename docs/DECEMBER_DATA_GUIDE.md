# 12月数据预测与精度验证指南

## 一、数据放置位置

### 1. 原始数据

根据项目现有目录结构，12月原始数据应放在：

#### BMS数据
```
raw_data/bms/
└── AESC_SHIYAN7_history_data_2025.xlsx  # 包含12月的BMS历史数据
```

如果12月数据是单独文件，建议命名：
```
raw_data/bms/
├── AESC_SHIYAN7_history_data_20250805.xlsx  # 1-7月数据（原有）
└── AESC_SHIYAN7_history_data_202512.xlsx    # 12月数据（新增）⭐
```

#### 天气数据
```
raw_data/weather/
├── weather_SHIYAN_250101-250331.csv  # 1-3月（原有）
├── weather_SHIYAN_250401-250731.csv  # 4-7月（原有）
└── weather_SHIYAN_251201-251231.csv  # 12月数据（新增）⭐
```

### 2. 处理后数据

```
processed_data/
├── shiyan7_features.csv              # 1-7月特征（原有）
├── dec2025_features.csv              # 12月特征（新增）⭐
└── dec2025_prediction_comparison.csv  # 12月预测对比（新生成）⭐
```

### 3. 报告文件

```
reports/
├── dec2025_prediction_report.txt     # 12月预测报告（新生成）⭐
└── dec2025_accuracy_assessment.txt  # 12月精度评估（新生成）⭐
```

---

## 二、数据准备清单

### 必需数据

#### BMS数据 (12月)
必须包含以下列：
- `time`: 时间戳
- `CH1-CH-CH_raw_temp_chwr`, `CH1-CH-CH_raw_temp_chws` (CH1供回水温度)
- `CH2-CH-CH_raw_temp_chwr`, `CH2-CH-CH_raw_temp_chws` (CH2供回水温度)
- `CH3-CH-CH_raw_temp_chwr`, `CH3-CH-CH_raw_temp_chws` (CH3供回水温度)
- `CH4-CH-CH_raw_temp_chwr`, `CH4-CH-CH_raw_temp_chws` (CH4供回水温度)
- `CH5-CH-CH_raw_temp_chwr`, `CH5-CH-CH_raw_temp_chws` (CH5供回水温度)
- `CH1-percentage_current` ~ `CH5-percentage_current` (冷机电流百分比)
- `CWP1-CWP-PUMP_raw_power_active_total`, `CWP2-CWP-PUMP_raw_power_active_total` (冷却水泵)
- `CHWP1-CHWP-PUMP_raw_power_active_total`, `CHWP2-CHWP-PUMP_raw_power_active_total` (冷冻水泵)
- `CT1_5-CT-CT_raw_power_active_total`, `CT5_10-CT-CT_raw_power_active_total` (冷却塔)
- `CHPL-CHPL-CHPL_raw_temp_chwr_header_low`, `CHPL-CHPL-CHPL_raw_temp_chws_header_low` (低温水总管)
- `CHPL-CHPL-CHPL_raw_temp_chwr_header_medium`, `CHPL-CHPL-CHPL_raw_temp_chws_header_medium` (中温水总管)
- `CHPL-CHPL-CHPL_raw_temp_cws_header`, `CHPL-CHPL-CHPL_raw_temp_cwr_header` (冷却水总管)
- `CHPL-CHPL-CHPL_raw_delta_temperature_header` (总管温差)
- `CHPL-CHPL-CHPL_raw_load_building` (建筑负荷，可选)

#### 天气数据 (12月)
必须包含以下列：
- `time`: 时间戳
- `temperature`: 室外温度 (°C)
- `RH`: 相对湿度 (%)
- `pres`: 气压 (hPa)
- `ws`: 风速 (m/s)
- `wd`: 风向 (°)
- `ssrd`: 太阳辐射 (W/m²)

#### 真实能耗数据 (12月)
需要以下之一：
1. 从BMS数据计算: `total_plant_power = 冷机总能耗 + 水泵总能耗 + 冷却塔总能耗`
2. 单独的能耗记录文件

---

## 三、预测流程

### 步骤1: 数据合并
将12月的BMS数据和天气数据合并

### 步骤2: 特征计算
计算v3.0模型需要的10个特征：
1. hour (小时)
2. temperature (室外温度)
3. RH (相对湿度)
4. dew_point (露点温度)
5. wet_bulb (湿球温度)
6. ssrd (太阳辐射)
7. total_chiller_load (总冷机负荷)
8. load_group (负荷分组)
9. temp_ma3 (温度移动平均)
10. temp_x_rh (温度×湿度)

### 步骤3: 模型预测
使用v3.0最终模型预测12月能耗

### 步骤4: 精度评估
对比预测值与真实值，计算：
- R² (决定系数)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- MAPE (平均百分比误差)

### 步骤5: 生成报告
生成预测对比报告和精度评估报告

---

## 四、推荐做法

### 数据文件命名

```
raw_data/bms/
└── AESC_SHIYAN7_history_data_2025.xlsx  # 将1-12月合并到一个文件

或

raw_data/bms/
├── AESC_SHIYAN7_history_data_2501-07.xlsx  # 1-7月
└── AESC_SHIYAN7_history_data_2512.xlsx      # 12月
```

```
raw_data/weather/
├── weather_SHIYAN_250101-250731.csv  # 1-7月
└── weather_SHIYAN_251201-251231.csv  # 12月
```

### 使用脚本处理

可以使用以下脚本处理12月数据：
1. `scripts/predict_december.py` - 预测12月能耗
2. `scripts/evaluate_december_accuracy.py` - 评估12月精度

（这两个脚本需要根据实际情况创建）

---

## 五、快速开始

### 1. 准备数据
```bash
# 将12月BMS数据放到
cp /path/to/december_bms.xlsx raw_data/bms/

# 将12月天气数据放到
cp /path/to/december_weather.csv raw_data/weather/
```

### 2. 运行预测脚本
```bash
python3 scripts/predict_december.py
```

### 3. 查看结果
```bash
# 查看预测对比数据
cat processed_data/dec2025_prediction_comparison.csv

# 查看精度评估报告
cat reports/dec2025_accuracy_assessment.txt
```

---

## 六、注意事项

1. **时间格式**: 确保时间格式一致（建议：YYYY-MM-DD HH:MM:SS）
2. **数据完整性**: 12月数据应尽量完整，避免大量缺失值
3. **特征计算**: temp_ma3需要历史数据，前2小时可能无法计算
4. **真实能耗**: 确保真实能耗的计算方式与训练数据一致

---

## 七、预期输出

### 预测对比数据 (CSV)
```
time,actual_energy,predicted_energy,abs_error,rel_error(%),savings,savings_rate(%)
2025-12-01 00:00:00,1500.0,1485.5,14.5,0.97,-14.5,-0.98
2025-12-01 01:00:00,1450.0,1462.3,12.3,0.85,12.3,0.84
...
```

### 精度评估报告 (TXT)
```
================================================================================
           12月预测精度评估报告
================================================================================

数据范围: 2025-12-01 至 2025-12-31
数据量: 744 条

精度指标:
  R² (决定系数): 0.XXXX
  RMSE (均方根误差): XX.XX kW
  MAE (平均绝对误差): XX.XX kW
  MAPE (平均百分比误差): X.XX%

与测试集对比:
  测试集 MAPE: 2.55%
  12月 MAPE: X.XX%
  差异: ±X.XX%

结论:
  [评估结论]
```

---

## 八、联系与支持

如有问题，请检查：
1. 数据格式是否正确
2. 特征计算是否完整
3. 模型文件是否存在: `models/advanced/best_model_final.pkl`

项目路径: `/home/long/energy_baseline_training/projects/shiyan7/`
