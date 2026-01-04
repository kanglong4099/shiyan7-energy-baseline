# 数据说明文档

## 数据目录结构

```
raw_data/              # 原始数据（不上传到Git）
├── bms/              # BMS系统数据
│   └── AESC_SHIYAN7_history_data_202512.xlsx  # 12月BMS数据
├── weather/          # 气象数据
│   └── weather_SHIYAN_251201-251230.csv      # 12月气象数据
├── energy/           # 能耗数据
│   └── SHIYAN7_energy_20251201_20251230.csv  # 12月实际能耗数据
└── assets/           # 设备台账
    └── shiyan7_chiller_assets.xlsx          # 冷机设备清单

processed_data/        # 处理后的数据（不上传到Git）
├── shiyan7_features.csv                    # 训练特征数据
├── dec2025_prediction_only.csv             # 12月预测结果
└── dec2025_accuracy_detail.csv              # 12月精度评估
```

## 数据获取方式

由于数据文件包含敏感信息，不上传到 Git 仓库。如需获取数据：

1. **从本地数据库导出**
2. **联系数据管理员**
3. **按照以下格式准备数据**

---

## 数据格式说明

### 1. BMS数据 (raw_data/bms/)

**文件名**: `AESC_SHIYAN7_history_data_YYYYMM.xlsx`

**必需列**:
```csv
normalized_time, CH1_CH_raw_temp_cwr, CH1_CH_raw_temp_chws, CH1_CH_raw_fla,
CH2_CH_raw_temp_cwr, CH2_CH_raw_temp_chws, CH2_CH_raw_fla,
...
CH5_CH_raw_temp_cwr, CH5_CH_raw_temp_chws, CH5_CH_raw_fla
```

**列说明**:
- `normalized_time`: 时间戳 (YYYY-MM-DD HH:00:00)
- `CHx_CH_raw_temp_cwr`: 冷却水回水温度 (°C)
- `CHx_CH_raw_temp_chws`: 冷冻水供水温度 (°C)
- `CHx_CH_raw_fla`: 冷机电流百分比 (%)

**数据要求**:
- 时间范围: 至少覆盖一年
- 数据粒度: 小时级
- 完整性: 无缺失值

---

### 2. 气象数据 (raw_data/weather/)

**文件名**: `weather_SHIYAN_YYMMDD-YYMMDD.csv`

**必需列**:
```csv
time, temperature, ssrd, pres, RH
```

**列说明**:
- `time`: 时间戳
- `temperature`: 室外干球温度 (°C)
- `ssrd`: 太阳辐射 (W/m²)
- `pres`: 大气压力 (Pa)
- `RH`: 相对湿度 (%)

**数据要求**:
- 与BMS数据时间范围一致
- 数据粒度: 小时级

---

### 3. 能耗数据 (raw_data/energy/)

**文件名**: `SHIYAN7_energy_YYYYMMDD_YYYYMMDD.csv`

**必需列**:
```csv
time, CHPL_cal_power_plant
```

**列说明**:
- `time`: 时间戳
- `CHPL_cal_power_plant**: 冷冻站总能耗 (kW)

**数据要求**:
- 用于精度验证
- 与BMS数据时间范围一致

---

### 4. 设备台账 (raw_data/assets/)

**文件名**: `shiyan7_chiller_assets.xlsx`

**必需列**:
```csv
设备编号, 设备名称, 额定制冷量(kW)
```

**示例**:
```csv
CH1, 冷机1, 9845
CH2, 冷机2, 9845
CH3, 冷机3, 9845
CH4, 冷机4, 9845
CH5, 冷机5, 3517
```

---

## 数据预处理流程

### 1. 合并数据
```python
import pandas as pd

# 读取数据
bms_df = pd.read_excel('raw_data/bms/AESC_SHIYAN7_history_data_202512.xlsx')
weather_df = pd.read_csv('raw_data/weather/weather_SHIYAN_251201-251230.csv')

# 合并
merged = pd.merge(bms_df, weather_df, on='time', how='inner')
```

### 2. 计算总冷机负荷
```python
chiller_capacities = [9845, 9845, 9845, 9845, 3517]
chiller_percentages = [
    'CH1_CH_raw_fla', 'CH2_CH_raw_fla',
    'CH3_CH_raw_fla', 'CH4_CH_raw_fla',
    'CH5_CH_raw_fla'
]

merged['total_chiller_load'] = sum(
    merged[col] * cap / 100
    for col, cap in zip(chiller_percentages, chiller_capacities)
)
```

### 3. 计算负荷分组
```python
import numpy as np

merged['load_group'] = pd.cut(
    merged['total_chiller_load'],
    bins=[-np.inf, 6919, 10512, 13614, np.inf],
    labels=[1, 2, 3, 4]
).astype(int)
```

### 4. 计算衍生特征
```python
# 露点温度
merged['dew_point'] = merged['temperature'] - ((100 - merged['RH']) / 5)

# 湿球温度
merged['wet_bulb'] = merged['temperature'] * 0.6 + merged['dew_point'] * 0.4

# 温度×湿度
merged['temp_x_rh'] = merged['temperature'] * merged['RH'] / 100

# 温度移动平均
merged['temp_ma3'] = merged['temperature'].rolling(window=3, min_periods=1).mean()

# 小时
merged['hour'] = pd.to_datetime(merged['time']).dt.hour
```

---

## 数据质量检查

### 检查清单

- [ ] 数据时间范围完整（无缺失时间段）
- [ ] 数据粒度正确（小时级）
- [ ] 无异常值（如温度 > 50°C）
- [ ] 无缺失值
- [ ] BMS数据和气象数据时间对齐

### 异常值处理

```python
# 温度范围检查
merged = merged[(merged['temperature'] >= -20) & (merged['temperature'] <= 50)]
merged = merged[(merged['RH'] >= 0) & (merged['RH'] <= 100)]

# 冷机负荷范围检查
merged = merged[merged['total_chiller_load'] >= 0]
merged = merged[merged['total_chiller_load'] <= 50000]
```

---

## 数据隐私和安全

**注意事项**:
1. ❌ 不要上传包含敏感信息的BMS数据
2. ❌ 不要上传包含能耗计费信息的能耗数据
3. ❌ 不要上传日志文件（可能包含调试信息）
4. ✓ 可以上传匿名化的样本数据
5. ✓ 可以上传数据格式说明文档

---

## 数据备份

建议定期备份：
1. 原始BMS数据（每月备份）
2. 气象数据（自动同步）
3. 能耗数据（每月备份）
4. 模型文件（每次训练后备份）

备份位置：
- 本地服务器
- 云存储（OSS/S3）
- 异地备份

---

## 联系方式

如有数据相关问题，请联系：
- 数据管理员: [联系方式]
- 技术支持: [联系方式]

---

**最后更新**: 2025-01-04
**版本**: v1.0
