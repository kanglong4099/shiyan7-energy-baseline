#!/usr/bin/env python3
"""
7#冷冻站能耗基线预测脚本（高级版本 - v3.0最终版本）

使用极端随机森林模型预测冷冻站能耗基线
模型精度: R²=0.9897, RMSE=88.53 kW, MAPE=2.55%
模型版本: v3.0 (最终版本，不含可优化特征）
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib


class Shiyan7PredictorAdvanced:
    """7#冷冻站高级预测器（使用极端随机森林）"""

    def __init__(self):
        """初始化预测器"""
        self.model_dir = os.path.join(
            os.path.dirname(__file__),
            '../models'
        )

        # 加载v3.0最终模型
        self.model = joblib.load(f'{self.model_dir}/advanced/best_model_final.pkl')
        self.feature_columns = joblib.load(f'{self.model_dir}/metadata/final_feature_columns.pkl')

        print(f"✓ 最终模型已加载: {self.model_dir}/advanced/best_model_final.pkl")
        print(f"✓ 模型类型: 极端随机森林 (ExtraTreesRegressor)")
        print(f"✓ 模型版本: v3.0 (最终版本）")
        print(f"✓ 特征数量: {len(self.feature_columns)}")

    def predict_baseline(self, hour, temperature, RH, dew_point, wet_bulb, ssrd,
                         total_chiller_load, load_group, temp_ma3, temp_x_rh):
        """
        预测能耗基线（使用所有特征）

        参数:
            hour: 小时 (0-23)
            temperature: 室外温度 (°C)
            RH: 相对湿度 (%)
            dew_point: 露点温度 (°C)
            wet_bulb: 湿球温度 (°C)
            ssrd: 太阳辐射 (W/m²)
            total_chiller_load: 总冷机负荷 (kW)
            load_group: 负荷分组 (int)
            temp_ma3: 温度3小时移动平均 (°C)
            temp_x_rh: 温度×湿度交互特征

        返回:
            predicted_energy: 预测能耗 (kW)
        """
        # 准备特征（按v3.0顺序：10个特征）
        features = [[
            hour, temperature, RH, dew_point, wet_bulb, ssrd,
            total_chiller_load, load_group, temp_ma3, temp_x_rh
        ]]

        # 创建DataFrame
        features_df = pd.DataFrame(features, columns=self.feature_columns)

        # 预测
        predicted_energy = self.model.predict(features_df)[0]

        return predicted_energy

    def predict_baseline_simple(self, temperature, RH, total_chiller_load, load_group):
        """
        预测能耗基线（简化版本，自动计算其他特征）

        参数:
            temperature: 室外温度 (°C)
            RH: 相对湿度 (%)
            total_chiller_load: 总冷机负荷 (kW)
            load_group: 负荷分组 (1-4)

        返回:
            predicted_energy: 预测能耗 (kW)
        """
        # 计算衍生特征
        dew_point = temperature - ((100 - RH) / 5)
        wet_bulb = temperature * 0.6 + dew_point * 0.4
        temp_ma3 = temperature  # 简化处理
        hour = 12  # 默认值
        ssrd = 200  # 默认值
        temp_x_rh = temperature * RH / 100

        return self.predict_baseline(
            hour, temperature, RH, dew_point, wet_bulb, ssrd,
            total_chiller_load, load_group, temp_ma3, temp_x_rh
        )

    def calculate_savings(self, actual_energy, predicted_energy):
        """
        计算节能量和节能率

        参数:
            actual_energy: 实际能耗 (kW)
            predicted_energy: 预测基线能耗 (kW)

        返回:
            savings: 节能量 (kW)
            savings_rate: 节能率 (%)
        """
        savings = actual_energy - predicted_energy
        savings_rate = (savings / predicted_energy) * 100
        return savings, savings_rate

    def predict_batch(self, input_file, output_file=None):
        """
        批量预测

        参数:
            input_file: 输入CSV文件路径
            output_file: 输出CSV文件路径（可选）

        返回:
            result_df: 包含预测结果的DataFrame
        """
        # 读取数据
        df = pd.read_csv(input_file)

        # 检查所需特征
        missing = [f for f in self.feature_columns if f not in df.columns]
        if missing:
            raise ValueError(f"数据文件缺失特征: {missing}")

        # 提取需要的特征
        X = df[self.feature_columns]

        # 预测
        df['predicted_energy'] = self.model.predict(X)

        # 计算节能效果（如果包含实际能耗）
        if 'actual_energy' in df.columns:
            df['savings'] = df['actual_energy'] - df['predicted_energy']
            df['savings_rate'] = (df['savings'] / df['predicted_energy']) * 100

            baseline_total = df['predicted_energy'].sum()
            actual_total = df['actual_energy'].sum()
            savings_total = baseline_total - actual_total
            savings_percent = (savings_total / baseline_total) * 100

            print(f"\n节能效果汇总:")
            print(f"  基线能耗: {baseline_total:.2f} kWh")
            print(f"  实际能耗: {actual_total:.2f} kWh")
            print(f"  节能量: {savings_total:.2f} kWh")
            print(f"  节能率: {savings_percent:.2f}%")
            print(f"  节约成本: ¥{savings_total * 0.8:.2f}")

        # 保存结果
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n✓ 预测结果已保存到: {output_file}")

        return df

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_columns),
            'features': self.feature_columns,
            'n_estimators': self.model.n_estimators if hasattr(self.model, 'n_estimators') else None,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_.tolist()))
        }


def main():
    """示例使用"""
    print("=" * 70)
    print(" " * 15 + "7#冷冻站能耗基线预测（最终版本v3.0）")
    print("=" * 70)
    print(f"\n模型性能: R²=0.9897, RMSE=88.53 kW, MAPE=2.55%")
    print(f"算法: 极端随机森林 (Extra Trees Regressor)")
    print(f"版本: v3.0 (最终版本，不含可优化特征）")

    # 初始化预测器
    predictor = Shiyan7PredictorAdvanced()

    # 示例1: 简化预测
    print("【示例1: 简化预测】")
    print("-" * 70)
    predicted = predictor.predict_baseline_simple(
        temperature=30.0,
        RH=60.0,
        total_chiller_load=15000.0,
        load_group=3
    )

    print(f"输入条件:")
    print(f"  室外温度: 30.0°C")
    print(f"  相对湿度: 60%")
    print(f"  总冷机负荷: 15000.0 kW")
    print(f"  负荷分组: 3")
    print(f"\n预测基线能耗: {predicted:.2f} kW")

    # 示例2: 计算节能量
    print("\n【示例2: 计算节能量】")
    print("-" * 70)
    actual_energy = 3500.0
    predicted_energy = predicted

    savings, savings_rate = predictor.calculate_savings(actual_energy, predicted_energy)

    print(f"实际能耗: {actual_energy:.2f} kW")
    print(f"基线能耗: {predicted_energy:.2f} kW")
    print(f"节能量: {savings:.2f} kW")
    print(f"节能率: {savings_rate:.2f}%")

    if savings > 0:
        print(f"✓ 结论: 能耗高于基线 {abs(savings_rate):.2f}%，需优化运行")
    else:
        print(f"✓ 结论: 能耗低于基线，实现节能 {abs(savings_rate):.2f}%")

    # 示例3: 特征重要性
    print("\n【示例3: 特征重要性 Top 5】")
    print("-" * 70)
    model_info = predictor.get_model_info()
    importance = sorted(model_info['feature_importance'].items(), key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(importance[:5], 1):
        print(f"  {i}. {feat}: {imp:.2%}")

    # 示例4: 模型信息
    print("\n【示例4: 模型信息】")
    print("-" * 70)
    print(f"模型类型: {model_info['model_type']}")
    print(f"特征数量: {model_info['n_features']}")
    if model_info['n_estimators']:
        print(f"树的数量: {model_info['n_estimators']}")

    # 示例5: 批量预测提示
    print("\n【示例5: 批量预测】")
    print("-" * 70)
    print("提示: 使用CSV文件进行批量预测")
    print("CSV文件需包含所有10个特征:")
    print(f"  {', '.join(model_info['features'])}")
    print("\n调用示例:")
    print("  predictor.predict_batch('input_data.csv', 'predictions.csv')")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
