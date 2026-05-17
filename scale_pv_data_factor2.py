# -*- coding: utf-8 -*-
"""
光伏数据扩大工具（2倍）
生成扩大2倍的光伏数据文件，用于强化学习训练
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

SCALE_FACTOR = 2.0
INPUT_FILE = 'date_file/按日累计-台区R1-总光伏出力.csv'
OUTPUT_FILE = f'date_file/按日累计-台区R1-总光伏出力_扩大{SCALE_FACTOR:.1f}倍.csv'

print('=' * 80)
print(f"光伏数据扩大工具 - 扩大倍数: {SCALE_FACTOR}x")
print('=' * 80)

print("\n[步骤1] 读取原始光伏数据...")
try:
    pv_data = pd.read_csv(INPUT_FILE, encoding='utf-8')
except UnicodeDecodeError:
    pv_data = pd.read_csv(INPUT_FILE, encoding='gbk')

print(f"原始数据形状: {pv_data.shape}")

print("\n[步骤2] 处理光伏数据...")
dates = pv_data.iloc[:, 0]
if pv_data.shape[1] == 98:
    print("检测到98列数据，移除最后一列...")
    pv_values = pv_data.iloc[:, 1:97].values.astype(np.float32)
else:
    pv_values = pv_data.iloc[:, 1:].values.astype(np.float32)

print("原始数据统计:")
print(f"  最小值: {pv_values.min():.2f} kW")
print(f"  最大值: {pv_values.max():.2f} kW")
print(f"  平均值: {pv_values.mean():.2f} kW")

pv_values = np.abs(pv_values)

pv_q99 = np.percentile(pv_values, 99)
outlier_count = np.sum(pv_values > pv_q99)
if outlier_count > 0:
    print(f"\n检测到异常值: {outlier_count}个点超过99%分位数({pv_q99:.2f} kW)")
    print(f"已将异常值限制在 {pv_q99:.2f} kW 以内")
    pv_values = np.clip(pv_values, 0, pv_q99)

print("\n处理后数据统计:")
print(f"  最小值: {pv_values.min():.2f} kW")
print(f"  最大值: {pv_values.max():.2f} kW")
print(f"  平均值: {pv_values.mean():.2f} kW")

print(f"\n[步骤3] 将光伏数据扩大 {SCALE_FACTOR} 倍...")
pv_values_scaled = pv_values * SCALE_FACTOR

print("\n扩大后数据统计:")
print(f"  最小值: {pv_values_scaled.min():.2f} kW")
print(f"  最大值: {pv_values_scaled.max():.2f} kW")
print(f"  平均值: {pv_values_scaled.mean():.2f} kW")

print("\n[步骤4] 构建新数据...")
p_columns = [f'P{i}' for i in range(1, 97)]
scaled_df = pd.DataFrame(pv_values_scaled, columns=p_columns)
scaled_df.insert(0, dates.name if dates.name else '日期', dates.values)

print("\n[步骤5] 保存扩大后的数据...")
scaled_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"成功保存到: {OUTPUT_FILE}")

print("\n[步骤6] 验证保存结果...")
verify_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
verify_values = verify_df.iloc[:, 1:].values
print(f"验证数据形状: {verify_df.shape}")
print(f"验证数据均值: {verify_values.mean():.2f} kW")
print(f"数据一致性检查: {'通过' if np.allclose(verify_values, pv_values_scaled) else '失败'}")

print("\n[步骤7] 生成简要对比指标...")
original_total = pv_values.sum()
scaled_total = pv_values_scaled.sum()
print(f"  原始总发电量: {original_total / 1000:.2f} MWh")
print(f"  扩大后总发电量: {scaled_total / 1000:.2f} MWh")
print(f"  倍数校验: {scaled_total / original_total:.2f}x")

print("\n操作完成，可在 rural_env.py 中改用新的光伏数据文件。")
print('=' * 80)

