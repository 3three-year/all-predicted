"""
将时间序列格式的光伏数据转换为按日累计格式
输入: 处理后的全年光伏数据_含预测.xlsx (35040行×2列，每15分钟一行)
输出: 按日累计-台区R1-总光伏出力_扩大2.0倍_含预测.csv (365行×97列，每天一行)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("将时间序列格式转换为按日累计格式")
print("=" * 80)

# ==================== 步骤1: 读取时间序列数据 ====================
print("\n【步骤1】读取时间序列数据...")

input_file = 'date_file_with_prediction/处理后的全年光伏数据_含预测.xlsx'
print(f"  输入文件: {input_file}")

df_timeseries = pd.read_excel(input_file)
print(f"  数据形状: {df_timeseries.shape}")
print(f"  列名: {df_timeseries.columns.tolist()}")

# 显示数据预览
print(f"\n  数据预览:")
print(df_timeseries.head())

# 确保日期列是datetime格式
date_col = df_timeseries.columns[0]
value_col = df_timeseries.columns[1]
df_timeseries[date_col] = pd.to_datetime(df_timeseries[date_col])

print(f"\n  日期范围: {df_timeseries[date_col].min()} 至 {df_timeseries[date_col].max()}")
print(f"  总时间点数: {len(df_timeseries)}")
print(f"  预计天数: {len(df_timeseries) / 96:.0f}")

# ==================== 步骤2: 转换为按日累计格式 ====================
print("\n【步骤2】转换为按日累计格式...")

# 提取日期（不含时间）
df_timeseries['日期'] = df_timeseries[date_col].dt.date

# 按日期分组
grouped = df_timeseries.groupby('日期')

print(f"  分组后天数: {len(grouped)}")

# 创建结果DataFrame
result_data = []

for date, group in grouped:
    # 确保每天有96个时间点
    if len(group) != 96:
        print(f"  ⚠ 警告: {date} 有 {len(group)} 个时间点（应该是96个）")
    
    # 提取该天的96个功率值
    power_values = group[value_col].values
    
    # 如果不足96个，用0填充；如果超过96个，只取前96个
    if len(power_values) < 96:
        power_values = np.pad(power_values, (0, 96 - len(power_values)), 'constant', constant_values=0)
    elif len(power_values) > 96:
        power_values = power_values[:96]
    
    # 创建一行数据：日期 + P1-P96
    row_data = [date] + power_values.tolist()
    result_data.append(row_data)

# 创建列名：日期, P1, P2, ..., P96
columns = ['日期'] + [f'P{i}' for i in range(1, 97)]

# 创建DataFrame
df_daily = pd.DataFrame(result_data, columns=columns)

print(f"\n  转换后数据形状: {df_daily.shape}")
print(f"  列数: {len(df_daily.columns)} (应该是97: 1个日期列 + 96个P列)")
print(f"  行数: {len(df_daily)} (应该是365天)")

# 显示转换后的数据预览
print(f"\n  转换后数据预览:")
print(df_daily.head())

# ==================== 步骤3: 保存为CSV文件 ====================
print("\n【步骤3】保存为CSV文件...")

output_dir = 'date_file_with_prediction'
output_file = os.path.join(output_dir, '按日累计-台区R1-总光伏出力_含预测.csv')

# 保存为CSV（覆盖之前的文件）
df_daily.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  ✓ 文件已保存: {output_file}")

# ==================== 步骤4: 验证转换结果 ====================
print("\n【步骤4】验证转换结果...")

# 重新读取保存的文件
df_verify = pd.read_csv(output_file, encoding='utf-8-sig')
print(f"\n  验证读取:")
print(f"    文件形状: {df_verify.shape}")
print(f"    列名: {df_verify.columns.tolist()[:5]}... (共{len(df_verify.columns)}列)")

# 检查特定日期的数据
target_dates = ['2022-09-16', '2022-09-25']
print(f"\n  检查目标日期的数据:")

for target_date in target_dates:
    # 在转换后的数据中查找
    mask = df_daily['日期'].astype(str) == target_date
    if mask.sum() > 0:
        row = df_daily[mask].iloc[0]
        p_cols = [f'P{i}' for i in range(1, 97)]
        values = row[p_cols].values
        
        print(f"\n  {target_date}:")
        print(f"    数据点数: {len(values)}")
        print(f"    平均值: {values.mean():.2f} kW")
        print(f"    最大值: {values.max():.2f} kW")
        print(f"    最小值: {values.min():.2f} kW")
        print(f"    前3个值: {values[:3]}")
        print(f"    后3个值: {values[-3:]}")
        
        # 与原始时间序列数据对比
        date_obj = pd.to_datetime(target_date).date()
        mask_ts = df_timeseries['日期'] == date_obj
        if mask_ts.sum() > 0:
            ts_values = df_timeseries[mask_ts][value_col].values
            print(f"    原始时间序列平均值: {ts_values.mean():.2f} kW")
            print(f"    数据是否一致: {np.allclose(values, ts_values, rtol=1e-5)}")
    else:
        print(f"  ✗ 未找到 {target_date}")

# ==================== 步骤5: 与参考格式对比 ====================
print("\n【步骤5】与参考格式对比...")

reference_file = 'date_file_with_prediction/按日累计-居民台区2-充电负荷数据.csv'
print(f"  参考文件: {reference_file}")

try:
    df_reference = pd.read_csv(reference_file, encoding='utf-8-sig')
    print(f"  参考文件形状: {df_reference.shape}")
    print(f"  参考文件列名: {df_reference.columns.tolist()[:5]}... (共{len(df_reference.columns)}列)")
    
    print(f"\n  格式对比:")
    print(f"    新文件: {df_daily.shape[0]}行 × {df_daily.shape[1]}列")
    print(f"    参考文件: {df_reference.shape[0]}行 × {df_reference.shape[1]}列")
    
    if df_daily.shape == df_reference.shape:
        print(f"    ✓ 格式一致！")
    else:
        print(f"    ⚠ 格式不完全一致")
        
    # 检查列名是否一致
    if df_daily.columns.tolist() == df_reference.columns.tolist():
        print(f"    ✓ 列名一致！")
    else:
        print(f"    ⚠ 列名不完全一致")
        print(f"      新文件第一列: {df_daily.columns[0]}")
        print(f"      参考文件第一列: {df_reference.columns[0]}")
        
except Exception as e:
    print(f"  ⚠ 无法读取参考文件: {e}")

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"\n转换结果:")
print(f"  输入: {input_file}")
print(f"    格式: 时间序列 ({len(df_timeseries)}行 × {len(df_timeseries.columns)}列)")
print(f"  输出: {output_file}")
print(f"    格式: 按日累计 ({len(df_daily)}行 × {len(df_daily.columns)}列)")
print(f"\n数据说明:")
print(f"  - 包含 {len(df_daily)} 天的数据")
print(f"  - 每天 96 个时间点（15分钟间隔）")
print(f"  - 其中 2022-09-16 和 2022-09-25 使用预测值")
print(f"  - 其他天数保持原始数据")
print("\n文件可以直接用于强化学习环境！")
print("=" * 80)
