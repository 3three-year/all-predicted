# -*- coding: utf-8 -*-
"""
从充电桩数据-2022中提取指定台区编号的数据，并按日期累加
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# 目标台区编号（转换为整数，因为文件中是int64类型）
target_stations = [13007403, 11984688, 10589545, 14588386, 14738916]

print("="*60)
print("提取指定台区的充电负荷数据")
print("="*60)
print(f"目标台区编号: {target_stations}")

# 读取数据文件
data_file = 'date_file/充电桩数据-2022.csv'
print(f"\n读取数据文件: {data_file}")
print(f"文件大小: {os.path.getsize(data_file) / 1024 / 1024:.2f} MB")

# 分块读取数据（文件太大，需要分块处理）
print("\n开始分块读取和筛选数据...")
chunk_size = 100000  # 每次读取10万行
chunks = []
total_rows = 0
matched_rows = 0

for chunk in pd.read_csv(data_file, chunksize=chunk_size, encoding='utf-8'):
    total_rows += len(chunk)
    
    # 筛选目标台区
    filtered_chunk = chunk[chunk['台区编号'].isin(target_stations)]
    
    if len(filtered_chunk) > 0:
        matched_rows += len(filtered_chunk)
        chunks.append(filtered_chunk)
        print(f"  已处理 {total_rows:,} 行，找到 {matched_rows:,} 条匹配数据...")
    
    # 每处理100万行显示一次进度
    if total_rows % 1000000 == 0:
        print(f"  进度: {total_rows:,} 行，已匹配 {matched_rows:,} 条...")

print(f"\n总共处理 {total_rows:,} 行")
print(f"找到匹配数据 {matched_rows:,} 条")

# 合并所有筛选后的数据
if chunks:
    print("\n合并筛选后的数据...")
    filtered_data = pd.concat(chunks, ignore_index=True)
    print(f"合并后数据行数: {len(filtered_data):,}")
    
    # 显示每个台区的数据量
    print("\n各台区数据量:")
    station_counts = filtered_data['台区编号'].value_counts()
    print(station_counts)
    
    # 检查日期范围
    filtered_data['日期'] = pd.to_datetime(filtered_data['日期'])
    print(f"\n日期范围: {filtered_data['日期'].min()} 至 {filtered_data['日期'].max()}")
    print(f"总天数: {filtered_data['日期'].nunique()}")
    
    # 提取功率列（P1-P96）
    p_columns = [f'P{i}' for i in range(1, 97)]
    
    # 按日期分组，累加所有台区的功率
    print("\n按日期累加功率数据（96个时间点）...")
    
    # 按日期分组，对每个时间点的功率求和
    daily_sum = filtered_data.groupby('日期')[p_columns].sum().reset_index()
    
    print(f"\n累加后的数据形状: {daily_sum.shape}")
    print(f"日期范围: {daily_sum['日期'].min()} 至 {daily_sum['日期'].max()}")
    print(f"总天数: {len(daily_sum)}")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(daily_sum.head())
    
    # 检查数据统计
    print("\n数据统计:")
    all_power = daily_sum[p_columns].values.flatten()
    non_zero_power = all_power[all_power > 0.1]
    print(f"  总数据点: {len(all_power):,}")
    print(f"  非零数据点: {len(non_zero_power):,} ({len(non_zero_power)/len(all_power)*100:.1f}%)")
    print(f"  平均功率: {np.mean(non_zero_power):.4f} kW")
    print(f"  最大功率: {np.max(all_power):.4f} kW")
    print(f"  每日平均总充电量: {np.mean(daily_sum[p_columns].sum(axis=1)) * 0.25:.2f} kWh")
    
    # 准备输出数据（格式与现有文件一致）
    output_data = []
    for idx, row in daily_sum.iterrows():
        row_data = {'日期': row['日期'].strftime('%Y-%m-%d')}
        for i in range(1, 97):
            row_data[f'P{i}'] = row[f'P{i}']
        output_data.append(row_data)
    
    output_df = pd.DataFrame(output_data)
    
    # 保存数据
    output_file = 'date_file/按日累计-居民台区2-充电负荷数据.csv'
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n数据已保存: {output_file}")
    print(f"输出数据形状: {output_df.shape}")
    
    # 验证保存的数据
    print("\n验证保存的数据...")
    verify_df = pd.read_csv(output_file, nrows=5, encoding='utf-8')
    print("前5行:")
    print(verify_df.head())
    
    # 生成统计报告
    print("\n" + "="*60)
    print("提取结果统计")
    print("="*60)
    print(f"目标台区编号: {target_stations}")
    print(f"提取的数据行数: {len(filtered_data):,}")
    print(f"累加后的天数: {len(daily_sum)}")
    print(f"日期范围: {daily_sum['日期'].min()} 至 {daily_sum['日期'].max()}")
    print(f"平均每日总充电量: {np.mean(daily_sum[p_columns].sum(axis=1)) * 0.25:.2f} kWh")
    print(f"最大单点功率: {np.max(all_power):.4f} kW")
    print(f"输出文件: {output_file}")
    
else:
    print("\n[错误] 未找到目标台区的数据")
    print("请检查:")
    print("1. 台区编号是否正确")
    print("2. 数据文件中是否包含这些台区编号")
    print("3. 列名是否正确（应为'台区编号'）")

print("\n" + "="*60)
print("处理完成！")
print("="*60)
