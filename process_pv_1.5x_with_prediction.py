"""
处理1.5倍光伏数据并整合预测值
1. 读取处理后的全年光伏数据_1.5倍.xlsx
2. 替换9月16日和9月25日为对应的1.5倍预测数据
3. 转换为按日累计格式（365行×97列）
4. 保存到date_file_with_prediction目录
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("处理1.5倍光伏数据并整合预测值")
print("=" * 80)

# ==================== 步骤1: 读取全年1.5倍光伏数据 ====================
print("\n【步骤1】读取全年1.5倍光伏数据...")

yearly_file = 'DATE_FILE_predict/处理后的全年光伏数据_1.5倍.xlsx'
print(f"  输入文件: {yearly_file}")

df_yearly = pd.read_excel(yearly_file)
print(f"  数据形状: {df_yearly.shape}")
print(f"  列名: {df_yearly.columns.tolist()}")
print(f"\n  数据预览:")
print(df_yearly.head())

# 确保日期列是datetime格式
date_col = df_yearly.columns[0]
value_col = df_yearly.columns[1]
df_yearly[date_col] = pd.to_datetime(df_yearly[date_col])

print(f"\n  日期范围: {df_yearly[date_col].min()} 至 {df_yearly[date_col].max()}")
print(f"  总时间点数: {len(df_yearly)}")
print(f"  预计天数: {len(df_yearly) / 96:.0f}")

# ==================== 步骤2: 读取1.5倍预测数据 ====================
print("\n【步骤2】读取1.5倍预测数据...")

# 2.1 读取9月16日 1.5倍预测数据
sep16_file = 'DATE_FILE_predict/单日数据_2022-09-16_1.5倍.xlsx'
print(f"\n  读取文件: {sep16_file}")
sep16_df = pd.read_excel(sep16_file)
print(f"  数据形状: {sep16_df.shape}")
print(f"  列名: {sep16_df.columns.tolist()}")

# 提取预测值列
if '预测值' in sep16_df.columns:
    sep16_predictions = sep16_df['预测值'].values
elif 'Predicted' in sep16_df.columns:
    sep16_predictions = sep16_df['Predicted'].values
else:
    numeric_cols = sep16_df.select_dtypes(include=[np.number]).columns
    print(f"  可用的数值列: {numeric_cols.tolist()}")
    sep16_predictions = sep16_df[numeric_cols[-1]].values

print(f"  9月16日预测数据: {len(sep16_predictions)} 个时间点")
print(f"    平均值: {sep16_predictions.mean():.2f} kW")
print(f"    最大值: {sep16_predictions.max():.2f} kW")
print(f"    最小值: {sep16_predictions.min():.2f} kW")

# 2.2 读取9月25日 1.5倍预测数据
sep25_file = 'DATE_FILE_predict/单日数据_2022-09-25_1.5倍.xlsx'
print(f"\n  读取文件: {sep25_file}")
sep25_df = pd.read_excel(sep25_file)
print(f"  数据形状: {sep25_df.shape}")
print(f"  列名: {sep25_df.columns.tolist()}")

# 提取预测值列
if '预测值' in sep25_df.columns:
    sep25_predictions = sep25_df['预测值'].values
elif 'Predicted' in sep25_df.columns:
    sep25_predictions = sep25_df['Predicted'].values
else:
    numeric_cols = sep25_df.select_dtypes(include=[np.number]).columns
    print(f"  可用的数值列: {numeric_cols.tolist()}")
    sep25_predictions = sep25_df[numeric_cols[-1]].values

print(f"  9月25日预测数据: {len(sep25_predictions)} 个时间点")
print(f"    平均值: {sep25_predictions.mean():.2f} kW")
print(f"    最大值: {sep25_predictions.max():.2f} kW")
print(f"    最小值: {sep25_predictions.min():.2f} kW")

# 将预测数据存储到字典中
pv_predictions = {
    '2022-09-16': sep16_predictions,
    '2022-09-25': sep25_predictions
}

# ==================== 步骤3: 替换预测数据 ====================
print("\n【步骤3】替换预测数据...")

# 创建副本
df_yearly_new = df_yearly.copy()

target_dates = ['2022-09-16', '2022-09-25']
replaced_count = 0
replaced_points = 0

for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    
    # 找到该日期的所有时间点
    mask = df_yearly_new[date_col].dt.date == target_dt.date()
    
    if mask.sum() > 0:
        print(f"\n  处理 {target_date}:")
        print(f"    找到 {mask.sum()} 个时间点")
        
        # 获取预测数据
        predictions = pv_predictions[target_date]
        print(f"    预测数据: {len(predictions)} 个值")
        
        # 替换该日期的所有行
        indices = df_yearly_new[mask].index
        
        # 替换每个时间点的值
        for i, idx in enumerate(indices):
            if i < len(predictions):
                old_val = df_yearly_new.loc[idx, value_col]
                df_yearly_new.loc[idx, value_col] = predictions[i]
                if i == 0:
                    print(f"    第1个时间点: {old_val:.4f} → {predictions[i]:.2f} kW")
                if i == len(indices) - 1:
                    print(f"    第{i+1}个时间点: {old_val:.4f} → {predictions[i]:.2f} kW")
        
        replaced_points += min(len(indices), len(predictions))
        replaced_count += 1
        print(f"    ✓ 成功替换 {min(len(indices), len(predictions))} 个时间点")
    else:
        print(f"  ✗ 未找到 {target_date}")

print(f"\n  总计替换了 {replaced_count} 天的数据，共 {replaced_points} 个时间点")

# ==================== 步骤4: 转换为按日累计格式 ====================
print("\n【步骤4】转换为按日累计格式...")

# 提取日期（不含时间）
df_yearly_new['日期'] = df_yearly_new[date_col].dt.date

# 按日期分组
grouped = df_yearly_new.groupby('日期')
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

# ==================== 步骤5: 保存文件 ====================
print("\n【步骤5】保存文件...")

output_dir = 'date_file_with_prediction'
os.makedirs(output_dir, exist_ok=True)

# 保存为CSV格式
output_csv_file = os.path.join(output_dir, '按日累计-台区R1-总光伏出力_1.5倍_含预测.csv')
df_daily.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
print(f"  ✓ CSV文件已保存: {output_csv_file}")

# 同时保存Excel格式（可选）
output_excel_file = os.path.join(output_dir, '处理后的全年光伏数据_1.5倍_含预测.xlsx')
df_yearly_new.to_excel(output_excel_file, index=False, engine='openpyxl')
print(f"  ✓ Excel文件已保存: {output_excel_file}")

# ==================== 步骤6: 验证结果 ====================
print("\n【步骤6】验证结果...")

# 重新读取保存的文件
df_verify = pd.read_csv(output_csv_file, encoding='utf-8-sig')
print(f"\n  验证读取:")
print(f"    文件形状: {df_verify.shape}")
print(f"    列名: {df_verify.columns.tolist()[:5]}... (共{len(df_verify.columns)}列)")

# 检查特定日期的数据
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
        
        # 与预测数据对比
        pred_values = pv_predictions[target_date]
        print(f"    预测数据平均值: {pred_values.mean():.2f} kW")
        print(f"    数据是否一致: {np.allclose(values, pred_values, rtol=1e-5)}")
    else:
        print(f"  ✗ 未找到 {target_date}")

# ==================== 步骤7: 与参考格式对比 ====================
print("\n【步骤7】与参考格式对比...")

reference_file = 'date_file_with_prediction/按日累计-台区R1-总光伏出力_含预测.csv'
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
        
except Exception as e:
    print(f"  ⚠ 无法读取参考文件: {e}")

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"\n生成的文件:")
print(f"  1. {output_csv_file}")
print(f"     格式: 按日累计 (365行 × 97列)")
print(f"     用途: 直接用于强化学习环境")
print(f"\n  2. {output_excel_file}")
print(f"     格式: 时间序列 (35040行 × 2列)")
print(f"     用途: 数据分析和可视化")
print(f"\n数据说明:")
print(f"  - 包含 {len(df_daily)} 天的1.5倍光伏数据")
print(f"  - 每天 96 个时间点（15分钟间隔）")
print(f"  - 其中 2022-09-16 和 2022-09-25 使用1.5倍预测值")
print(f"  - 其他天数保持原始1.5倍数据")
print("\n文件可以直接用于强化学习环境测试！")
print("=" * 80)
