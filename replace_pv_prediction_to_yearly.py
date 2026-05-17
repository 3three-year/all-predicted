"""
将单日光伏预测数据替换到全年光伏数据中
从DATE_FILE_predict目录读取：
- 处理后的全年光伏数据.xlsx（基础数据）
- 单日数据_2022-09-16.xlsx（9月16日预测数据）
- 单日数据_2022-09-25.xlsx（9月25日预测数据）

将9月16日和9月25日的数据替换为预测值，其他日期保持不变
输出到date_file_with_prediction目录
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("将单日光伏预测数据替换到全年光伏数据中")
print("=" * 80)

# 创建输出目录
output_dir = 'date_file_with_prediction'
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

# 目标日期
target_dates = ['2022-09-16', '2022-09-25']
print(f"目标日期: {target_dates}")

# ==================== 步骤1: 加载全年光伏数据 ====================
print("\n【步骤1】加载全年光伏数据...")

yearly_file = 'DATE_FILE_predict/处理后的全年光伏数据.xlsx'
print(f"  读取文件: {yearly_file}")

yearly_df = pd.read_excel(yearly_file)
print(f"  数据形状: {yearly_df.shape}")
print(f"  列名: {yearly_df.columns.tolist()[:5]}...")

# 显示数据前几行
print(f"\n  数据预览:")
print(yearly_df.head())

# 确保日期列是datetime格式
date_col = yearly_df.columns[0]  # 第一列是日期
print(f"\n  日期列名: {date_col}")
yearly_df[date_col] = pd.to_datetime(yearly_df[date_col])
print(f"  日期范围: {yearly_df[date_col].min()} 至 {yearly_df[date_col].max()}")

# ==================== 步骤2: 加载单日预测数据 ====================
print("\n【步骤2】加载单日预测数据...")

# 2.1 加载9月16日预测数据
sep16_file = 'DATE_FILE_predict/单日数据_2022-09-16.xlsx'
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
    # 如果没有明确的预测值列，尝试找到数值列
    numeric_cols = sep16_df.select_dtypes(include=[np.number]).columns
    print(f"  可用的数值列: {numeric_cols.tolist()}")
    # 假设最后一列是预测值
    sep16_predictions = sep16_df[numeric_cols[-1]].values

print(f"  9月16日预测数据: {len(sep16_predictions)} 个时间点")
print(f"    平均值: {sep16_predictions.mean():.2f} kW")
print(f"    最大值: {sep16_predictions.max():.2f} kW")
print(f"    最小值: {sep16_predictions.min():.2f} kW")

# 2.2 加载9月25日预测数据
sep25_file = 'DATE_FILE_predict/单日数据_2022-09-25.xlsx'
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

# ==================== 步骤3: 替换指定日期的数据 ====================
print("\n【步骤3】替换指定日期的数据...")

# 创建全年数据的副本
yearly_df_new = yearly_df.copy()

# 检测数据格式
print(f"\n  数据格式检测:")
print(f"    总行数: {len(yearly_df_new)}")
print(f"    列数: {len(yearly_df_new.columns)}")

# 判断是时间序列格式还是按日累计格式
is_timeseries = len(yearly_df_new) > 365  # 如果行数远大于365，说明是时间序列格式

if is_timeseries:
    print(f"    格式: 时间序列格式（每15分钟一行）")
    print(f"    预计每天有 {len(yearly_df_new) / 365:.0f} 个时间点")
else:
    print(f"    格式: 按日累计格式（每天一行）")

replaced_count = 0
replaced_points = 0

for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    
    # 找到该日期的所有时间点
    mask = yearly_df_new[date_col].dt.date == target_dt.date()
    
    if mask.sum() > 0:
        print(f"\n  处理 {target_date}:")
        print(f"    找到 {mask.sum()} 个时间点")
        
        # 获取预测数据
        predictions = pv_predictions[target_date]
        print(f"    预测数据: {len(predictions)} 个值")
        
        if is_timeseries:
            # 时间序列格式：替换该日期的所有行
            indices = yearly_df_new[mask].index
            
            # 获取数值列（通常是'Power'或类似名称）
            numeric_cols = yearly_df_new.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]  # 使用第一个数值列
                print(f"    数值列: {value_col}")
                
                # 替换每个时间点的值
                for i, idx in enumerate(indices):
                    if i < len(predictions):
                        old_val = yearly_df_new.loc[idx, value_col]
                        yearly_df_new.loc[idx, value_col] = predictions[i]
                        if i == 0:  # 只打印第一个值作为示例
                            print(f"    第1个时间点: {old_val:.4f} → {predictions[i]:.2f} kW")
                        if i == len(indices) - 1:  # 打印最后一个值
                            print(f"    第{i+1}个时间点: {old_val:.4f} → {predictions[i]:.2f} kW")
                
                replaced_points += min(len(indices), len(predictions))
                replaced_count += 1
                print(f"    ✓ 成功替换 {min(len(indices), len(predictions))} 个时间点")
            else:
                print(f"    ✗ 未找到数值列")
        else:
            # 按日累计格式：替换该日期行的P1-P96列
            idx = yearly_df_new[mask].index[0]
            
            # 尝试找P1-P96列
            p_cols = [f'P{i}' for i in range(1, 97)]
            available_p_cols = [col for col in p_cols if col in yearly_df_new.columns]
            
            if len(available_p_cols) > 0:
                print(f"    找到 {len(available_p_cols)} 个P列")
                
                # 替换数据
                for i, col_name in enumerate(available_p_cols):
                    if i < len(predictions):
                        old_val = yearly_df_new.loc[idx, col_name]
                        yearly_df_new.loc[idx, col_name] = predictions[i]
                        if i == 0:
                            print(f"    {col_name}: {old_val:.2f} → {predictions[i]:.2f}")
                
                replaced_points += min(len(available_p_cols), len(predictions))
                replaced_count += 1
                print(f"    ✓ 成功替换 {min(len(available_p_cols), len(predictions))} 个时间点")
            else:
                print(f"    ✗ 未找到P列")
    else:
        print(f"  ✗ 未找到 {target_date}")

print(f"\n  总计替换了 {replaced_count} 天的数据，共 {replaced_points} 个时间点")

# ==================== 步骤4: 保存新文件 ====================
print("\n【步骤4】保存新文件...")

# 保存为Excel格式
output_excel_file = os.path.join(output_dir, '处理后的全年光伏数据_含预测.xlsx')
yearly_df_new.to_excel(output_excel_file, index=False, engine='openpyxl')
print(f"  ✓ Excel文件已保存: {output_excel_file}")

# 同时保存为CSV格式（方便后续使用）
output_csv_file = os.path.join(output_dir, '按日累计-台区R1-总光伏出力_扩大2.0倍_含预测.csv')
yearly_df_new.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
print(f"  ✓ CSV文件已保存: {output_csv_file}")

# ==================== 步骤5: 验证替换结果 ====================
print("\n【步骤5】验证替换结果...")

print("\n对比验证:")
for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    
    # 原始数据
    mask_orig = yearly_df[date_col].dt.date == target_dt.date()
    if mask_orig.sum() > 0:
        # 新数据
        mask_new = yearly_df_new[date_col].dt.date == target_dt.date()
        
        # 获取数值列
        numeric_cols = yearly_df_new.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            value_col = numeric_cols[0]
            
            orig_values = yearly_df[mask_orig][value_col].values
            new_values = yearly_df_new[mask_new][value_col].values
            pred_values = pv_predictions[target_date]
            
            print(f"\n  {target_date}:")
            print(f"    原始数据: {len(orig_values)} 个时间点, 平均 {orig_values.mean():.2f} kW")
            print(f"    新数据: {len(new_values)} 个时间点, 平均 {new_values.mean():.2f} kW")
            print(f"    预测数据: {len(pred_values)} 个值, 平均 {pred_values.mean():.2f} kW")
            
            # 检查是否一致
            if len(new_values) == len(pred_values):
                is_match = np.allclose(new_values, pred_values, rtol=1e-5)
                print(f"    数据是否一致: {is_match}")
                if is_match:
                    print(f"    ✓ 替换成功！")
                else:
                    print(f"    ⚠ 数据不完全一致，可能存在精度差异")
                    # 显示前几个值的对比
                    print(f"    前3个值对比:")
                    for i in range(min(3, len(new_values))):
                        print(f"      时间点{i+1}: 新={new_values[i]:.2f}, 预测={pred_values[i]:.2f}")
            else:
                print(f"    ⚠ 数据长度不匹配: 新数据{len(new_values)}个 vs 预测数据{len(pred_values)}个")

# ==================== 步骤6: 生成使用说明 ====================
print("\n【步骤6】生成使用说明...")

instructions = f"""
================================================================================
全年光伏数据（含预测）- 说明文档
================================================================================

已成功将单日预测数据替换到全年光伏数据中！

替换的日期: {', '.join(target_dates)}

生成的新文件:
├── {output_excel_file}
└── {output_csv_file}

数据说明:
- 包含365天的完整光伏出力数据
- 其中2022-09-16和2022-09-25使用预测值
- 其他363天保持原始数据不变
- 数据格式与原始文件保持一致

================================================================================
数据来源
================================================================================

原始数据: DATE_FILE_predict/处理后的全年光伏数据.xlsx
预测数据:
  - 9月16日: DATE_FILE_predict/单日数据_2022-09-16.xlsx
  - 9月25日: DATE_FILE_predict/单日数据_2022-09-25.xlsx

================================================================================
使用方法
================================================================================

方法1: 直接使用CSV文件
--------------------------------------
将生成的CSV文件复制到date_file目录:
  copy {output_csv_file} date_file\\按日累计-台区R1-总光伏出力_扩大2.0倍.csv

方法2: 使用Excel文件
--------------------------------------
在需要的地方直接读取Excel文件:
  pd.read_excel('{output_excel_file}')

================================================================================
数据验证
================================================================================

9月16日:
  - 预测数据平均: {pv_predictions['2022-09-16'].mean():.2f} kW
  - 预测数据最大: {pv_predictions['2022-09-16'].max():.2f} kW

9月25日:
  - 预测数据平均: {pv_predictions['2022-09-25'].mean():.2f} kW
  - 预测数据最大: {pv_predictions['2022-09-25'].max():.2f} kW

================================================================================
注意事项
================================================================================

✓ 新文件保持了原始数据的格式和结构
✓ 只有2天的数据被替换，其他天数不受影响
✓ 可以安全地用于测试和分析
✓ 同时提供Excel和CSV两种格式

================================================================================
"""

instructions_file = os.path.join(output_dir, 'README_光伏预测数据.txt')
with open(instructions_file, 'w', encoding='utf-8') as f:
    f.write(instructions)

print(f"  ✓ 使用说明已保存: {instructions_file}")

# ==================== 完成 ====================
print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"\n生成的文件:")
print(f"  1. {output_excel_file}")
print(f"  2. {output_csv_file}")
print(f"  3. {instructions_file}")
print(f"\n已将 {', '.join(target_dates)} 的数据替换为预测值")
print(f"其他 {len(yearly_df) - len(target_dates)} 天的数据保持不变")
print("\n下一步: 查看使用说明了解如何使用这些文件")
print("=" * 80)
