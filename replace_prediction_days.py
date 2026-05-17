"""
将预测数据替换到原始数据文件中
用9月16日和9月25日的预测值替换原始数据中对应日期的值
生成新的数据文件用于测试
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("将预测数据替换到原始数据文件中")
print("=" * 80)

# 创建输出目录
output_dir = 'date_file_with_prediction'
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

# 目标日期
target_dates = ['2022-09-16', '2022-09-25']
print(f"目标日期: {target_dates}")

# ==================== 步骤1: 加载预测数据 ====================
print("\n【步骤1】加载预测数据...")

# 1.1 光伏预测数据
pv_16 = pd.read_excel('DATE_FILE_predict/光伏出力预测_2022年09月16日.xlsx')
pv_25 = pd.read_excel('DATE_FILE_predict/光伏出力预测_2022年09月25日.xlsx')

pv_predictions = {
    '2022-09-16': pv_16['预测值'].values,
    '2022-09-25': pv_25['预测值'].values
}
print(f"  光伏预测数据:")
print(f"    2022-09-16: {len(pv_predictions['2022-09-16'])} 个时间点, 平均 {pv_predictions['2022-09-16'].mean():.2f} kW")
print(f"    2022-09-25: {len(pv_predictions['2022-09-25'])} 个时间点, 平均 {pv_predictions['2022-09-25'].mean():.2f} kW")

# 1.2 基础负荷预测数据
base_load_df = pd.read_excel('DATE_FILE_predict/base_load_prediction_data.xlsx')
base_load_predictions = {
    '2022-09-16': base_load_df[base_load_df['Date'] == '2022-09-16']['BaseLoad_Predicted'].values,
    '2022-09-25': base_load_df[base_load_df['Date'] == '2022-09-25']['BaseLoad_Predicted'].values
}
print(f"  基础负荷预测数据:")
print(f"    2022-09-16: {len(base_load_predictions['2022-09-16'])} 个时间点, 平均 {base_load_predictions['2022-09-16'].mean():.2f} kW")
print(f"    2022-09-25: {len(base_load_predictions['2022-09-25'])} 个时间点, 平均 {base_load_predictions['2022-09-25'].mean():.2f} kW")

# 1.3 EV充电负荷预测数据
ev_df = pd.read_excel('DATE_FILE_predict/ev_charging_prediction_data.xlsx')
ev_predictions = {
    '2022-09-16': ev_df[ev_df['Date'] == '2022-09-16']['EV_Predicted'].values,
    '2022-09-25': ev_df[ev_df['Date'] == '2022-09-25']['EV_Predicted'].values
}
print(f"  EV充电负荷预测数据:")
print(f"    2022-09-16: {len(ev_predictions['2022-09-16'])} 个时间点, 平均 {ev_predictions['2022-09-16'].mean():.2f} kW")
print(f"    2022-09-25: {len(ev_predictions['2022-09-25'])} 个时间点, 平均 {ev_predictions['2022-09-25'].mean():.2f} kW")

# ==================== 步骤2: 处理光伏数据 ====================
print("\n【步骤2】处理光伏数据...")

pv_file = 'date_file/按日累计-台区R1-总光伏出力_扩大2.0倍.csv'
print(f"  读取原始文件: {pv_file}")

try:
    pv_df = pd.read_csv(pv_file, encoding='utf-8')
except:
    pv_df = pd.read_csv(pv_file, encoding='gbk')

print(f"  原始数据形状: {pv_df.shape}")
print(f"  列名: {pv_df.columns.tolist()[:5]}...")

# 确保日期列是datetime格式
date_col = pv_df.columns[0]  # 第一列是日期
pv_df[date_col] = pd.to_datetime(pv_df[date_col])

# 替换预测日期的数据
replaced_count = 0
for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    mask = pv_df[date_col].dt.date == target_dt.date()
    
    if mask.sum() > 0:
        # 找到对应行
        idx = pv_df[mask].index[0]
        # 替换P1-P96列的值
        for i, val in enumerate(pv_predictions[target_date], 1):
            col_name = f'P{i}'
            if col_name in pv_df.columns:
                old_val = pv_df.loc[idx, col_name]
                pv_df.loc[idx, col_name] = val
        replaced_count += 1
        print(f"  ✓ 替换 {target_date}: 行索引 {idx}")
    else:
        print(f"  ✗ 未找到 {target_date}")

# 保存新文件
output_pv_file = os.path.join(output_dir, '按日累计-台区R1-总光伏出力_扩大2.0倍.csv')
pv_df.to_csv(output_pv_file, index=False, encoding='utf-8-sig')
print(f"  保存到: {output_pv_file}")
print(f"  替换了 {replaced_count} 天的数据")

# ==================== 步骤3: 处理基础负荷数据 ====================
print("\n【步骤3】处理基础负荷数据...")

base_load_file = 'date_file/按日累计-台区R1-基础负荷.csv'
print(f"  读取原始文件: {base_load_file}")

try:
    base_load_df = pd.read_csv(base_load_file, encoding='utf-8')
except:
    base_load_df = pd.read_csv(base_load_file, encoding='gbk')

print(f"  原始数据形状: {base_load_df.shape}")
print(f"  列名: {base_load_df.columns.tolist()[:5]}...")

# 确保日期列是datetime格式
date_col = base_load_df.columns[0]
base_load_df[date_col] = pd.to_datetime(base_load_df[date_col])

# 替换预测日期的数据
replaced_count = 0
for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    mask = base_load_df[date_col].dt.date == target_dt.date()
    
    if mask.sum() > 0:
        idx = base_load_df[mask].index[0]
        for i, val in enumerate(base_load_predictions[target_date], 1):
            col_name = f'P{i}'
            if col_name in base_load_df.columns:
                old_val = base_load_df.loc[idx, col_name]
                base_load_df.loc[idx, col_name] = val
        replaced_count += 1
        print(f"  ✓ 替换 {target_date}: 行索引 {idx}")
    else:
        print(f"  ✗ 未找到 {target_date}")

# 保存新文件
output_base_load_file = os.path.join(output_dir, '按日累计-台区R1-基础负荷.csv')
base_load_df.to_csv(output_base_load_file, index=False, encoding='utf-8-sig')
print(f"  保存到: {output_base_load_file}")
print(f"  替换了 {replaced_count} 天的数据")

# ==================== 步骤4: 处理EV充电负荷数据 ====================
print("\n【步骤4】处理EV充电负荷数据...")

ev_file = 'date_file/按日累计-居民台区2-充电负荷数据.csv'
print(f"  读取原始文件: {ev_file}")

try:
    ev_df_orig = pd.read_csv(ev_file, encoding='utf-8')
except:
    ev_df_orig = pd.read_csv(ev_file, encoding='gbk')

print(f"  原始数据形状: {ev_df_orig.shape}")
print(f"  列名: {ev_df_orig.columns.tolist()[:5]}...")

# 确保日期列是datetime格式
date_col = ev_df_orig.columns[0]
ev_df_orig[date_col] = pd.to_datetime(ev_df_orig[date_col])

# 替换预测日期的数据
replaced_count = 0
for target_date in target_dates:
    target_dt = pd.to_datetime(target_date)
    mask = ev_df_orig[date_col].dt.date == target_dt.date()
    
    if mask.sum() > 0:
        idx = ev_df_orig[mask].index[0]
        for i, val in enumerate(ev_predictions[target_date], 1):
            col_name = f'P{i}'
            if col_name in ev_df_orig.columns:
                old_val = ev_df_orig.loc[idx, col_name]
                ev_df_orig.loc[idx, col_name] = val
        replaced_count += 1
        print(f"  ✓ 替换 {target_date}: 行索引 {idx}")
    else:
        print(f"  ✗ 未找到 {target_date}")

# 保存新文件
output_ev_file = os.path.join(output_dir, '按日累计-居民台区2-充电负荷数据.csv')
ev_df_orig.to_csv(output_ev_file, index=False, encoding='utf-8-sig')
print(f"  保存到: {output_ev_file}")
print(f"  替换了 {replaced_count} 天的数据")

# ==================== 步骤5: 验证替换结果 ====================
print("\n【步骤5】验证替换结果...")

print("\n对比验证（以2022-09-16为例）:")

# 验证光伏数据
pv_df_new = pd.read_csv(output_pv_file, encoding='utf-8-sig')
pv_df_new[pv_df_new.columns[0]] = pd.to_datetime(pv_df_new[pv_df_new.columns[0]])
mask = pv_df_new[pv_df_new.columns[0]].dt.date == pd.to_datetime('2022-09-16').date()
if mask.sum() > 0:
    row = pv_df_new[mask].iloc[0]
    p_cols = [f'P{i}' for i in range(1, 97)]
    new_values = row[p_cols].values
    print(f"  光伏数据:")
    print(f"    预测值平均: {pv_predictions['2022-09-16'].mean():.2f} kW")
    print(f"    新文件平均: {new_values.mean():.2f} kW")
    print(f"    是否一致: {np.allclose(new_values, pv_predictions['2022-09-16'], rtol=1e-5)}")

# 验证基础负荷数据
base_load_df_new = pd.read_csv(output_base_load_file, encoding='utf-8-sig')
base_load_df_new[base_load_df_new.columns[0]] = pd.to_datetime(base_load_df_new[base_load_df_new.columns[0]])
mask = base_load_df_new[base_load_df_new.columns[0]].dt.date == pd.to_datetime('2022-09-16').date()
if mask.sum() > 0:
    row = base_load_df_new[mask].iloc[0]
    p_cols = [f'P{i}' for i in range(1, 97)]
    new_values = row[p_cols].values
    print(f"  基础负荷数据:")
    print(f"    预测值平均: {base_load_predictions['2022-09-16'].mean():.2f} kW")
    print(f"    新文件平均: {new_values.mean():.2f} kW")
    print(f"    是否一致: {np.allclose(new_values, base_load_predictions['2022-09-16'], rtol=1e-5)}")

# 验证EV数据
ev_df_new = pd.read_csv(output_ev_file, encoding='utf-8-sig')
ev_df_new[ev_df_new.columns[0]] = pd.to_datetime(ev_df_new[ev_df_new.columns[0]])
mask = ev_df_new[ev_df_new.columns[0]].dt.date == pd.to_datetime('2022-09-16').date()
if mask.sum() > 0:
    row = ev_df_new[mask].iloc[0]
    p_cols = [f'P{i}' for i in range(1, 97)]
    new_values = row[p_cols].values
    print(f"  EV充电负荷数据:")
    print(f"    预测值平均: {ev_predictions['2022-09-16'].mean():.2f} kW")
    print(f"    新文件平均: {new_values.mean():.2f} kW")
    print(f"    是否一致: {np.allclose(new_values, ev_predictions['2022-09-16'], rtol=1e-5)}")

# ==================== 步骤6: 生成使用说明 ====================
print("\n【步骤6】生成使用说明...")

instructions = f"""
================================================================================
使用新数据文件进行测试 - 说明文档
================================================================================

已成功将预测数据替换到原始数据文件中！

替换的日期: {', '.join(target_dates)}

生成的新文件位于: {output_dir}/
├── 按日累计-台区R1-总光伏出力_扩大2.0倍.csv
├── 按日累计-台区R1-基础负荷.csv
└── 按日累计-居民台区2-充电负荷数据.csv

这些文件包含：
- 365天的完整数据
- 其中2022-09-16和2022-09-25使用预测值
- 其他363天保持原始数据不变

================================================================================
使用方法
================================================================================

方法1: 临时替换（推荐）
--------------------------------------
1. 备份原始date_file目录:
   xcopy date_file date_file_backup /E /I /H

2. 复制新文件到date_file目录:
   copy {output_dir}\\*.csv date_file\\

3. 运行测试（只测试这2天）:
   python train_simplified.py --mode test --num_days 2

4. 恢复原始文件:
   xcopy date_file_backup date_file /E /I /H /Y
   rmdir /s /q date_file_backup

方法2: 修改环境配置指向新目录
--------------------------------------
1. 打开 rural_env.py

2. 找到数据加载部分（约第203行）:
   data_dir = os.path.join(project_root, 'date_file')

3. 临时修改为:
   data_dir = os.path.join(project_root, '{output_dir}')

4. 运行测试:
   python train_simplified.py --mode test --num_days 2

5. 测试完成后恢复修改

================================================================================
测试命令
================================================================================

# 完整测试（测试集中的所有天数，包括被替换的2天）
python train_simplified.py --mode test

# 只测试2天（需要确保测试集包含这2天）
python train_simplified.py --mode test --num_days 2

注意: 由于测试集是从2022-09-13开始的，9月16日和9月25日都在测试集中！

================================================================================
验证结果
================================================================================

测试完成后，检查results/目录中的输出：
- 负荷曲线图
- 成本对比
- 光伏消纳率
- 其他性能指标

对比9月16日和9月25日的结果，看模型在预测数据上的表现。

================================================================================
重要提示
================================================================================

✓ 新文件保持了原始数据的格式和结构
✓ 只有2天的数据被替换，其他天数不受影响
✓ 可以安全地用于测试，不会破坏原始数据
✓ 测试完成后记得恢复原始配置

================================================================================
"""

instructions_file = os.path.join(output_dir, 'README.txt')
with open(instructions_file, 'w', encoding='utf-8') as f:
    f.write(instructions)

print(f"  使用说明已保存: {instructions_file}")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"\n新数据文件已生成在: {output_dir}/")
print(f"\n包含文件:")
print(f"  - 按日累计-台区R1-总光伏出力_扩大2.0倍.csv")
print(f"  - 按日累计-台区R1-基础负荷.csv")
print(f"  - 按日累计-居民台区2-充电负荷数据.csv")
print(f"  - README.txt (使用说明)")
print(f"\n已将 {', '.join(target_dates)} 的数据替换为预测值")
print(f"\n下一步: 查看 {instructions_file} 了解如何使用这些文件进行测试")
print("=" * 80)
