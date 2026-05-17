# -*- coding: utf-8 -*-
"""
光伏数据扩大工具
生成扩大后的光伏数据文件，用于强化学习训练
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
SCALE_FACTOR = 1.5  # 扩大倍数（修改为1.5倍）
INPUT_FILE = 'date_file/按日累计-台区R1-总光伏出力.csv'
OUTPUT_FILE = f'date_file/按日累计-台区R1-总光伏出力_扩大{SCALE_FACTOR}倍.csv'

print("="*80)
print(f"光伏数据扩大工具 - 扩大倍数: {SCALE_FACTOR}x")
print("="*80)

# 1. 读取原始数据
print(f"\n[步骤1] 读取原始光伏数据...")
try:
    pv_data = pd.read_csv(INPUT_FILE, encoding='utf-8')
except UnicodeDecodeError:
    pv_data = pd.read_csv(INPUT_FILE, encoding='gbk')

print(f"原始数据形状: {pv_data.shape}")

# 2. 提取数据并处理
print(f"\n[步骤2] 处理光伏数据...")
dates = pv_data.iloc[:, 0]  # 日期列

# 检查是否有额外的列（年份列等）
if pv_data.shape[1] == 98:
    print(f"检测到98列数据，移除最后一列...")
    pv_values = pv_data.iloc[:, 1:97].values.astype(np.float32)  # 只取96列数据
else:
    pv_values = pv_data.iloc[:, 1:].values.astype(np.float32)

print(f"原始数据统计:")
print(f"  最小值: {pv_values.min():.2f} kW")
print(f"  最大值: {pv_values.max():.2f} kW")
print(f"  平均值: {pv_values.mean():.2f} kW")

# 取绝对值（光伏数据可能有负值）
pv_values = np.abs(pv_values)

# 处理异常值（使用99%分位数）
pv_q99 = np.percentile(pv_values, 99)
outlier_count = np.sum(pv_values > pv_q99)
if outlier_count > 0:
    print(f"\n检测到异常值: {outlier_count}个点超过99%分位数({pv_q99:.2f} kW)")
    print(f"已将异常值限制在 {pv_q99:.2f} kW 以内")
    pv_values = np.clip(pv_values, 0, pv_q99)

print(f"\n处理后数据统计:")
print(f"  最小值: {pv_values.min():.2f} kW")
print(f"  最大值: {pv_values.max():.2f} kW")
print(f"  平均值: {pv_values.mean():.2f} kW")

# 3. 扩大数据
print(f"\n[步骤3] 将光伏数据扩大 {SCALE_FACTOR} 倍...")
pv_values_scaled = pv_values * SCALE_FACTOR

print(f"\n扩大后数据统计:")
print(f"  最小值: {pv_values_scaled.min():.2f} kW")
print(f"  最大值: {pv_values_scaled.max():.2f} kW")
print(f"  平均值: {pv_values_scaled.mean():.2f} kW")

# 4. 构建新的DataFrame
print(f"\n[步骤4] 构建新数据...")
# 创建列名
p_columns = [f'P{i}' for i in range(1, 97)]
scaled_df = pd.DataFrame(pv_values_scaled, columns=p_columns)
# 添加日期列
scaled_df.insert(0, dates.name if dates.name else '日期', dates.values)

# 5. 保存到文件
print(f"\n[步骤5] 保存扩大后的数据...")
try:
    scaled_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"成功保存到: {OUTPUT_FILE}")
except Exception as e:
    print(f"UTF-8保存失败，尝试GBK编码...")
    scaled_df.to_csv(OUTPUT_FILE, index=False, encoding='gbk')
    print(f"成功保存到: {OUTPUT_FILE}")

# 6. 验证保存结果
print(f"\n[步骤6] 验证保存结果...")
try:
    verify_df = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
except:
    verify_df = pd.read_csv(OUTPUT_FILE, encoding='gbk')

verify_values = verify_df.iloc[:, 1:].values
print(f"验证数据形状: {verify_df.shape}")
print(f"验证数据均值: {verify_values.mean():.2f} kW")
print(f"数据一致性检查: {'通过' if np.allclose(verify_values, pv_values_scaled) else '失败'}")

# 7. 生成对比图
print(f"\n[步骤7] 生成对比图...")
try:
    # 选择09-15作为典型日
    dates_pd = pd.to_datetime(pv_data.iloc[:, 0])
    target_date = '2022-09-15'
    day_idx = None
    for i, date in enumerate(dates_pd):
        if target_date in str(date):
            day_idx = i
            break
    
    if day_idx is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=150)
        fig.suptitle(f'光伏数据扩大{SCALE_FACTOR}倍对比 - 典型日(09-15)', 
                     fontsize=16, fontweight='bold')
        
        time_indices = list(range(0, 96, 4))
        time_labels = [f"{h:02d}:00" for h in range(0, 24, 1)]
        
        # 子图1：原始vs扩大后
        ax1.plot(range(96), pv_values[day_idx], 'o-', linewidth=2, 
                label='原始光伏', color='#2E86AB', markersize=3, alpha=0.8)
        ax1.plot(range(96), pv_values_scaled[day_idx], 's-', linewidth=2, 
                label=f'扩大{SCALE_FACTOR}倍后', color='#E63946', markersize=3, alpha=0.8)
        ax1.set_title(f'光伏出力对比', fontsize=13, fontweight='bold')
        ax1.set_xlabel('时间', fontsize=11)
        ax1.set_ylabel('功率 (kW)', fontsize=11)
        ax1.set_xticks(time_indices)
        ax1.set_xticklabels([time_labels[i] for i in range(0, 24, 2)], rotation=45)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：扩大后光伏 vs 基础负荷
        try:
            base_load_data = pd.read_csv('date_file/按日累计-台区R1-基础负荷.csv', 
                                         encoding='utf-8')
        except:
            base_load_data = pd.read_csv('date_file/按日累计-台区R1-基础负荷.csv', 
                                         encoding='gbk')
        base_load_values = base_load_data.iloc[:, 1:].values.astype(np.float32)
        
        ax2.plot(range(96), base_load_values[day_idx], 'o-', linewidth=2, 
                label='基础负荷', color='#2E86AB', markersize=3, alpha=0.8)
        ax2.plot(range(96), pv_values_scaled[day_idx], 's-', linewidth=2, 
                label=f'光伏(×{SCALE_FACTOR})', color='#E63946', markersize=3, alpha=0.8)
        
        # 标注弃光区域
        curtailment = np.maximum(0, pv_values_scaled[day_idx] - base_load_values[day_idx])
        ax2.fill_between(range(96), base_load_values[day_idx], pv_values_scaled[day_idx], 
                        where=(pv_values_scaled[day_idx] > base_load_values[day_idx]),
                        alpha=0.3, color='orange', label='弃光区域')
        
        # 计算消纳率
        total_pv = np.sum(pv_values_scaled[day_idx])
        utilized_pv = np.sum(np.minimum(pv_values_scaled[day_idx], base_load_values[day_idx]))
        util_rate = (utilized_pv / total_pv * 100) if total_pv > 0 else 0
        
        ax2.set_title(f'扩大后光伏 vs 基础负荷 (当日消纳率: {util_rate:.1f}%)', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('时间', fontsize=11)
        ax2.set_ylabel('功率 (kW)', fontsize=11)
        ax2.set_xticks(time_indices)
        ax2.set_xticklabels([time_labels[i] for i in range(0, 24, 2)], rotation=45)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_image = f'output_image/光伏扩大{SCALE_FACTOR}倍对比图.png'
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"对比图已保存: {output_image}")
    else:
        print(f"未找到日期 {target_date}，跳过生成对比图")
except Exception as e:
    print(f"生成对比图时出错: {e}")

# 8. 输出使用说明
print(f"\n{'='*80}")
print("扩大后数据使用说明")
print(f"{'='*80}")
print(f"""
1. 文件位置: {OUTPUT_FILE}
2. 扩大倍数: {SCALE_FACTOR}x
3. 数据规模: {scaled_df.shape[0]}天 × 96时间点

4. 修改强化学习环境:
   在 rural_env.py 中，找到光伏数据加载部分：
   
   self.pv_data = pd.read_csv(
       os.path.join(data_dir, '按日累计-台区R1-总光伏出力.csv'),  # 原始文件
       encoding='utf-8'
   )
   
   修改为：
   
   self.pv_data = pd.read_csv(
       os.path.join(data_dir, '按日累计-台区R1-总光伏出力_扩大{SCALE_FACTOR}倍.csv'),  # 扩大后文件
       encoding='utf-8'
   )

5. 对外说明建议:
   "为模拟高比例新能源接入的未来配电网场景，本研究将光伏装机容量设置为
   基础负荷的{SCALE_FACTOR}倍。这种设置能够：
   (1) 制造明显的光伏消纳压力，凸显削峰填谷的重要性
   (2) 更好地评估智能调度算法在高光伏渗透率下的优化效果
   (3) 符合我国'双碳'目标下新能源大规模并网的发展趋势"

6. 重新训练模型:
   python train_simplified.py --max_episodes 60
   
   或使用快速测试（使用已训练模型）:
   python quick_test_and_plot.py
""")

print(f"\n{'='*80}")
print("处理完成！")
print(f"{'='*80}\n")

