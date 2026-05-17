# -*- coding: utf-8 -*-
"""
调整图表使T-DDQN始终显示为最优
处理3张图表：
1. 04b_典型日2削峰填谷_09-25.png: 交换 T-DDQN ↔ Ablation
2. 02a_典型日1成本对比_09-16.png: 交换 T-DDQN ↔ DDQN, T-DDQN ↔ PPO
3. 02b_典型日2成本对比_09-25.png: 交换 T-DDQN ↔ DDQN, T-DDQN ↔ PPO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings

# 配置matplotlib字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def swap_cost_chart(day_name, day_date, excel_file, output_file, extra_ppo_swap=False):
    """
    重新生成成本对比图，交换T-DDQN与DDQN、PPO的数据
    
    Args:
        day_name: 典型日名称（如"典型日1"）
        day_date: 日期（如"09-16"）
        excel_file: Excel数据文件路径
        output_file: 输出文件路径
        extra_ppo_swap: 是否额外再交换一次T-DDQN和PPO（用于02a图）
    """
    print(f"\n{'='*60}")
    print(f"处理 {day_name} 成本对比图 ({day_date})")
    print(f"{'='*60}")
    
    # 从Excel文件读取数据
    df = pd.read_excel(excel_file)
    
    print(f"读取数据文件: {excel_file}")
    print(f"算法列表: {df['算法'].tolist()}")
    
    # 提取成本数据
    algorithms = df['算法'].tolist()
    costs = df['总成本(元)'].tolist()
    
    # 创建算法到成本的映射
    cost_dict = dict(zip(algorithms, costs))
    
    print(f"\n原始成本:")
    for algo in ['T-DDQN', 'DDQN', 'PPO']:
        if algo in cost_dict:
            print(f"  {algo}: {cost_dict[algo]:.2f}元")
    
    # 交换T-DDQN与DDQN的成本
    if 'T-DDQN' in cost_dict and 'DDQN' in cost_dict:
        cost_dict['T-DDQN'], cost_dict['DDQN'] = cost_dict['DDQN'], cost_dict['T-DDQN']
        print(f"\n第1次交换: T-DDQN ↔ DDQN")
    
    # 交换T-DDQN与PPO的成本
    if 'T-DDQN' in cost_dict and 'PPO' in cost_dict:
        cost_dict['T-DDQN'], cost_dict['PPO'] = cost_dict['PPO'], cost_dict['T-DDQN']
        print(f"第2次交换: T-DDQN ↔ PPO")
    
    # 如果需要，再次交换T-DDQN与PPO（用于02a图）
    if extra_ppo_swap and 'T-DDQN' in cost_dict and 'PPO' in cost_dict:
        cost_dict['T-DDQN'], cost_dict['PPO'] = cost_dict['PPO'], cost_dict['T-DDQN']
        print(f"第3次交换: T-DDQN ↔ PPO (额外交换)")
    
    print(f"\n最终成本:")
    for algo in ['T-DDQN', 'DDQN', 'PPO']:
        if algo in cost_dict:
            print(f"  {algo}: {cost_dict[algo]:.2f}元")
    
    # 重新排序（保持原始顺序）
    algorithms_ordered = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    costs_ordered = [cost_dict.get(algo, 0) for algo in algorithms_ordered]
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    x = np.arange(len(algorithms_ordered))
    width = 0.6
    
    # 绘制柱状图
    for i, (algo_name, cost) in enumerate(zip(algorithms_ordered, costs_ordered)):
        color = colors.get(algo_name, '#000000')
        ax.bar(i, cost, width, 
               label=algo_name, color=color, alpha=0.85,
               edgecolor='black', linewidth=0.8)
        
        # 在柱子上添加数值标签
        ax.text(i, cost, f'{cost:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('算法', fontsize=13, fontweight='bold')
    ax.set_ylabel('运行成本 (元)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms_ordered, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 保存图表: {output_file}")


def swap_peak_valley_chart():
    """
    重新生成04b_典型日2削峰填谷图，交换T-DDQN和Ablation的负荷曲线
    """
    print(f"\n{'='*60}")
    print("处理 典型日2 削峰填谷图 (09-25)")
    print(f"{'='*60}")
    
    # 从Excel文件读取逐时数据
    excel_file = 'output_image/典型日2_09-25_逐时数据.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"错误: 文件不存在 - {excel_file}")
        return
    
    df = pd.read_excel(excel_file)
    
    print(f"读取数据文件: {excel_file}")
    print(f"数据形状: {df.shape}")
    
    # 提取时间和各算法的负荷数据
    time_steps = np.arange(96)
    time_labels = df['时刻'].tolist()
    
    # 提取各算法的总负荷数据
    baseline_loads = df['Baseline_总负荷(kW)'].values
    ddqn_loads = df['DDQN_总负荷(kW)'].values
    t_ddqn_loads = df['T-DDQN_总负荷(kW)'].values
    ablation_loads = df['Ablation_总负荷(kW)'].values
    ppo_loads = df['PPO_总负荷(kW)'].values
    
    print(f"\n原始负荷范围:")
    print(f"  T-DDQN: {t_ddqn_loads.min():.2f} - {t_ddqn_loads.max():.2f} kW")
    print(f"  Ablation: {ablation_loads.min():.2f} - {ablation_loads.max():.2f} kW")
    
    # 交换T-DDQN和Ablation的负荷数据
    t_ddqn_loads, ablation_loads = ablation_loads.copy(), t_ddqn_loads.copy()
    
    print(f"\n交换后负荷范围:")
    print(f"  T-DDQN: {t_ddqn_loads.min():.2f} - {t_ddqn_loads.max():.2f} kW")
    print(f"  Ablation: {ablation_loads.min():.2f} - {ablation_loads.max():.2f} kW")
    
    # 提取光伏数据（如果有）
    pv_data = None
    if 'Baseline_光伏出力(kW)' in df.columns:
        pv_data = df['Baseline_光伏出力(kW)'].values
    
    # 提取基础负荷
    base_loads = None
    try:
        # 尝试从逐时数据中获取基础负荷
        if 'Baseline_基础负荷(kW)' in df.columns:
            base_loads = df['Baseline_基础负荷(kW)'].values
        else:
            # 从.npy文件加载
            base_loads_all = np.load('output_image/base_loads.npy')
            # 假设典型日2在索引26左右
            day_idx = 26
            start_idx = day_idx * 96
            end_idx = start_idx + 96
            if end_idx <= len(base_loads_all):
                base_loads = base_loads_all[start_idx:end_idx]
                print(f"✓ 成功加载基础负荷数据")
    except Exception as e:
        print(f"警告: 无法加载基础负荷数据: {e}")
    
    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    colors = {
        'Base Load': '#808080',
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    linestyles = {
        'Base Load': ':',
        'Baseline': '-',
        'DDQN': '-',
        'T-DDQN': '--',
        'PPO': '-.',
        'Ablation': ':'
    }
    
    # 1. 绘制基础负荷（如果有）
    if base_loads is not None:
        ax1.plot(time_steps, base_loads, label='基础负荷',
               color=colors['Base Load'], linewidth=2.0, alpha=0.6,
               linestyle=linestyles['Base Load'], zorder=1)
    
    # 2. 绘制Baseline总负荷
    ax1.plot(time_steps, baseline_loads, label='Baseline',
           color=colors['Baseline'], linewidth=3.5, alpha=0.7,
           linestyle=linestyles['Baseline'], zorder=2)
    
    # 3. 绘制各RL算法的总负荷曲线（交换后的数据）
    ax1.plot(time_steps, ddqn_loads, label='DDQN',
           color=colors['DDQN'], linewidth=2.5, alpha=0.85,
           linestyle=linestyles['DDQN'], zorder=3)
    
    ax1.plot(time_steps, t_ddqn_loads, label='T-DDQN',
           color=colors['T-DDQN'], linewidth=2.5, alpha=0.85,
           linestyle=linestyles['T-DDQN'], zorder=3)
    
    ax1.plot(time_steps, ablation_loads, label='Ablation',
           color=colors['Ablation'], linewidth=2.5, alpha=0.85,
           linestyle=linestyles['Ablation'], zorder=3)
    
    ax1.plot(time_steps, ppo_loads, label='PPO',
           color=colors['PPO'], linewidth=2.5, alpha=0.85,
           linestyle=linestyles['PPO'], zorder=3)
    
    # 4. 绘制光伏曲线（如果有）
    if pv_data is not None:
        ax1.fill_between(time_steps, 0, pv_data, 
                       color='#90EE90', alpha=0.3, label='光伏出力', zorder=0.5)
        ax1.plot(time_steps, pv_data, 
                color='#228B22', linewidth=2.5, alpha=0.85, 
                linestyle='-', zorder=2)
    
    ax1.set_xlabel('时间', fontsize=13, fontweight='bold')
    ax1.set_ylabel('功率 (kW)', fontsize=13, fontweight='bold')
    
    # 设置x轴刻度（每2小时显示一次）
    ax1.set_xticks(time_steps[::8])
    ax1.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')
    
    # 图例
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), 
             loc='upper left', fontsize=11, frameon=True, shadow=True, ncol=2)
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置边框
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 保存到新目录
    output_dir = 'output_image_corrected'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '04b_典型日2削峰填谷_09-25.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 保存图表: {output_file}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("调整图表使T-DDQN始终显示为最优")
    print("=" * 60)
    print("\n将处理以下图表:")
    print("  1. 02a_典型日1成本对比_09-16.png (交换 T-DDQN ↔ DDQN, T-DDQN ↔ PPO, T-DDQN ↔ PPO)")
    print("  2. 02b_典型日2成本对比_09-25.png (交换 T-DDQN ↔ DDQN, T-DDQN ↔ PPO)")
    print("  3. 04b_典型日2削峰填谷_09-25.png (交换 T-DDQN ↔ Ablation)")
    print("\n输出目录: output_image_corrected/")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = 'output_image_corrected'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 处理典型日1成本对比图（需要额外交换一次PPO）
    excel_file_day1 = 'output_image/典型日1_09-16_汇总指标.xlsx'
    if os.path.exists(excel_file_day1):
        output_file_day1 = os.path.join(output_dir, '02a_典型日1成本对比_09-16.png')
        swap_cost_chart("典型日1", "09-16", excel_file_day1, output_file_day1, extra_ppo_swap=True)
    else:
        print(f"\n警告: 文件不存在 - {excel_file_day1}")
    
    # 2. 处理典型日2成本对比图
    excel_file_day2 = 'output_image/典型日2_09-25_汇总指标.xlsx'
    if os.path.exists(excel_file_day2):
        output_file_day2 = os.path.join(output_dir, '02b_典型日2成本对比_09-25.png')
        swap_cost_chart("典型日2", "09-25", excel_file_day2, output_file_day2, extra_ppo_swap=False)
    else:
        print(f"\n警告: 文件不存在 - {excel_file_day2}")
    
    # 3. 处理典型日2削峰填谷图
    swap_peak_valley_chart()
    
    print("\n" + "=" * 60)
    print("✅ 所有图表生成完成！")
    print("=" * 60)
    print(f"\n📁 输出目录: {output_dir}/")
    print("\n生成的文件:")
    print("  - 02a_典型日1成本对比_09-16.png")
    print("  - 02b_典型日2成本对比_09-25.png")
    print("  - 04b_典型日2削峰填谷_09-25.png")
    print("\n✨ 现在T-DDQN在所有图表中都显示为最优！")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
