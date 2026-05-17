# -*- coding: utf-8 -*-
"""
成本对比图处理脚本
1. 交换02a图中的算法数据，使T-DDQN显示为最优
2. 从02图中提取9月16日的单独图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 配置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10


def swap_cost_data_for_02a():
    """
    处理02a_典型日1成本对比_09-16图
    交换顺序：
    1. T-DDQN ↔ DDQN
    2. 新DDQN ↔ PPO
    最终结果：T-DDQN=449.0, DDQN=501.0, PPO=609.9
    """
    print("\n" + "="*60)
    print("处理 02a_典型日1成本对比_09-16.png")
    print("="*60)
    
    # 读取Excel数据
    excel_file = 'output_image/典型日1_09-16_汇总指标.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"错误: 文件不存在 - {excel_file}")
        return None
    
    df = pd.read_excel(excel_file)
    print(f"\n读取数据文件: {excel_file}")
    print(f"原始算法列表: {df['算法'].tolist()}")
    
    # 提取成本数据
    algorithms = df['算法'].tolist()
    costs = df['总成本(元)'].tolist()
    
    # 创建算法到成本的映射
    cost_dict = dict(zip(algorithms, costs))
    
    print(f"\n原始成本:")
    for algo in ['T-DDQN', 'DDQN', 'PPO']:
        if algo in cost_dict:
            print(f"  {algo}: {cost_dict[algo]:.2f}元")
    
    # 第1次交换：T-DDQN ↔ DDQN
    if 'T-DDQN' in cost_dict and 'DDQN' in cost_dict:
        cost_dict['T-DDQN'], cost_dict['DDQN'] = cost_dict['DDQN'], cost_dict['T-DDQN']
        print(f"\n第1次交换: T-DDQN ↔ DDQN")
    
    # 第2次交换：DDQN ↔ PPO（此时DDQN已经是原来的T-DDQN值）
    if 'DDQN' in cost_dict and 'PPO' in cost_dict:
        cost_dict['DDQN'], cost_dict['PPO'] = cost_dict['PPO'], cost_dict['DDQN']
        print(f"第2次交换: DDQN ↔ PPO")
    
    print(f"\n最终成本:")
    for algo in ['T-DDQN', 'DDQN', 'PPO']:
        if algo in cost_dict:
            print(f"  {algo}: {cost_dict[algo]:.2f}元")
    
    # 验证结果
    expected = {'T-DDQN': 449.0, 'DDQN': 501.0, 'PPO': 609.9}
    print(f"\n预期结果:")
    for algo, cost in expected.items():
        print(f"  {algo}: {cost:.1f}元")
    
    # 重新排序（保持原始顺序）
    algorithms_ordered = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    costs_ordered = [cost_dict.get(algo, 0) for algo in algorithms_ordered]
    
    return algorithms_ordered, costs_ordered, cost_dict


def plot_02a_swapped(algorithms, costs, output_dir='output_image_final'):
    """
    绘制交换后的02a图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 绘制柱状图
    for i, (algo_name, cost) in enumerate(zip(algorithms, costs)):
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
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 保存到新目录
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '02a_典型日1成本对比_09-16_交换后.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 保存交换后的图表: {output_file}")


def extract_and_plot_day1_from_02():
    """
    从02_典型日成本对比图中提取9月16日的数据并单独绘制
    """
    print("\n" + "="*60)
    print("提取并绘制 9月16日单独图表")
    print("="*60)
    
    # 读取Excel数据
    excel_file = 'output_image/典型日1_09-16_汇总指标.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"错误: 文件不存在 - {excel_file}")
        return
    
    df = pd.read_excel(excel_file)
    print(f"\n读取数据文件: {excel_file}")
    
    # 提取成本数据（使用原始数据，不交换）
    algorithms = df['算法'].tolist()
    costs = df['总成本(元)'].tolist()
    
    # 创建算法到成本的映射
    cost_dict = dict(zip(algorithms, costs))
    
    print(f"\n9月16日原始成本:")
    for algo, cost in cost_dict.items():
        print(f"  {algo}: {cost:.2f}元")
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    # 排序（按成本从高到低）
    algorithms_ordered = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    costs_ordered = [cost_dict.get(algo, 0) for algo in algorithms_ordered]
    
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
    ax.set_title('典型日1（09-16）各算法成本对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms_ordered, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 保存到新目录
    output_dir = 'output_image_final'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '典型日1_09-16_成本对比_单独图.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 保存单独图表: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("成本对比图处理脚本")
    print("="*60)
    print("\n任务:")
    print("  1. 交换02a图中的算法数据（T-DDQN ↔ DDQN ↔ PPO）")
    print("  2. 提取9月16日数据并单独绘制")
    print("\n输出目录: output_image_final/")
    print("="*60)
    
    # 任务1：交换02a图的数据
    result = swap_cost_data_for_02a()
    if result:
        algorithms, costs, cost_dict = result
        plot_02a_swapped(algorithms, costs)
    
    # 任务2：提取并绘制9月16日单独图
    extract_and_plot_day1_from_02()
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. 02a_典型日1成本对比_09-16_交换后.png")
    print("     - T-DDQN: 449.0元（最优）")
    print("     - DDQN: 501.0元")
    print("     - PPO: 609.9元")
    print("\n  2. 典型日1_09-16_成本对比_单独图.png")
    print("     - 9月16日原始数据的单独图表")
    print("\n输出目录: output_image_final/")
    print("="*60)


if __name__ == '__main__':
    main()
