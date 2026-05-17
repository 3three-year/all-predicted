# -*- coding: utf-8 -*-
"""
最终优化图表布局
1. 02_典型日1成本对比_09-16_提取_统一配色.png: 图例移至右上角，删除标题
2. 02a_典型日1成本对比_09-16_交换后_优化布局.png: 删除标题
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 配置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10


def plot_extracted_chart_final():
    """
    最终版本：提取图表
    - 图例在右上角
    - 无标题
    """
    print("\n" + "="*60)
    print("绘制提取图表最终版（图例右上角，无标题）")
    print("="*60)
    
    # 数据（典型日1，09-16）
    algorithms = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    costs = [550.1, 426.5, 415.8, 479.1, 471.9]
    
    # 统一的配色方案
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'Ablation': '#C73E1D',
        'PPO': '#F18F01'
    }
    
    print(f"\n数据:")
    for algo, cost in zip(algorithms, costs):
        print(f"  {algo}: {cost}元")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制柱状图
    x = np.arange(len(algorithms))
    width = 0.6
    
    for i, (algo_name, cost) in enumerate(zip(algorithms, costs)):
        color = colors.get(algo_name, '#000000')
        ax.bar(i, cost, width, 
               color=color, alpha=0.85,
               edgecolor='black', linewidth=0.8)
        
        # 在柱子上添加数值标签
        ax.text(i, cost + 8, f'{cost:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置坐标轴（无标题）
    ax.set_xlabel('算法', fontsize=13, fontweight='bold')
    ax.set_ylabel('运行成本 (元)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    
    # 添加图例（右上角）
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[algo], alpha=0.85, 
                                    edgecolor='black', linewidth=0.8) 
                      for algo in algorithms]
    ax.legend(legend_elements, algorithms, 
             loc='upper right', fontsize=11, frameon=True, shadow=True,
             framealpha=0.95)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # 设置Y轴范围
    ax.set_ylim([0, max(costs) * 1.15])
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'output_image_final'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '02_典型日1成本对比_09-16_最终版.png')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 保存最终版图表: {output_file}")


def plot_swapped_chart_final():
    """
    最终版本：交换后图表
    - 图例在左上角（避免遮挡PPO）
    - 无标题
    """
    print("\n" + "="*60)
    print("绘制交换后图表最终版（无标题）")
    print("="*60)
    
    # 交换后的数据
    algorithms = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    costs = [557.5, 501.0, 449.0, 491.6, 609.9]
    
    # 统一的配色方案
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'Ablation': '#C73E1D',
        'PPO': '#F18F01'
    }
    
    print(f"\n交换后的数据:")
    for algo, cost in zip(algorithms, costs):
        print(f"  {algo}: {cost}元")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制柱状图
    x = np.arange(len(algorithms))
    width = 0.6
    
    for i, (algo_name, cost) in enumerate(zip(algorithms, costs)):
        color = colors.get(algo_name, '#000000')
        ax.bar(i, cost, width, 
               color=color, alpha=0.85,
               edgecolor='black', linewidth=0.8)
        
        # 在柱子上添加数值标签
        ax.text(i, cost + 10, f'{cost:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置坐标轴（无标题）
    ax.set_xlabel('算法', fontsize=13, fontweight='bold')
    ax.set_ylabel('运行成本 (元)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    
    # 添加图例（左上角，避免遮挡右侧的PPO）
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors[algo], alpha=0.85, 
                                    edgecolor='black', linewidth=0.8) 
                      for algo in algorithms]
    ax.legend(legend_elements, algorithms, 
             loc='upper left', fontsize=11, frameon=True, shadow=True,
             framealpha=0.95)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    ax.set_axisbelow(True)
    
    # 设置Y轴范围
    ax.set_ylim([0, max(costs) * 1.15])
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'output_image_final'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '02a_典型日1成本对比_09-16_最终版.png')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 保存最终版图表: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("最终优化图表布局")
    print("="*60)
    print("\n任务:")
    print("  1. 提取图表: 图例移至右上角，删除标题")
    print("  2. 交换后图表: 删除标题")
    print("\n输出目录: output_image_final/")
    print("="*60)
    
    # 任务1：提取图表最终版
    plot_extracted_chart_final()
    
    # 任务2：交换后图表最终版
    plot_swapped_chart_final()
    
    print("\n" + "="*60)
    print("✅ 所有任务完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. 02_典型日1成本对比_09-16_最终版.png")
    print("     - 原始数据（T-DDQN: 415.8元）")
    print("     - 图例在右上角")
    print("     - 无标题")
    print("\n  2. 02a_典型日1成本对比_09-16_最终版.png")
    print("     - 交换后数据（T-DDQN: 449.0元，最优）")
    print("     - 图例在左上角（避免遮挡PPO）")
    print("     - 无标题")
    print("\n配色方案:")
    print("  - Baseline: 灰色 (#6C757D)")
    print("  - DDQN: 蓝色 (#2E86AB)")
    print("  - T-DDQN: 紫红色 (#A23B72)")
    print("  - Ablation: 橙红色 (#C73E1D)")
    print("  - PPO: 黄色 (#F18F01)")
    print("="*60)


if __name__ == '__main__':
    main()
