# -*- coding: utf-8 -*-
"""
居民台区2充电负荷数据平滑处理
使用混合策略：移动平均 + 高斯噪声 + 总量守恒
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# 导入字体配置
try:
    import font_config
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def smooth_ev_data(input_file, output_file, smooth_window=5, noise_ratio=0.06, gaussian_sigma=1.0):
    """
    平滑EV负荷数据
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        smooth_window: 移动平均窗口大小
        noise_ratio: 高斯噪声比例（相对于原值的标准差）
        gaussian_sigma: 高斯滤波的sigma参数
    """
    print("="*60)
    print("居民台区2充电负荷数据平滑处理")
    print("="*60)
    
    # 读取原始数据
    print(f"\n读取数据: {input_file}")
    ev_data = pd.read_csv(input_file, encoding='utf-8')
    date_col = ev_data.columns[0]
    p_columns = [f'P{i}' for i in range(1, 97)]
    ev_values = ev_data[p_columns].values.astype(np.float32)
    
    print(f"原始数据形状: {ev_values.shape}")
    
    # 统计原始数据
    all_values = ev_values.flatten()
    non_zero_values = all_values[all_values > 0.1]
    original_mean = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0
    
    print(f"\n原始数据统计:")
    print(f"  非零均值: {original_mean:.4f} kW")
    print(f"  最大值: {np.max(all_values):.4f} kW")
    print(f"  最小值: {np.min(all_values):.4f} kW")
    
    # 创建平滑后的数据
    smoothed_values = np.zeros_like(ev_values)
    
    print(f"\n开始平滑处理...")
    print(f"  移动平均窗口: {smooth_window}")
    print(f"  高斯噪声比例: {noise_ratio*100:.1f}%")
    print(f"  高斯滤波sigma: {gaussian_sigma}")
    
    for i in range(len(ev_values)):
        day_data = ev_values[i].copy()
        original_total = np.sum(day_data) * 0.25  # kWh
        
        # 步骤1: 移动平均平滑
        smoothed_day = np.zeros_like(day_data)
        half_window = smooth_window // 2
        
        for j in range(96):
            start = max(0, j - half_window)
            end = min(96, j + half_window + 1)
            smoothed_day[j] = np.mean(day_data[start:end])
        
        # 步骤2: 高斯滤波（进一步平滑）
        smoothed_day = gaussian_filter1d(smoothed_day, sigma=gaussian_sigma)
        
        # 步骤3: 添加高斯噪声（仅对非零值）
        for j in range(96):
            if smoothed_day[j] > 0.1:
                noise_std = smoothed_day[j] * noise_ratio
                noise = np.random.normal(0, noise_std)
                smoothed_day[j] = max(0.0, smoothed_day[j] + noise)
        
        # 步骤4: 总量守恒
        smoothed_total = np.sum(smoothed_day) * 0.25  # kWh
        if smoothed_total > 1e-6:
            adjustment_factor = original_total / smoothed_total
            smoothed_day = smoothed_day * adjustment_factor
        
        # 确保非负
        smoothed_day = np.maximum(smoothed_day, 0.0)
        
        smoothed_values[i] = smoothed_day
    
    # 统计平滑后的数据
    smoothed_all = smoothed_values.flatten()
    smoothed_nonzero = smoothed_all[smoothed_all > 0.1]
    smoothed_mean = np.mean(smoothed_nonzero) if len(smoothed_nonzero) > 0 else 0
    
    print("\n" + "="*60)
    print("平滑后统计特征")
    print("="*60)
    print(f"非零均值: {smoothed_mean:.4f} kW")
    print(f"最大值: {np.max(smoothed_all):.4f} kW")
    print(f"最小值: {np.min(smoothed_all):.4f} kW")
    
    # 检查唯一值数量（评估平滑效果）
    unique_before = len(np.unique(np.round(ev_values.flatten(), 2)))
    unique_after = len(np.unique(np.round(smoothed_values.flatten(), 2)))
    print(f"\n唯一值数量:")
    print(f"  平滑前: {unique_before:,}")
    print(f"  平滑后: {unique_after:,}")
    print(f"  增加: {unique_after - unique_before:,} ({((unique_after - unique_before) / unique_before * 100):.1f}%)")
    
    # 保存平滑后的数据
    print(f"\n保存平滑后的数据: {output_file}")
    smoothed_data = ev_data.copy()
    smoothed_data[p_columns] = smoothed_values
    smoothed_data.to_csv(output_file, index=False, encoding='utf-8')
    
    print("="*60)
    print("数据平滑完成！")
    print("="*60)
    
    return ev_values, smoothed_values

def visualize_comparison(original_file, smoothed_file, output_dir, target_dates=['2022-09-16', '2022-09-25']):
    """对比原始数据和平滑后数据"""
    print("\n" + "="*60)
    print("生成对比可视化")
    print("="*60)
    
    # 读取数据
    orig_data = pd.read_csv(original_file, encoding='utf-8')
    smooth_data = pd.read_csv(smoothed_file, encoding='utf-8')
    
    date_col = orig_data.columns[0]
    orig_data['date'] = pd.to_datetime(orig_data[date_col])
    smooth_data['date'] = pd.to_datetime(smooth_data[date_col])
    
    p_columns = [f'P{i}' for i in range(1, 97)]
    hours = np.arange(0, 24, 0.25)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 典型日对比（原始 vs 平滑）
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('居民台区2 - EV充电负荷平滑前后对比', fontsize=16, fontweight='bold')
    
    for idx, target_date in enumerate(target_dates):
        ax = axes[idx]
        
        orig_day = orig_data[orig_data['date'] == target_date]
        smooth_day = smooth_data[smooth_data['date'] == target_date]
        
        if len(orig_day) > 0 and len(smooth_day) > 0:
            orig_values = orig_day[p_columns].values[0]
            smooth_values = smooth_day[p_columns].values[0]
            
            orig_total = np.sum(orig_values) * 0.25
            smooth_total = np.sum(smooth_values) * 0.25
            
            ax.plot(hours, orig_values, label='原始数据', linewidth=2, alpha=0.7, color='steelblue')
            ax.plot(hours, smooth_values, label='平滑后数据', linewidth=2, alpha=0.8, color='coral', linestyle='-')
            
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('功率 (kW)', fontsize=12)
            ax.set_title(f'{target_date} - 平滑前后对比', fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
            ax.set_ylim(bottom=0)
            
            # 添加统计信息
            textstr = f'原始: 总量={orig_total:.2f} kWh\n'
            textstr += f'平滑: 总量={smooth_total:.2f} kWh\n'
            textstr += f'差异: {abs(orig_total - smooth_total):.2f} kWh ({abs(orig_total - smooth_total)/orig_total*100:.2f}%)'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'taiqu2_ev_smoothing_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()
    
    # 图2: 平滑后的典型日曲线（单独展示）
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('居民台区2 - 平滑后EV充电负荷典型日', fontsize=16, fontweight='bold')
    
    for idx, target_date in enumerate(target_dates):
        ax = axes[idx]
        smooth_day = smooth_data[smooth_data['date'] == target_date]
        
        if len(smooth_day) > 0:
            smooth_values = smooth_day[p_columns].values[0]
            smooth_total = np.sum(smooth_values) * 0.25
            smooth_mean = np.mean(smooth_values)
            smooth_max = np.max(smooth_values)
            
            ax.plot(hours, smooth_values, linewidth=2.5, alpha=0.9, color='coral', marker='o', markersize=2)
            ax.fill_between(hours, 0, smooth_values, alpha=0.3, color='coral')
            
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('功率 (kW)', fontsize=12)
            ax.set_title(f'{target_date} - 平滑后EV充电负荷曲线', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 24)
            ax.set_ylim(bottom=0)
            
            # 添加统计信息
            textstr = f'总充电量: {smooth_total:.2f} kWh\n'
            textstr += f'平均功率: {smooth_mean:.2f} kW\n'
            textstr += f'最大功率: {smooth_max:.2f} kW'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'taiqu2_ev_smoothed_typical_days.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()
    
    # 图3: 两个典型日对比（平滑后）
    fig, ax = plt.subplots(figsize=(16, 6))
    
    for target_date in target_dates:
        smooth_day = smooth_data[smooth_data['date'] == target_date]
        if len(smooth_day) > 0:
            smooth_values = smooth_day[p_columns].values[0]
            ax.plot(hours, smooth_values, label=target_date, linewidth=2.5, alpha=0.8, marker='o', markersize=2)
    
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('功率 (kW)', fontsize=12)
    ax.set_title('居民台区2 - 平滑后典型日对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'taiqu2_ev_smoothed_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 已保存: {output_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("可视化完成！")
    print(f"所有图表已保存到: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    # 设置参数
    input_file = 'date_file/按日累计-居民台区2-充电负荷数据.csv'
    output_file = 'date_file/按日累计-居民台区2-充电负荷数据_平滑.csv'
    output_dir = 'taiqu2_ev_smoothed'
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 执行平滑处理
    original_values, smoothed_values = smooth_ev_data(
        input_file=input_file,
        output_file=output_file,
        smooth_window=5,      # 移动平均窗口
        noise_ratio=0.06,     # 6%的高斯噪声
        gaussian_sigma=1.0    # 高斯滤波参数
    )
    
    # 生成对比可视化
    visualize_comparison(input_file, output_file, output_dir)
    
    print("\n" + "="*60)
    print("全部处理完成！")
    print(f"平滑后的数据文件: {output_file}")
    print(f"可视化图表目录: {output_dir}")
    print("="*60)
