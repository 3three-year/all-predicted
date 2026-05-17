# -*- coding: utf-8 -*-
"""
强化学习结果可视化工具 - 清晰版
只包含4个核心对比图，基于真实测试数据
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
from datetime import datetime, timedelta

# 配置matplotlib字体设置，支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 禁用字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def plot_training_rewards_comparison(algorithm_rewards, output_dir='output_image'):
    """
    图1: 所有强化学习算法的训练奖励对比曲线
    
    Args:
        algorithm_rewards: dict, {算法名: {'rewards': [], 'avg_rewards': []}}
        output_dir: 输出目录
    """
    plt.figure(figsize=(14, 8))
    
    colors = {
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D',
        'Baseline': '#6C757D'
    }
    
    for algo_name, data in algorithm_rewards.items():
        if 'avg_rewards' in data and len(data['avg_rewards']) > 0:
            episodes = np.arange(1, len(data['avg_rewards']) + 1)
            color = colors.get(algo_name, '#000000')
            
            # 绘制平滑曲线
            plt.plot(episodes, data['avg_rewards'], 
                    label=f'{algo_name}',
                    linewidth=2.5, color=color, alpha=0.9)
            
            # 如果有原始奖励，用浅色背景显示
            if 'rewards' in data and len(data['rewards']) > 0:
                plt.plot(episodes, data['rewards'], 
                        color=color, alpha=0.15, linewidth=1)
    
    plt.xlabel('训练轮次 (Episodes)', fontsize=13, fontweight='bold')
    plt.ylabel('平均奖励 (Average Reward)', fontsize=13, fontweight='bold')
    # plt.title('强化学习算法训练奖励对比', fontsize=15, fontweight='bold', pad=15)  # 删除标题
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置坐标轴
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '01_训练奖励对比曲线.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[DONE] Generated: {output_dir}/01_训练奖励对比曲线.png")


def plot_test_cost_comparison(algorithm_costs, test_dates, output_dir='output_image'):
    """
    图2: 典型日的各算法成本对比图（每个典型日生成一张独立的图）
    
    Args:
        algorithm_costs: dict, {算法名: [各典型日的成本列表]}
        test_dates: list, 典型日期列表（格式：'MM-DD'）
        output_dir: 输出目录
    """
    algorithms = list(algorithm_costs.keys())
    n_days = len(test_dates)
    
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    # 为每个典型日生成一张独立的图
    for day_num in range(n_days):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = np.arange(len(algorithms))
        width = 0.6  # 柱子宽度
        
        # 提取当天各算法的成本
        day_costs = []
        for algo_name in algorithms:
            if day_num < len(algorithm_costs[algo_name]):
                day_costs.append(algorithm_costs[algo_name][day_num])
            else:
                day_costs.append(0)
        
        # 绘制柱状图
        bars = []
        for i, (algo_name, cost) in enumerate(zip(algorithms, day_costs)):
            color = colors.get(algo_name, '#000000')
            bar = ax.bar(i, cost, width, 
                        label=algo_name, color=color, alpha=0.85,
                        edgecolor='black', linewidth=0.8)
            bars.append(bar)
            
            # 在柱子上添加数值标签
            ax.text(i, cost, f'{cost:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('算法', fontsize=13, fontweight='bold')
        ax.set_ylabel('运行成本 (元)', fontsize=13, fontweight='bold')
        # ax.set_title(f'典型日{day_num+1} - 各算法成本对比', fontsize=15, fontweight='bold', pad=15)  # 删除标题
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成独立的文件名
        date_str = test_dates[day_num] if day_num < len(test_dates) else f'day{day_num+1}'
        filename = f'02{chr(96+day_num+1)}_典型日{day_num+1}成本对比_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DONE] Generated: {output_dir}/{filename}")


def plot_test_pv_utilization_comparison(algorithm_pv_data, test_dates, day_indices, output_dir='output_image'):
    """
    图3: 典型日的各算法光伏消纳率对比图（每个典型日生成一张图，显示96个时间步）
    
    Args:
        algorithm_pv_data: dict, {算法名: [所有测试日的逐时消纳率数据]} (110天×96点)
        test_dates: list, 典型日期列表（格式：'MM-DD'）
        day_indices: list, 典型日在测试集中的索引
        output_dir: 输出目录
    """
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    markers = {
        'Baseline': 'o',
        'DDQN': 's',
        'T-DDQN': '^',
        'PPO': 'D',
        'Ablation': 'v'
    }
    
    # 为每个典型日生成一张图
    for day_num, (day_idx, date_str) in enumerate(zip(day_indices, test_dates), 1):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 时间轴：96个点，每15分钟一个（0:00-23:45）
        time_steps = np.arange(96)
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]
        
        # 绘制每个算法的消纳率曲线
        for algo_name, pv_data in algorithm_pv_data.items():
            # 提取当天96个点的数据
            start_idx = day_idx * 96
            end_idx = start_idx + 96
            
            if end_idx > len(pv_data):
                print(f"警告：{algo_name} 第{day_idx}天数据超出范围")
                continue
            
            day_pv_util = np.array(pv_data[start_idx:end_idx]) * 100  # 转换为百分比
            
            color = colors.get(algo_name, '#000000')
            marker = markers.get(algo_name, 'o')
            
            # 绘制曲线
            ax.plot(time_steps, day_pv_util, label=algo_name,
                   color=color, linewidth=2.5, alpha=0.85,
                   marker=marker, markersize=4, markevery=8,  # 每8个点标记一次
                   markeredgecolor='black', markeredgewidth=0.8)
        
        # 添加光伏有效时段的背景色（8:00-18:00，时间步32-72）
        ax.axvspan(32, 72, alpha=0.05, color='yellow', label='光伏有效时段', zorder=0)
        
        ax.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax.set_ylabel('光伏消纳率 (%)', fontsize=13, fontweight='bold')
        # ax.set_title(f'典型日{day_num} ({date_str}) - 各算法光伏消纳率对比（逐时变化）\n'
        #             f'注：当光伏<5kW时设为0%，5-10kW渐进过渡，≥10kW正常计算', 
        #             fontsize=15, fontweight='bold', pad=15)  # 删除标题
        
        # 设置x轴刻度（每2小时显示一次）
        ax.set_xticks(time_steps[::8])  # 每8个点=2小时
        ax.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')
        
        ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_ylim([0, 105])
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成独立的文件名
        filename = f'03{chr(96+day_num)}_典型日{day_num}光伏消纳率_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DONE] Generated: {output_dir}/{filename}")


def plot_test_peak_valley_comparison(algorithm_loads, baseline_loads, test_dates, output_dir='output_image', 
                                    day_indices=None, base_loads=None, pv_data=None):
    """
    图4: 典型日的各算法削峰填谷效果对比（每个典型日生成一张图，显示96个时间步的负荷曲线）
    
    Args:
        algorithm_loads: dict, {算法名: [所有测试日的负荷数据]} (110天×96点)
        baseline_loads: list, 基线负荷数据（所有测试日）
        test_dates: list, 典型日期列表（格式：'MM-DD'）
        output_dir: 输出目录
        day_indices: list, 典型日在测试集中的索引
        base_loads: list, 基础负荷数据（固定负荷，不可调度部分）
        pv_data: list, 光伏出力数据（所有测试日，110天×96点）
    """
    # 如果没有提供索引，使用顺序索引
    if day_indices is None:
        day_indices = list(range(len(test_dates)))
    
    colors = {
        'Base Load': '#808080',  # 基础负荷：灰色
        'Baseline': '#6C757D',   # Baseline：深灰色
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    linestyles = {
        'Base Load': ':',      # 基础负荷：虚线
        'Baseline': '-',       # Baseline：实线
        'DDQN': '-',
        'T-DDQN': '--',
        'PPO': '-.',
        'Ablation': ':'
    }
    
    # 为每个典型日生成一张图
    for day_num, (day_idx, date_str) in enumerate(zip(day_indices, test_dates), 1):
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # 时间轴：96个点，每15分钟一个（0:00-23:45）
        time_steps = np.arange(96)
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]
        
        # 提取当天基线负荷数据
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        
        if end_idx > len(baseline_loads):
            print(f"警告：第{day_idx}天基线数据超出范围")
            continue
        
        # 1. 绘制基础负荷（固定部分，作为底线参考）
        if base_loads is not None and end_idx <= len(base_loads):
            base_load_day = base_loads[start_idx:end_idx]
            ax1.plot(time_steps, base_load_day, label='基础负荷',
                   color=colors['Base Load'], linewidth=2.0, alpha=0.6,
                   linestyle=linestyles['Base Load'], zorder=1)
        
        # 2. 绘制Baseline总负荷（规则优化：简单规则调度电池）
        base_loads_day = baseline_loads[start_idx:end_idx]
        ax1.plot(time_steps, base_loads_day, label='Baseline',
               color=colors['Baseline'], linewidth=3.5, alpha=0.7,
               linestyle=linestyles['Baseline'], zorder=2)
        
        # 3. 绘制各RL算法的总负荷曲线（优化后）
        for algo_name, loads_data in algorithm_loads.items():
            if algo_name == 'Baseline':
                continue  # 已经绘制过了
            
            # 提取当天96个点的数据
            if end_idx > len(loads_data):
                print(f"警告：{algo_name} 第{day_idx}天数据超出范围")
                continue
            
            algo_loads_day = loads_data[start_idx:end_idx]
            
            color = colors.get(algo_name, '#000000')
            linestyle = linestyles.get(algo_name, '-')
            
            # 绘制曲线
            ax1.plot(time_steps, algo_loads_day, label=f'{algo_name}',
                   color=color, linewidth=2.5, alpha=0.85,
                   linestyle=linestyle, zorder=3)
        
        # 注释掉峰谷时段背景颜色
        # # 标注峰谷时段
        # # 峰时段：8:00-11:00, 18:00-22:00
        # peak_periods = [(32, 44), (72, 88)]  # 对应的时间步索引
        # for start, end in peak_periods:
        #     ax1.axvspan(start, end, alpha=0.1, color='red', label='峰时段' if start == 32 else '')
        # 
        # # 谷时段：23:00-7:00
        # valley_periods = [(0, 28), (92, 96)]  # 对应的时间步索引
        # for start, end in valley_periods:
        #     ax1.axvspan(start, end, alpha=0.1, color='blue', label='谷时段' if start == 0 else '')
        
        # ==================== 修复：将光伏曲线绘制在同一个y轴（左侧），避免双y轴混淆 ====================
        if pv_data is not None and end_idx <= len(pv_data):
            pv_day = pv_data[start_idx:end_idx]
            
            # 绘制光伏曲线（填充区域）- 绿色，与负荷曲线共用左侧y轴
            ax1.fill_between(time_steps, 0, pv_day, 
                           color='#90EE90', alpha=0.3, label='光伏出力', zorder=0.5)  # 浅绿色填充，zorder较低，作为背景
            ax1.plot(time_steps, pv_day, 
                    color='#228B22', linewidth=2.5, alpha=0.85, 
                    linestyle='-', label='光伏出力', zorder=2)  # 深绿色曲线，zorder适中
        # ==============================================================
        
        ax1.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax1.set_ylabel('功率 (kW)', fontsize=13, fontweight='bold')  # 统一标签：功率（包含负荷和光伏）
        # ax1.set_title(f'典型日{day_num} - 各算法削峰填谷效果对比（负荷曲线）', 
        #             fontsize=15, fontweight='bold', pad=15)  # 删除标题
        
        # 设置x轴刻度（每2小时显示一次）
        ax1.set_xticks(time_steps[::8])  # 每8个点=2小时
        ax1.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')
        
        # 简化图例处理（单y轴，无需合并）
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        ax1.legend(by_label.values(), by_label.keys(), 
                 loc='upper left', fontsize=11, frameon=True, shadow=True, ncol=2)
        
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # 设置边框
        for spine in ax1.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成独立的文件名
        filename = f'04{chr(96+day_num)}_典型日{day_num}削峰填谷_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DONE] Generated: {output_dir}/{filename}")


def plot_battery_soc_comparison(algorithm_socs, test_dates, day_indices, output_dir='output_image'):
    """
    图5: 典型日的各算法储能SOC变化对比图（每个典型日生成一张图，显示96个时间步）
    
    Args:
        algorithm_socs: dict, {算法名: [所有测试日的SOC数据]} (110天×96点)
        test_dates: list, 典型日期列表（格式：'MM-DD'）
        day_indices: list, 典型日在测试集中的索引
        output_dir: 输出目录
    """
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    markers = {
        'Baseline': 'o',
        'DDQN': 's',
        'T-DDQN': '^',
        'PPO': 'D',
        'Ablation': 'v'
    }
    
    # 为每个典型日生成一张图
    for day_num, (day_idx, date_str) in enumerate(zip(day_indices, test_dates), 1):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 时间轴：96个点，每15分钟一个（0:00-23:45）
        time_steps = np.arange(96)
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]
        
        # 绘制每个算法的SOC曲线
        for algo_name, soc_data in algorithm_socs.items():
            # 提取当天96个点的数据
            start_idx = day_idx * 96
            end_idx = start_idx + 96
            
            if end_idx > len(soc_data):
                print(f"警告：{algo_name} 第{day_idx}天SOC数据超出范围")
                continue
            
            day_soc = np.array(soc_data[start_idx:end_idx]) * 100  # 转换为百分比
            
            color = colors.get(algo_name, '#000000')
            marker = markers.get(algo_name, 'o')
            
            # 绘制曲线
            ax.plot(time_steps, day_soc, label=algo_name,
                   color=color, linewidth=2.5, alpha=0.85,
                   marker=marker, markersize=4, markevery=8,  # 每8个点标记一次
                   markeredgecolor='black', markeredgewidth=0.8)
        
        # 添加SOC安全范围标记
        ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='SOC下限 (15%)')
        ax.axhline(y=85, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='SOC上限 (85%)')
        ax.axhline(y=50, color='green', linestyle=':', linewidth=1.0, alpha=0.3, label='SOC目标 (50%)')
        
        # 填充安全运行区间
        ax.axhspan(15, 85, alpha=0.05, color='green', label='安全运行区间')
        
        ax.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax.set_ylabel('储能SOC (%)', fontsize=13, fontweight='bold')
        # ax.set_title(f'典型日{day_num} ({date_str}) - 各算法储能SOC变化对比', 
        #             fontsize=15, fontweight='bold', pad=15)  # 删除标题
        
        # 设置x轴刻度（每2小时显示一次）
        ax.set_xticks(time_steps[::8])  # 每8个点=2小时
        ax.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')
        
        # 图例去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=11, frameon=True, shadow=True, ncol=2)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_ylim([0, 100])
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成独立的文件名
        filename = f'05{chr(96+day_num)}_典型日{day_num}储能SOC变化_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DONE] Generated: {output_dir}/{filename}")


def plot_battery_power_comparison(algorithm_battery_powers, test_dates, day_indices, output_dir='output_image'):
    """
    典型日储能功率曲线对比图

    约定：
    - 正功率：储能充电
    - 负功率：储能放电
    """
    colors = {
        'Baseline': '#6C757D',
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }

    linestyles = {
        'Baseline': '-',
        'DDQN': '-',
        'T-DDQN': '--',
        'PPO': '-.',
        'Ablation': ':'
    }

    for day_num, (day_idx, date_str) in enumerate(zip(day_indices, test_dates), 1):
        fig, ax = plt.subplots(figsize=(16, 8))

        time_steps = np.arange(96)
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]

        for algo_name, power_data in algorithm_battery_powers.items():
            start_idx = day_idx * 96
            end_idx = start_idx + 96

            if end_idx > len(power_data):
                print(f"警告：{algo_name} 在第 {day_idx} 天的储能功率数据不足")
                continue

            day_power = np.array(power_data[start_idx:end_idx])
            color = colors.get(algo_name, '#000000')
            linestyle = linestyles.get(algo_name, '-')

            ax.plot(
                time_steps, day_power, label=algo_name,
                color=color, linewidth=2.5, alpha=0.9,
                linestyle=linestyle, zorder=3
            )

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='零功率线')

        ax.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax.set_ylabel('储能功率 (kW)', fontsize=13, fontweight='bold')

        ax.set_xticks(time_steps[::8])
        ax.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper right', fontsize=11, frameon=True, shadow=True, ncol=2)

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)

        filename = f'07{chr(96+day_num)}_典型日{day_num}储能功率曲线_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[DONE] Generated: {output_dir}/{filename}")


def plot_ev_load_comparison(algorithm_ev_loads, baseline_ev_loads, real_ev_loads, 
                            test_dates, day_indices, output_dir='output_image'):
    """
    图6: 典型日的EV负荷调度前后对比图（每个典型日生成一张图，显示96个时间步）
    
    Args:
        algorithm_ev_loads: dict, {算法名: [所有测试日的EV负荷数据]} (110天×96点)
        baseline_ev_loads: list, 基线EV负荷数据（Baseline算法）
        real_ev_loads: list, 真实EV负荷数据（调度前的参考基准）
        test_dates: list, 典型日期列表（格式：'MM-DD'）
        day_indices: list, 典型日在测试集中的索引
        output_dir: 输出目录
    """
    colors = {
        'Real EV': '#808080',    # 真实EV：灰色
        'Baseline': '#6C757D',   # Baseline：深灰色
        'DDQN': '#2E86AB',
        'T-DDQN': '#A23B72',
        'PPO': '#F18F01',
        'Ablation': '#C73E1D'
    }
    
    linestyles = {
        'Real EV': ':',          # 真实EV：虚线
        'Baseline': '-',         # Baseline：实线
        'DDQN': '-',
        'T-DDQN': '--',
        'PPO': '-.',
        'Ablation': ':'
    }
    
    # 为每个典型日生成一张图
    for day_num, (day_idx, date_str) in enumerate(zip(day_indices, test_dates), 1):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 时间轴：96个点，每15分钟一个（0:00-23:45）
        time_steps = np.arange(96)
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]
        
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        
        # 1. 绘制真实EV负荷（调度前的参考基准，也是Baseline的结果）
        if real_ev_loads is not None and end_idx <= len(real_ev_loads):
            real_ev_day = real_ev_loads[start_idx:end_idx]
            ax.plot(time_steps, real_ev_day, label='真实EV负荷（未调度，Baseline）',
                   color=colors['Real EV'], linewidth=3.0, alpha=0.8,
                   linestyle=linestyles['Real EV'], zorder=1)
        
        # 注意：Baseline不调控EV，所以与真实EV负荷一致，不需要单独绘制
        # 如果传入了baseline_ev_loads但它与real_ev_loads不同，才单独绘制
        if baseline_ev_loads is not None and real_ev_loads is not None:
            baseline_ev_day = baseline_ev_loads[start_idx:end_idx]
            real_ev_day = real_ev_loads[start_idx:end_idx]
            # 检查是否不同（允许小误差）
            if not np.allclose(baseline_ev_day, real_ev_day, rtol=0.01):
                ax.plot(time_steps, baseline_ev_day, label='Baseline（异常：应与真实EV一致）',
                       color=colors['Baseline'], linewidth=2.0, alpha=0.5,
                       linestyle='--', zorder=2)
        
        # 3. 绘制各RL算法的EV负荷（调度后）
        for algo_name, ev_loads_data in algorithm_ev_loads.items():
            if algo_name == 'Baseline':
                continue  # 已经绘制过了
            
            # 提取当天96个点的数据
            if end_idx > len(ev_loads_data):
                print(f"警告：{algo_name} 第{day_idx}天EV数据超出范围")
                continue
            
            algo_ev_day = ev_loads_data[start_idx:end_idx]
            
            color = colors.get(algo_name, '#000000')
            linestyle = linestyles.get(algo_name, '-')
            
            # 绘制曲线
            ax.plot(time_steps, algo_ev_day, label=f'{algo_name}（智能调度）',
                   color=color, linewidth=2.5, alpha=0.85,
                   linestyle=linestyle, zorder=3)
        
        # 注释掉峰谷时段背景颜色（不显示在图例中）
        # # 峰时段：8:00-11:00, 18:00-22:00
        # peak_periods = [(32, 44), (72, 88)]  # 对应的时间步索引
        # for start, end in peak_periods:
        #     ax.axvspan(start, end, alpha=0.1, color='red', label='')
        # 
        # # 谷时段：23:00-7:00
        # valley_periods = [(0, 28), (92, 96)]  # 对应的时间步索引
        # for start, end in valley_periods:
        #     ax.axvspan(start, end, alpha=0.1, color='blue', label='')
        
        ax.set_xlabel('时间', fontsize=13, fontweight='bold')
        ax.set_ylabel('EV充电功率 (kW)', fontsize=13, fontweight='bold')
        # ax.set_title(f'典型日{day_num} ({date_str}) - EV负荷调度前后对比', 
        #             fontsize=15, fontweight='bold', pad=15)  # 删除标题
        
        # 设置x轴刻度（每2小时显示一次）
        ax.set_xticks(time_steps[::8])  # 每8个点=2小时
        ax.set_xticklabels([time_labels[i] for i in range(0, 96, 8)], rotation=45, ha='right')
        
        # 图例去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='upper right', fontsize=11, frameon=True, shadow=True, ncol=2)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成独立的文件名
        filename = f'06{chr(96+day_num)}_典型日{day_num}EV负荷调度对比_{date_str}.png'
        plt.savefig(os.path.join(output_dir, filename), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[DONE] Generated: {output_dir}/{filename}")


if __name__ == '__main__':
    # 测试示例
    print("强化学习结果可视化工具 - 清晰版")
    print("请在train.py中调用这些函数进行绘图")
