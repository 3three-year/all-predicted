# -*- coding: utf-8 -*-
"""
用户舒适度与温度分析可视化脚本
生成两个图表：
1. 综合满意度随时间变化曲线（有调度 vs 无调度）
2. 室内温度与设定温度对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import torch as T
from rural_env import CountrysideEnv

# 导入模型类
from DRL import DuelingDDQN, TDuelingDDQN, AblationTDuelingDDQN
from stable_baselines3 import PPO

# 配置matplotlib字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10


def get_latest_checkpoint(ckpt_dir, prefix):
    """获取最新的checkpoint文件"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"模型文件夹不存在：{ckpt_dir}")
    files = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"未找到以 {prefix} 开头的模型文件")
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    return os.path.join(ckpt_dir, files[0])


def load_trained_model(algo_name, ckpt_dir='./checkpoints'):
    """
    加载训练好的模型
    
    Args:
        algo_name: 算法名称 ('DDQN', 'T-DDQN', 'PPO', 'Ablation')
        ckpt_dir: checkpoint目录
    
    Returns:
        训练好的模型
    """
    state_dim = 15  # 状态空间维度
    action_dim = 5  # 动作空间维度
    
    try:
        if algo_name == "DDQN":
            agent = DuelingDDQN(
                alpha=0.0003, state_dim=state_dim, action_dim=action_dim,
                fc1_dim=256, fc2_dim=256, ckpt_dir=ckpt_dir,
                gamma=0.99, tau=0.01, eps_dec=1e-5,
                max_size=100000, batch_size=128
            )
            eval_path = get_latest_checkpoint(os.path.join(ckpt_dir, 'Q_eval'), 'DuelingDDQN_q_eval_')
            agent.q_eval.load_checkpoint(eval_path)
            agent.epsilon = 0.0
            print(f"  已加载DDQN模型: {eval_path}")
            return agent
            
        elif algo_name == "T-DDQN":
            agent = TDuelingDDQN(
                alpha=0.0003, state_dim=state_dim, action_dim=action_dim,
                fc1_dim=256, fc2_dim=256, ckpt_dir=ckpt_dir,
                gamma=0.99, tau=0.01, eps_dec=1e-5,
                max_size=100000, batch_size=128, sequence_length=12
            )
            eval_path = get_latest_checkpoint(os.path.join(ckpt_dir, 'Q_eval'), 'TransformerDuelingDQN_q_eval_')
            agent.q_eval.load_checkpoint(eval_path)
            agent.epsilon = 0.0
            print(f"  已加载T-DDQN模型: {eval_path}")
            return agent
            
        elif algo_name == "Ablation":
            agent = AblationTDuelingDDQN(
                alpha=0.0003, state_dim=state_dim, action_dim=action_dim,
                fc1_dim=256, fc2_dim=256, ckpt_dir=ckpt_dir,
                gamma=0.99, tau=0.01, eps_dec=1e-5,
                max_size=100000, batch_size=128, sequence_length=12
            )
            eval_path = get_latest_checkpoint(os.path.join(ckpt_dir, 'Q_eval'), 'AblationTransformerDuelingDQN_q_eval_')
            agent.q_eval.load_checkpoint(eval_path)
            agent.epsilon = 0.0
            print(f"  已加载Ablation模型: {eval_path}")
            return agent
            
        elif algo_name == "PPO":
            model_path = os.path.join(ckpt_dir, 'PPO', 'ppo_model.zip')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"PPO模型文件不存在：{model_path}")
            model = PPO.load(model_path)
            print(f"  已加载PPO模型: {model_path}")
            return model
            
        else:
            raise ValueError(f"未知算法: {algo_name}")
            
    except Exception as e:
        print(f"  警告：加载{algo_name}模型失败: {e}")
        print(f"  将使用规则策略代替")
        return None


def run_episode_and_collect_data(env, algo_name, test_date, use_trained_model=False, model=None):
    """
    运行一个episode并收集满意度和温度数据
    
    Args:
        env: 环境实例
        algo_name: 算法名称
        test_date: 测试日期（用于定位数据）
        use_trained_model: 是否使用训练好的模型
        model: 训练好的模型实例（如果use_trained_model=True）
    
    Returns:
        dict: 包含满意度、温度等数据的字典
    """
    state = env.reset()
    done = False
    step = 0
    
    # 数据收集列表
    satisfaction_history = []
    indoor_temp_history = []
    comfort_temp_history = []
    outdoor_temp_history = []
    time_steps = []
    
    # 固定室外温度（与环境中的设置一致）
    outdoor_temp_fixed = 30.0  # 夏季室外温度
    
    if use_trained_model and model is not None:
        print(f"\n开始收集 {algo_name} ({test_date}) 的数据（使用训练模型）...")
    else:
        print(f"\n开始收集 {algo_name} ({test_date}) 的数据（使用规则策略）...")
    
    while not done and step < 96:  # 一天96个时间步
        # 记录当前数据
        satisfaction = env._calculate_user_satisfaction()
        indoor_temp = env.ac_load.current_temp
        comfort_temp = env.ac_load.comfort_temp
        
        satisfaction_history.append(satisfaction)
        indoor_temp_history.append(indoor_temp)
        comfort_temp_history.append(comfort_temp)
        outdoor_temp_history.append(outdoor_temp_fixed)
        time_steps.append(step)
        
        # 选择动作
        if algo_name == "Baseline":
            action = -1  # Baseline不执行任何调度
        elif use_trained_model and model is not None:
            # 使用训练好的模型选择动作
            if step == 0:
                action = -1  # 第一步保持不变
            else:
                if algo_name == "PPO":
                    # PPO使用predict方法
                    action, _ = model.predict(state, deterministic=True)
                else:
                    # DQN系列使用choose_action方法
                    action = model.choose_action(state, isTrain=False)
        else:
            # 使用规则策略（当模型加载失败时的备选方案）
            pv_output = env._get_pv_output()
            current_load = env.load_demand
            battery_soc = env.battery_soc
            
            # 简化的规则策略
            if pv_output > 10:
                action = 2  # 光伏高发时调整空调
            elif pv_output < 2:
                action = 2  # 光伏低发时调整空调
            else:
                action = -1  # 其他时段不调整
        
        # 执行动作
        state, reward, done, info = env.step(action)
        step += 1
    
    print(f"  收集完成：{len(satisfaction_history)} 个时间步")
    
    return {
        'satisfaction': satisfaction_history,
        'indoor_temp': indoor_temp_history,
        'comfort_temp': comfort_temp_history,
        'outdoor_temp': outdoor_temp_history,
        'time_steps': time_steps
    }


def plot_satisfaction_comparison_all_algorithms(all_data, test_date, output_dir='comfort_analysis'):
    """
    图1：所有算法的综合满意度随时间变化曲线
    
    Args:
        all_data: 包含所有算法数据的字典
        test_date: 测试日期（用于文件名）
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 算法配置：名称、颜色、线型、标记
    algo_configs = {
        'Baseline': {'color': '#6C757D', 'linestyle': '--', 'marker': 'o', 'label': 'Baseline'},
        'DDQN': {'color': '#2E86AB', 'linestyle': '-', 'marker': 's', 'label': 'DDQN'},
        'T-DDQN': {'color': '#A23B72', 'linestyle': '-', 'marker': '^', 'label': 'T-DDQN'},
        'PPO': {'color': '#F18F01', 'linestyle': '-', 'marker': 'D', 'label': 'PPO'},
        'Ablation': {'color': '#C73E1D', 'linestyle': '-.', 'marker': 'v', 'label': 'Ablation'}
    }
    
    # 时间轴（小时）
    time_hours = np.array(all_data['Baseline']['time_steps']) / 4  # 转换为小时
    
    # 绘制所有算法的满意度曲线
    for algo_name, data in all_data.items():
        if algo_name in algo_configs:
            config = algo_configs[algo_name]
            ax.plot(time_hours, data['satisfaction'],
                   label=config['label'], color=config['color'], 
                   linewidth=2.5, alpha=0.85, linestyle=config['linestyle'],
                   marker=config['marker'], markersize=4, markevery=8)
    
    # 添加满意度阈值线
    ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, 
              alpha=0.7, label='满意度目标 (0.8)')
    ax.axhline(y=0.6, color='orange', linestyle=':', linewidth=1.5, 
              alpha=0.7, label='满意度警戒 (0.6)')
    
    # 填充满意区间
    ax.axhspan(0.8, 1.0, alpha=0.05, color='green', label='满意区间')
    ax.axhspan(0.6, 0.8, alpha=0.05, color='yellow', label='可接受区间')
    
    ax.set_xlabel('时间 (小时)', fontsize=13, fontweight='bold')
    ax.set_ylabel('综合满意度', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    
    # 设置x轴刻度
    ax.set_xticks(np.arange(0, 25, 2))
    
    # 图例去重并分两列显示
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='lower right', fontsize=11, frameon=True, shadow=True, ncol=2)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'01_综合满意度对比_{test_date}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[DONE] 已生成: {filename}")
    
    # 计算并打印统计数据
    print(f"\n  === {test_date} 满意度统计 ===")
    satisfaction_stats = {}
    for algo_name, data in all_data.items():
        if algo_name in algo_configs:
            avg_satisfaction = np.mean(data['satisfaction'])
            min_satisfaction = np.min(data['satisfaction'])
            max_satisfaction = np.max(data['satisfaction'])
            # 计算满意度达标率（≥0.8）
            target_rate = np.sum(np.array(data['satisfaction']) >= 0.8) / len(data['satisfaction']) * 100
            # 计算可接受率（≥0.6）
            acceptable_rate = np.sum(np.array(data['satisfaction']) >= 0.6) / len(data['satisfaction']) * 100
            
            satisfaction_stats[algo_name] = {
                'avg': avg_satisfaction,
                'min': min_satisfaction,
                'max': max_satisfaction,
                'target_rate': target_rate,
                'acceptable_rate': acceptable_rate
            }
            
            print(f"  {algo_name:>8}: 平均={avg_satisfaction:.3f}, 最小={min_satisfaction:.3f}, "
                  f"达标率={target_rate:.1f}%, 可接受率={acceptable_rate:.1f}%")
    
    return satisfaction_stats


def plot_temperature_comparison_all_algorithms(all_data, test_date, output_dir='comfort_analysis'):
    """
    图2：所有算法的室内温度对比图
    
    Args:
        all_data: 包含所有算法数据的字典
        test_date: 测试日期（用于文件名）
        output_dir: 输出目录
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 算法配置：名称、颜色、线型、标记
    algo_configs = {
        'Baseline': {'color': '#6C757D', 'linestyle': '--', 'marker': 'o', 'label': 'Baseline'},
        'DDQN': {'color': '#2E86AB', 'linestyle': '-', 'marker': 's', 'label': 'DDQN'},
        'T-DDQN': {'color': '#A23B72', 'linestyle': '-', 'marker': '^', 'label': 'T-DDQN'},
        'PPO': {'color': '#F18F01', 'linestyle': '-', 'marker': 'D', 'label': 'PPO'},
        'Ablation': {'color': '#C73E1D', 'linestyle': '-.', 'marker': 'v', 'label': 'Ablation'}
    }
    
    # 时间轴（小时）
    time_hours = np.array(all_data['Baseline']['time_steps']) / 4
    
    # === 上图：室内温度对比 ===
    for algo_name, data in all_data.items():
        if algo_name in algo_configs:
            config = algo_configs[algo_name]
            ax1.plot(time_hours, data['indoor_temp'],
                    label=f'{config["label"]} 室内温度', color=config['color'], 
                    linewidth=2.5, alpha=0.85, linestyle=config['linestyle'],
                    marker=config['marker'], markersize=3, markevery=12)
    
    # 绘制设定温度（使用Baseline的数据，所有算法应该相同）
    ax1.plot(time_hours, all_data['Baseline']['comfort_temp'],
            label='设定温度 $T_{set}$', color='black', linewidth=2.5, 
            alpha=0.7, linestyle=':', marker='*', markersize=4, markevery=12)
    
    # 绘制室外温度（参考）
    ax1.plot(time_hours, all_data['Baseline']['outdoor_temp'],
            label='室外温度 $T_{out}$', color='red', linewidth=2.0, 
            alpha=0.6, linestyle=':', marker='x', markersize=3, markevery=12)
    
    # 计算舒适区间（设定温度 ± 2℃）
    comfort_temp_array = np.array(all_data['Baseline']['comfort_temp'])
    comfort_upper = comfort_temp_array + 2.0
    comfort_lower = comfort_temp_array - 2.0
    
    # 绘制舒适区间
    ax1.fill_between(time_hours, comfort_lower, comfort_upper,
                    alpha=0.15, color='green', label='舒适区间 ($T_{set} ± 2°C$)')
    
    ax1.set_xlabel('时间 (小时)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('温度 (°C)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.set_ylim(16, 35)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_title('室内温度对比', fontsize=14, fontweight='bold', pad=15)
    
    # === 下图：温度偏差对比 ===
    for algo_name, data in all_data.items():
        if algo_name in algo_configs:
            config = algo_configs[algo_name]
            # 计算温度偏差
            indoor_temps = np.array(data['indoor_temp'])
            comfort_temps = np.array(data['comfort_temp'])
            temp_deviations = np.abs(indoor_temps - comfort_temps)
            
            ax2.plot(time_hours, temp_deviations,
                    label=f'{config["label"]} 偏差', color=config['color'], 
                    linewidth=2.5, alpha=0.85, linestyle=config['linestyle'],
                    marker=config['marker'], markersize=3, markevery=12)
    
    # 添加舒适度阈值线
    ax2.axhline(y=2.0, color='green', linestyle=':', linewidth=2.0, 
               alpha=0.7, label='舒适阈值 (2°C)')
    ax2.axhline(y=1.0, color='blue', linestyle=':', linewidth=1.5, 
               alpha=0.5, label='理想阈值 (1°C)')
    
    # 填充舒适区间
    ax2.axhspan(0, 1.0, alpha=0.1, color='blue', label='理想区间')
    ax2.axhspan(1.0, 2.0, alpha=0.1, color='green', label='舒适区间')
    ax2.axhspan(2.0, 5.0, alpha=0.1, color='orange', label='不适区间')
    
    ax2.set_xlabel('时间 (小时)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('温度偏差 (°C)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 5)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_title('温度偏差对比', fontsize=14, fontweight='bold', pad=15)
    
    # 设置边框
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'02_室内温度对比_{test_date}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[DONE] 已生成: {filename}")
    
    # 计算并打印统计数据
    print(f"\n  === {test_date} 温度控制统计 ===")
    temp_stats = {}
    for algo_name, data in all_data.items():
        if algo_name in algo_configs:
            indoor_temps = np.array(data['indoor_temp'])
            comfort_temps = np.array(data['comfort_temp'])
            temp_deviations = np.abs(indoor_temps - comfort_temps)
            
            avg_deviation = np.mean(temp_deviations)
            max_deviation = np.max(temp_deviations)
            comfort_rate = np.sum(temp_deviations <= 2.0) / len(temp_deviations) * 100
            ideal_rate = np.sum(temp_deviations <= 1.0) / len(temp_deviations) * 100
            
            temp_stats[algo_name] = {
                'avg_deviation': avg_deviation,
                'max_deviation': max_deviation,
                'comfort_rate': comfort_rate,
                'ideal_rate': ideal_rate
            }
            
            print(f"  {algo_name:>8}: 平均偏差={avg_deviation:.2f}°C, 最大偏差={max_deviation:.2f}°C, "
                  f"舒适率={comfort_rate:.1f}%, 理想率={ideal_rate:.1f}%")
    
    return temp_stats


def generate_comfort_summary_table(all_satisfaction_stats, all_temp_stats, output_dir='comfort_analysis'):
    """
    生成舒适度汇总表格
    
    Args:
        all_satisfaction_stats: 所有日期的满意度统计数据
        all_temp_stats: 所有日期的温度统计数据
        output_dir: 输出目录
    """
    import pandas as pd
    
    # 准备数据
    summary_data = []
    
    for date in all_satisfaction_stats.keys():
        satisfaction_stats = all_satisfaction_stats[date]
        temp_stats = all_temp_stats[date]
        
        for algo in satisfaction_stats.keys():
            summary_data.append({
                '日期': date,
                '算法': algo,
                '平均满意度': satisfaction_stats[algo]['avg'],
                '最小满意度': satisfaction_stats[algo]['min'],
                '满意度达标率(%)': satisfaction_stats[algo]['target_rate'],
                '满意度可接受率(%)': satisfaction_stats[algo]['acceptable_rate'],
                '平均温度偏差(°C)': temp_stats[algo]['avg_deviation'],
                '最大温度偏差(°C)': temp_stats[algo]['max_deviation'],
                '温度舒适率(%)': temp_stats[algo]['comfort_rate'],
                '温度理想率(%)': temp_stats[algo]['ideal_rate']
            })
    
    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    
    # 保存为Excel文件
    excel_filename = os.path.join(output_dir, '舒适度分析汇总表.xlsx')
    df.to_excel(excel_filename, index=False, sheet_name='舒适度统计')
    
    print(f"[DONE] 已生成汇总表: {excel_filename}")
    
    # 计算跨日期平均值
    print(f"\n  === 跨日期平均性能排名 ===")
    
    # 按算法分组计算平均值
    avg_by_algo = df.groupby('算法').agg({
        '平均满意度': 'mean',
        '满意度达标率(%)': 'mean',
        '平均温度偏差(°C)': 'mean',
        '温度舒适率(%)': 'mean'
    }).round(3)
    
    # 按平均满意度排序
    avg_by_algo_sorted = avg_by_algo.sort_values('平均满意度', ascending=False)
    
    print("  算法排名（按平均满意度）:")
    for i, (algo, row) in enumerate(avg_by_algo_sorted.iterrows(), 1):
        print(f"  {i}. {algo:>8}: 满意度={row['平均满意度']:.3f}, "
              f"达标率={row['满意度达标率(%)']:.1f}%, "
              f"温度偏差={row['平均温度偏差(°C)']:.2f}°C, "
              f"舒适率={row['温度舒适率(%)']:.1f}%")
    
    return df, avg_by_algo_sorted


def get_test_index_for_date(target_date_str):
    """
    根据日期字符串（如'09-16'）查找对应的test_index
    
    Args:
        target_date_str: 日期字符串，格式为'MM-DD'
    
    Returns:
        int: 对应的test_index，如果未找到则返回0
    """
    import datetime
    
    # 创建临时环境来获取test_dates列表
    temp_env = CountrysideEnv(algo="baseline", num_days=1, mode='test', verbose=False)
    
    # 解析目标日期（假设年份为2022，因为数据是22年的）
    month, day = map(int, target_date_str.split('-'))
    target_date = datetime.date(2022, month, day)
    
    # 在test_dates中查找匹配的日期
    if hasattr(temp_env, 'test_dates'):
        for idx, test_date in enumerate(temp_env.test_dates):
            if test_date == target_date:
                print(f"  找到日期 {target_date_str} 对应的test_index: {idx}")
                return idx
    
    print(f"  警告：未找到日期 {target_date_str}，使用默认test_index=0")
    return 0


def main():
    """主函数"""
    print("=" * 60)
    print("用户舒适度与温度分析 - 所有算法对比（使用真实训练模型）")
    print("=" * 60)
    
    output_dir = 'comfort_analysis'
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = './checkpoints'
    
    # 定义所有算法配置
    algorithms = {
        'Baseline': {'algo': 'baseline', 'use_trained_model': False, 'model': None},
        'DDQN': {'algo': 'dueling_dqn', 'use_trained_model': True, 'model': None},
        'T-DDQN': {'algo': 't_dueling_dqn', 'use_trained_model': True, 'model': None},
        'PPO': {'algo': 'ppo', 'use_trained_model': True, 'model': None},
        'Ablation': {'algo': 'ablation', 'use_trained_model': True, 'model': None}
    }
    
    # 加载所有训练好的模型
    print("\n[步骤0] 加载训练好的模型...")
    for algo_name, config in algorithms.items():
        if config['use_trained_model']:
            try:
                config['model'] = load_trained_model(algo_name, ckpt_dir)
            except Exception as e:
                print(f"  警告：{algo_name}模型加载失败: {e}")
                print(f"  将使用规则策略代替")
                config['use_trained_model'] = False
    
    # 定义测试日期及其对应的test_index
    test_date_configs = [
        {'date_str': '09-16', 'test_index': None},
        {'date_str': '09-25', 'test_index': None}
    ]
    
    # 查找每个日期对应的test_index
    print("\n[步骤1] 查找测试日期对应的索引...")
    for config in test_date_configs:
        config['test_index'] = get_test_index_for_date(config['date_str'])
    
    # 存储所有统计数据
    all_satisfaction_stats = {}
    all_temp_stats = {}
    
    # 为每个测试日期生成图表
    for config in test_date_configs:
        test_date = config['date_str']
        test_index = config['test_index']
        
        print(f"\n{'='*60}")
        print(f"处理测试日期: {test_date} (test_index={test_index})")
        print(f"{'='*60}")
        
        # 收集所有算法的数据
        all_data = {}
        
        for algo_name, algo_config in algorithms.items():
            print(f"\n[步骤{list(algorithms.keys()).index(algo_name)+2}] 收集 {algo_name} 数据 ({test_date})...")
            
            try:
                env = CountrysideEnv(algo=algo_config['algo'], num_days=1, mode='test', 
                                   verbose=False, test_index=test_index)
                data = run_episode_and_collect_data(
                    env, algo_name, test_date, 
                    use_trained_model=algo_config['use_trained_model'],
                    model=algo_config['model']
                )
                all_data[algo_name] = data
                print(f"  [OK] {algo_name} 数据收集成功")
                
            except Exception as e:
                print(f"  [FAIL] {algo_name} 数据收集失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_data) < 2:
            print(f"  警告：{test_date} 可用算法数据不足，跳过图表生成")
            continue
        
        # 生成图1：综合满意度对比
        print(f"\n[图表1] 生成综合满意度对比图 ({test_date})...")
        satisfaction_stats = plot_satisfaction_comparison_all_algorithms(all_data, test_date, output_dir)
        all_satisfaction_stats[test_date] = satisfaction_stats
        
        # 生成图2：室内温度对比
        print(f"\n[图表2] 生成室内温度对比图 ({test_date})...")
        temp_stats = plot_temperature_comparison_all_algorithms(all_data, test_date, output_dir)
        all_temp_stats[test_date] = temp_stats
    
    # 生成汇总表格
    if all_satisfaction_stats and all_temp_stats:
        print(f"\n[汇总] 生成舒适度分析汇总表...")
        summary_df, ranking = generate_comfort_summary_table(all_satisfaction_stats, all_temp_stats, output_dir)
    
    print("\n" + "=" * 60)
    print(f"所有图表已生成到目录: {output_dir}/")
    print("=" * 60)
    print("\n说明：本次分析使用了真实的训练模型进行动作选择")
    print("生成的图表反映了各算法训练后的实际运行效果")
    print("=" * 60)
    
    # 生成更新的说明文件
    readme_content = """# 用户舒适度与温度分析结果 - 所有算法对比

## 生成的图表

### 典型日1（9月16日）

#### 1. 综合满意度对比 (01_综合满意度对比_09-16.png)
- 横轴：时间（24小时）
- 纵轴：综合满意度（0-1）
- 曲线：包含所有算法（Baseline、DDQN、T-DDQN、PPO、Ablation）
- 参考线：
  - 满意度目标（0.8）：绿色虚线
  - 满意度警戒（0.6）：橙色虚线
- 区间：
  - 满意区间（0.8-1.0）：绿色阴影
  - 可接受区间（0.6-0.8）：黄色阴影

#### 2. 室内温度对比 (02_室内温度对比_09-16.png)
- 上图：所有算法的室内温度曲线
- 下图：所有算法的温度偏差对比
- 参考线：
  - 设定温度：黑色点线
  - 室外温度：红色点线
  - 舒适阈值（2°C）：绿色虚线
  - 理想阈值（1°C）：蓝色虚线

### 典型日2（9月25日）

#### 3. 综合满意度对比 (01_综合满意度对比_09-25.png)
- 与典型日1格式相同

#### 4. 室内温度对比 (02_室内温度对比_09-25.png)
- 与典型日1格式相同

### 汇总分析

#### 5. 舒适度分析汇总表 (舒适度分析汇总表.xlsx)
- 包含所有算法在两个典型日的详细统计数据
- 指标包括：
  - 平均满意度、最小满意度
  - 满意度达标率（≥0.8）、可接受率（≥0.6）
  - 平均温度偏差、最大温度偏差
  - 温度舒适率（偏差≤2°C）、理想率（偏差≤1°C）

## 算法配置说明

| 算法 | 环境配置 | 模型使用 | 颜色 | 线型 |
|------|---------|---------|------|------|
| Baseline | baseline | 无 | 灰色 | 虚线 |
| DDQN | dueling_dqn | 训练模型 | 蓝色 | 实线 |
| T-DDQN | t_dueling_dqn | 训练模型 | 紫色 | 实线 |
| PPO | ppo | 训练模型 | 橙色 | 实线 |
| Ablation | ablation | 训练模型 | 红色 | 点划线 |

## 评价指标说明

### 满意度指标
- **平均满意度**：全天96个时间点的满意度平均值
- **满意度达标率**：满意度≥0.8的时间点占比
- **满意度可接受率**：满意度≥0.6的时间点占比

### 温度控制指标
- **平均温度偏差**：|室内温度 - 设定温度|的平均值
- **温度舒适率**：温度偏差≤2°C的时间点占比
- **温度理想率**：温度偏差≤1°C的时间点占比

## 数据说明

- 数据来源：运行环境模拟一天（96个时间步，每步15分钟）
- 测试日期：9月16日和9月25日（与削峰填谷分析保持一致）
- 所有算法使用相同的环境参数和测试条件
- 训练模型算法使用预训练的模型权重

## 使用方法

运行 `train_simplified.py` 完成训练后，单独运行：
```bash
python plot_comfort_analysis.py
```

或者在 `train_simplified.py` 的最后添加：
```python
from plot_comfort_analysis import main as plot_comfort
plot_comfort()
```

## 生成的文件列表

```
comfort_analysis/
├── 01_综合满意度对比_09-16.png
├── 02_室内温度对比_09-16.png
├── 01_综合满意度对比_09-25.png
├── 02_室内温度对比_09-25.png
├── 舒适度分析汇总表.xlsx
└── README.md
```

## 分析要点

### 多目标优化权衡
- 观察各算法在成本优化与舒适度保持之间的权衡
- 分析哪个算法在保证舒适度的同时实现了最佳经济效益

### 跨日期稳定性
- 比较各算法在不同日期下的舒适度表现一致性
- 评估算法的鲁棒性和适应性

### 温度控制精度
- 分析各算法的温度控制精度和稳定性
- 评估智能调度对用户舒适度的实际影响

### 综合性能评价
- 结合满意度、温度控制、成本效益等多维度指标
- 为实际应用选择最适合的算法提供依据
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n说明文件已生成: {output_dir}/README.md")
    print("\n生成的文件列表:")
    print(f"  - 01_综合满意度对比_09-16.png")
    print(f"  - 02_室内温度对比_09-16.png")
    print(f"  - 01_综合满意度对比_09-25.png")
    print(f"  - 02_室内温度对比_09-25.png")
    print(f"  - 舒适度分析汇总表.xlsx")
    print(f"  - README.md")


if __name__ == '__main__':
    main()
