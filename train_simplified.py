# -*- coding: utf-8 -*-
"""
简化版强化学习训练脚本
修复数据集划分问题，简化测试逻辑
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

# 抑制gym的Box精度警告（这些警告不影响功能）
warnings.filterwarnings('ignore', category=UserWarning, module='gym.spaces.box')

# 导入字体配置
import font_config
from ddqn_trainer import DuelingDDQNTrainer, TDuelingDDQNTrainer, AblationTDuelingDDQNTrainer
from ppo_trainer import PPOTrainer
from baseline_trainer import BaselineTrainer
from rural_env import CountrysideEnv

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=60)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/')
args = parser.parse_args()

def main():
    # 设置随机种子
    import time
    current_seed = int(time.time()) % 10000
    print(f"使用随机种子: {current_seed}")
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    
    # 设置输出目录
    output_dir = "output_image"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # ========== 1. 训练各算法 ==========
    print("\n" + "="*60)
    print("开始训练强化学习算法")
    print("="*60)
    
    # 重置测试数据索引
    CountrysideEnv.test_data_index = 0
    
    # 1.1 训练DDQN
    print("\n开始训练 DDQN...")
    ddqn_trainer = DuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    ddqn_rewards, ddqn_avg_rewards, ddqn_loads, ddqn_pvs, ddqn_hour_to_loads, ddqn_hour_to_pvs = ddqn_trainer.train()
    
    # 1.2 训练T-DDQN
    print("\n开始训练 T-DDQN...")
    t_ddqn_trainer = TDuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    t_ddqn_rewards, t_ddqn_avg_rewards, t_ddqn_loads, t_ddqn_pvs, t_ddqn_hour_to_loads, t_ddqn_hour_to_pvs = t_ddqn_trainer.train()
    
    # 1.3 训练消融实验
    print("\n开始训练 Ablation (无注意力)...")
    ablation_trainer = AblationTDuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    ablation_rewards, ablation_avg_rewards, ablation_loads, ablation_pvs, ablation_hour_to_loads, ablation_hour_to_pvs = ablation_trainer.train()
    
    # 1.4 训练PPO
    print("\n开始训练 PPO...")
    ppo_trainer = PPOTrainer(args.max_episodes, output_dir)
    ppo_rewards, ppo_avg_rewards = ppo_trainer.train()
    
    # ========== 2. 测试各算法 ==========
    print("\n" + "="*60)
    print("开始在标准测试集上测试所有算法")
    print("="*60)
    
    # 获取测试集信息（verbose=False避免打印日志）
    env_temp = CountrysideEnv(algo="baseline", num_days=1, mode='test', verbose=False)
    test_dates_count = len(env_temp.test_dates)
    print(f"\n测试集共 {test_dates_count} 天")
    print(f"测试日期范围: {env_temp.test_dates[0]} 至 {env_temp.test_dates[-1]}\n")
    
    # 2.1 测试Baseline
    print("测试 Baseline...")
    baseline_trainer = BaselineTrainer(output_dir)
    baseline_result = baseline_trainer.test(test_index=0, num_test_days=test_dates_count)
    if len(baseline_result) == 8:  # 新版本返回8个值（包含EV负荷和储能功率）
        before_loads, before_pvs, before_socs, before_pv_util, before_prices, base_loads, baseline_ev_loads, baseline_battery_powers = baseline_result
    elif len(baseline_result) == 7:  # 兼容旧版本（仅包含EV负荷）
        before_loads, before_pvs, before_socs, before_pv_util, before_prices, base_loads, baseline_ev_loads = baseline_result
        baseline_battery_powers = None
    else:  # 旧版本返回6个值
        before_loads, before_pvs, before_socs, before_pv_util, before_prices, base_loads = baseline_result
        baseline_ev_loads = None
        baseline_battery_powers = None
    
    # 2.2 测试DDQN
    print("测试 DDQN...")
    ddqn_test_loads, ddqn_test_pvs, ddqn_test_socs, ddqn_test_pv_util, ddqn_prices, ddqn_ev_loads, ddqn_battery_powers = ddqn_trainer.test(test_index=0, num_test_days=test_dates_count)
    # 保存DDQN总负荷数据
    np.save(os.path.join(output_dir, 'ddqn_loads.npy'), ddqn_test_loads)
    print(f"   [DONE] Saved DDQN loads: {len(ddqn_test_loads)} points")
    
    # 2.3 测试T-DDQN
    print("测试 T-DDQN...")
    t_ddqn_test_loads, t_ddqn_test_pvs, t_ddqn_test_socs, t_ddqn_test_pv_util, t_ddqn_prices, t_ddqn_base_loads, t_ddqn_ev_loads, t_ddqn_battery_powers = t_ddqn_trainer.test(test_index=0, num_test_days=test_dates_count)
    # 保存T-DDQN总负荷数据
    np.save(os.path.join(output_dir, 't_ddqn_loads.npy'), t_ddqn_test_loads)
    print(f"   [DONE] Saved T-DDQN loads: {len(t_ddqn_test_loads)} points")
    
    # 2.4 测试Ablation
    print("测试 Ablation...")
    ablation_result = ablation_trainer.test(test_index=0, num_test_days=test_dates_count)
    if len(ablation_result) == 6:
        ablation_test_loads, ablation_test_pvs, ablation_test_socs, ablation_test_pv_util, ablation_prices, ablation_base_loads = ablation_result
        ablation_ev_loads = None  # Ablation没有单独的EV负荷数据
        ablation_battery_powers = None
    elif len(ablation_result) == 7:
        ablation_test_loads, ablation_test_pvs, ablation_test_socs, ablation_test_pv_util, ablation_prices, ablation_base_loads, ablation_ev_loads = ablation_result
        ablation_battery_powers = None
    else:
        ablation_test_loads, ablation_test_pvs, ablation_test_socs, ablation_test_pv_util, ablation_prices, ablation_base_loads, ablation_ev_loads, ablation_battery_powers = ablation_result
    # 保存Ablation总负荷数据
    np.save(os.path.join(output_dir, 'ablation_loads.npy'), ablation_test_loads)
    print(f"   [DONE] Saved Ablation loads: {len(ablation_test_loads)} points")
    
    # 2.5 测试PPO
    print("测试 PPO...")
    ppo_test_loads, ppo_test_pvs, ppo_test_socs, ppo_test_pv_util, ppo_prices, ppo_ev_loads, ppo_battery_powers = ppo_trainer.test(test_index=0, num_test_days=test_dates_count)
    # 保存PPO总负荷数据
    np.save(os.path.join(output_dir, 'ppo_loads.npy'), ppo_test_loads)
    print(f"   [DONE] Saved PPO loads: {len(ppo_test_loads)} points")
    
    # ========== 3. 生成对比图 ==========
    print("\n" + "="*60)
    print("📊 开始生成清晰的强化学习结果对比图（最多10张图）")
    print("="*60 + "\n")
    
    # 导入绘图工具
    from rl_plot_utils import (
        plot_training_rewards_comparison,
        plot_test_cost_comparison,
        plot_test_pv_utilization_comparison,
        plot_test_peak_valley_comparison,
        plot_battery_soc_comparison,
        plot_ev_load_comparison,
        plot_battery_power_comparison
    )
    
    # 3.1 收集训练奖励数据
    print("📈 收集训练奖励数据...")
    algorithm_rewards = {
        'DDQN': {'rewards': ddqn_rewards, 'avg_rewards': ddqn_avg_rewards},
        'T-DDQN': {'rewards': t_ddqn_rewards, 'avg_rewards': t_ddqn_avg_rewards},
        'Ablation': {'rewards': ablation_rewards, 'avg_rewards': ablation_avg_rewards},
        'PPO': {'rewards': ppo_rewards, 'avg_rewards': ppo_avg_rewards}
    }
    
    # 3.2 从测试集中选择2个典型日
    print("📊 选择典型日期...")
    all_test_dates = env_temp.test_dates
    
    # 修改：使用预测数据的日期（09-16和09-25）
    typical_day_indices = []
    typical_day_dates_str = []
    target_dates = ['2022-09-16', '2022-09-25']
    
    for target in target_dates:
        for idx, date in enumerate(all_test_dates):
            if target in str(date):
                typical_day_indices.append(idx)
                typical_day_dates_str.append(str(date)[5:10])
                print(f"找到典型日: {target} (索引 {idx})")
                break
    
    # 如果找不到，使用备选日期
    if len(typical_day_indices) < 2:
        print("警告：目标日期不在测试集，使用备选日期")
        typical_day_indices = [5, min(26, test_dates_count-1)]
        typical_day_dates_str = [str(all_test_dates[i])[5:10] for i in typical_day_indices]
    
    print(f"最终选择的典型日: {typical_day_dates_str}")
    print(f"对应日期: {[str(all_test_dates[i]) for i in typical_day_indices]}")
    
    # 3.3 计算典型日指标
    def calculate_test_cost(loads, pvs, prices):
        """计算测试成本"""
        total_cost = 0
        time_interval = 0.25  # 15分钟
        for i in range(len(loads)):
            net_load = max(0, loads[i] - pvs[i])
            total_cost += net_load * prices[i] * time_interval
        return total_cost
    
    # 准备绘图数据
    algorithm_costs = {'Baseline': [], 'DDQN': [], 'T-DDQN': [], 'Ablation': [], 'PPO': []}
    algorithm_pv_utils_avg = {'Baseline': [], 'DDQN': [], 'T-DDQN': [], 'Ablation': [], 'PPO': []}  # 用于成本图
    algorithm_loads = {'Baseline': before_loads, 'DDQN': ddqn_test_loads, 
                       'T-DDQN': t_ddqn_test_loads, 'Ablation': ablation_test_loads, 'PPO': ppo_test_loads}
    
    # 计算所有测试日的逐时光伏消纳率（110天×96点）
    algorithm_pv_utils_timeseries = {
        'Baseline': [],
        'DDQN': [],
        'T-DDQN': [],
        'Ablation': [],
        'PPO': []
    }
    
    # 计算所有110天的逐时消纳率
    test_dates_count = len(before_loads) // 96
    for day_i in range(test_dates_count):
        start_i = day_i * 96
        end_i = start_i + 96
        
        # Baseline
        day_loads_bl = before_loads[start_i:end_i]
        day_pvs_bl = before_pvs[start_i:end_i]
        for t in range(96):
            # 优化：渐进过渡方案，平滑从0%过渡到实际消纳率
            pv_value = day_pvs_bl[t]
            if pv_value < 5.0:
                # PV<5kW: 设为0%
                pv_util_t = 0.0
            elif pv_value >= 10.0:
                # PV≥10kW: 正常计算
                pv_util_t = min(pv_value, day_loads_bl[t]) / pv_value
            else:
                # 5kW≤PV<10kW: 线性插值过渡
                actual_util = min(pv_value, day_loads_bl[t]) / pv_value
                transition_factor = (pv_value - 5.0) / (10.0 - 5.0)  # 0到1的插值因子
                pv_util_t = actual_util * transition_factor
            algorithm_pv_utils_timeseries['Baseline'].append(pv_util_t)
        
        # DDQN
        day_loads_dq = ddqn_test_loads[start_i:end_i]
        day_pvs_dq = ddqn_test_pvs[start_i:end_i]
        for t in range(96):
            # 优化：渐进过渡方案，平滑从0%过渡到实际消纳率
            pv_value = day_pvs_dq[t]
            if pv_value < 5.0:
                # PV<5kW: 设为0%
                pv_util_t = 0.0
            elif pv_value >= 10.0:
                # PV≥10kW: 正常计算
                pv_util_t = min(pv_value, day_loads_dq[t]) / pv_value
            else:
                # 5kW≤PV<10kW: 线性插值过渡
                actual_util = min(pv_value, day_loads_dq[t]) / pv_value
                transition_factor = (pv_value - 5.0) / (10.0 - 5.0)  # 0到1的插值因子
                pv_util_t = actual_util * transition_factor
            algorithm_pv_utils_timeseries['DDQN'].append(pv_util_t)
        
        # T-DDQN
        day_loads_td = t_ddqn_test_loads[start_i:end_i]
        day_pvs_td = t_ddqn_test_pvs[start_i:end_i]
        for t in range(96):
            # 优化：渐进过渡方案，平滑从0%过渡到实际消纳率
            pv_value = day_pvs_td[t]
            if pv_value < 5.0:
                # PV<5kW: 设为0%
                pv_util_t = 0.0
            elif pv_value >= 10.0:
                # PV≥10kW: 正常计算
                pv_util_t = min(pv_value, day_loads_td[t]) / pv_value
            else:
                # 5kW≤PV<10kW: 线性插值过渡
                actual_util = min(pv_value, day_loads_td[t]) / pv_value
                transition_factor = (pv_value - 5.0) / (10.0 - 5.0)  # 0到1的插值因子
                pv_util_t = actual_util * transition_factor
            algorithm_pv_utils_timeseries['T-DDQN'].append(pv_util_t)
        
        # PPO
        day_loads_pp = ppo_test_loads[start_i:end_i]
        day_pvs_pp = ppo_test_pvs[start_i:end_i]
        for t in range(96):
            # 优化：渐进过渡方案，平滑从0%过渡到实际消纳率
            pv_value = day_pvs_pp[t]
            if pv_value < 5.0:
                # PV<5kW: 设为0%
                pv_util_t = 0.0
            elif pv_value >= 10.0:
                # PV≥10kW: 正常计算
                pv_util_t = min(pv_value, day_loads_pp[t]) / pv_value
            else:
                # 5kW≤PV<10kW: 线性插值过渡
                actual_util = min(pv_value, day_loads_pp[t]) / pv_value
                transition_factor = (pv_value - 5.0) / (10.0 - 5.0)  # 0到1的插值因子
                pv_util_t = actual_util * transition_factor
            algorithm_pv_utils_timeseries['PPO'].append(pv_util_t)
        
        # Ablation
        day_loads_ab = ablation_test_loads[start_i:end_i]
        day_pvs_ab = ablation_test_pvs[start_i:end_i]
        for t in range(96):
            # 优化：渐进过渡方案，平滑从0%过渡到实际消纳率
            pv_value = day_pvs_ab[t]
            if pv_value < 5.0:
                # PV<5kW: 设为0%
                pv_util_t = 0.0
            elif pv_value >= 10.0:
                # PV≥10kW: 正常计算
                pv_util_t = min(pv_value, day_loads_ab[t]) / pv_value
            else:
                # 5kW≤PV<10kW: 线性插值过渡
                actual_util = min(pv_value, day_loads_ab[t]) / pv_value
                transition_factor = (pv_value - 5.0) / (10.0 - 5.0)  # 0到1的插值因子
                pv_util_t = actual_util * transition_factor
            algorithm_pv_utils_timeseries['Ablation'].append(pv_util_t)
    
    # 计算典型日的平均成本和消纳率（用于成本对比图）
    for day_idx in typical_day_indices:
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        
        if end_idx > len(before_loads):
            print(f"警告：第{day_idx}天数据超出范围")
            continue
        
        # Baseline
        day_loads = before_loads[start_idx:end_idx]
        day_pvs = before_pvs[start_idx:end_idx]
        day_prices = before_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['Baseline'].append(cost)
        algorithm_pv_utils_avg['Baseline'].append(pv_util)
        
        # DDQN
        day_loads = ddqn_test_loads[start_idx:end_idx]
        day_pvs = ddqn_test_pvs[start_idx:end_idx]
        day_prices = ddqn_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['DDQN'].append(cost)
        algorithm_pv_utils_avg['DDQN'].append(pv_util)
        
        # T-DDQN
        day_loads = t_ddqn_test_loads[start_idx:end_idx]
        day_pvs = t_ddqn_test_pvs[start_idx:end_idx]
        day_prices = t_ddqn_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['T-DDQN'].append(cost)
        algorithm_pv_utils_avg['T-DDQN'].append(pv_util)
        
        # Ablation
        day_loads = ablation_test_loads[start_idx:end_idx]
        day_pvs = ablation_test_pvs[start_idx:end_idx]
        day_prices = ablation_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['Ablation'].append(cost)
        algorithm_pv_utils_avg['Ablation'].append(pv_util)
        
        # PPO
        day_loads = ppo_test_loads[start_idx:end_idx]
        day_pvs = ppo_test_pvs[start_idx:end_idx]
        day_prices = ppo_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['PPO'].append(cost)
        algorithm_pv_utils_avg['PPO'].append(pv_util)
    
    # 3.4 生成图表（训练奖励1张 + 成本对比1张 + 光伏消纳率2张 + 削峰填谷2张 + 储能SOC2张 + EV负荷2张）
    print("\n🎨 开始生成图表...\n")
    
    # 图1: 训练奖励对比曲线
    plot_training_rewards_comparison(algorithm_rewards, output_dir)
    
    # 图2: 典型日成本对比（整天平均）
    plot_test_cost_comparison(algorithm_costs, typical_day_dates_str, output_dir)
    
    # 图3a和3b: 典型日光伏消纳率对比（逐时曲线，2张图）
    plot_test_pv_utilization_comparison(algorithm_pv_utils_timeseries, typical_day_dates_str, 
                                       typical_day_indices, output_dir)
    
    # 图4a和4b: 典型日削峰填谷对比（负荷曲线，2张图）
    # 传递光伏数据以显示光伏出力曲线
    plot_test_peak_valley_comparison(algorithm_loads, before_loads, typical_day_dates_str, 
                                    output_dir, typical_day_indices, base_loads, before_pvs)
    
    # 图5a和5b: 典型日储能SOC变化对比（2张图）
    algorithm_socs = {
        'Baseline': before_socs,
        'DDQN': ddqn_test_socs,
        'T-DDQN': t_ddqn_test_socs,
        'Ablation': ablation_test_socs,
        'PPO': ppo_test_socs
    }
    plot_battery_soc_comparison(algorithm_socs, typical_day_dates_str, typical_day_indices, output_dir)

    # 图7a/7b: 典型日储能功率曲线
    # 由SOC按天反推储能功率：正值表示充电，负值表示放电
    battery_capacity_kwh = env_temp.battery_capacity_kwh if hasattr(env_temp, 'battery_capacity_kwh') else 100.0
    time_interval = 0.25

    algorithm_battery_powers = {
        'Baseline': baseline_battery_powers if baseline_battery_powers is not None else [0.0] * len(before_loads),
        'DDQN': ddqn_battery_powers,
        'T-DDQN': t_ddqn_battery_powers,
        'Ablation': ablation_battery_powers if ablation_battery_powers is not None else [0.0] * len(ablation_test_loads),
        'PPO': ppo_battery_powers
    }
    plot_battery_power_comparison(algorithm_battery_powers, typical_day_dates_str, typical_day_indices, output_dir)
    
    # 图6a和6b: 典型日EV负荷调度对比（2张图）
    # 修复：正确获取真实EV负荷数据
    real_ev_loads = None
    if hasattr(env_temp, 'real_ev_data') and env_temp.real_ev_data is not None:
        # 修复：根据测试集的实际日期获取对应的真实EV数据
        real_ev_loads = []
        for day_idx in range(test_dates_count):
            # 获取测试集中该天对应的实际日期
            test_date = env_temp.test_dates[day_idx]
            
            # 在real_ev_data中查找对应日期的数据
            if test_date in env_temp.real_ev_data['date_mapping']:
                real_data_idx = env_temp.real_ev_data['date_mapping'][test_date]
                real_ev_loads.extend(env_temp.real_ev_data['data'][real_data_idx])
            else:
                # 如果找不到对应日期，使用0填充
                print(f"警告：未找到日期 {test_date} 的真实EV数据")
                real_ev_loads.extend([0] * 96)
    
    # 绘制EV负荷调度对比图
    # 注意：Baseline算法不调控EV，所以Baseline等于真实EV负荷，不需要单独绘制
    if real_ev_loads is not None:
        algorithm_ev_loads = {
            # Baseline不调控EV，不需要单独显示（与真实EV一样）
            # 添加所有强化学习算法的EV负荷调度结果
            'DDQN': ddqn_ev_loads,
            'T-DDQN': t_ddqn_ev_loads,
            'PPO': ppo_ev_loads
        }
        # 只有当ablation有EV数据时才添加
        if ablation_ev_loads is not None:
            algorithm_ev_loads['Ablation'] = ablation_ev_loads
        # 传入None作为baseline_ev_loads，避免重复绘制与真实EV相同的曲线
        plot_ev_load_comparison(algorithm_ev_loads, None, real_ev_loads,
                               typical_day_dates_str, typical_day_indices, output_dir)
    else:
        print("警告：未找到真实EV数据，跳过EV负荷调度对比图")
    
    print("\n" + "="*60)
    print("✅ 所有对比图生成完成！")
    print(f"📁 图片保存位置: {output_dir}/")
    print("\n【生成的图片】")
    print("   01_训练奖励对比曲线.png")
    print("   02_典型日成本对比.png")
    print(f"   03a_典型日1光伏消纳率_{typical_day_dates_str[0]}.png")
    print(f"   03b_典型日2光伏消纳率_{typical_day_dates_str[1]}.png")
    print(f"   04a_典型日1削峰填谷_{typical_day_dates_str[0]}.png")
    print(f"   04b_典型日2削峰填谷_{typical_day_dates_str[1]}.png")
    print(f"   05a_典型日1储能SOC变化_{typical_day_dates_str[0]}.png")
    print(f"   05b_典型日2储能SOC变化_{typical_day_dates_str[1]}.png")
    print(f"   07a_典型日1储能功率曲线_{typical_day_dates_str[0]}.png")
    print(f"   07b_典型日2储能功率曲线_{typical_day_dates_str[1]}.png")
    if baseline_ev_loads is not None:
        print(f"   06a_典型日1EV负荷调度对比_{typical_day_dates_str[0]}.png")
        print(f"   06b_典型日2EV负荷调度对比_{typical_day_dates_str[1]}.png")
    print(f"\n📅 典型日期: {', '.join(typical_day_dates_str)}")
    print("="*60 + "\n")
    
    # ==================== 新增：输出典型日的详细指标数据 ====================
    print("\n" + "="*60)
    print("📊 开始输出典型日的详细指标数据...")
    print("="*60 + "\n")
    
    import pandas as pd
    
    # 准备算法列表
    algorithms = ['Baseline', 'DDQN', 'T-DDQN', 'Ablation', 'PPO']
    
    # 准备数据字典
    all_data = {
        'Baseline': {
            'loads': before_loads,
            'pvs': before_pvs,
            'socs': before_socs,
            'prices': before_prices,
            'ev_loads': baseline_ev_loads if baseline_ev_loads is not None else None
        },
        'DDQN': {
            'loads': ddqn_test_loads,
            'pvs': ddqn_test_pvs,
            'socs': ddqn_test_socs,
            'prices': ddqn_prices,
            'ev_loads': ddqn_ev_loads
        },
        'T-DDQN': {
            'loads': t_ddqn_test_loads,
            'pvs': t_ddqn_test_pvs,
            'socs': t_ddqn_test_socs,
            'prices': t_ddqn_prices,
            'ev_loads': t_ddqn_ev_loads
        },
        'Ablation': {
            'loads': ablation_test_loads,
            'pvs': ablation_test_pvs,
            'socs': ablation_test_socs,
            'prices': ablation_prices,
            'ev_loads': ablation_ev_loads if ablation_ev_loads is not None else None
        },
        'PPO': {
            'loads': ppo_test_loads,
            'pvs': ppo_test_pvs,
            'socs': ppo_test_socs,
            'prices': ppo_prices,
            'ev_loads': ppo_ev_loads
        }
    }
    
    # 为每个典型日生成数据文件
    for day_num, (day_idx, date_str) in enumerate(zip(typical_day_indices, typical_day_dates_str), 1):
        print(f"\n处理典型日{day_num} ({date_str})...")
        
        # 提取当天数据的索引
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        
        # 创建时间列
        time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 15, 30, 45]]
        hours = [i / 4 for i in range(96)]  # 0.00, 0.25, 0.50, ..., 23.75
        
        # ========== 1. 输出逐时数据（96个时间点的详细数据）==========
        print(f"  生成逐时数据文件...")
        
        # 准备DataFrame
        df_timeseries = pd.DataFrame({
            '时间点': range(96),
            '时刻': time_labels,
            '小时': hours
        })
        
        # 添加各算法的数据
        for algo in algorithms:
            if algo not in all_data:
                continue
            
            data = all_data[algo]
            
            # 提取当天数据
            loads_day = data['loads'][start_idx:end_idx]
            pvs_day = data['pvs'][start_idx:end_idx]
            socs_day = data['socs'][start_idx:end_idx]
            prices_day = data['prices'][start_idx:end_idx]
            
            # 计算净负荷和光伏消纳率
            net_loads_day = [max(0, loads_day[i] - pvs_day[i]) for i in range(96)]
            pv_utils_day = []
            for i in range(96):
                if pvs_day[i] < 5.0:
                    pv_utils_day.append(0.0)
                elif pvs_day[i] >= 10.0:
                    pv_utils_day.append(min(pvs_day[i], loads_day[i]) / pvs_day[i])
                else:
                    actual_util = min(pvs_day[i], loads_day[i]) / pvs_day[i]
                    transition_factor = (pvs_day[i] - 5.0) / 5.0
                    pv_utils_day.append(actual_util * transition_factor)
            
            # 添加到DataFrame
            df_timeseries[f'{algo}_总负荷(kW)'] = loads_day
            df_timeseries[f'{algo}_光伏出力(kW)'] = pvs_day
            df_timeseries[f'{algo}_净负荷(kW)'] = net_loads_day
            df_timeseries[f'{algo}_储能SOC'] = socs_day
            df_timeseries[f'{algo}_电价(元/kWh)'] = prices_day
            df_timeseries[f'{algo}_光伏消纳率'] = pv_utils_day
            
            # 添加EV负荷（如果有）
            if data['ev_loads'] is not None and len(data['ev_loads']) > end_idx:
                ev_loads_day = data['ev_loads'][start_idx:end_idx]
                df_timeseries[f'{algo}_EV负荷(kW)'] = ev_loads_day
        
        # 保存逐时数据
        timeseries_file = os.path.join(output_dir, f'典型日{day_num}_{date_str}_逐时数据.xlsx')
        df_timeseries.to_excel(timeseries_file, index=False)
        print(f"    ✓ 保存: {timeseries_file}")
        
        # ========== 2. 输出汇总指标（每个算法的关键指标）==========
        print(f"  生成汇总指标文件...")
        
        summary_data = []
        
        for algo in algorithms:
            if algo not in all_data:
                continue
            
            data = all_data[algo]
            
            # 提取当天数据
            loads_day = data['loads'][start_idx:end_idx]
            pvs_day = data['pvs'][start_idx:end_idx]
            socs_day = data['socs'][start_idx:end_idx]
            prices_day = data['prices'][start_idx:end_idx]
            
            # 计算关键指标
            total_load_energy = sum(loads_day) * 0.25  # kWh
            total_pv_energy = sum(pvs_day) * 0.25  # kWh
            
            # 计算成本
            total_cost = 0
            for i in range(96):
                net_load = max(0, loads_day[i] - pvs_day[i])
                total_cost += net_load * prices_day[i] * 0.25
            
            # 计算平均光伏消纳率
            pv_utils_day = []
            for i in range(96):
                if pvs_day[i] < 5.0:
                    pv_utils_day.append(0.0)
                elif pvs_day[i] >= 10.0:
                    pv_utils_day.append(min(pvs_day[i], loads_day[i]) / pvs_day[i])
                else:
                    actual_util = min(pvs_day[i], loads_day[i]) / pvs_day[i]
                    transition_factor = (pvs_day[i] - 5.0) / 5.0
                    pv_utils_day.append(actual_util * transition_factor)
            avg_pv_util = sum(pv_utils_day) / len(pv_utils_day)
            
            # 峰谷差
            peak_load = max(loads_day)
            valley_load = min(loads_day)
            peak_valley_diff = peak_load - valley_load
            
            # EV相关指标
            if data['ev_loads'] is not None and len(data['ev_loads']) > end_idx:
                ev_loads_day = data['ev_loads'][start_idx:end_idx]
                total_ev_charge = sum([ev for ev in ev_loads_day if ev > 0]) * 0.25  # kWh
                total_ev_discharge = sum([abs(ev) for ev in ev_loads_day if ev < 0]) * 0.25  # kWh
                max_ev_charge = max([ev for ev in ev_loads_day if ev > 0], default=0)
                max_ev_discharge = abs(min([ev for ev in ev_loads_day if ev < 0], default=0))
            else:
                total_ev_charge = 0
                total_ev_discharge = 0
                max_ev_charge = 0
                max_ev_discharge = 0
            
            # 储能相关指标
            soc_range = max(socs_day) - min(socs_day)
            avg_soc = sum(socs_day) / len(socs_day)
            
            # 添加到汇总数据
            summary_data.append({
                '算法': algo,
                '总成本(元)': round(total_cost, 2),
                '平均光伏消纳率(%)': round(avg_pv_util * 100, 2),
                '峰值负荷(kW)': round(peak_load, 2),
                '谷值负荷(kW)': round(valley_load, 2),
                '峰谷差(kW)': round(peak_valley_diff, 2),
                '总负荷电量(kWh)': round(total_load_energy, 2),
                '总光伏发电量(kWh)': round(total_pv_energy, 2),
                'EV总充电量(kWh)': round(total_ev_charge, 2),
                'EV总放电量(kWh)': round(total_ev_discharge, 2),
                'EV最大充电功率(kW)': round(max_ev_charge, 2),
                'EV最大放电功率(kW)': round(max_ev_discharge, 2),
                '储能SOC范围': round(soc_range, 3),
                '储能平均SOC': round(avg_soc, 3)
            })
        
        # 创建汇总DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # 计算相对Baseline的改善率
        if 'Baseline' in df_summary['算法'].values:
            baseline_cost = df_summary[df_summary['算法'] == 'Baseline']['总成本(元)'].values[0]
            baseline_pv_util = df_summary[df_summary['算法'] == 'Baseline']['平均光伏消纳率(%)'].values[0]
            baseline_peak_valley = df_summary[df_summary['算法'] == 'Baseline']['峰谷差(kW)'].values[0]
            
            df_summary['成本降低率(%)'] = df_summary['总成本(元)'].apply(
                lambda x: round((baseline_cost - x) / baseline_cost * 100, 2)
            )
            df_summary['光伏消纳提升(%)'] = df_summary['平均光伏消纳率(%)'].apply(
                lambda x: round(x - baseline_pv_util, 2)
            )
            df_summary['峰谷差降低率(%)'] = df_summary['峰谷差(kW)'].apply(
                lambda x: round((baseline_peak_valley - x) / baseline_peak_valley * 100, 2)
            )
        
        # 保存汇总指标
        summary_file = os.path.join(output_dir, f'典型日{day_num}_{date_str}_汇总指标.xlsx')
        df_summary.to_excel(summary_file, index=False)
        print(f"    ✓ 保存: {summary_file}")
        
        # 打印关键指标
        print(f"\n  【典型日{day_num} ({date_str}) 关键指标】")
        for _, row in df_summary.iterrows():
            print(f"    {row['算法']:10s}: 成本 {row['总成本(元)']:7.2f}元, "
                  f"光伏消纳率 {row['平均光伏消纳率(%)']:5.2f}%, "
                  f"峰谷差 {row['峰谷差(kW)']:6.2f}kW")
    
    print("\n" + "="*60)
    print("✅ 典型日详细指标数据输出完成！")
    print(f"📁 数据文件保存位置: {output_dir}/")
    print("\n【生成的数据文件】")
    for day_num, date_str in enumerate(typical_day_dates_str, 1):
        print(f"   典型日{day_num}_{date_str}_逐时数据.xlsx")
        print(f"   典型日{day_num}_{date_str}_汇总指标.xlsx")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

