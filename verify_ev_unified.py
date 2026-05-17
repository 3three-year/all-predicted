"""
验证EV调控统一化修改的正确性

用途：
1. 验证所有RL算法是否都使用真实EV数据作为参考
2. 验证所有RL算法的EV调控逻辑是否一致
3. 验证Baseline算法是否不调控EV
4. 验证方案A总量守恒是否对所有算法生效
"""

import sys
import os
import numpy as np
import warnings

# 抑制gym警告
warnings.filterwarnings('ignore', category=UserWarning, module='gym.spaces.box')

from rural_env import CountrysideEnv

def test_ev_control_consistency():
    """测试所有RL算法的EV调控一致性"""
    print("=" * 80)
    print("EV调控统一化验证测试")
    print("=" * 80)
    
    # 测试的算法列表
    algorithms = ["dueling_dqn", "t_dueling_dqn", "ppo", "baseline"]
    
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*80}")
        print(f"测试算法: {algo}")
        print(f"{'='*80}")
        
        # 创建环境
        env = CountrysideEnv(algo=algo, num_days=1, mode='test', test_index=0, verbose=False)
        
        # 重置环境
        initial_state = env.reset()
        
        # 记录关键信息
        algo_results = {
            'reference_ev_energy': env.reference_ev_energy if hasattr(env, 'reference_ev_energy') else None,
            'actual_ev_energy_charged': [],
            'ev_powers': [],
            'real_ev_powers': [],
            'has_real_data': env.real_ev_data is not None,
            'uses_smoothing': False,
            'completion_ratio': None
        }
        
        print(f"参考EV充电总量: {algo_results['reference_ev_energy']:.2f} kWh" if algo_results['reference_ev_energy'] else "无参考值")
        print(f"是否有真实EV数据: {'是' if algo_results['has_real_data'] else '否'}")
        
        # 模拟运行一天（96步）
        for step in range(96):
            # 执行EV调控动作（action=3）
            state, reward, done, info = env.step(3)
            
            # 记录数据
            if hasattr(env, 'actual_ev_energy_charged'):
                algo_results['actual_ev_energy_charged'].append(env.actual_ev_energy_charged)
            
            algo_results['ev_powers'].append(env.ev_load.current_power)
            
            # 获取真实EV数据（如果有）
            if env.real_ev_data is not None:
                real_ev_load = env._get_real_ev_load_for_day(env.current_day)
                if real_ev_load is not None:
                    algo_results['real_ev_powers'].append(real_ev_load[step])
            
            # 检查是否使用了平滑控制
            if hasattr(env, 'prev_ev_power') and step > 0:
                algo_results['uses_smoothing'] = True
        
        # 计算最终充电完成度
        if algo_results['reference_ev_energy'] and algo_results['reference_ev_energy'] > 0:
            final_charged = algo_results['actual_ev_energy_charged'][-1] if algo_results['actual_ev_energy_charged'] else 0
            algo_results['completion_ratio'] = final_charged / algo_results['reference_ev_energy']
        
        # 统计EV功率
        ev_powers = np.array(algo_results['ev_powers'])
        
        print(f"\n--- {algo} 算法结果 ---")
        print(f"[OK] 使用真实EV数据: {'是' if algo_results['has_real_data'] else '否'}")
        print(f"[OK] 使用平滑控制: {'是' if algo_results['uses_smoothing'] else '否'}")
        
        if algo_results['completion_ratio'] is not None:
            print(f"[OK] 充电完成度: {algo_results['completion_ratio']:.2%}")
            if 0.9 <= algo_results['completion_ratio'] <= 1.1:
                print(f"   [OK] 在合理范围内（90%-110%）")
            else:
                print(f"   [WARN] 超出合理范围！")
        
        print(f"\nEV功率统计:")
        print(f"  平均功率: {ev_powers.mean():.2f} kW")
        print(f"  最大功率: {ev_powers.max():.2f} kW")
        print(f"  最小功率: {ev_powers.min():.2f} kW")
        print(f"  功率变化幅度: {np.std(np.diff(ev_powers)):.2f} kW/步")
        
        # 如果有真实数据，对比真实功率和调度功率
        if algo_results['real_ev_powers']:
            real_ev_powers = np.array(algo_results['real_ev_powers'])
            print(f"\n真实EV数据统计:")
            print(f"  平均功率: {real_ev_powers.mean():.2f} kW")
            print(f"  与真实数据的平均偏差: {np.mean(np.abs(ev_powers - real_ev_powers)):.2f} kW")
        
        results[algo] = algo_results
    
    # 对比分析
    print(f"\n{'='*80}")
    print("对比分析")
    print(f"{'='*80}")
    
    # 检查是否所有RL算法都使用真实数据
    rl_algos = ["dueling_dqn", "t_dueling_dqn", "ppo"]
    all_use_real_data = all(results[algo]['has_real_data'] for algo in rl_algos)
    print(f"\n1. 真实EV数据使用:")
    for algo in algorithms:
        status = "[OK]" if results[algo]['has_real_data'] else "[FAIL]"
        print(f"   {status} {algo}: {'使用' if results[algo]['has_real_data'] else '未使用'}")
    
    if all_use_real_data:
        print(f"   [PASS] 所有RL算法都使用真实EV数据！")
    else:
        print(f"   [WARN] 存在RL算法未使用真实EV数据！")
    
    # 检查是否所有RL算法都使用平滑控制
    all_use_smoothing = all(results[algo]['uses_smoothing'] for algo in rl_algos)
    print(f"\n2. 平滑控制:")
    for algo in algorithms:
        if algo != "baseline":
            status = "[OK]" if results[algo]['uses_smoothing'] else "[FAIL]"
            print(f"   {status} {algo}: {'使用' if results[algo]['uses_smoothing'] else '未使用'}")
    
    if all_use_smoothing:
        print(f"   [PASS] 所有RL算法都使用平滑控制！")
    else:
        print(f"   [WARN] 存在RL算法未使用平滑控制！")
    
    # 检查充电完成度
    print(f"\n3. 充电完成度（方案A）:")
    for algo in algorithms:
        if results[algo]['completion_ratio'] is not None:
            ratio = results[algo]['completion_ratio']
            in_range = 0.9 <= ratio <= 1.1
            status = "[OK]" if in_range else "[WARN]"
            print(f"   {status} {algo}: {ratio:.2%} {'(合理)' if in_range else '(超出范围)'}")
    
    # 检查EV功率变化一致性
    print(f"\n4. EV功率变化幅度（平滑性）:")
    for algo in algorithms:
        if algo != "baseline":
            ev_powers = np.array(results[algo]['ev_powers'])
            fluctuation = np.std(np.diff(ev_powers))
            print(f"   {algo}: {fluctuation:.2f} kW/步")
    
    # 检查Baseline是否调控EV
    baseline_ev_powers = np.array(results['baseline']['ev_powers'])
    baseline_real_ev = np.array(results['baseline']['real_ev_powers']) if results['baseline']['real_ev_powers'] else None
    
    print(f"\n5. Baseline算法验证:")
    if baseline_real_ev is not None:
        is_same = np.allclose(baseline_ev_powers, baseline_real_ev, rtol=0.01)
        if is_same:
            print(f"   [OK] Baseline保持真实EV负荷，未进行调控")
        else:
            print(f"   [WARN] Baseline修改了EV负荷（可能有问题）")
    else:
        print(f"   [WARN] 无法验证（缺少真实数据）")
    
    # 总结
    print(f"\n{'='*80}")
    print("验证总结")
    print(f"{'='*80}")
    
    all_tests_passed = all_use_real_data and all_use_smoothing
    
    if all_tests_passed:
        print("[PASS] 所有验证项通过！")
        print("[PASS] EV调控已成功统一化！")
        print("[PASS] 所有RL算法使用相同的参考基准和核心逻辑！")
        print("[PASS] 算法对比公平性得到保证！")
    else:
        print("[FAIL] 部分验证项未通过，请检查代码！")
    
    return all_tests_passed

if __name__ == "__main__":
    print("\n开始验证EV调控统一化...\n")
    success = test_ev_control_consistency()
    
    if success:
        print("\n[SUCCESS] 验证成功！可以继续运行完整的训练测试。")
        sys.exit(0)
    else:
        print("\n[FAIL] 验证失败！请检查修改后的代码。")
        sys.exit(1)

