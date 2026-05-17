import numpy as np
import os
from rural_env import CountrysideEnv


class BaselineTrainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def test(self, test_index=0, num_test_days=20):
        """
        测试Baseline算法
        Args:
            test_index: 测试起始索引
            num_test_days: 要测试的天数（默认20天）
        """
        # 设置固定的随机种子
        np.random.seed(42)
        
        from rural_env import CountrysideEnv
        CountrysideEnv.global_test_index = test_index
        
        # 修改：循环测试多个单日
        all_before_loads, all_before_pvs, all_before_socs = [], [], []
        all_before_pv_util, all_before_prices = [], []
        all_base_loads = []  # 新增：基础负荷（固定部分）
        all_ev_loads = []  # 新增：EV负荷数据
        all_battery_powers = []  # 新增：储能原始功率数据
        
        max_steps = 96  # 单日96步
        
        for day_idx in range(num_test_days):
            current_test_idx = test_index + day_idx
            
            # 为每一天重新创建环境（verbose=False减少日志）
            env = CountrysideEnv(
                algo="baseline", 
                num_days=1, 
                mode='test',
                test_index=current_test_idx,
                verbose=False  # 关闭verbose日志
            )
            env.test_date_index = current_test_idx
            env._ensure_consistent_test_data()
            
            # 每天的数据容器
            before_loads, before_pvs, before_socs = [], [], []
            before_pv_util, before_prices = [], []
            
            state = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < max_steps:
                step_count += 1
                
                # 记录数据
                raw_pv = env._get_pv_output()
                current_step_idx = env.current_step % 96
                # 修复：使用charge_indices正确映射到实际数据索引
                actual_charge_idx = env.charge_indices[env.current_day]
                base_load = env.charging_data.iloc[actual_charge_idx, 1 + current_step_idx]
                
                # 修复：记录总负荷（基础负荷 + 柔性负荷）
                total_flexible_power = (env.ac_load.current_power +
                                       env.ev_load.current_power +
                                       env.shiftable_power)
                total_load = base_load + total_flexible_power
                
                before_loads.append(total_load)  # 修复：记录总负荷而不是只记录基础负荷
                before_pvs.append(raw_pv)
                before_socs.append(env.battery_soc)
                before_pv_util.append(env._get_pv_utilization_ratio())
                before_prices.append(env._get_electricity_price())
                
                # 修复：直接从real_ev_data获取真实EV负荷（Baseline不调控EV）
                ev_power = 0.0
                if hasattr(env, 'real_ev_data') and env.real_ev_data is not None:
                    real_ev_load = env._get_real_ev_load_for_day(env.current_day)
                    if real_ev_load is not None:
                        ev_power = real_ev_load[current_step_idx]
                all_ev_loads.append(ev_power)
                
                # 执行动作
                if env.battery_soc > 0.6 and env._is_peak_hour():
                    action = 0
                elif env.battery_soc < 0.4 and env._is_valley_hour():
                    action = 1
                else:
                    action = -1
                
                next_state, _, done, _ = env.step(action)
                all_battery_powers.append(env.battery_power_kw)
            
            # 获取基础负荷（从环境历史记录中）
            if hasattr(env, 'base_load_history') and len(env.base_load_history) > 0:
                all_base_loads.extend(env.base_load_history)
            else:
                # 如果没有记录，使用charging_values作为基础负荷
                all_base_loads.extend(before_loads)
            
            # 将当天数据添加到总结果中
            all_before_loads.extend(before_loads)
            all_before_pvs.extend(before_pvs)
            all_before_socs.extend(before_socs)
            all_before_pv_util.extend(before_pv_util)
            all_before_prices.extend(before_prices)
            
            if (day_idx + 1) % 10 == 0:
                print(f"  已完成 {day_idx + 1}/{num_test_days} 天测试", flush=True)
        
        print(f"\n[DONE] Completed {num_test_days} days of testing")
        print(f"   总步数: {len(all_before_loads)} (预期: {num_test_days * 96})")
        
        np.save(os.path.join(self.output_dir, "before_loads.npy"), all_before_loads)
        np.save(os.path.join(self.output_dir, "before_pvs.npy"), all_before_pvs)
        np.save(os.path.join(self.output_dir, "before_socs.npy"), all_before_socs)
        np.save(os.path.join(self.output_dir, "before_pv_util.npy"), all_before_pv_util)
        np.save(os.path.join(self.output_dir, "before_prices.npy"), all_before_prices)
        np.save(os.path.join(self.output_dir, "base_loads.npy"), all_base_loads)  # 新增：保存基础负荷
        np.save(os.path.join(self.output_dir, "baseline_ev_loads.npy"), all_ev_loads)  # 新增：保存EV负荷
        np.save(os.path.join(self.output_dir, "baseline_battery_powers.npy"), all_battery_powers)

        return all_before_loads, all_before_pvs, all_before_socs, all_before_pv_util, all_before_prices, all_base_loads, all_ev_loads, all_battery_powers
