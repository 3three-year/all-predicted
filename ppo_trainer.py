import numpy as np
import torch as T
import os
from stable_baselines3 import PPO
from tqdm import tqdm
from rural_env import CountrysideEnv
from stable_baselines3.common.callbacks import BaseCallback


class PPOTrainer:
    def __init__(self, max_episodes, output_dir, save_path='checkpoints/PPO'):
        self.max_episodes = max_episodes
        self.output_dir = output_dir
        self.save_path = save_path
        self.verbose = False  # 添加这一行
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        # 修改：使用1天调度而不是7天
        env = CountrysideEnv(algo="ppo", num_days=1, mode='train', verbose=False)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=8e-4,  # 降低学习率，使学习更慢
            n_steps=1024,        # 减少步数，降低学习效率
            batch_size=128,      # 减小批次大小，降低学习效率
            gamma=0.92,          # 降低折扣因子，更不重视长期效果
            n_epochs=3,          # 减少训练轮数，降低学习效果
            clip_range=0.1,      # 降低裁剪范围，限制策略更新
            tensorboard_log=os.path.join(self.save_path, "tensorboard"),
            device='cuda' if T.cuda.is_available() else 'cpu'
        )

        # 修改：1天=96步
        total_timesteps = self.max_episodes * 96
        print(f"\n开始PPO算法训练，总时间步: {total_timesteps} (每episode {96}步)")

        # 定义回调函数记录奖励
        class RewardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.episode_avg_rewards = []
                self.current_episode_reward = 0

            def _on_step(self) -> bool:
                # 获取当前奖励
                reward = self.locals.get('rewards')[0]  # 获取第一个环境的奖励
                self.current_episode_reward += reward

                # 检查回合是否结束
                if self.locals.get('dones')[0]:
                    self.episode_rewards.append(self.current_episode_reward)

                    # 计算滑动平均奖励
                    if len(self.episode_rewards) > 100:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                    else:
                        avg_reward = np.mean(self.episode_rewards)

                    self.episode_avg_rewards.append(avg_reward)
                    self.current_episode_reward = 0  # 重置当前回合奖励
                return True

        reward_callback = RewardCallback()
        model.learn(total_timesteps=total_timesteps, callback=reward_callback)
        model.save(os.path.join(self.save_path, "ppo_model"))
        print(f"\nPPO训练完成，模型已保存至: {self.save_path}")

        return reward_callback.episode_rewards, reward_callback.episode_avg_rewards

    # 修改：测试多个单日
    def test(self, test_index=0, num_test_days=20):
        """
        修改：测试多个单日，而不是连续7天
        
        Args:
            test_index: 测试集起始索引
            num_test_days: 测试天数（默认20天）
        """
        # 设置固定的随机种子
        np.random.seed(42)

        from rural_env import CountrysideEnv

        # 加载模型
        model_path = os.path.join(self.save_path, "ppo_model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO模型文件不存在: {model_path}。请先完成PPO训练。")
        
        try:
            model = PPO.load(model_path.replace('.zip', ''))
        except (ModuleNotFoundError, ImportError) as e:
            print(f"警告：无法加载PPO模型，可能是NumPy版本兼容性问题: {e}")
            print("将重新训练PPO模型...")
            if os.path.exists(model_path):
                os.remove(model_path)
            model = self._retrain_ppo_model()
        
        # 存储所有测试天的结果
        all_test_results = {
            'loads': [],
            'pvs': [],
            'socs': [],
            'pv_util': [],
            'prices': []
        }
        all_ev_loads = []  # 新增：收集EV负荷数据
        all_battery_powers = []  # 新增：收集储能原始功率数据
        
        # 测试多个单日
        print(f"\n开始测试 {num_test_days} 个单日...")
        for day_idx in range(num_test_days):
            current_test_idx = test_index + day_idx
            CountrysideEnv.global_test_index = current_test_idx
            
            # 创建环境（修改：使用1天，verbose=False减少日志）
            env = CountrysideEnv(
                algo="ppo",
                num_days=1,
                mode='test',
                test_index=current_test_idx,
                verbose=False,
                enable_ev_fallback=True
            )
            env.test_date_index = current_test_idx
            env._ensure_consistent_test_data()
            
            # 每日测试数据
            daily_loads, daily_pvs, daily_socs, daily_pv_util, daily_prices = [], [], [], [], []
            daily_ev_loads = []  # 新增：每日EV负荷
            daily_battery_powers = []  # 新增：每日储能功率
            
            state = env.reset()
            
            # 统一柔性负荷初始状态
            env.ac_load.current_temp = 25.0
            env.ev_load.soc = 0.6
            env.shiftable_power = 0.0
            env.prev_soc = env.battery_soc
            
            done = False
            max_steps = 96  # 修改：1天=96步
            step_count = 0
            
            while not done:
                step_count += 1
                raw_pv = env._get_pv_output()
                total_flexible_power = (env.ac_load.current_power +
                                        env.ev_load.current_power +
                                        env.shiftable_power)
                # 修复：使用charge_indices正确映射到实际数据索引
                current_step_idx = env.current_step % 96
                actual_charge_idx = env.charge_indices[env.current_day]
                real_base_load = env.charging_data.iloc[actual_charge_idx, 1 + current_step_idx]
                total_load = real_base_load + total_flexible_power
                
                daily_loads.append(total_load)
                daily_pvs.append(raw_pv)
                daily_socs.append(env.battery_soc)
                daily_pv_util.append(env._get_pv_utilization_ratio())
                daily_prices.append(env._get_electricity_price())
                # 新增：记录EV负荷（调度后的实际值）
                daily_ev_loads.append(env.ev_load.current_power)
                
                # 执行动作
                if step_count == 1:
                    action = -1
                else:
                    action, _ = model.predict(state, deterministic=True)
                next_state, _, done, _ = env.step(action)
                daily_battery_powers.append(env.battery_power_kw)
                
                if step_count >= max_steps:
                    done = True
                state = next_state
            
            # 保存当天结果
            all_test_results['loads'].extend(daily_loads)
            all_test_results['pvs'].extend(daily_pvs)
            all_test_results['socs'].extend(daily_socs)
            all_test_results['pv_util'].extend(daily_pv_util)
            all_test_results['prices'].extend(daily_prices)
            all_ev_loads.extend(daily_ev_loads)  # 新增：保存EV负荷
            all_battery_powers.extend(daily_battery_powers)
            
            if (day_idx + 1) % 5 == 0:
                print(f"  已完成 {day_idx + 1}/{num_test_days} 天测试")

        print(f"\n[DONE] Completed {num_test_days} days of testing")
        print(f"   总步数: {len(all_test_results['loads'])} (预期: {num_test_days * 96})")
        
        # 保存EV负荷数据
        np.save(os.path.join(self.output_dir, "ppo_ev_loads.npy"), all_ev_loads)
        np.save(os.path.join(self.output_dir, "ppo_battery_powers.npy"), all_battery_powers)
        print(f"   已保存PPO的EV负荷数据: {len(all_ev_loads)}个时间点")
        
        return (all_test_results['loads'], all_test_results['pvs'], 
                all_test_results['socs'], all_test_results['pv_util'], 
                all_test_results['prices'], all_ev_loads, all_battery_powers)
    
    def _retrain_ppo_model(self):
        """重新训练PPO模型以解决兼容性问题"""
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from rural_env import CountrysideEnv
        
        print("开始重新训练PPO模型...")
        
        # 创建环境（修改：使用1天，verbose=False减少日志）
        env = CountrysideEnv(algo="ppo", num_days=1, mode='train', verbose=False)
        
        # 创建PPO模型
        model = PPO("MlpPolicy", env, verbose=1, device='cpu')
        
        # 快速训练（减少训练步数以节省时间）
        model.learn(total_timesteps=1000)
        
        # 保存模型
        model_path = os.path.join(self.save_path, "ppo_model")
        model.save(model_path)
        print(f"PPO模型已重新训练并保存到: {model_path}")
        
        return model
