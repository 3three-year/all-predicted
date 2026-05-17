import numpy as np
import torch as T
import os
import json  # 添加缺失的导入
import torch.optim as optim  # 添加这一行
from tqdm import tqdm
from collections import deque  # 添加这行
from DRL import DuelingDDQN
from DRL import TDuelingDDQN
from DRL import AblationTDuelingDDQN
from rural_env import CountrysideEnv
from utils import plot_daily_profile, plot_combined_curves, plot_soc_comparison, plot_pv_utilization_comparison


class DuelingDDQNTrainer:
    def __init__(self, ckpt_dir, max_episodes, output_dir):
        self.ckpt_dir = ckpt_dir
        self.max_episodes = max_episodes
        self.output_dir = output_dir
        self.verbose = False  # 添加这一行
        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        # 修改：使用1天调度而不是7天（verbose=False减少日志）
        env_train = CountrysideEnv(algo="dueling_dqn", num_days=1, mode='train', verbose=False)
        env_val = CountrysideEnv(algo="dueling_dqn", num_days=1, mode='val', verbose=False)
        agent = DuelingDDQN(
            alpha=8e-4,  # 降低学习率，使学习更慢
            state_dim=env_train.state_dim,
            action_dim=env_train.action_dim,
            fc1_dim=128,  # 减小网络维度，降低学习能力
            fc2_dim=128,
            ckpt_dir=self.ckpt_dir,
            gamma=0.95,  # 降低折扣因子，更不重视长期效果
            tau=0.01,
            eps_dec=1e-5,
            max_size=100000,
            batch_size=32  # 减小批次大小，降低学习效率
        )

        global_rewards, global_avg_rewards = [], []
        global_hourly_loads, global_hourly_pvs = [], []
        # 修改：1天=24小时
        hour_to_loads, hour_to_pvs = [[] for _ in range(24)], [[] for _ in range(24)]
        global_socs, global_pv_utilization = [], []

        for episode in tqdm(range(self.max_episodes)):
            state = env_train.reset()
            total_reward = 0
            done = False
            while not done:
                # 修改：1天内的小时（0-23）
                hour = (env_train.current_day * 24) + (env_train.current_step // 4)
                if 0 <= hour < 24:
                    total_load = env_train.load_demand + env_train.ac_load.current_power + env_train.ev_load.current_power + env_train.shiftable_power
                    global_hourly_loads.append(total_load)
                    global_hourly_pvs.append(env_train._get_pv_output())
                    hour_to_loads[hour].append(total_load)
                    hour_to_pvs[hour].append(env_train._get_pv_output())
                    global_socs.append(env_train.battery_soc)
                    global_pv_utilization.append(env_train._get_pv_utilization_ratio())
                action = agent.choose_action(state)
                next_state, reward, done, _ = env_train.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                total_reward += reward
                state = next_state
            global_rewards.append(total_reward)
            avg_reward = np.mean(global_rewards[-100:]) if len(global_rewards) >= 100 else np.mean(global_rewards)
            global_avg_rewards.append(avg_reward)
            print(f'DDQN训练 Episode: {episode + 1}, 总奖励: {total_reward:.2f}, 平均奖励: {avg_reward:.2f}')
            if episode % 10 == 0:
                val_total_reward = 0
                val_state = env_val.reset()
                val_done = False
                while not val_done:
                    val_action = agent.choose_action(val_state, isTrain=False)
                    val_state, val_reward, val_done, _ = env_val.step(val_action)
                    val_total_reward += val_reward
                print(f'验证集奖励: {val_total_reward:.2f}')
            if (episode + 1) % 50 == 0 or (episode + 1) == self.max_episodes:
                agent.save_models(episode + 1)

        np.save(os.path.join(self.output_dir, "ddqn_rewards.npy"), global_rewards)
        #plot_soc_comparison(global_socs, [], [], os.path.join(self.output_dir, "ddqn_soc_pv.png"))
        #plot_combined_curves(
            #global_rewards, global_avg_rewards,
           # global_hourly_loads, global_hourly_pvs,
          #  hour_to_loads, hour_to_pvs,
         #   filename=os.path.join(self.output_dir, "ddqn_combined.png")
        #)
        average_loads = [np.mean(hour_to_loads[h]) for h in range(24)]
        average_pvs = [np.mean(hour_to_pvs[h]) for h in range(24)]
        plot_daily_profile(average_loads, average_pvs, os.path.join(self.output_dir, "ddqn_daily_profile.png"))

        return global_rewards, global_avg_rewards, global_hourly_loads, global_hourly_pvs, hour_to_loads, hour_to_pvs

    def test(self, test_index=0, num_test_days=20):
        """
        修改：测试多个单日，而不是连续7天
        
        Args:
            test_index: 测试集起始索引
            num_test_days: 测试天数（默认20天）
        
        Returns:
            每天的测试结果列表
        """
        # 设置固定的随机种子
        np.random.seed(42)
        T.manual_seed(42)

        from rural_env import CountrysideEnv
        
        # 创建环境（修改：使用1天而不是7天）
        env_ddqn = CountrysideEnv(
            algo="dueling_dqn",
            num_days=1,
            mode='test',
            test_index=test_index,
            enable_ev_fallback=True
        )
        
        if not hasattr(self, 'state_dim'):
            self.state_dim = env_ddqn.observation_space.shape[0]
        if not hasattr(self, 'action_dim'):
            self.action_dim = env_ddqn.action_dim

        # 初始化智能体
        agent_ddqn = DuelingDDQN(
            alpha=0.0003,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            fc1_dim=256,
            fc2_dim=256,
            ckpt_dir=self.ckpt_dir,
            gamma=0.99,
            tau=0.01,
            eps_dec=1e-5,
            max_size=100000,
            batch_size=128
        )

        # 加载模型
        eval_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_eval'), 'DuelingDDQN_q_eval_')
        agent_ddqn.q_eval.load_checkpoint(eval_path)
        target_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_target'), 'DuelingDDQN_Q_target_')
        agent_ddqn.q_target.load_checkpoint(target_path)
        agent_ddqn.epsilon = 0.0  # 关闭探索

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
            # 使用统一的测试索引
            current_test_idx = test_index + day_idx
            CountrysideEnv.global_test_index = current_test_idx
            
            # 重新创建环境以测试新的一天（verbose=False减少日志）
            env_ddqn = CountrysideEnv(
                algo="dueling_dqn",
                num_days=1,
                mode='test',
                test_index=current_test_idx,
                verbose=False,
                enable_ev_fallback=True
            )
            env_ddqn._ensure_consistent_test_data()
            
            # 每日测试数据
            daily_loads, daily_pvs, daily_socs, daily_pv_util, daily_prices = [], [], [], [], []
            daily_ev_loads = []  # 新增：每日EV负荷
            daily_battery_powers = []  # 新增：每日储能功率
            
            state = env_ddqn.reset()
            if state is None:
                print(f"Warning: 测试日{day_idx+1}数据reset失败，跳过")
                continue

            # 统一柔性负荷初始状态
            env_ddqn.ac_load.current_temp = 25.0
            env_ddqn.ev_load.soc = 0.6
            env_ddqn.shiftable_power = 0.0
            env_ddqn.prev_soc = env_ddqn.battery_soc

            done = False
            max_steps = 96  # 修改：1天=96步
            step_count = 0
            
            while not done:
                step_count += 1
                # 记录当前状态数据
                raw_pv = env_ddqn._get_pv_output()
                total_flexible_power = (env_ddqn.ac_load.current_power +
                                        env_ddqn.ev_load.current_power +
                                        env_ddqn.shiftable_power)
                current_day_idx = env_ddqn.test_date_index % len(env_ddqn.charging_values)
                current_step_idx = env_ddqn.current_step % 96
                real_base_load = env_ddqn.charging_values[current_day_idx][current_step_idx]
                total_load = real_base_load + total_flexible_power
                
                daily_loads.append(total_load)
                daily_pvs.append(raw_pv)
                daily_socs.append(env_ddqn.battery_soc)
                daily_pv_util.append(env_ddqn._get_pv_utilization_ratio())
                daily_prices.append(env_ddqn._get_electricity_price())
                # 新增：记录EV负荷（调度后的实际值）
                daily_ev_loads.append(env_ddqn.ev_load.current_power)
                
                # 执行动作
                if step_count == 1:
                    action = -1
                else:
                    action = agent_ddqn.choose_action(state, isTrain=False)
                next_state, _, done, _ = env_ddqn.step(action)
                daily_battery_powers.append(env_ddqn.battery_power_kw)
                
                if step_count >= max_steps:
                    done = True
                if next_state is None:
                    break
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
        np.save(os.path.join(self.output_dir, "ddqn_ev_loads.npy"), all_ev_loads)
        np.save(os.path.join(self.output_dir, "ddqn_battery_powers.npy"), all_battery_powers)
        print(f"   已保存DDQN的EV负荷数据: {len(all_ev_loads)}个时间点")
        
        return (all_test_results['loads'], all_test_results['pvs'], 
                all_test_results['socs'], all_test_results['pv_util'], 
                all_test_results['prices'], all_ev_loads, all_battery_powers)

    def get_latest_checkpoint(self, ckpt_dir, prefix):
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"模型文件夹不存在：{ckpt_dir}")
        files = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)]
        if not files:
            raise FileNotFoundError(f"未找到以 {prefix} 开头的模型文件")
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        return os.path.join(ckpt_dir, files[0])

class TDuelingDDQNTrainer(DuelingDDQNTrainer):
    def __init__(self, ckpt_dir, max_episodes, output_dir, config_path='best_tdqn_params.json'):
        super().__init__(ckpt_dir, max_episodes, output_dir)
        # 优化后的参数设置，专门针对削峰填谷任务
        self.alpha = 0.0003  # 提高学习率，加快收敛
        self.gamma = 0.98    # 提高折扣因子，更重视长期奖励
        self.tau = 0.005     # 提高软更新率，加快目标网络更新
        self.batch_size = 256 # 增大批次大小，提高训练稳定性
        self.sequence_length = 32  # 增加序列长度，更好地捕获时序模式
        self.fc1_dim = 512   # 增加网络容量
        self.fc2_dim = 512   # 增加网络容量
        self.nhead = 16      # 增加注意力头数，提高表达能力
        self.num_layers = 4  # 增加层数，提高网络深度
        self.eps_dec = 5e-7  # 调整探索衰减
        
        # T-DuelingDDQN专用训练优化参数
        self.learn_interval = 1  # 每步都学习，提高学习效率
        self.target_update_interval = 50  # 更频繁地更新目标网络

        # 尝试加载最优参数
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    best_params = json.load(f)

                # 检查参数完整性
                required_keys = ['alpha', 'gamma', 'tau', 'batch_size', 'sequence_length',
                                 'fc1_dim', 'fc2_dim', 'nhead', 'num_layers', 'eps_dec']

                if all(key in best_params for key in required_keys):
                    print(f"成功加载最优超参数: {best_params}")
                    self.alpha = best_params['alpha']
                    self.gamma = best_params['gamma']
                    self.tau = best_params['tau']
                    self.batch_size = int(best_params['batch_size'])
                    self.sequence_length = int(best_params['sequence_length'])
                    self.fc1_dim = int(best_params['fc1_dim'])
                    self.fc2_dim = int(best_params['fc2_dim'])
                    self.nhead = int(best_params['nhead'])
                    self.num_layers = int(best_params['num_layers'])
                    self.eps_dec = best_params['eps_dec']
                else:
                    print("警告：最优超参数文件不完整，使用默认参数")
            else:
                print("警告：未找到最优超参数文件，使用默认参数")

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"加载超参数出错: {str(e)}，使用默认参数")
    def train(self):
        # 创建环境（verbose=False减少日志）
        env_train = CountrysideEnv(
            algo="t_dueling_dqn",
            num_days=1,
            mode='train',
            verbose=False
        )
        env_val = CountrysideEnv(
            algo="t_dueling_dqn",
            num_days=1,
            mode='val',
            verbose=False
        )
        # 保存状态维度用于测试
        self.state_dim = env_train.observation_space.shape[0]
        self.action_dim = env_train.action_dim
        # 新增：从环境中获取电池参数
        battery_params = {
            'capacity_kwh': env_train.battery_capacity_kwh,
            'max_charge_kw': env_train.max_charge_power_kw,
            'max_discharge_kw': env_train.max_discharge_power_kw
        }

        # 创建智能体（使用优化参数）
        # 创建智能体（使用优化参数）
        agent = TDuelingDDQN(
            alpha=self.alpha,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            fc1_dim=int(self.fc1_dim),
            fc2_dim=int(self.fc2_dim),
            ckpt_dir=self.ckpt_dir,
            gamma=self.gamma,  # 使用优化后的折扣因子
            tau=self.tau,  # 使用优化后的更新率
            eps_dec=1e-6,  # 更慢的探索衰减
            max_size=500000,  # 增大经验池
            batch_size=int(self.batch_size),  # 转换为 int
            nhead=int(self.nhead),  # 转换为 int
            num_layers=int(self.num_layers),  # 转换为 int
            sequence_length=int(self.sequence_length),  # 转换为 int
            battery_params=battery_params  # 传递电池参数
        )
        # 新增：学习率调度器 - 为T-DuelingDQN优化
        if self.__class__.__name__ == 'TDuelingDDQNTrainer':
            # T-DuelingDQN使用优化的学习率调度，专门针对削峰填谷任务
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                agent.q_eval.optimizer,
                T_0=150,        # 缩短重启周期，加快收敛
                T_mult=2,       # 周期倍增因子
                eta_min=5e-7    # 更小的最小学习率
            )
            
            # 添加梯度裁剪，提高训练稳定性
            import torch
            torch.nn.utils.clip_grad_norm_(agent.q_eval.parameters(), max_norm=1.0)
        else:
            # 其他算法使用原有策略
            lr_scheduler = optim.lr_scheduler.StepLR(
                agent.q_eval.optimizer,
                step_size=200,
                gamma=0.9
            )

        # 训练过程中增加注意力可视化
        global_rewards, global_avg_rewards = [], []
        global_hourly_loads, global_hourly_pvs = [], []
        hour_to_loads = [[] for _ in range(7 * 24)]
        hour_to_pvs = [[] for _ in range(7 * 24)]
        global_socs, global_pv_utilization = [], []

        # === 修复1: 增加数据收集频率 ===
        data_collection_interval = 1  # 每步收集数据（原为5）

        # === 修复2: 移除注意力临时文件保存 ===
        attention_save_interval = max(1, self.max_episodes // 10)
        state_sequences = []
        # 动态探索率调整
        exploration_decay = 0.999  # 更平缓的衰减

        for episode in tqdm(range(self.max_episodes)):
            state = env_train.reset()
            # 修复1：确保状态缓冲区正确初始化（不重置）
            if not hasattr(agent, 'state_buffer') or len(agent.state_buffer) == 0:
                agent.state_buffer = deque(maxlen=agent.sequence_length * 2)
                for _ in range(agent.sequence_length):
                    agent.state_buffer.append(np.zeros_like(state))
            total_reward = 0
            done = False
            step_count = 0  # 添加步数计数器

            # 关键修改3：确保状态缓冲区正确初始化
            # 修复2：用当前状态填充缓冲区
            while len(agent.state_buffer) < agent.sequence_length:
                agent.state_buffer.append(state.copy())
            for _ in range(agent.sequence_length):
                agent.state_buffer.append(state.copy())

            # 动态调整探索率 - 为T-DuelingDQN优化
            if self.__class__.__name__ == 'TDuelingDDQNTrainer':
                # T-DuelingDQN使用优化的探索策略，专门针对削峰填谷任务
                if episode > self.max_episodes // 6:  # 更早开始衰减
                    # 使用指数衰减，但保持更长的探索期
                    decay_rate = 0.9998
                    agent.epsilon = max(agent.eps_min, agent.epsilon * decay_rate)
                # 在训练后期增加少量探索，避免局部最优
                elif episode > self.max_episodes * 0.8:
                    agent.epsilon = max(agent.eps_min * 2, agent.epsilon)
            else:
                # 其他算法使用原有策略
                if episode > self.max_episodes // 3:
                    agent.epsilon = max(agent.eps_min, agent.epsilon * exploration_decay)

            while not done:
                # === 修复3: 确保连续数据收集 ===
                hour = (env_train.current_day * 24) + (env_train.current_step // 4)
                if 0 <= hour < 7 * 24:
                    total_load = env_train.load_demand + env_train.ac_load.current_power + env_train.ev_load.current_power + env_train.shiftable_power
                    global_hourly_loads.append(total_load)
                    global_hourly_pvs.append(env_train._get_pv_output())
                    hour_to_loads[hour].append(total_load)
                    hour_to_pvs[hour].append(env_train._get_pv_output())
                    global_socs.append(env_train.battery_soc)
                    global_pv_utilization.append(env_train._get_pv_utilization_ratio())

                # 修复状态序列收集
                if len(agent.state_buffer) >= agent.sequence_length:
                    state_list = list(agent.state_buffer)
                    state_seq = np.stack(state_list[-agent.sequence_length:])
                    state_sequences.append(state_seq)


                action = agent.choose_action(state)
                next_state, reward, done, _ = env_train.step(action)
                # 更新状态缓冲区
                agent.state_buffer.append(next_state.copy())  # 添加新的状态
                if len(agent.state_buffer) > agent.sequence_length * 2:
                    agent.state_buffer.popleft()  # 使用popleft()移除最左边的元素
                agent.remember(state, action, reward, next_state, done)

                # T-DuelingDDQN优化的学习策略
                if self.__class__.__name__ == 'TDuelingDDQNTrainer':
                    # 每步都学习，提高学习效率
                    if agent.memory.ready():
                        agent.learn()
                    # 更频繁地更新目标网络
                    if step_count % self.target_update_interval == 0:
                        agent.update_network_parameters(tau=1.0)
                else:
                    # 其他算法使用原有策略
                    if agent.memory.ready():
                        batch = agent.memory.sample_buffer()
                        if batch is not None:
                            agent.learn()

                total_reward += reward
                state = next_state
                step_count += 1

            # 新增：更新学习率
            lr_scheduler.step()

            global_rewards.append(total_reward)
            avg_reward = np.mean(global_rewards[-100:]) if len(global_rewards) >= 100 else np.mean(global_rewards)
            global_avg_rewards.append(avg_reward)

            # ... [验证和模型保存逻辑保持不变] ...

        np.save(os.path.join(self.output_dir, "t_ddqn_rewards.npy"), global_rewards)

        # ... [绘图代码保持不变] ...
        #plot_combined_curves(
            #global_rewards, global_avg_rewards,
           # global_hourly_loads, global_hourly_pvs,
          #  hour_to_loads, hour_to_pvs,
         #   filename=os.path.join(self.output_dir, "t_ddqn_combined.png")
        #)
        average_loads = [np.mean(hour_to_loads[h]) for h in range(24)]
        average_pvs = [np.mean(hour_to_pvs[h]) for h in range(24)]
        plot_daily_profile(average_loads, average_pvs, os.path.join(self.output_dir, "t_ddqn_daily_profile.png"))

        return global_rewards, global_avg_rewards, global_hourly_loads, global_hourly_pvs, hour_to_loads, hour_to_pvs

    def test(self, test_index=0, num_test_days=20):
        """
        测试T-DDQN算法
        Args:
            test_index: 测试起始索引
            num_test_days: 要测试的天数（默认20天）
        """
        # 设置固定的随机种子
        np.random.seed(42)
        T.manual_seed(42)

        from rural_env import CountrysideEnv
        # 使用统一的测试索引
        CountrysideEnv.global_test_index = test_index

        env_tdqn = CountrysideEnv(
            algo="t_dueling_dqn",
            num_days=1,  # 改为单日测试
            mode='test',
            test_index = test_index
        )

        # 确保环境使用统一数据
        env_tdqn._ensure_consistent_test_data()

        # 新增：如果没有设置state_dim和action_dim，则从环境中获取
        if not hasattr(self, 'state_dim'):
            self.state_dim = env_tdqn.observation_space.shape[0]
        if not hasattr(self, 'action_dim'):
            self.action_dim = env_tdqn.action_dim

        # 电池参数：测试时不需要，因为模型会覆盖，这里传递None
        battery_params = None

        # 使用训练时的统一参数
        agent_tdqn = TDuelingDDQN(
            alpha=self.alpha,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            fc1_dim=self.fc1_dim,
            fc2_dim=self.fc2_dim,
            ckpt_dir=self.ckpt_dir,
            gamma=0.99,
            tau=0.01,
            eps_dec=1e-6,
            max_size=100000,
            batch_size=64,
            nhead=self.nhead,
            num_layers=self.num_layers,
            sequence_length=self.sequence_length,
            battery_params=battery_params  # 传递电池参数
        )

        # 修复：T-DuelingDDQN应该加载自己的检查点，而不是DuelingDDQN的检查点
        # 查找T-DuelingDDQN专用的检查点文件
        try:
            eval_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_eval'), 'TDuelingDDQN_q_eval_')
            if eval_path is None:
                raise FileNotFoundError("未找到T-DuelingDDQN专用检查点")
            print(f"使用T-DuelingDDQN专用检查点: {eval_path}")
        except FileNotFoundError:
            # 如果没有找到T-DuelingDDQN专用检查点，使用通用检查点
            eval_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_eval'), 'DuelingDDQN_q_eval_')
            print(f"警告：未找到T-DuelingDDQN专用检查点，使用通用检查点: {eval_path}")
            print("注意：T-DuelingDDQN将使用DuelingDDQN的权重，可能影响性能")
        
        agent_tdqn.q_eval.load_checkpoint(eval_path)
        
        try:
            target_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_target'), 'TDuelingDDQN_Q_target_')
            if target_path is None:
                raise FileNotFoundError("未找到T-DuelingDDQN专用目标网络检查点")
            print(f"使用T-DuelingDDQN专用目标网络检查点: {target_path}")
        except FileNotFoundError:
            target_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_target'), 'DuelingDDQN_Q_target_')
            print(f"警告：未找到T-DuelingDDQN专用目标网络检查点，使用通用检查点: {target_path}")
        
        agent_tdqn.q_target.load_checkpoint(target_path)
        agent_tdqn.epsilon = 0.0  # 强制设为 0，关闭随机探索

        # 修改：循环测试多个单日，而不是连续7天
        all_test_results = {
            'loads': [],
            'pvs': [],
            'socs': [],
            'pv_util': [],
            'prices': [],
            'base_loads': []
        }
        all_ev_loads = []
        all_battery_powers = []
        all_battery_powers = []
        all_battery_powers = []
        all_battery_powers = []
        all_battery_powers = []  # 新增：收集储能原始功率数据
        
        max_steps = 96  # 单日96步
        
        for day_idx in range(num_test_days):
            current_test_idx = test_index + day_idx
            
            # 为每一天重新创建环境（verbose=False减少日志）
            env_tdqn = CountrysideEnv(
                algo="t_dueling_dqn",
                num_days=1,
                mode='test',
                test_index=current_test_idx,
                verbose=False
            )
            env_tdqn._ensure_consistent_test_data()
            
            # 每天的数据容器
            daily_loads, daily_pvs, daily_socs = [], [], []
            daily_pv_util, daily_prices, daily_base_loads = [], [], []
            daily_ev_loads = []
            daily_battery_powers = []
            daily_battery_powers = []
            daily_battery_powers = []
            daily_battery_powers = []
            daily_battery_powers = []
            daily_battery_powers = []  # 新增：每日储能功率
            
            state = env_tdqn.reset()
            done = False
            step_count = 0
            
            # 统一初始状态
            env_tdqn.ac_load.current_temp = 25.0
            env_tdqn.ev_load.soc = 0.6
            env_tdqn.shiftable_power = 0.0
            env_tdqn.prev_soc = env_tdqn.battery_soc
            
            while not done and step_count < max_steps:
                step_count += 1
                
                # 记录数据
                raw_pv = env_tdqn._get_pv_output()
                total_flexible_power = (env_tdqn.ac_load.current_power +
                                        env_tdqn.ev_load.current_power +
                                        env_tdqn.shiftable_power)
                current_day_idx = env_tdqn.test_date_index % len(env_tdqn.charging_values)
                current_step_idx = env_tdqn.current_step % 96
                real_base_load = env_tdqn.charging_values[current_day_idx][current_step_idx]
                total_load = real_base_load + total_flexible_power
                
                daily_loads.append(total_load)
                daily_pvs.append(raw_pv)
                daily_socs.append(env_tdqn.battery_soc)
                daily_pv_util.append(env_tdqn._get_pv_utilization_ratio())
                daily_prices.append(env_tdqn._get_electricity_price())
                daily_base_loads.append(real_base_load)
                # 新增：记录EV负荷（调度后的实际值）
                daily_ev_loads.append(env_tdqn.ev_load.current_power)
                
                # 执行动作
                if step_count == 1:
                    action = -1
                else:
                    action = agent_tdqn.choose_action(state, isTrain=False)
                next_state, _, done, _ = env_tdqn.step(action)
                daily_battery_powers.append(env_tdqn.battery_power_kw)
                state = next_state
            
            # 将当天数据添加到总结果中
            all_test_results['loads'].extend(daily_loads)
            all_test_results['pvs'].extend(daily_pvs)
            all_test_results['socs'].extend(daily_socs)
            all_test_results['pv_util'].extend(daily_pv_util)
            all_test_results['prices'].extend(daily_prices)
            all_test_results['base_loads'].extend(daily_base_loads)
            all_ev_loads.extend(daily_ev_loads)  # 新增：保存EV负荷
            all_battery_powers.extend(daily_battery_powers)
            
            if (day_idx + 1) % 10 == 0:
                print(f"  已完成 {day_idx + 1}/{num_test_days} 天测试", flush=True)
        
        print(f"\n[DONE] Completed {num_test_days} days of testing")
        print(f"   总步数: {len(all_test_results['loads'])} (预期: {num_test_days * 96})")
        
        # 保存EV负荷数据
        np.save(os.path.join(self.output_dir, "t_ddqn_ev_loads.npy"), all_ev_loads)
        np.save(os.path.join(self.output_dir, "t_ddqn_battery_powers.npy"), all_battery_powers)
        print(f"   已保存T-DDQN的EV负荷数据: {len(all_ev_loads)}个时间点")
        
        return (all_test_results['loads'], all_test_results['pvs'], 
                all_test_results['socs'], all_test_results['pv_util'],
                all_test_results['prices'], all_test_results['base_loads'], all_ev_loads, all_battery_powers)


class EnhancedTDuelingDDQNTrainer(TDuelingDDQNTrainer):
    def __init__(self, ckpt_dir, max_episodes, output_dir):
        super().__init__(ckpt_dir, max_episodes, output_dir)
        # 更平滑的课程难度级别
        self.curriculum_levels = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        self.current_level = 0
        # 动态升级阈值（初始0.7，随级别提高而增加）
        self.base_level_up_threshold = 0.7
        # 更平缓的探索衰减
        self.eps_decay = 0.9995
        self.eps_min = 0.02  # 保留少量探索

    def train(self):
        # 创建智能体（使用增强网络）
        env_train = CountrysideEnv(algo="enhanced_t_ddqn", num_days=7, mode='train', difficulty=0.1)
        agent = TDuelingDDQN(
            alpha=2e-5,  # 微调学习率
            state_dim=env_train.state_dim,
            action_dim=env_train.action_dim,
            fc1_dim=512,
            fc2_dim=512,  # 增加维度
            ckpt_dir=self.ckpt_dir,
            gamma=0.995,  # 更高的折扣因子
            tau=0.001,
            eps_dec=1e-6,
            max_size=2000000,  # 更大的经验池
            batch_size=512,  # 更大的批大小
            nhead=16,  # 更多注意力头
            num_layers=6,  # 更深的网络
            sequence_length=48  # 更长的序列
        )

        # 自适应探索策略（基于课程难度调整）
        def adaptive_exploration(ep, current_level):
            # 基础探索率随训练进行衰减
            base_eps = max(self.eps_min, 0.5 * (1 - ep / self.max_episodes))
            # 新难度级别时增加探索
            level_bonus = 0.1 if ep % 50 == 0 and current_level > 0 else 0
            return base_eps + level_bonus + 0.05 * np.random.randn()

        # 训练循环
        global_rewards, global_avg_rewards = [], []
        best_val_reward = -np.inf

        for episode in tqdm(range(self.max_episodes)):
            # 动态调整升级阈值（随当前级别提高）
            dynamic_threshold = self.base_level_up_threshold + 0.15 * (
                    self.current_level / (len(self.curriculum_levels) - 1))

            # 创建当前难度环境
            env_train = CountrysideEnv(
                algo="enhanced_t_ddqn",
                num_days=7,
                mode='train',
                difficulty=self.curriculum_levels[self.current_level]
            )

            # 设置当前探索率
            agent.epsilon = adaptive_exploration(episode, self.current_level)

            # 初始化状态缓冲区（确保完整序列）
            state = env_train.reset()
            agent.state_buffer = deque(
                [state.copy()] * agent.sequence_length,
                maxlen=agent.sequence_length * 2
            )

            total_reward = 0
            done = False
            step_count = 0

            while not done:
                # 构建完整状态序列
                state_seq = np.array(list(agent.state_buffer)[-agent.sequence_length:])

                # 选择动作
                action = agent.choose_action(state_seq)

                # 执行动作
                next_state, reward, done, _ = env_train.step(action)
                total_reward += reward

                # 更新状态缓冲区
                agent.state_buffer.append(next_state.copy())

                # 存储经验
                next_state_seq = np.array(list(agent.state_buffer)[-agent.sequence_length:])
                agent.remember(state_seq, action, reward, next_state_seq, done)

                # 学习（每4步学习一次）
                if step_count % 4 == 0:
                    agent.learn()

                step_count += 1

            # 更新全局奖励记录
            global_rewards.append(total_reward)
            avg_reward = np.mean(global_rewards[-100:]) if len(global_rewards) >= 100 else np.mean(global_rewards)
            global_avg_rewards.append(avg_reward)

            # 每10回合验证一次
            if episode % 10 == 0:
                val_total_reward = 0
                val_env = CountrysideEnv(
                    algo="enhanced_t_ddqn",
                    num_days=7,
                    mode='val',
                    difficulty=self.curriculum_levels[self.current_level]
                )
                val_state = val_env.reset()
                val_done = False

                while not val_done:
                    val_action = agent.choose_action(val_state, isTrain=False)
                    val_state, val_reward, val_done, _ = val_env.step(val_action)
                    val_total_reward += val_reward

                # 动态课程升级
                if val_total_reward > dynamic_threshold and self.current_level < len(self.curriculum_levels) - 1:
                    self.current_level += 1
                    print(
                        f"\n升级课程难度至: {self.curriculum_levels[self.current_level]} (验证奖励: {val_total_reward:.2f})")

                # 保存最佳模型
                if val_total_reward > best_val_reward:
                    best_val_reward = val_total_reward
                    agent.save_models("best")
                    print(f"保存最佳模型 (验证奖励: {val_total_reward:.2f})")

            # 定期保存模型
            if episode % 50 == 0:
                agent.save_models(episode)

        # 保存增强的注意力数据
        if hasattr(agent.q_eval, 'last_attention'):
            np.save(os.path.join(self.output_dir, "enhanced_attention.npy"),
                    agent.q_eval.last_attention.cpu().numpy())
        # 返回训练结果
        return global_rewards, global_avg_rewards, [], [], [], []

# 在文件末尾添加以下代码

class AblationTDuelingDDQNTrainer(TDuelingDDQNTrainer):
    """??????????????????????"""

    def __init__(self, ckpt_dir, max_episodes, output_dir):
        super().__init__(ckpt_dir, max_episodes, output_dir, config_path='nonexistent_abl_params.json')
        self.output_dir = os.path.join(output_dir, "ablation")
        os.makedirs(self.output_dir, exist_ok=True)
        self.alpha = 6e-4
        self.gamma = 0.96
        self.tau = 0.001
        self.batch_size = 64
        self.sequence_length = 16
        self.fc1_dim = 128
        self.fc2_dim = 128
        self.nhead = 4
        self.num_layers = 2
        self.eps_dec = 1e-5
        self.learn_interval = 2
        self.target_update_interval = 100

    def test(self, test_index=0, num_test_days=20):
        """
        ??Ablation???????EV???????
        Args:
            test_index: ??????
            num_test_days: ??????
        """
        np.random.seed(42)
        T.manual_seed(42)

        from rural_env import CountrysideEnv
        CountrysideEnv.global_test_index = test_index

        env_adqn = CountrysideEnv(
            algo="ablation_t_ddqn",
            num_days=1,
            mode='test',
            test_index=test_index,
            verbose=False
        )
        env_adqn._ensure_consistent_test_data()

        if not hasattr(self, 'state_dim'):
            self.state_dim = env_adqn.observation_space.shape[0]
        if not hasattr(self, 'action_dim'):
            self.action_dim = env_adqn.action_dim

        battery_params = {
            'capacity_kwh': env_adqn.battery_capacity_kwh,
            'max_charge_kw': env_adqn.max_charge_power_kw,
            'max_discharge_kw': env_adqn.max_discharge_power_kw
        }

        agent_adqn = AblationTDuelingDDQN(
            alpha=self.alpha,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            fc1_dim=self.fc1_dim,
            fc2_dim=self.fc2_dim,
            ckpt_dir=self.ckpt_dir,
            gamma=0.99,
            tau=0.01,
            eps_dec=1e-5,
            max_size=100000,
            batch_size=128,
            num_layers=self.num_layers,
            sequence_length=self.sequence_length,
            battery_params=battery_params
        )

        eval_path = self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_eval'), 'DuelingDDQN_q_eval_')
        agent_adqn.q_eval.load_checkpoint(eval_path)
        agent_adqn.q_target.load_checkpoint(
            self.get_latest_checkpoint(os.path.join(self.ckpt_dir, 'Q_target'), 'DuelingDDQN_Q_target_')
        )
        agent_adqn.epsilon = 0.0

        all_test_results = {
            'loads': [],
            'pvs': [],
            'socs': [],
            'pv_util': [],
            'prices': [],
            'base_loads': []
        }
        all_ev_loads = []
        all_battery_powers = []

        max_steps = 96
        for day_idx in range(num_test_days):
            current_test_idx = test_index + day_idx

            env_adqn = CountrysideEnv(
                algo="ablation_t_ddqn",
                num_days=1,
                mode='test',
                test_index=current_test_idx,
                verbose=False
            )
            env_adqn._ensure_consistent_test_data()

            daily_loads, daily_pvs, daily_socs = [], [], []
            daily_pv_util, daily_prices, daily_base_loads = [], [], []
            daily_ev_loads = []
            daily_battery_powers = []

            state = env_adqn.reset()
            done = False
            step_count = 0

            env_adqn.ac_load.current_temp = 25.0
            env_adqn.ev_load.soc = 0.6
            env_adqn.shiftable_power = 0.0
            env_adqn.prev_soc = env_adqn.battery_soc
            agent_adqn.state_buffer = deque(maxlen=agent_adqn.sequence_length * 2)
            for _ in range(agent_adqn.sequence_length):
                agent_adqn.state_buffer.append(np.zeros_like(state))

            while not done and step_count < max_steps:
                step_count += 1

                raw_pv = env_adqn._get_pv_output()
                total_flexible_power = (
                    env_adqn.ac_load.current_power +
                    env_adqn.ev_load.current_power +
                    env_adqn.shiftable_power
                )
                current_day_idx = env_adqn.test_date_index % len(env_adqn.charging_values)
                current_step_idx = env_adqn.current_step % 96
                real_base_load = env_adqn.charging_values[current_day_idx][current_step_idx]
                total_load = real_base_load + total_flexible_power

                daily_loads.append(total_load)
                daily_pvs.append(raw_pv)
                daily_socs.append(env_adqn.battery_soc)
                daily_pv_util.append(env_adqn._get_pv_utilization_ratio())
                daily_prices.append(env_adqn._get_electricity_price())
                daily_base_loads.append(real_base_load)
                daily_ev_loads.append(env_adqn.ev_load.current_power)

                if step_count == 1:
                    action = -1
                else:
                    action = agent_adqn.choose_action(state, isTrain=False)
                next_state, _, done, _ = env_adqn.step(action)
                if 'daily_battery_powers' not in locals():
                    daily_battery_powers = []
                daily_battery_powers.append(env_adqn.battery_power_kw)
                state = next_state

            all_test_results['loads'].extend(daily_loads)
            all_test_results['pvs'].extend(daily_pvs)
            all_test_results['socs'].extend(daily_socs)
            all_test_results['pv_util'].extend(daily_pv_util)
            all_test_results['prices'].extend(daily_prices)
            all_test_results['base_loads'].extend(daily_base_loads)
            all_ev_loads.extend(daily_ev_loads)
            if 'all_battery_powers' not in locals():
                all_battery_powers = []
            all_battery_powers.extend(daily_battery_powers)

            if (day_idx + 1) % 10 == 0:
                print(f"  ??? {day_idx + 1}/{num_test_days} ???", flush=True)

        print(f"\n[DONE] Completed {num_test_days} days of testing")
        print(f"   ???: {len(all_test_results['loads'])} (??: {num_test_days * 96})")
        np.save(os.path.join(self.output_dir, "ablation_ev_loads.npy"), all_ev_loads)
        np.save(os.path.join(self.output_dir, "ablation_battery_powers.npy"), all_battery_powers)
        print(f"   ???Ablation?EV????: {len(all_ev_loads)}????")

        return (
            all_test_results['loads'],
            all_test_results['pvs'],
            all_test_results['socs'],
            all_test_results['pv_util'],
            all_test_results['prices'],
            all_test_results['base_loads'],
            all_ev_loads,
            all_battery_powers
        )
