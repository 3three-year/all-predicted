import argparse
import os
import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

# 导入字体配置，禁用字体警告
import font_config
from ddqn_trainer import DuelingDDQNTrainer, TDuelingDDQNTrainer, AblationTDuelingDDQNTrainer
from ppo_trainer import PPOTrainer
from baseline_trainer import BaselineTrainer
from rural_env import CountrysideEnv

def clear_output_images():
    """清除输出图片文件"""
    # 确保清理根目录的output_image文件夹
    current_dir = os.getcwd()
    if 'rural-revitalization' in current_dir:
        # 找到rural-revitalization的位置，然后获取其父目录
        parts = current_dir.split(os.sep)
        rural_index = parts.index('rural-revitalization')
        project_root = os.sep.join(parts[:rural_index + 1])
        output_dir = os.path.join(project_root, 'output_image')
    else:
        output_dir = "output_image"
    
    # 清除所有图片文件（更彻底的清理）
    import glob
    
    # 清除所有指定文件
    specific_files = [
        "load_pv_comparison.png",
        "load_pv_comparison.svg",
        "load_pv_comparison_font_fixed.png",
        "load_pv_comparison_final_fixed.png",
        "load_pv_comparison_optimized.png",
        "load_pv_comparison_optimized_english.png",
        "average_metrics_comparison.svg", 
        "pv_utilization_comparison.svg",
        "font_test.png",
        "true_peak_valley_comparison.png",
        "true_peak_valley_comparison_sunny.png",
        "true_peak_valley_comparison_rainy.png",
        "true_peak_valley_comparison_real_data.png"
    ]
    
    for image_file in specific_files:
        file_path = os.path.join(output_dir, image_file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除旧图片: {file_path}")
    
    # 清除所有以特定前缀开头的图片文件
    image_extensions = ['.png', '.svg', '.jpg', '.jpeg', '.gif', '.bmp']
    for ext in image_extensions:
        pattern = os.path.join(output_dir, f"*{ext}")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            # 只删除我们生成的图表文件
            if any(prefix in filename for prefix in ['load_pv_comparison', 'true_peak_valley_comparison', 'average_metrics', 'pv_utilization', 'soc_comparison', 'economic_analysis', 'transformer_', 'all_algorithms', 'initial_load_consistency']):
                os.remove(file_path)
                print(f"已删除旧图片: {file_path}")
    
     # 同时清除venv目录下的图片文件
    venv_output_dir = "rural-revitalization/venv/output_image"
    if os.path.exists(venv_output_dir):
        for file in os.listdir(venv_output_dir):
            if file.endswith(('.png', '.svg')):
                file_path = os.path.join(venv_output_dir, file)
                os.remove(file_path)
                print(f"已删除venv目录下的图片: {file_path}")
    
    # 清除venv根目录下的调试图片文件
    venv_root = "rural-revitalization/venv"
    if os.path.exists(venv_root):
        for file in os.listdir(venv_root):
            if file.endswith(('.png', '.svg', '.jpg', '.jpeg')) and not file.startswith('true_peak_valley_comparison'):
                file_path = os.path.join(venv_root, file)
                os.remove(file_path)
                print(f"已删除venv根目录下的调试图片: {file_path}")
    
    print("图片文件清除完成！")

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=100)  # 调试阶段使用100轮，验证代码正确后再增加到1000
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/')
args = parser.parse_args()


# 添加统一测试函数
# 在train.py中的unified_test_all_algorithms函数中，修改PPO测试结果处理部分
# 修改 train.py 中的 unified_test_all_algorithms 函数
def unified_test_all_algorithms(output_dir, num_tests=5, ckpt_dir=None, max_episodes=None):
    """统一测试所有算法，确保使用相同的数据"""
    from rural_env import CountrysideEnv
    from datetime import datetime

    # 保存当前索引以便恢复
    original_index = getattr(CountrysideEnv, 'global_test_index', 0)

    # 设置随机种子确保可重复性（使用传入的种子或默认值）
    # 注意：这里使用固定种子是为了确保不同训练轮次的测试结果具有可比性
    # 但模型权重不同，所以结果仍然会不同
    test_seed = 42  # 测试时使用固定种子确保一致性
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    # 获取测试集日期数量
    env_temp = CountrysideEnv(algo="baseline", num_days=7, mode='test')
    test_dates_count = len(env_temp.test_dates)
    
    print(f"测试集总日期数量: {test_dates_count}")
    print(f"测试集日期范围: {env_temp.test_dates[0]} 至 {env_temp.test_dates[-1]}")

    # 确保有足够的测试数据
    if test_dates_count < 7:
        raise ValueError(f"测试集数据不足，需要至少7天，当前只有{test_dates_count}天")

    # 寻找7-9月的高光伏发电测试日期范围（光伏高发期）
    print("=== 分析光伏数据，选择高光伏发电日期进行测试 ===")
    
    # 分析所有测试日期的光伏数据
    pv_analysis = []
    for i, date_str in enumerate(env_temp.test_dates):
        try:
            # 尝试解析日期字符串
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            else:
                date_obj = date_str
            
            # 获取光伏数据
            pv_idx, charge_idx = env_temp.date_to_indices[date_str]
            pv_data = np.abs(env_temp.pv_data_values[pv_idx, :])
            pv_total = np.sum(pv_data)
            pv_max = np.max(pv_data)
            
            pv_analysis.append({
                'index': i,
                'date': date_str,
                'month': date_obj.month,
                'pv_total': pv_total,
                'pv_max': pv_max
            })
        except (ValueError, AttributeError, KeyError):
            # 如果日期解析或数据获取失败，跳过
            continue
    
    # 按光伏发电总量排序
    pv_analysis.sort(key=lambda x: x['pv_total'], reverse=True)
    
    print(f"光伏数据分析完成，共分析 {len(pv_analysis)} 天数据")
    print("前10个高光伏发电日期:")
    for i, data in enumerate(pv_analysis[:10]):
        print(f"  {i+1:2d}. {data['date']} (第{data['month']}月): 总量={data['pv_total']:8.1f}kW, 峰值={data['pv_max']:6.1f}kW")
    
    # 优先选择7-9月的高光伏发电日期（光伏高发期）
    summer_high_pv = []
    other_high_pv = []
    
    for data in pv_analysis:
        if data['month'] in [7, 8, 9]:
            summer_high_pv.append(data)
        else:
            other_high_pv.append(data)
    
    print(f"找到7-9月高光伏发电日期: {len(summer_high_pv)} 天")
    print(f"其他月份高光伏发电日期: {len(other_high_pv)} 天")
    
    # 选择最优的测试日期
    if len(summer_high_pv) >= 7:
        # 优先使用7-9月的高光伏发电日期
        selected_dates = summer_high_pv[:7]
        print("使用7-9月高光伏发电日期进行测试")
    elif len(summer_high_pv) + len(other_high_pv) >= 7:
        # 7-9月不足时，补充其他月份的高光伏发电日期
        selected_dates = summer_high_pv + other_high_pv[:7-len(summer_high_pv)]
        print("使用7-9月高光伏发电日期 + 其他月份高光伏发电日期进行测试")
    else:
        # 如果高光伏发电日期不足，使用所有可用日期
        selected_dates = pv_analysis[:min(7, len(pv_analysis))]
        print("高光伏发电日期不足，使用所有可用日期")
    
    # 获取选中的日期索引
    available_indices = [data['index'] for data in selected_dates]
    
    print(f"最终选择的测试日期:")
    for i, data in enumerate(selected_dates):
        print(f"  {i+1}. {data['date']} (第{data['month']}月): 光伏总量={data['pv_total']:.1f}kW")
    
    print(f"选择的测试日期索引: {available_indices}")
    
    # 使用选中的高光伏发电日期进行测试
    if len(available_indices) < 7:
        raise ValueError(f"高光伏发电日期不足，需要至少7天，当前只有{len(available_indices)}天")
    
    # 使用选中的7个高光伏发电日期进行测试
    test_start_indices = available_indices[:7]  # 使用前7个最高光伏发电日期
    
    print(f"使用的高光伏发电日期索引: {test_start_indices}")
    print("测试日期安排:")
    for i, start_idx in enumerate(test_start_indices):
        print(f"  第{i+1}天: {env_temp.test_dates[start_idx]} (光伏总量={selected_dates[i]['pv_total']:.1f}kW)")
    
    # 使用选定的7个高光伏发电日期进行测试
    # 为了保持原有的多次测试逻辑，我们进行5次测试，每次使用不同的起始日期
    num_tests = min(5, len(available_indices))
    print(f"将进行 {num_tests} 次测试，使用选定的高光伏发电日期") 

    all_test_results = []

    # 使用选定的高光伏发电日期进行多次测试
    for test_idx in range(num_tests):
        if test_idx < len(test_start_indices):
            start_index = test_start_indices[test_idx]
            print(f"\n=== 测试 {test_idx + 1}/{num_tests}，使用高光伏发电日期 ===")
            print(f"起始索引: {start_index}")
            print(f"测试日期: {env_temp.test_dates[start_index]} (光伏总量={selected_dates[test_idx]['pv_total']:.1f}kW)")
        else:
            # 如果高光伏发电日期不足，使用其他日期
            start_index = test_start_indices[0] + test_idx
            print(f"\n=== 测试 {test_idx + 1}/{num_tests}，使用补充日期 ===")
            print(f"起始索引: {start_index}")
            print(f"测试日期: {env_temp.test_dates[start_index]}")

        # 测试各算法（传递正确的start_index）
        baseline_trainer = BaselineTrainer(output_dir)
        baseline_results = baseline_trainer.test(test_index=start_index)

        # 使用传入的参数或默认值
        if ckpt_dir is None:
            ckpt_dir = args.ckpt_dir
        if max_episodes is None:
            max_episodes = args.max_episodes
            
        print(f"🔧 使用参数: ckpt_dir={ckpt_dir}, max_episodes={max_episodes}")

        # 使用已经训练好的模型进行测试
        ddqn_trainer = DuelingDDQNTrainer(ckpt_dir, max_episodes, output_dir)
        ddqn_results = ddqn_trainer.test(test_index=start_index)

        # 使用已经训练好的T-DDQN模型进行测试
        t_ddqn_trainer = TDuelingDDQNTrainer(ckpt_dir, max_episodes, output_dir)
        t_ddqn_results = t_ddqn_trainer.test(test_index=start_index)

        # 使用已经训练好的PPO模型进行测试
        ppo_trainer = PPOTrainer(max_episodes, output_dir)
        ppo_results = ppo_trainer.test(test_index=start_index)

        # 使用已经训练好的消融实验模型进行测试
        ablation_trainer = AblationTDuelingDDQNTrainer(ckpt_dir, max_episodes, output_dir)
        ablation_results = ablation_trainer.test(test_index=start_index)

        # 验证数据一致性
        pv_data_list = [
            baseline_results[1],  # 光伏数据
            ddqn_results[1],
            t_ddqn_results[1],
            ppo_results[1],
            ablation_results[1]
        ]

        algorithm_names = ['基线', 'DDQN', 'T-DuelingDDQN', 'PPO', '消融实验']
        consistent = validate_pv_consistency(pv_data_list, algorithm_names, output_dir)

        if not consistent:
            print(f"警告: 数据不一致!")

        # 保存当前测试结果
        all_test_results.append({
            'start_index': start_index,
            'dates': env_temp.test_dates[start_index:start_index + 7],
            'baseline': baseline_results,
            'ddqn': ddqn_results,
            't_ddqn': t_ddqn_results,
            'ppo': ppo_results,
            'ablation': ablation_results,
            'consistent': consistent
        })

    # 恢复原始索引
    CountrysideEnv.global_test_index = original_index

    # 计算平均指标
    avg_results = calculate_average_metrics(all_test_results)
    
    # 保存测试结果到pickle文件，供real_data_peak_valley_analysis.py使用
    test_results_data = {
        'all_test_results': all_test_results,
        'avg_results': avg_results,
        'test_dates': env_temp.test_dates,
        'test_start_indices': test_start_indices
    }
    
    # 保存到output_image目录
    os.makedirs(output_dir, exist_ok=True)
    test_results_file = os.path.join(output_dir, "latest_test_results.pkl")
    with open(test_results_file, 'wb') as f:
        pickle.dump(test_results_data, f)
    print(f"✓ 已保存测试结果到: {test_results_file}")

    return all_test_results, avg_results


def calculate_average_metrics(all_test_results):
    """计算单次测试的3天指标（与削峰填谷对比图保持一致）"""
    algorithms = ['baseline', 'ddqn', 't_ddqn', 'ppo', 'ablation']
    metrics = {}
    
    # 定义前3天的数据长度（3天 × 96步/天 = 288步）
    first_3_days_steps = 288

    # 只使用第一次测试的结果，与削峰填谷对比图保持一致
    first_test = all_test_results[0]
    print("✓ 使用第一次测试的3天数据计算指标，与削峰填谷对比图保持一致")

    for algo in algorithms:
        # 只取第一次测试的前3天数据
        first_3_days_loads = first_test[algo][0][:first_3_days_steps]
        first_3_days_pvs = first_test[algo][1][:first_3_days_steps]
        first_3_days_prices = first_test[algo][4][:first_3_days_steps]
        
        # 计算这次测试前3天的成本
        cost = calculate_test_cost(first_3_days_loads, first_3_days_pvs, first_3_days_prices)

        # 计算这次测试前3天的平均消纳率
        avg_utilization = np.mean(first_test[algo][3][:first_3_days_steps])

        # 记录这次测试第3天结束时的SOC（第288步的SOC）
        final_soc = first_test[algo][2][first_3_days_steps-1] if len(first_test[algo][2]) >= first_3_days_steps else 0.5

        # 单次测试结果，不需要标准差
        metrics[algo] = {
            'avg_cost': cost,
            'std_cost': 0.0,  # 单次测试，标准差为0
            'avg_utilization': avg_utilization,
            'std_utilization': 0.0,  # 单次测试，标准差为0
            'avg_final_soc': final_soc,
            'std_final_soc': 0.0  # 单次测试，标准差为0
        }

    return metrics


def calculate_test_cost(loads, pvs, prices):
    """计算单次测试的总成本"""
    total_cost = 0
    time_interval = 0.25  # 15分钟=0.25小时

    for i in range(len(loads)):
        # 净负荷 = 总负荷 - 光伏出力
        net_load = max(0, loads[i] - pvs[i])
        # 成本 = 净负荷 * 电价 * 时间间隔
        total_cost += net_load * prices[i] * time_interval

    return total_cost


# 添加验证函数
# 增强validate_pv_consistency函数
def validate_pv_consistency(pv_data_list, algorithm_names, output_dir):
    """验证所有算法的光伏数据是否一致（增强版）"""
    # 检查数据长度
    lengths = [len(data) for data in pv_data_list]
    if len(set(lengths)) > 1:
        print(f"数据长度不一致: {dict(zip(algorithm_names, lengths))}")
        return False

    # 检查数据值一致性
    consistent = True
    max_diff = 0
    diff_positions = []

    # 以基线数据为基准
    base_data = np.array(pv_data_list[0])

    for i in range(1, len(pv_data_list)):
        current_data = np.array(pv_data_list[i])

        # 计算差异
        differences = np.abs(base_data - current_data)
        max_diff = max(max_diff, np.max(differences))

        # 查找差异位置
        diff_indices = np.where(differences > 1e-5)[0]
        if len(diff_indices) > 0:
            consistent = False
            diff_positions.extend([(i, pos, differences[pos]) for pos in diff_indices[:10]])  # 记录前10个差异点

    if not consistent:
        print(f"光伏数据不一致! 最大差异: {max_diff:.6f}")
        print("前10个差异点:")
        for algo_idx, pos, diff in diff_positions[:10]:
            print(f"  {algorithm_names[algo_idx]} 在位置 {pos}: 差异={diff:.6f}")

        # 绘制差异图
        plt.figure(figsize=(12, 8))
        time_steps = np.arange(len(base_data))

        for i, (data, name) in enumerate(zip(pv_data_list, algorithm_names)):
            plt.plot(time_steps, data, label=name, alpha=0.7)

        plt.title("光伏数据对比")
        plt.xlabel("时间步")
        plt.ylabel("光伏出力 (kW)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "pv_data_comparison.png"), dpi=300)
        plt.close()

    return consistent

def main():
    # 设置动态随机种子，确保每次训练结果不同
    import time
    current_seed = int(time.time()) % 10000  # 使用时间戳作为种子
    print(f"使用随机种子: {current_seed}")
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    
    # 清除旧的图片文件
    clear_output_images()
    
    # 确保output_dir指向正确的路径
    current_dir = os.getcwd()
    if 'rural-revitalization' in current_dir and 'venv' in current_dir:
        # 如果在rural-revitalization/venv目录下，需要回到项目根目录的output_image
        # 找到rural-revitalization的位置，然后获取其父目录
        parts = current_dir.split(os.sep)
        rural_index = parts.index('rural-revitalization')
        project_root = os.sep.join(parts[:rural_index + 1])
        output_dir = os.path.join(project_root, 'output_image')
    else:
        # 如果在项目根目录下
        output_dir = "output_image"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录设置为: {os.path.abspath(output_dir)}")
    # 清理所有旧图像文件（新增代码），包括true_peak_valley_comparison.png
    for file in glob.glob(os.path.join(output_dir, "*.png")):
        try:
            os.remove(file)
            print(f"已清理旧图像文件: {file}")
        except Exception as e:
            print(f"清理图像文件失败 {file}: {str(e)}")

    # 保留原有的注意力文件清理
    for file in glob.glob("output_image/t_ddqn_attention_*.npy"):
        try:
            os.remove(file)
            print(f"已清理旧文件: {file}")
        except Exception as e:
            print(f"清理文件失败 {file}: {str(e)}")

    # 确保每次运行使用相同的测试数据
    CountrysideEnv.test_data_index = 0  # 重置测试数据索引

    # 1. 训练和测试DuelingDDQN算法
    ddqn_trainer = DuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    ddqn_rewards, ddqn_avg_rewards, ddqn_loads, ddqn_pvs, ddqn_hour_to_loads, ddqn_hour_to_pvs = ddqn_trainer.train()
    #ddqn_test_loads, ddqn_test_pvs, ddqn_test_socs, ddqn_test_pv_util, ddqn_prices = ddqn_trainer.test()

    # T-DuelingDDQN训练
    print("\n" + "=" * 50 + "\n开始T-Duelingdqn实验\n" + "=" * 50)
    t_ddqn_trainer = TDuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    t_ddqn_rewards, t_ddqn_avg_rewards, t_ddqn_loads, t_ddqn_pvs, t_ddqn_hour_to_loads, t_ddqn_hour_to_pvs = t_ddqn_trainer.train()
    (
        t_ddqn_test_loads,
        t_ddqn_test_pvs,
        t_ddqn_test_socs,
        t_ddqn_test_pv_util,
        t_ddqn_prices,
        t_ddqn_base_loads,
        t_ddqn_ev_loads,
    ) = t_ddqn_trainer.test(test_index=0)

    # 新增：训练和测试消融实验（无自注意力的Transformer）
    print("\n" + "=" * 50 + "\n开始消融实验（无自注意力Transformer）\n" + "=" * 50)
    ablation_trainer = AblationTDuelingDDQNTrainer(args.ckpt_dir, args.max_episodes, output_dir)
    ablation_rewards, ablation_avg_rewards, ablation_loads, ablation_pvs, ablation_hour_to_loads, ablation_hour_to_pvs = ablation_trainer.train()

    # 2. 训练PPO算法（移到这里，确保在统一测试前完成）
    print("\n" + "-" * 50 + "\n开始运行PPO算法训练\n" + "-" * 50)
    ppo_trainer = PPOTrainer(args.max_episodes, output_dir)
    ppo_rewards, ppo_avg_rewards = ppo_trainer.train()
    print(f"PPO Rewards length: {len(ppo_rewards)}, Avg Rewards length: {len(ppo_avg_rewards)}")

    # 4. 测试基线策略（移到统一测试前，确保所有训练完成）
    print("\n" + "-" * 50 + "\n开始基线策略测试\n" + "-" * 50)
    baseline_trainer = BaselineTrainer(output_dir)

    # 统一测试所有算法（进行多次测试）
    print("\n" + "=" * 50 + "\n开始统一测试所有算法\n" + "=" * 50)
    print(f"🔧 使用训练参数: ckpt_dir={args.ckpt_dir}, max_episodes={args.max_episodes}")
    all_test_results, avg_metrics = unified_test_all_algorithms(
        output_dir, 
        num_tests=5, 
        ckpt_dir=args.ckpt_dir, 
        max_episodes=args.max_episodes
    )

    # 使用第一次测试的结果进行绘图（保持原有绘图逻辑）
    first_test = all_test_results[0]

    # 从第一次测试结果中提取数据
    (before_loads, before_pvs, before_socs, before_pv_util, before_prices,
     ddqn_test_loads, ddqn_test_pvs, ddqn_test_socs, ddqn_test_pv_util, ddqn_prices,
     t_ddqn_test_loads, t_ddqn_test_pvs, t_ddqn_test_socs, t_ddqn_test_pv_util, t_ddqn_prices,
     ppo_test_loads, ppo_test_pvs, ppo_test_socs, ppo_test_pv_util, ppo_prices,
     ablation_test_loads, ablation_test_pvs, ablation_test_socs, ablation_test_pv_util,
     ablation_prices) = (
        first_test['baseline'][0], first_test['baseline'][1], first_test['baseline'][2],
        first_test['baseline'][3], first_test['baseline'][4],
        first_test['ddqn'][0], first_test['ddqn'][1], first_test['ddqn'][2],
        first_test['ddqn'][3], first_test['ddqn'][4],
        first_test['t_ddqn'][0], first_test['t_ddqn'][1], first_test['t_ddqn'][2],
        first_test['t_ddqn'][3], first_test['t_ddqn'][4],
        first_test['ppo'][0], first_test['ppo'][1], first_test['ppo'][2],
        first_test['ppo'][3], first_test['ppo'][4],
        first_test['ablation'][0], first_test['ablation'][1], first_test['ablation'][2],
        first_test['ablation'][3], first_test['ablation'][4]
    )

    # ============================================================
    # 新的清晰绘图系统 - 只生成4张核心对比图
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 开始生成清晰的强化学习结果对比图（4张核心图）")
    print("=" * 60 + "\n")
    
    # 导入新的绘图工具
    from rl_plot_utils import (
        plot_training_rewards_comparison,
        plot_test_cost_comparison,
        plot_test_pv_utilization_comparison,
        plot_test_peak_valley_comparison
    )
    
    # ========== 1. 收集训练奖励数据 ==========
    print("📈 收集训练奖励数据...")
    algorithm_rewards = {
        'DDQN': {
            'rewards': ddqn_rewards,
            'avg_rewards': ddqn_avg_rewards
        },
        'T-DDQN': {
            'rewards': t_ddqn_rewards,
            'avg_rewards': t_ddqn_avg_rewards
        },
        'PPO': {
            'rewards': ppo_rewards,
            'avg_rewards': ppo_avg_rewards
        }
    }
    
    # ========== 2. 收集测试数据（只收集2个典型日） ==========
    print("📊 收集测试数据（2个典型日）...")
    
    # 获取测试日期（CountrysideEnv已在文件顶部导入）
    env_temp = CountrysideEnv(algo="baseline", num_days=1, mode='test')
    all_test_dates = env_temp.test_dates
    
    # 选择2个典型日：09-15 和 09-25（在测试集范围内）
    # 测试集范围：2022-09-12 至 2022-09-30
    # 09-15是索引3，09-25是索引13
    typical_day_indices = []
    typical_day_dates_str = []
    
    # 查找09-15和09-25的索引
    target_dates = ['2022-09-15', '2022-09-25']
    for target in target_dates:
        for idx, date in enumerate(all_test_dates):
            date_str = str(date)
            if target in date_str:
                typical_day_indices.append(idx)
                typical_day_dates_str.append(date_str[5:10])
                break
    
    # 如果找不到目标日期，使用索引3和13
    if len(typical_day_indices) < 2:
        typical_day_indices = [3, 13] if len(all_test_dates) >= 14 else [0, min(5, len(all_test_dates)-1)]
        typical_day_dates_str = [str(all_test_dates[i])[5:10] for i in typical_day_indices]
    
    print(f"选择的典型日: {typical_day_dates_str}")
    print(f"对应索引: {typical_day_indices}")
    
    # 计算每个典型日的成本、光伏消纳率
    algorithm_costs = {'Baseline': [], 'DDQN': [], 'T-DDQN': [], 'PPO': []}
    algorithm_pv_utils = {'Baseline': [], 'DDQN': [], 'T-DDQN': [], 'PPO': []}
    algorithm_loads = {'Baseline': before_loads, 'DDQN': ddqn_test_loads, 
                       'T-DDQN': t_ddqn_test_loads, 'PPO': ppo_test_loads}
    
    # 按典型日计算指标
    for day_idx in typical_day_indices:
        start_idx = day_idx * 96
        end_idx = start_idx + 96
        
        # 检查数据范围
        if end_idx > len(before_loads):
            print(f"警告：第{day_idx}天数据超出范围，跳过")
            continue
        
        # Baseline
        day_loads = before_loads[start_idx:end_idx]
        day_pvs = before_pvs[start_idx:end_idx]
        day_prices = before_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['Baseline'].append(cost)
        algorithm_pv_utils['Baseline'].append(pv_util)
        
        # DDQN
        day_loads = ddqn_test_loads[start_idx:end_idx]
        day_pvs = ddqn_test_pvs[start_idx:end_idx]
        day_prices = ddqn_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['DDQN'].append(cost)
        algorithm_pv_utils['DDQN'].append(pv_util)
        
        # T-DDQN
        day_loads = t_ddqn_test_loads[start_idx:end_idx]
        day_pvs = t_ddqn_test_pvs[start_idx:end_idx]
        day_prices = t_ddqn_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['T-DDQN'].append(cost)
        algorithm_pv_utils['T-DDQN'].append(pv_util)
        
        # PPO
        day_loads = ppo_test_loads[start_idx:end_idx]
        day_pvs = ppo_test_pvs[start_idx:end_idx]
        day_prices = ppo_prices[start_idx:end_idx]
        cost = calculate_test_cost(day_loads, day_pvs, day_prices)
        pv_util = np.sum(np.minimum(day_pvs, day_loads)) / (np.sum(day_pvs) + 1e-8)
        algorithm_costs['PPO'].append(cost)
        algorithm_pv_utils['PPO'].append(pv_util)
    
    # ========== 3. 生成4张清晰对比图 ==========
    print("\n🎨 开始生成图表...\n")
    
    # 图1: 训练奖励对比曲线
    plot_training_rewards_comparison(algorithm_rewards, output_dir)
    
    # 图2: 典型日成本对比
    plot_test_cost_comparison(algorithm_costs, typical_day_dates_str, output_dir)
    
    # 图3: 典型日光伏消纳率对比
    plot_test_pv_utilization_comparison(algorithm_pv_utils, typical_day_dates_str, typical_day_indices, output_dir)
    
    # 图4: 典型日削峰填谷对比
    plot_test_peak_valley_comparison(algorithm_loads, before_loads, typical_day_dates_str, output_dir, typical_day_indices)
    
    print("\n" + "=" * 60)
    print("✅ 所有清晰对比图生成完成！")
    print(f"📁 图片保存位置: {output_dir}/")
    print("   01_训练奖励对比曲线.png")
    print("   02_典型日成本对比.png")
    print("   03_典型日光伏消纳率对比.png")
    print("   04_典型日削峰填谷对比.png")
    print(f"📅 典型日期: {typical_day_dates_str}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()