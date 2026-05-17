import os
import json
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from ddqn_trainer import TDuelingDDQNTrainer

# 1. 缩减超参数搜索空间（减少候选值范围，降低每次训练的计算量）
space = [
    Real(5e-5, 5e-4, name='alpha', prior='log-uniform'),  # 缩小学习率范围
    Real(0.92, 0.97, name='gamma'),  # 缩小折扣因子范围
    Real(0.01, 0.05, name='tau'),  # 缩小目标网络更新率范围
    Integer(64, 128, name='batch_size'),  # 缩小批大小范围（减少计算量）
    Integer(20, 30, name='sequence_length'),  # 缩小序列长度（减少数据处理量）
    Categorical([128, 256], name='fc1_dim'),  # 移除大网络（减少参数规模）
    Categorical([128, 256], name='fc2_dim'),
    Categorical([4, 8], name='nhead'),  # 减少注意力头数（降低Transformer计算量）
    Integer(1, 2, name='num_layers'),  # 减少Transformer层数
    Real(5e-7, 5e-5, name='eps_dec', prior='log-uniform')  # 缩小探索率衰减范围
]

# 2. 减少贝叶斯优化迭代次数
n_calls = 20  # 从50减少到20次

# 3. 减少训练轮数
training_episodes = 100  # 从200减少到100轮

# 4. 减少测试轮数
test_episodes = 10  # 从50减少到10轮

# 5. 使用更小的网络和更短的序列长度来加速训练
@use_named_args(space)
def objective(**params):
    """贝叶斯优化目标函数"""
    print(f"测试参数: {params}")
    
    try:
        # 创建训练器
        trainer = TDuelingDDQNTrainer(
            algo="bayesian_test",
            num_days=7,
            mode='train',
            verbose=False,
            **params
        )
        
        # 训练模型
        trainer.train(episodes=training_episodes)
        
        # 测试模型
        test_results = trainer.test(episodes=test_episodes)
        
        # 计算目标函数值（负的平均奖励，因为要最小化）
        avg_reward = np.mean([result['total_reward'] for result in test_results])
        objective_value = -avg_reward
        
        print(f"平均奖励: {avg_reward:.4f}, 目标值: {objective_value:.4f}")
        
        # 保存结果
        result = {
            'params': params,
            'avg_reward': avg_reward,
            'objective_value': objective_value,
            'test_results': test_results
        }
        
        # 保存到文件
        os.makedirs('checkpoints_bayesian', exist_ok=True)
        with open(f'checkpoints_bayesian/result_{len(os.listdir("checkpoints_bayesian"))}.json', 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return objective_value
        
    except Exception as e:
        print(f"训练失败: {e}")
        return 1000.0  # 返回一个很大的值表示失败

def main():
    """主函数"""
    print("开始贝叶斯优化...")
    print(f"搜索空间大小: {len(space)}")
    print(f"优化迭代次数: {n_calls}")
    print(f"每次训练轮数: {training_episodes}")
    print(f"每次测试轮数: {test_episodes}")
    
    # 运行贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=42,
        acq_func='EI'  # 使用期望改进
    )
    
    # 获取最佳参数
    best_params = {}
    for i, dim in enumerate(space):
        best_params[dim.name] = result.x[i]
    
    print('=' * 50)
    print("贝叶斯优化完成!")
    print(f"最佳目标值: {result.fun:.4f}")
    print(f"最佳参数: {best_params}")
    print('=' * 50)

    with open('best_tdqn_params.json', 'w') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    print(f"最佳参数已保存到 best_tdqn_params.json")


if __name__ == '__main__':
    main()