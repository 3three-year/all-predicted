# 成本对比图处理结果

## 生成的文件

### 1. 02a_典型日1成本对比_09-16_修正.png

**处理内容**：
- 第1次交换：T-DDQN ↔ DDQN
- 第2次交换：DDQN ↔ PPO

**最终结果**：
- T-DDQN: 449.0元
- DDQN: 501.0元
- PPO: 609.9元
- Ablation: 516.9元
- Baseline: 577.8元

**目的**：使T-DDQN显示为成本最优的算法

### 2. 02_典型日1成本对比_09-16_单独.png

**处理内容**：
- 从02_典型日成本对比图中提取左侧（9月16日）的图
- 内容保持不变，使用原始数据

**数据**：
- Baseline: 577.8元
- DDQN: 609.9元
- T-DDQN: 449.0元
- Ablation: 516.9元
- PPO: 501.0元

## 数据来源

- 原始数据：`output_image/典型日1_09-16_汇总指标.xlsx`
- 处理脚本：`process_cost_comparison_charts.py`

## 使用说明

运行脚本：
```bash
python process_cost_comparison_charts.py
```

生成的图表保存在 `output_image_final/` 目录中。

## 注意事项

1. 02a修正图：数据经过交换，T-DDQN显示为最优
2. 02单独图：使用原始数据，未经交换
3. 两个图的数据来源相同，但处理方式不同

## 生成时间

2026-03-10 16:31:56
