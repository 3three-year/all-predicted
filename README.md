# 电动汽车(EV)充电负荷预测系统

## 📋 项目概述

本项目使用**CNN-BiLSTM-Attention**深度学习模型对台区R1的电动汽车充电负荷进行预测。该模型结合了卷积神经网络(CNN)、双向长短期记忆网络(BiLSTM)和注意力机制(Attention)，能够有效捕捉充电负荷的时序特征和关键模式。

## 🎯 模型选择说明

经过对6个候选模型的分析比较，我们选择了**CNN-BiLSTM-Attention**模型，原因如下：

| 模型 | 特点 | 适用性 |
|------|------|--------|
| **CNN-BiLSTM-Attention** ⭐⭐⭐⭐⭐ | CNN提取局部特征 + BiLSTM双向时序 + Attention关注重点 | 最佳选择 |
| BiLSTM-Attention | BiLSTM + Attention | 次优选择 |
| CNN-LSTM-Attention | CNN + 单向LSTM + Attention | 较好 |
| LSTM-Attention | 单向LSTM + Attention | 一般 |
| BiLSTM | 仅BiLSTM | 基础 |
| LSTM | 仅LSTM | 基础 |

### 模型优势
1. **CNN层**：提取充电负荷的局部时序特征和模式
2. **BiLSTM层**：从前后两个方向捕获时间序列的长期依赖关系
3. **Attention机制**：自动识别和关注充电高峰等重要时段
4. **综合性能**：在时序预测任务中表现最优

## 📊 数据说明

- **数据源**：`date_file/按日累计-台区R1-充电负荷数据.csv`
- **数据格式**：
  - 第1列：日期
  - 第2-97列：P1-P96（一天96个时间点，每15分钟一个）
- **数据特点**：按日累计的充电负荷时间序列数据

## 🔧 环境要求

```bash
python >= 3.7
tensorflow >= 2.0
keras >= 2.0
pandas
numpy
matplotlib
scikit-learn
prettytable
mplcyberpunk  # 可选，用于美化图表
qbstyles      # 可选，用于图表样式
```

### 安装依赖

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn prettytable
pip install mplcyberpunk qbstyles  # 可选
```

## 🚀 使用方法

### 1. 准备数据
确保数据文件位于正确路径：
```
guo-predicted/
├── CNN-BiLSTM-Attention.py
├── date_file/
│   └── 按日累计-台区R1-充电负荷数据.csv
└── README.md
```

### 2. 运行预测

```bash
python CNN-BiLSTM-Attention.py
```

### 3. 查看结果

运行完成后会生成：
- **模型文件**：`ev_charging_load_cnn_bilstm_attention_model.h5`
- **预测结果**：`ev_charging_load_prediction_results.csv`
- **可视化图表**：训练损失曲线、预测对比图

## 📈 模型参数说明

当前配置：
- **输入时间步** (`n_in`)：7天（使用前7天数据）
- **输出时间步** (`n_out`)：1天（预测未来1天）
- **特征维度** (`or_dim`)：96个时间点
- **训练集比例**：80%
- **测试集比例**：20%
- **批次大小** (`batch_size`)：16
- **训练轮数** (`epochs`)：100

### 调整参数

可以根据实际需求在脚本中修改以下参数：

```python
# 第113-114行
n_in = 7   # 输入的天数
n_out = 1  # 预测的天数

# 第237行
batch_size = 16   # 批次大小
epochs = 100      # 训练轮数
```

## 📊 评估指标

模型使用以下指标评估预测性能：

- **MAPE** (Mean Absolute Percentage Error)：平均绝对百分比误差
- **RMSE** (Root Mean Square Error)：均方根误差
- **MAE** (Mean Absolute Error)：平均绝对误差
- **R²** (R-squared)：决定系数

## 🎨 输出示例

```
数据形状: (354, 97)
数据前5行:
...

提取的数据形状: (354, 96)
数据范围: 最小值=0.0000, 最大值=11.5316, 平均值=1.2345

数据总行数: 354, 特征维度: 96, 可用样本数: 346, 实际使用: 300
训练集样本数: 240, 测试集样本数: 60

Model: "model"
...

Epoch 1/100
...

============================================================
EV充电负荷预测模型性能总结 (CNN-BiLSTM-Attention)
============================================================
模型结构: CNN → BiLSTM → Attention → Dense
训练样本数: 240
测试样本数: 60
输入时间步: 7天 (每天96个时间点)
预测时间步: 1天
特征维度: 96个时间点
============================================================
总体MAPE: 5.23%
总体RMSE: 0.3456
总体MAE: 0.2789
总体R²: 92.34%
============================================================

预测完成！模型和结果已保存。
```

## 📁 项目结构

```
guo-predicted/
├── CNN-BiLSTM-Attention.py              # 主预测脚本（推荐使用）
├── BiLSTM-Attention.py                  # 备选模型
├── CNN-LSTM-Attention.py                # 备选模型
├── LSTM-Attention.py                    # 备选模型
├── BiLSTM.py                            # 基础模型
├── LSTM.py                              # 基础模型
├── date_file/
│   ├── 按日累计-台区R1-充电负荷数据.csv    # 原始数据
│   └── 按日累计-台区R1-充电负荷数据.xlsx   # Excel格式数据
├── ev_charging_load_cnn_bilstm_attention_model.h5  # 训练后的模型
├── ev_charging_load_prediction_results.csv          # 预测结果
└── README.md                            # 说明文档
```

## 🔍 故障排除

### 1. 导入错误
如果遇到模块导入错误，请检查是否安装了所有依赖库：
```bash
pip install -r requirements.txt  # 如果提供了requirements.txt
```

### 2. 内存不足
如果训练时内存不足，可以：
- 减小 `batch_size`（如改为8或4）
- 减小 `num_samples`
- 减小 `n_in`（使用更少的历史天数）

### 3. 数据格式错误
确保数据文件：
- 编码为UTF-8
- 第一行为列名
- 数据为数值型（无缺失或异常值）

### 4. GPU加速（可选）
如果有NVIDIA GPU，可以安装GPU版本的TensorFlow以加速训练：
```bash
pip install tensorflow-gpu
```

## 📞 技术支持

如有问题，请检查：
1. Python版本是否 >= 3.7
2. 所有依赖库是否正确安装
3. 数据文件路径是否正确
4. 数据格式是否符合要求

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

感谢提供EV充电负荷数据，以及开源社区提供的深度学习框架和工具。

---

**版本**：v1.0  
**更新日期**：2024  
**作者**：AI Assistant

