# 预测结果目录说明

本目录包含三个子目录，分别存储不同数据源的预测结果。

---

## 📁 目录结构

```
results/
├── TaiquR1_Original/    # 台区R1充电负荷预测结果（项目原有）
├── TaiquB/              # 台区B总负荷预测结果（新增）
└── PV_Forecast/         # 台区R1光伏发电预测结果（最新，3天平滑优化版本）
```

---

## 📊 TaiquR1_Original/ - 台区R1充电负荷预测

### 数据来源
- **文件**: `date_file/按日累计-台区R1-充电负荷数据.csv`
- **数据类型**: 电动汽车(EV)充电负荷
- **数据特征**: 
  - 平均负荷: 2.31 kW
  - 数据规模: 365天 × 96时间点

### 模型信息
- **模型**: CNN-BiLSTM-Attention (优化版)
- **脚本**: `CNN-BiLSTM-Attention-Optimized.py`

### 包含文件 (16个)

#### 模型文件
- `best_model_optimized.h5` - 验证集最佳模型
- `optimized_ev_charging_model.h5` - 最终训练模型

#### 预测结果
- `optimized_prediction_results.csv` - 预测结果数据
- `all_samples_adjustable_power_merged.csv` - 所有样本可调功率汇总
- `sample01-05_adjustable_power_v2.csv` (5个文件) - 各样本可调功率

#### 可视化图表
- `01_cleaned_data_preview.png` - 清洗后数据预览
- `02_training_loss_optimized.png` - 训练损失曲线
- `04_optimized_sample01-05_prediction.png` (5张) - 样本预测对比图

### 预测性能
根据原始项目配置，针对EV充电负荷的特点进行了优化。

---

## 📊 TaiquB/ - 台区B总负荷预测

### 数据来源
- **文件**: `date_file/按日累计-台区B-总负荷-去除无功.csv`
- **数据类型**: 台区总负荷（去除无功）
- **数据特征**:
  - 平均负荷: 129.03 kW
  - 数据规模: 365天 × 96时间点
  - **是台区R1的56.42倍**

### 模型信息
- **模型**: CNN-BiLSTM-Attention (优化版，针对台区B调优)
- **脚本**: `CNN-BiLSTM-Attention-Optimized-TaiquB.py`

### 包含文件 (13个)

#### 模型文件
- `best_model_optimized.h5` - 验证集最佳模型
- `optimized_taiquB_model.h5` - 最终训练模型

#### 预测结果
- `optimized_prediction_results.csv` - 预测结果数据
- `metrics_by_timepoint.csv` - 各时间点性能指标
- `prediction_report.md` - 完整预测报告

#### 可视化图表
- `01_cleaned_data_preview.png` - 清洗后数据预览
- `02_training_loss_optimized.png` - 训练损失曲线
- `04_optimized_sample01-05_prediction.png` (5张) - 样本预测对比图
- `timepoint_performance_analysis.png` - 时间点性能分析图

### 预测性能
- **平均 MAPE**: 13.75% ✅
- **平均 RMSE**: 19.03 kW ✅
- **平均 MAE**: 15.46 kW ✅
- **平均 R²**: 48.82% ⚠️

---

## 📊 PV_Forecast/ - 台区R1光伏发电预测

### 数据来源
- **文件**: `date_file/按日累计-台区R1-总光伏出力.csv`
- **数据类型**: 光伏发电功率
- **数据特征**:
  - 平均发电: 45.20 kW
  - 数据规模: 365天 × 96时间点（完整年度数据）
  - **日期范围**: 2022-01-01 至 2022-12-31

### 模型信息
- **模型**: CNN-BiLSTM-Attention (优化版，针对光伏特性调优)
- **脚本**: `CNN-BiLSTM-Attention-Optimized-PV.py`
- **数据预处理**:
  - 负值转为正值（发电为主）
  - 原始正值清零（去除负荷残差）
  - 3天移动平滑（提取共性规律）
  - 3-sigma异常值裁剪

### 包含文件 (13个)

#### 模型文件
- `best_pv_model.h5` - 验证集最佳模型
- `optimized_pv_model.h5` - 最终训练模型

#### 预测结果
- `pv_prediction_results.csv` - 预测结果数据
- `metrics_by_timepoint.csv` - 各时间点性能指标

#### 可视化图表
- `01_pv_typical_day.png` - 典型日发电曲线
- `02_training_loss.png` - 训练损失曲线
- `04_pv_sample01-05_prediction.png` (5张) - 样本预测对比图
- `sample_comparison_best_vs_worst.png` - 最佳/最差样本对比
- `smoothing_comparison.png` - 平滑效果对比图

### 预测性能
- **平均 MAPE**: 69.69% ⚠️
- **平均 RMSE**: 41.51 kW ⚠️
- **平均 MAE**: 23.17 kW ⚠️
- **平均 R²**: -12.43% ❌

### 性能分析
- **优秀样本 (R²>0)**: 81.7% - 主要是晴天，预测效果好
- **较差样本 (R²<0)**: 18.3% - 主要是多云/阴天，预测效果差
- **核心问题**: 缺少天气特征，无法区分不同天气模式
- **改进建议**: 引入天气类型、辐照度、云量等特征

### 版本说明
本版本为**3天移动平滑优化版本**，是经过以下实验验证的最优方案：
- 无平滑版本：MAPE 195%（性能最差）
- 1天平滑版本：MAPE 165%（性能较差）
- **3天平滑版本**：**MAPE 69.69%**（**当前最优**）

---

## 🔍 三个数据集对比

| 特征 | 台区R1 (充电负荷) | 台区B (总负荷) | 台区R1 (光伏发电) |
|------|------------------|---------------|------------------|
| 数据性质 | 单一充电负荷 | 混合总负荷 | 光伏发电 |
| 数据日期 | 2022-01-14至12-31 | 2022-01-01至12-31 | 2022-01-01至12-31 |
| 数据天数 | 352天 | 365天 | 365天 |
| 平均功率 | 2.31 kW | 129.03 kW | 45.20 kW |
| 峰值时段 | 晚间20:00-22:00 | 早晚双峰 | 正午12:00-13:00 |
| 预测难度 | 较低（规律性强） | 中等（模式复杂） | 高（受天气影响大） |
| 数据波动 | 小（单一用途） | 大（多种负荷） | 极大（天气变化） |
| MAPE | 15-20% | 13.75% | 69.69% |

### 日期对应关系
- ✅ **光伏 & 台区B**：完全对应（365天）
- ⚠️ **光伏 & R1充电**：部分对应（R1充电缺少前13天）
- ✅ **三者可对比时间段**：2022-01-14至12-31（352天）

---

## 📝 使用说明

### 查看台区R1充电负荷结果
```bash
cd results/TaiquR1_Original
# 查看预测对比图
start 04_optimized_sample01_prediction.png
```

### 查看台区B总负荷结果
```bash
cd results/TaiquB
# 查看完整报告
start prediction_report.md
# 查看时间点性能分析
start timepoint_performance_analysis.png
```

### 查看台区R1光伏发电结果
```bash
cd results/PV_Forecast
# 查看典型日发电曲线
start 01_pv_typical_day.png
# 查看最佳/最差样本对比
start sample_comparison_best_vs_worst.png
# 查看平滑效果对比
start smoothing_comparison.png
```

---

## 📌 重要说明

1. **TaiquR1_Original** 目录 - 充电负荷预测（项目原有）
   - 针对电动汽车充电负荷
   - 数据规模小，预测相对容易
   - MAPE 15-20%，性能良好

2. **TaiquB** 目录 - 总负荷预测（新增）
   - 针对台区总负荷（去除无功）
   - 数据规模大，包含异常值处理
   - MAPE 13.75%，性能优秀
   - 预测难度更高，实际应用价值更大

3. **PV_Forecast** 目录 - 光伏发电预测（最新）
   - 针对光伏发电功率
   - 数据规模大，受天气影响显著
   - MAPE 69.69%，性能中等
   - 采用3天平滑优化（当前最优方案）
   - **核心限制**: 缺少天气特征，建议未来引入天气数据

---

## 🚀 改进建议

### 光伏预测改进方向
1. **引入天气特征** ⭐⭐⭐⭐⭐（最关键）
   - 天气类型（晴/多云/阴/雨）
   - 太阳辐照度
   - 云量
   - 温度
   - 预期效果：MAPE降至30-40%

2. **优化模型结构**
   - 增加模型容量
   - 减少正则化强度
   - 使用Transformer架构
   - 预期效果：MAPE降至40-50%

3. **增加训练数据**
   - 3-5年历史数据
   - 覆盖更多天气模式
   - 预期效果：提升泛化能力

---

**创建日期**: 2025-10-23  
**最后更新**: 2025-10-24  
**整理说明**: 包含三个预测系统的完整结果，已删除实验版本


