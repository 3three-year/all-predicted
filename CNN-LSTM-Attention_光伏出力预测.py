'''
光伏出力预测 - CNN-LSTM-Attention模型
基于CNN-LSTM-Attention架构对台区R1的光伏出力进行预测
典型日展示：9月16日 和 9月25日
'''

import os
import math
import pandas as pd
import openpyxl
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from pandas import DataFrame
from pandas import concat
import keras.backend as K
from scipy.io import savemat, loadmat
from sklearn.neural_network import MLPRegressor
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input, Model, Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, LayerNormalization, Input, Conv1D, MaxPooling1D, Reshape, BatchNormalization
from sklearn.model_selection import KFold
import warnings
from prettytable import PrettyTable
from matplotlib import rcParams
warnings.filterwarnings("ignore")

# 创建结果目录
if not os.path.exists('results_pv_lstm'):
    os.makedirs('results_pv_lstm')
    print("已创建results_pv_lstm目录用于保存预测结果和图表")

print("="*80)
print("光伏出力预测系统 - CNN-LSTM-Attention模型")
print("="*80)

# 读取光伏出力数据
dataset = pd.read_csv("date_file\按日累计-台区R1-总光伏出力_扩大1.5倍.csv", encoding='utf-8')
print("\n数据基本信息:")
print("数据形状:", dataset.shape)
print("\n数据前5行:")
print(dataset.head())

# 保存日期列用于后续典型日分析
dates = pd.to_datetime(dataset.iloc[:, 0])
print(f"\n数据日期范围: {dates.min()} 至 {dates.max()}")

# 查找典型日的索引
typical_date_1 = '2022-09-16'
typical_date_2 = '2022-09-25'
try:
    idx_sep16 = dates[dates == typical_date_1].index[0]
    idx_sep25 = dates[dates == typical_date_2].index[0]
    print(f"\n典型日信息:")
    print(f"  9月16日 在数据中的索引: {idx_sep16}")
    print(f"  9月25日 在数据中的索引: {idx_sep25}")
except:
    print(f"\n警告: 未找到典型日数据，将使用默认样本进行展示")
    idx_sep16 = None
    idx_sep25 = None

# 数据预处理
values = dataset.iloc[:, 1:].values
print(f"\n提取的数据形状: {values.shape}")

# 转换为数值类型
values = pd.DataFrame(values).apply(pd.to_numeric, errors='coerce').values
values = values.astype('float32')

# 处理缺失值
if np.isnan(values).any():
    print(f"发现缺失值，正在处理...")
    values = pd.DataFrame(values).ffill().bfill().values
    values = values.astype('float32')

print(f"数据范围: {values.min():.2f} - {values.max():.2f} kW")


def data_collation(data, n_in, n_out, or_dim, scroll_window, num_samples):
    """数据整理函数，将时间序列数据整理为监督学习格式"""
    res = np.zeros((num_samples, n_in*or_dim + or_dim))
    for i in range(0, num_samples):
        h1 = values[scroll_window*i: n_in+scroll_window*i, 0:or_dim]
        h2 = h1.reshape(1, n_in*or_dim)
        h3 = values[n_in+scroll_window*i, :]
        h4 = h3[np.newaxis, :]
        h5 = np.hstack((h2, h4))
        res[i,:] = h5
    return res


# 参数设置
n_in = 7  # 使用前7天的数据
n_out = 96  # 预测96个时间点（一天）
or_dim = values.shape[1]  # 特征维度（96个时间点）
max_samples = len(values) - n_in - 1
num_samples = min(max_samples, 300)  # 使用300个样本
scroll_window = 1  # 滑动窗口为1

print(f"\n模型参数设置:")
print(f"  输入窗口: {n_in} 天")
print(f"  输出窗口: {n_out} 个时间点")
print(f"  特征维度: {or_dim}")
print(f"  使用样本数: {num_samples}")

res = data_collation(values, n_in, n_out, or_dim, scroll_window, num_samples)

# 划分训练集和测试集
values_processed = np.array(res)
n_train_number = int(num_samples * 0.8)
print(f"训练集样本数: {n_train_number}, 测试集样本数: {num_samples - n_train_number}")

Xtrain = values_processed[:n_train_number, :n_in*or_dim]
Ytrain = values_processed[:n_train_number, n_in*or_dim:]
Xtest = values_processed[n_train_number:, :n_in*or_dim]
Ytest = values_processed[n_train_number:, n_in*or_dim:]

# 归一化
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(Xtrain)
vp_test = m_in.transform(Xtest)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(Ytrain)
vt_test = m_out.transform(Ytest)

# 重塑数据
vp_train = vp_train.reshape((vp_train.shape[0], n_in, or_dim))
vp_test = vp_test.reshape((vp_test.shape[0], n_in, or_dim))

print(f"\n训练数据形状: {vp_train.shape}, {vt_train.shape}")
print(f"测试数据形状: {vp_test.shape}, {vt_test.shape}")


def attention_layer(inputs, time_steps):
    """注意力机制层"""
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def cnn_lstm_attention_model():
    """CNN-LSTM-Attention模型（优化版本）"""
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
    inputs = Input(shape=(vp_train.shape[1], vp_train.shape[2]))
    
    # CNN层 - 添加正则化和BatchNormalization
    conv1d = Conv1D(filters=80, kernel_size=2, activation='relu',
                    kernel_regularizer=l2(0.0002))(inputs)
    conv1d = BatchNormalization()(conv1d)
    conv1d = Dropout(0.18)(conv1d)
    
    maxpooling = MaxPooling1D(pool_size=2)(conv1d)
    reshaped = Reshape((-1, 80 * maxpooling.shape[1]))(maxpooling)
    
    # LSTM层（单向）- 添加正则化
    lstm_out = LSTM(160, return_sequences=True, 
                    dropout=0.18, recurrent_dropout=0.12,
                    kernel_regularizer=l2(0.0002))(reshaped)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Attention层
    attention_out = attention_layer(lstm_out, time_steps=reshaped.shape[1])
    attention_flatten = Flatten()(attention_out)
    
    # Dense层
    dense1 = Dense(320, activation='relu', kernel_regularizer=l2(0.0002))(attention_flatten)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.2)(dense1)
    
    dense2 = Dense(160, activation='relu', kernel_regularizer=l2(0.0002))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.18)(dense2)
    
    outputs = Dense(vt_train.shape[1])(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0006)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    print("\n" + "="*70)
    print("【CNN-LSTM-Attention模型配置】")
    print("CNN: 80滤波器 | LSTM: 160单元(单向) | Dense: 320→160→96")
    print("正则化: L2(0.0002) + Dropout(0.18-0.2)")
    print("损失函数: MSE")
    print("="*70 + "\n")
    
    model.summary()
    return model


print("\n开始训练模型...")
model = cnn_lstm_attention_model()

# 回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=1)
]

print("\n训练配置:")
print("  - Early Stopping: patience=25")
print("  - ReduceLROnPlateau: patience=12")
print("  - 初始学习率: 0.0006")
print("  - Batch Size: 12")
print("\n开始训练...\n")

history = model.fit(vp_train, vt_train,
                    batch_size=12,
                    epochs=200,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=2)

model.save('results_pv_lstm/pv_output_cnn_lstm_attention_model.h5')
print("\n[OK] 模型已保存")

# 绘制训练历史
fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
ax.plot(history.history['loss'], label='Training Loss', linewidth=2.5, color='#FF6B35', marker='o', markersize=4, alpha=0.8)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.5, color='#004E89', marker='s', markersize=4, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training History - PV Output Prediction (CNN-LSTM-Attention)', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_pv_lstm/02_训练损失曲线.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存训练曲线")
plt.close()

# 预测
yhat = model.predict(vp_test)
yhat = yhat.reshape(num_samples-n_train_number, n_out)
predicted_data = m_out.inverse_transform(yhat)


def mape(y_true, y_pred):
    """MAPE指标"""
    record = []
    for index in range(len(y_true)):
        if y_true[index] != 0:
            temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
            record.append(temp_mape)
    return np.mean(record) * 100 if len(record) > 0 else 0


def evaluate_forecasts(Ytest, predicted_data, n_out):
    """评估指标"""
    mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic = [], [], [], [], []
    table = PrettyTable(['指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    
    for i in range(min(n_out, 10)):  # 只显示前10个时间点
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        rmse = sqrt(mse)
        rmse_dic.append(rmse)
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        
        strr = f'P{i+1}'
        table.add_row([strr, f'{mse:.4f}', f'{rmse:.4f}', f'{mae:.4f}', f'{MApe:.2f}%', f'{r2*100:.2f}%'])
    
    # 计算所有时间点的平均指标
    for i in range(n_out):
        if i >= 10:
            actual = [float(row[i]) for row in Ytest]
            predicted = [float(row[i]) for row in predicted_data]
            mse_dic.append(mean_squared_error(actual, predicted))
            rmse_dic.append(sqrt(mean_squared_error(actual, predicted)))
            mae_dic.append(mean_absolute_error(actual, predicted))
            mape_dic.append(mape(actual, predicted))
            r2_dic.append(r2_score(actual, predicted))
    
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table


mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(Ytest, predicted_data, n_out)
print("\n预测指标（前10个时间点）:")
print(table)

avg_mape = np.mean(mape_dic)
avg_rmse = np.mean(rmse_dic)
avg_mae = np.mean(mae_dic)
avg_r2 = np.mean(r2_dic)
print(f"\n整体平均指标:")
print(f"MAPE: {avg_mape:.2f}% | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f} | R²: {avg_r2*100:.2f}%")

# 可视化设置
plt.style.use('default')
config = {
    "font.family": 'sans-serif',
    "font.sans-serif": ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
    "font.size": 10,
    'axes.unicode_minus': False,
}
rcParams.update(config)

# 时间标签
time_labels = []
for hour in range(24):
    for minute in [0, 15, 30, 45]:
        time_labels.append(f"{hour:02d}:{minute:02d}")

print("\n开始生成可视化图表...")

# 整体平均预测图
fig, ax = plt.subplots(figsize=(14, 5), dpi=300, facecolor='white')
avg_real = np.mean(Ytest, axis=0)
avg_pred = np.mean(predicted_data, axis=0)
x_points = range(1, 97)

std_real = np.std(Ytest, axis=0)
std_pred = np.std(predicted_data, axis=0)
ax.fill_between(x_points, avg_real - std_real, avg_real + std_real, alpha=0.25, color='#FF6B35')
ax.fill_between(x_points, avg_pred - std_pred, avg_pred + std_pred, alpha=0.25, color='#004E89')

ax.plot(x_points, avg_real, linestyle="-", linewidth=3, label='平均真实值', marker='o', markersize=4, color='#FF6B35')
ax.plot(x_points, avg_pred, linestyle="--", linewidth=3, label='平均预测值', marker='s', markersize=4, color='#004E89')

tick_positions = list(range(0, 96, 8))
tick_labels_list = [time_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_list, rotation=45, ha='right')

ax.set_xlabel('时间 (24小时)', fontsize=12, fontweight='bold')
ax.set_ylabel('光伏出力 (kW)', fontsize=12, fontweight='bold')
ax.set_title(f'整体平均预测对比 (CNN-LSTM-Attention)\nMAPE: {avg_mape:.2f}% | RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f} | R²: {avg_r2*100:.2f}%',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_pv_lstm/03_整体平均预测对比.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存整体平均预测图")
plt.close()

# 典型日预测图
test_start_idx = n_train_number + n_in

if idx_sep16 is not None and idx_sep25 is not None:
    # 9月16日预测图
    if idx_sep16 >= test_start_idx and idx_sep16 < test_start_idx + len(Ytest):
        idx_sep16_test = idx_sep16 - test_start_idx
        
        fig, ax = plt.subplots(figsize=(16, 6), dpi=300, facecolor='white')
        x_points = range(1, 97)
        
        sample_mape = mape(Ytest[idx_sep16_test, :], predicted_data[idx_sep16_test, :])
        sample_rmse = np.sqrt(mean_squared_error(Ytest[idx_sep16_test, :], predicted_data[idx_sep16_test, :]))
        sample_mae = mean_absolute_error(Ytest[idx_sep16_test, :], predicted_data[idx_sep16_test, :])
        sample_r2 = r2_score(Ytest[idx_sep16_test, :], predicted_data[idx_sep16_test, :])
        
        ax.plot(x_points, Ytest[idx_sep16_test, :], linestyle="-", linewidth=3, label='真实光伏出力', 
                marker='o', markersize=5, color='#FF6B35', alpha=0.9)
        ax.plot(x_points, predicted_data[idx_sep16_test, :], linestyle="--", linewidth=3, label='预测光伏出力', 
                marker='s', markersize=5, color='#004E89', alpha=0.9)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('时间 (24小时)', fontsize=14, fontweight='bold')
        ax.set_ylabel('光伏出力 (kW)', fontsize=14, fontweight='bold')
        ax.set_title('典型日一预测图', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, prop={'family': 'Microsoft YaHei', 'size': 12})
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results_pv_lstm/典型日_2022年09月16日_光伏出力预测.png', dpi=300, bbox_inches='tight')
        print(f"[OK] 已保存 9月16日 典型日预测图")
        plt.close()
    else:
        print("9月16日不在测试集范围内")
    
    # 9月25日预测图
    if idx_sep25 >= test_start_idx and idx_sep25 < test_start_idx + len(Ytest):
        idx_sep25_test = idx_sep25 - test_start_idx
        
        fig, ax = plt.subplots(figsize=(16, 6), dpi=300, facecolor='white')
        x_points = range(1, 97)
        
        sample_mape = mape(Ytest[idx_sep25_test, :], predicted_data[idx_sep25_test, :])
        sample_rmse = np.sqrt(mean_squared_error(Ytest[idx_sep25_test, :], predicted_data[idx_sep25_test, :]))
        sample_mae = mean_absolute_error(Ytest[idx_sep25_test, :], predicted_data[idx_sep25_test, :])
        sample_r2 = r2_score(Ytest[idx_sep25_test, :], predicted_data[idx_sep25_test, :])
        
        ax.plot(x_points, Ytest[idx_sep25_test, :], linestyle="-", linewidth=3, label='真实光伏出力', 
                marker='o', markersize=5, color='#FF6B35', alpha=0.9)
        ax.plot(x_points, predicted_data[idx_sep25_test, :], linestyle="--", linewidth=3, label='预测光伏出力', 
                marker='s', markersize=5, color='#004E89', alpha=0.9)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('时间 (24小时)', fontsize=14, fontweight='bold')
        ax.set_ylabel('光伏出力 (kW)', fontsize=14, fontweight='bold')
        ax.set_title('典型日二预测图', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, prop={'family': 'Microsoft YaHei', 'size': 12})
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results_pv_lstm/典型日_2022年09月25日_光伏出力预测.png', dpi=300, bbox_inches='tight')
        print(f"[OK] 已保存 9月25日 典型日预测图")
        plt.close()
    else:
        print("9月25日不在测试集范围内")

# 两个典型日对比图
if idx_sep16 is not None and idx_sep25 is not None:
    if (idx_sep16 >= test_start_idx and idx_sep16 < test_start_idx + len(Ytest) and
        idx_sep25 >= test_start_idx and idx_sep25 < test_start_idx + len(Ytest)):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=300, facecolor='white')
        x_points = range(1, 97)
        
        # 9月16日
        idx_sep16_test = idx_sep16 - test_start_idx
        sample_mape_sep16 = mape(Ytest[idx_sep16_test, :], predicted_data[idx_sep16_test, :])
        ax1.plot(x_points, Ytest[idx_sep16_test, :], linestyle="-", linewidth=2.5, 
                label='真实值', marker='o', markersize=4, color='#FF6B35')
        ax1.plot(x_points, predicted_data[idx_sep16_test, :], linestyle="--", linewidth=2.5, 
                label='预测值', marker='s', markersize=4, color='#004E89')
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels_list, rotation=45, ha='right')
        ax1.set_ylabel('光伏出力 (kW)', fontsize=12, fontweight='bold')
        ax1.set_title(f'9月16日光伏出力预测 - MAPE: {sample_mape_sep16:.2f}%', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 9月25日
        idx_sep25_test = idx_sep25 - test_start_idx
        sample_mape_sep25 = mape(Ytest[idx_sep25_test, :], predicted_data[idx_sep25_test, :])
        ax2.plot(x_points, Ytest[idx_sep25_test, :], linestyle="-", linewidth=2.5, 
                label='真实值', marker='o', markersize=4, color='#FF6B35')
        ax2.plot(x_points, predicted_data[idx_sep25_test, :], linestyle="--", linewidth=2.5, 
                label='预测值', marker='s', markersize=4, color='#004E89')
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels_list, rotation=45, ha='right')
        ax2.set_xlabel('时间 (24小时)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('光伏出力 (kW)', fontsize=12, fontweight='bold')
        ax2.set_title(f'9月25日光伏出力预测 - MAPE: {sample_mape_sep25:.2f}%', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results_pv_lstm/典型日对比_9月16日vs9月25日.png', dpi=300, bbox_inches='tight')
        print("[OK] 已保存典型日对比图")
        plt.close()

# 保存预测结果 - 只保存9月16日和9月25日的数据（Excel格式）
if idx_sep16 is not None and idx_sep25 is not None:
    if (idx_sep16 >= test_start_idx and idx_sep16 < test_start_idx + len(Ytest) and
        idx_sep25 >= test_start_idx and idx_sep25 < test_start_idx + len(Ytest)):
        
        idx_sep16_test = idx_sep16 - test_start_idx
        idx_sep25_test = idx_sep25 - test_start_idx
        
        # 9月16日数据
        sep16_df = pd.DataFrame({
            '时间点': [f'P{i+1}' for i in range(96)],
            '实际值': Ytest[idx_sep16_test, :],
            '预测值': predicted_data[idx_sep16_test, :]
        })
        sep16_df.to_excel('results_pv_lstm/光伏出力预测_2022年09月16日.xlsx', index=False, engine='openpyxl')
        print("[OK] 9月16日预测数据已保存（Excel格式）")
        
        # 9月25日数据
        sep25_df = pd.DataFrame({
            '时间点': [f'P{i+1}' for i in range(96)],
            '实际值': Ytest[idx_sep25_test, :],
            '预测值': predicted_data[idx_sep25_test, :]
        })
        sep25_df.to_excel('results_pv_lstm/光伏出力预测_2022年09月25日.xlsx', index=False, engine='openpyxl')
        print("[OK] 9月25日预测数据已保存（Excel格式）")
        
        # 合并两天的数据到一个Excel文件，使用多个工作表
        with pd.ExcelWriter('results_pv_lstm/光伏出力预测_典型日汇总.xlsx', engine='openpyxl') as writer:
            # 9月16日工作表
            sep16_df.to_excel(writer, sheet_name='2022-09-16', index=False)
            # 9月25日工作表
            sep25_df.to_excel(writer, sheet_name='2022-09-25', index=False)
            # 汇总工作表
            combined_df = pd.DataFrame({
                '日期': ['2022-09-16'] * 96 + ['2022-09-25'] * 96,
                '时间点': [f'P{i+1}' for i in range(96)] * 2,
                '实际值': np.concatenate([Ytest[idx_sep16_test, :], Ytest[idx_sep25_test, :]]),
                '预测值': np.concatenate([predicted_data[idx_sep16_test, :], predicted_data[idx_sep25_test, :]])
            })
            combined_df.to_excel(writer, sheet_name='汇总', index=False)
        print("[OK] 典型日汇总数据已保存（Excel格式，包含3个工作表）")
    else:
        print("[警告] 典型日不在测试集范围内，无法保存预测数据")
else:
    print("[警告] 未找到典型日数据，无法保存预测数据")

print("\n" + "="*80)
print("光伏出力预测完成！(CNN-LSTM-Attention)")
print("="*80)
print(f"模型: CNN-LSTM-Attention")
print(f"平均MAPE: {avg_mape:.2f}%")
print(f"平均RMSE: {avg_rmse:.4f}")
print(f"平均MAE: {avg_mae:.4f}")
print(f"平均R²: {avg_r2*100:.2f}%")
print("="*80)
print("所有结果已保存到 results_pv_lstm 目录")
print("="*80)
