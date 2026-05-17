# -*- coding: utf-8 -*-
"""
计算光伏出力预测数据的评估指标
计算9月16日和9月25日的RMSE、MAPE和R²指标
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import openpyxl  # 用于读取Excel文件

def calculate_mape(y_true, y_pred):
    """
    计算MAPE（平均绝对百分比误差）
    MAPE = (1/n) * Σ|（实际值 - 预测值）/ 实际值| * 100%
    
    注意：当实际值接近0时，MAPE会非常大，因此只计算实际值>阈值的点
    """
    # 设置阈值，只计算实际值大于10 kW的点（避免除以接近0的数）
    threshold = 10.0
    mask = y_true > threshold
    
    if np.sum(mask) == 0:
        return 0.0
    
    # 只对实际值大于阈值的点计算MAPE
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def calculate_rmse_percentage(y_true, y_pred, max_value):
    """
    计算RMSE并转换为百分比形式
    RMSE% = RMSE / 历史最大值 * 100%
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_percentage = (rmse / max_value) * 100
    return rmse, rmse_percentage

def calculate_r2(y_true, y_pred):
    """
    计算R²（决定系数）
    R² = 1 - (SS_res / SS_tot)
    """
    return r2_score(y_true, y_pred)

def load_prediction_data():
    """
    加载预测数据（从两个独立的Excel文件）
    """
    print("=" * 60)
    print("加载光伏出力预测数据...")
    print("=" * 60)
    
    # 预测数据路径
    pred_dir = 'DATE_FILE_predict'
    pred_files = {
        '2022-09-16': os.path.join(pred_dir, '光伏出力预测_2022年09月16日.xlsx'),
        '2022-09-25': os.path.join(pred_dir, '光伏出力预测_2022年09月25日.xlsx')
    }
    
    # 读取两个文件并合并
    df_list = []
    for date_str, pred_file in pred_files.items():
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"预测文件不存在: {pred_file}")
        
        # 读取Excel文件
        df_day = pd.read_excel(pred_file)
        print(f"✓ 成功加载预测数据: {pred_file}")
        print(f"  数据形状: {df_day.shape}")
        
        # 添加日期列
        df_day['日期'] = pd.to_datetime(date_str)
        df_list.append(df_day)
    
    # 合并数据
    df_pred = pd.concat(df_list, ignore_index=True)
    print(f"\n✓ 合并后的预测数据形状: {df_pred.shape}")
    print(f"  列名: {df_pred.columns.tolist()}")
    
    return df_pred

def load_actual_data():
    """
    加载实际数据（真实值）
    """
    print("\n加载实际光伏出力数据...")
    
    # 实际数据路径
    actual_dir = 'date_file'
    actual_file = os.path.join(actual_dir, '按日累计-台区R1-总光伏出力_扩大2.0倍.csv')
    
    if not os.path.exists(actual_file):
        raise FileNotFoundError(f"实际数据文件不存在: {actual_file}")
    
    # 读取实际数据
    df_actual = pd.read_csv(actual_file, encoding='utf-8')
    print(f"✓ 成功加载实际数据: {actual_file}")
    print(f"  数据形状: {df_actual.shape}")
    
    # 转换日期列
    df_actual['日期'] = pd.to_datetime(df_actual.iloc[:, 0])
    
    return df_actual

def extract_day_data(df, target_date, is_prediction=False):
    """
    提取指定日期的数据
    
    参数:
        df: 数据DataFrame
        target_date: 目标日期
        is_prediction: 是否为预测数据（预测数据格式不同）
    """
    # 转换日期格式
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    # 筛选数据
    mask = df['日期'].dt.date == target_date.date()
    day_data = df[mask]
    
    if len(day_data) == 0:
        print(f"警告: 未找到日期 {target_date.date()} 的数据")
        return None
    
    # 提取96个时间点的数据
    if is_prediction:
        # 预测数据可能使用不同的列名格式
        # 尝试多种可能的列名格式
        if '预测值' in df.columns:
            # 如果有单列预测值，直接使用
            values = day_data['预测值'].values
        elif 'P1' in df.columns:
            # 如果有P1-P96列
            p_columns = [f'P{i}' for i in range(1, 97)]
            values = day_data[p_columns].values.flatten()
        else:
            # 尝试使用数值列（跳过日期列）
            numeric_cols = day_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 96:
                values = day_data[numeric_cols[:96]].values.flatten()
            else:
                print(f"错误: 无法识别预测数据格式")
                print(f"  可用列: {df.columns.tolist()}")
                return None
    else:
        # 实际数据使用P1-P96列
        p_columns = [f'P{i}' for i in range(1, 97)]
        if p_columns[0] not in df.columns:
            print(f"错误: 未找到P列，可用列: {df.columns.tolist()}")
            return None
        values = day_data[p_columns].values.flatten()
    
    return values

def calculate_metrics_for_day(actual_values, pred_values, date_str, max_value):
    """
    计算单日的评估指标
    """
    print(f"\n{'='*60}")
    print(f"计算 {date_str} 的评估指标")
    print(f"{'='*60}")
    
    # 数据验证
    if actual_values is None or pred_values is None:
        print(f"错误: {date_str} 的数据缺失")
        return None
    
    if len(actual_values) != len(pred_values):
        print(f"错误: 实际值和预测值长度不匹配")
        print(f"  实际值长度: {len(actual_values)}")
        print(f"  预测值长度: {len(pred_values)}")
        return None
    
    # 数据统计
    print(f"\n数据统计:")
    print(f"  实际值范围: {actual_values.min():.2f} - {actual_values.max():.2f} kW")
    print(f"  预测值范围: {pred_values.min():.2f} - {pred_values.max():.2f} kW")
    print(f"  实际值平均: {actual_values.mean():.2f} kW")
    print(f"  预测值平均: {pred_values.mean():.2f} kW")
    
    # 计算指标
    rmse, rmse_percentage = calculate_rmse_percentage(actual_values, pred_values, max_value)
    mape = calculate_mape(actual_values, pred_values)
    r2 = calculate_r2(actual_values, pred_values)
    
    # 打印结果
    print(f"\n评估指标:")
    print(f"  RMSE: {rmse:.4f} kW")
    print(f"  RMSE%: {rmse_percentage:.2f}%（相对于历史最大值 {max_value:.2f} kW）")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")
    
    return {
        '日期': date_str,
        'RMSE (kW)': rmse,
        'RMSE (%)': rmse_percentage,
        'MAPE (%)': mape,
        'R²': r2,
        '实际值平均 (kW)': actual_values.mean(),
        '预测值平均 (kW)': pred_values.mean(),
        '实际值最大 (kW)': actual_values.max(),
        '预测值最大 (kW)': pred_values.max(),
        '历史最大值 (kW)': max_value
    }

def main():
    """
    主函数
    """
    print("\n" + "=" * 60)
    print("光伏出力预测评估指标计算")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df_pred = load_prediction_data()
        df_actual = load_actual_data()
        
        # 2. 计算历史最大值
        p_columns = [f'P{i}' for i in range(1, 97)]
        historical_max = df_actual[p_columns].values.max()
        print(f"\n✓ 历史最大值: {historical_max:.2f} kW")
        
        # 3. 目标日期
        target_dates = ['2022-09-16', '2022-09-25']
        
        # 4. 计算每个日期的指标
        results = []
        
        for date_str in target_dates:
            # 提取实际值
            actual_values = extract_day_data(df_actual, date_str, is_prediction=False)
            
            # 提取预测值
            pred_values = extract_day_data(df_pred, date_str, is_prediction=True)
            
            # 计算指标
            metrics = calculate_metrics_for_day(actual_values, pred_values, date_str, historical_max)
            
            if metrics is not None:
                results.append(metrics)
        
        # 5. 保存结果
        if len(results) > 0:
            output_dir = 'pv_prediction_metrics'
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建DataFrame
            df_results = pd.DataFrame(results)
            
            # 保存为Excel
            output_file = os.path.join(output_dir, '光伏出力预测评估指标.xlsx')
            df_results.to_excel(output_file, index=False)
            print(f"\n{'='*60}")
            print(f"✓ 评估指标已保存: {output_file}")
            print(f"{'='*60}")
            
            # 保存为CSV（备份）
            output_csv = os.path.join(output_dir, '光伏出力预测评估指标.csv')
            df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"✓ CSV备份已保存: {output_csv}")
            
            # 打印汇总表格
            print(f"\n{'='*60}")
            print("评估指标汇总表")
            print(f"{'='*60}")
            print(df_results.to_string(index=False))
            print(f"{'='*60}")
            
            # 计算平均指标
            print(f"\n平均指标:")
            print(f"  平均RMSE: {df_results['RMSE (kW)'].mean():.4f} kW")
            print(f"  平均RMSE%: {df_results['RMSE (%)'].mean():.2f}%")
            print(f"  平均MAPE: {df_results['MAPE (%)'].mean():.2f}%")
            print(f"  平均R²: {df_results['R²'].mean():.4f}")
            
        else:
            print("\n错误: 没有成功计算任何指标")
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
