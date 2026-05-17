# -*- coding: utf-8 -*-
"""
计算预测结果的评估指标
包括：RMSE（百分比形式）、MAPE、R²
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import os

def calculate_metrics(y_true, y_pred, historical_max):
    """
    计算评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        historical_max: 历史最大值（用于RMSE归一化）
    
    返回:
        dict: 包含RMSE(%)、MAPE(%)、R²的字典
    """
    # 确保是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_percent = (rmse / historical_max) * 100  # 转换为百分比
    
    # 计算MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # 转换为百分比
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'RMSE(%)': rmse_percent,
        'MAPE(%)': mape,
        'R²': r2
    }


def calculate_pv_metrics():
    """
    计算光伏出力预测的评估指标
    """
    print("=" * 60)
    print("计算光伏出力预测指标")
    print("=" * 60)
    
    # 读取两天的预测数据
    file1 = 'DATE_FILE_predict/光伏出力预测_2022年09月16日.xlsx'
    file2 = 'DATE_FILE_predict/光伏出力预测_2022年09月25日.xlsx'
    
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    
    print(f"\n读取文件:")
    print(f"  - {file1}")
    print(f"  - {file2}")
    
    # 提取真实值和预测值
    # 假设列名为'真实值'和'预测值'，需要根据实际情况调整
    if '真实值' in df1.columns and '预测值' in df1.columns:
        true_col = '真实值'
        pred_col = '预测值'
    elif 'actual' in df1.columns and 'predicted' in df1.columns:
        true_col = 'actual'
        pred_col = 'predicted'
    else:
        # 打印列名帮助调试
        print(f"\n可用列名: {df1.columns.tolist()}")
        # 尝试自动识别
        cols = df1.columns.tolist()
        if len(cols) >= 2:
            true_col = cols[0]
            pred_col = cols[1]
            print(f"自动识别: 真实值列='{true_col}', 预测值列='{pred_col}'")
        else:
            raise ValueError("无法识别真实值和预测值列")
    
    # 9月16日
    y_true_day1 = df1[true_col].values
    y_pred_day1 = df1[pred_col].values
    
    # 9月25日
    y_true_day2 = df2[true_col].values
    y_pred_day2 = df2[pred_col].values
    
    # 计算历史最大值（从date_file目录的原始数据中获取）
    try:
        # 读取原始光伏数据
        pv_original = pd.read_csv('date_file/按日累计-台区R1-总光伏出力_扩大2.0倍.csv', encoding='utf-8')
        # 提取P1-P96列
        p_columns = [f'P{i}' for i in range(1, 97)]
        pv_values = pv_original[p_columns].values
        historical_max = np.max(pv_values)
        print(f"\n光伏出力历史最大值: {historical_max:.2f} kW")
    except Exception as e:
        print(f"警告: 无法读取原始数据，使用预测数据的最大值: {e}")
        historical_max = max(np.max(y_true_day1), np.max(y_true_day2))
    
    # 计算指标
    print(f"\n9月16日数据点数: {len(y_true_day1)}")
    metrics_day1 = calculate_metrics(y_true_day1, y_pred_day1, historical_max)
    
    print(f"9月25日数据点数: {len(y_true_day2)}")
    metrics_day2 = calculate_metrics(y_true_day2, y_pred_day2, historical_max)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("光伏出力预测评估指标")
    print(f"{'='*60}")
    print(f"\n9月16日:")
    print(f"  RMSE: {metrics_day1['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day1['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day1['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day1['R²']:.4f}")
    
    print(f"\n9月25日:")
    print(f"  RMSE: {metrics_day2['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day2['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day2['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day2['R²']:.4f}")
    
    # 计算平均指标
    avg_metrics = {
        'RMSE': (metrics_day1['RMSE'] + metrics_day2['RMSE']) / 2,
        'RMSE(%)': (metrics_day1['RMSE(%)'] + metrics_day2['RMSE(%)']) / 2,
        'MAPE(%)': (metrics_day1['MAPE(%)'] + metrics_day2['MAPE(%)']) / 2,
        'R²': (metrics_day1['R²'] + metrics_day2['R²']) / 2
    }
    
    print(f"\n平均指标:")
    print(f"  RMSE: {avg_metrics['RMSE']:.4f} kW")
    print(f"  RMSE(%): {avg_metrics['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {avg_metrics['MAPE(%)']:.2f}%")
    print(f"  R²: {avg_metrics['R²']:.4f}")
    
    return {
        '9月16日': metrics_day1,
        '9月25日': metrics_day2,
        '平均': avg_metrics,
        '历史最大值': historical_max
    }


def calculate_load_metrics():
    """
    计算台区总负荷预测的评估指标
    """
    print("\n" + "=" * 60)
    print("计算台区总负荷预测指标")
    print("=" * 60)
    
    # 读取预测数据
    file = 'DATE_FILE_predict/load_prediction_data.xlsx'
    df = pd.read_excel(file)
    
    print(f"\n读取文件: {file}")
    print(f"数据形状: {df.shape}")
    
    # 分离两天的数据
    df_day1 = df[df['Date'] == '2022-09-16']
    df_day2 = df[df['Date'] == '2022-09-25']
    
    # 提取真实值和预测值
    y_true_day1 = df_day1['Load_Actual'].values
    y_pred_day1 = df_day1['Load_Predicted'].values
    
    y_true_day2 = df_day2['Load_Actual'].values
    y_pred_day2 = df_day2['Load_Predicted'].values
    
    # 计算历史最大值
    try:
        # 读取原始负荷数据
        load_original = pd.read_csv('date_file/按日累计-台区R1-基础负荷.csv', encoding='utf-8')
        p_columns = [f'P{i}' for i in range(1, 97)]
        load_values = load_original[p_columns].values
        historical_max = np.max(load_values)
        print(f"\n台区总负荷历史最大值: {historical_max:.2f} kW")
    except Exception as e:
        print(f"警告: 无法读取原始数据，使用预测数据的最大值: {e}")
        historical_max = max(np.max(y_true_day1), np.max(y_true_day2))
    
    # 计算指标
    print(f"\n9月16日数据点数: {len(y_true_day1)}")
    metrics_day1 = calculate_metrics(y_true_day1, y_pred_day1, historical_max)
    
    print(f"9月25日数据点数: {len(y_true_day2)}")
    metrics_day2 = calculate_metrics(y_true_day2, y_pred_day2, historical_max)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("台区总负荷预测评估指标")
    print(f"{'='*60}")
    print(f"\n9月16日:")
    print(f"  RMSE: {metrics_day1['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day1['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day1['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day1['R²']:.4f}")
    
    print(f"\n9月25日:")
    print(f"  RMSE: {metrics_day2['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day2['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day2['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day2['R²']:.4f}")
    
    # 计算平均指标
    avg_metrics = {
        'RMSE': (metrics_day1['RMSE'] + metrics_day2['RMSE']) / 2,
        'RMSE(%)': (metrics_day1['RMSE(%)'] + metrics_day2['RMSE(%)']) / 2,
        'MAPE(%)': (metrics_day1['MAPE(%)'] + metrics_day2['MAPE(%)']) / 2,
        'R²': (metrics_day1['R²'] + metrics_day2['R²']) / 2
    }
    
    print(f"\n平均指标:")
    print(f"  RMSE: {avg_metrics['RMSE']:.4f} kW")
    print(f"  RMSE(%): {avg_metrics['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {avg_metrics['MAPE(%)']:.2f}%")
    print(f"  R²: {avg_metrics['R²']:.4f}")
    
    return {
        '9月16日': metrics_day1,
        '9月25日': metrics_day2,
        '平均': avg_metrics,
        '历史最大值': historical_max
    }


def calculate_ev_metrics():
    """
    计算EV充电负荷预测的评估指标
    """
    print("\n" + "=" * 60)
    print("计算EV充电负荷预测指标")
    print("=" * 60)
    
    # 读取预测数据
    file = 'DATE_FILE_predict/ev_charging_prediction_data.xlsx'
    df = pd.read_excel(file)
    
    print(f"\n读取文件: {file}")
    print(f"数据形状: {df.shape}")
    
    # 分离两天的数据
    df_day1 = df[df['Date'] == '2022-09-16']
    df_day2 = df[df['Date'] == '2022-09-25']
    
    # 提取真实值和预测值
    y_true_day1 = df_day1['EV_Actual'].values
    y_pred_day1 = df_day1['EV_Predicted'].values
    
    y_true_day2 = df_day2['EV_Actual'].values
    y_pred_day2 = df_day2['EV_Predicted'].values
    
    # 计算历史最大值
    try:
        # 读取原始EV数据
        ev_original = pd.read_csv('date_file/按日累计-台区R1-EV充电负荷_平滑后.csv', encoding='utf-8')
        p_columns = [f'P{i}' for i in range(1, 97)]
        ev_values = ev_original[p_columns].values
        historical_max = np.max(ev_values)
        print(f"\nEV充电负荷历史最大值: {historical_max:.2f} kW")
    except Exception as e:
        print(f"警告: 无法读取原始数据，使用预测数据的最大值: {e}")
        historical_max = max(np.max(y_true_day1), np.max(y_true_day2))
    
    # 计算指标
    print(f"\n9月16日数据点数: {len(y_true_day1)}")
    metrics_day1 = calculate_metrics(y_true_day1, y_pred_day1, historical_max)
    
    print(f"9月25日数据点数: {len(y_true_day2)}")
    metrics_day2 = calculate_metrics(y_true_day2, y_pred_day2, historical_max)
    
    # 打印结果
    print(f"\n{'='*60}")
    print("EV充电负荷预测评估指标")
    print(f"{'='*60}")
    print(f"\n9月16日:")
    print(f"  RMSE: {metrics_day1['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day1['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day1['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day1['R²']:.4f}")
    
    print(f"\n9月25日:")
    print(f"  RMSE: {metrics_day2['RMSE']:.4f} kW")
    print(f"  RMSE(%): {metrics_day2['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {metrics_day2['MAPE(%)']:.2f}%")
    print(f"  R²: {metrics_day2['R²']:.4f}")
    
    # 计算平均指标
    avg_metrics = {
        'RMSE': (metrics_day1['RMSE'] + metrics_day2['RMSE']) / 2,
        'RMSE(%)': (metrics_day1['RMSE(%)'] + metrics_day2['RMSE(%)']) / 2,
        'MAPE(%)': (metrics_day1['MAPE(%)'] + metrics_day2['MAPE(%)']) / 2,
        'R²': (metrics_day1['R²'] + metrics_day2['R²']) / 2
    }
    
    print(f"\n平均指标:")
    print(f"  RMSE: {avg_metrics['RMSE']:.4f} kW")
    print(f"  RMSE(%): {avg_metrics['RMSE(%)']:.2f}%")
    print(f"  MAPE(%): {avg_metrics['MAPE(%)']:.2f}%")
    print(f"  R²: {avg_metrics['R²']:.4f}")
    
    return {
        '9月16日': metrics_day1,
        '9月25日': metrics_day2,
        '平均': avg_metrics,
        '历史最大值': historical_max
    }


def main():
    """
    主函数
    """
    print("\n" + "=" * 60)
    print("预测结果评估指标计算")
    print("=" * 60)
    print("\n将计算以下指标:")
    print("  1. RMSE (Root Mean Square Error)")
    print("  2. RMSE(%) (归一化RMSE，除以历史最大值)")
    print("  3. MAPE(%) (Mean Absolute Percentage Error)")
    print("  4. R² (决定系数)")
    print("\n" + "=" * 60)
    
    # 创建输出目录
    output_dir = 'prediction_metrics_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}/")
    
    # 计算三类预测指标
    pv_results = calculate_pv_metrics()
    load_results = calculate_load_metrics()
    ev_results = calculate_ev_metrics()
    
    # 保存光伏出力预测结果
    output_file_pv = os.path.join(output_dir, '光伏出力预测评估指标.xlsx')
    results_df_pv = pd.DataFrame({
        '日期': ['9月16日', '9月25日', '平均'],
        'RMSE (kW)': [
            pv_results['9月16日']['RMSE'],
            pv_results['9月25日']['RMSE'],
            pv_results['平均']['RMSE']
        ],
        'RMSE (%)': [
            pv_results['9月16日']['RMSE(%)'],
            pv_results['9月25日']['RMSE(%)'],
            pv_results['平均']['RMSE(%)']
        ],
        'MAPE (%)': [
            pv_results['9月16日']['MAPE(%)'],
            pv_results['9月25日']['MAPE(%)'],
            pv_results['平均']['MAPE(%)']
        ],
        'R²': [
            pv_results['9月16日']['R²'],
            pv_results['9月25日']['R²'],
            pv_results['平均']['R²']
        ]
    })
    info_df_pv = pd.DataFrame({
        '日期': ['说明'],
        'RMSE (kW)': [f"历史最大值: {pv_results['历史最大值']:.2f} kW"],
        'RMSE (%)': ['RMSE / 历史最大值 × 100%'],
        'MAPE (%)': ['平均绝对百分比误差'],
        'R²': ['决定系数（越接近1越好）']
    })
    final_df_pv = pd.concat([results_df_pv, info_df_pv], ignore_index=True)
    final_df_pv.to_excel(output_file_pv, index=False)
    print(f"\n✓ 保存结果: {output_file_pv}")
    
    # 保存台区总负荷预测结果
    output_file_load = os.path.join(output_dir, '台区总负荷预测评估指标.xlsx')
    results_df_load = pd.DataFrame({
        '日期': ['9月16日', '9月25日', '平均'],
        'RMSE (kW)': [
            load_results['9月16日']['RMSE'],
            load_results['9月25日']['RMSE'],
            load_results['平均']['RMSE']
        ],
        'RMSE (%)': [
            load_results['9月16日']['RMSE(%)'],
            load_results['9月25日']['RMSE(%)'],
            load_results['平均']['RMSE(%)']
        ],
        'MAPE (%)': [
            load_results['9月16日']['MAPE(%)'],
            load_results['9月25日']['MAPE(%)'],
            load_results['平均']['MAPE(%)']
        ],
        'R²': [
            load_results['9月16日']['R²'],
            load_results['9月25日']['R²'],
            load_results['平均']['R²']
        ]
    })
    info_df_load = pd.DataFrame({
        '日期': ['说明'],
        'RMSE (kW)': [f"历史最大值: {load_results['历史最大值']:.2f} kW"],
        'RMSE (%)': ['RMSE / 历史最大值 × 100%'],
        'MAPE (%)': ['平均绝对百分比误差'],
        'R²': ['决定系数（越接近1越好）']
    })
    final_df_load = pd.concat([results_df_load, info_df_load], ignore_index=True)
    final_df_load.to_excel(output_file_load, index=False)
    print(f"✓ 保存结果: {output_file_load}")
    
    # 保存EV充电负荷预测结果
    output_file_ev = os.path.join(output_dir, 'EV充电负荷预测评估指标.xlsx')
    results_df_ev = pd.DataFrame({
        '日期': ['9月16日', '9月25日', '平均'],
        'RMSE (kW)': [
            ev_results['9月16日']['RMSE'],
            ev_results['9月25日']['RMSE'],
            ev_results['平均']['RMSE']
        ],
        'RMSE (%)': [
            ev_results['9月16日']['RMSE(%)'],
            ev_results['9月25日']['RMSE(%)'],
            ev_results['平均']['RMSE(%)']
        ],
        'MAPE (%)': [
            ev_results['9月16日']['MAPE(%)'],
            ev_results['9月25日']['MAPE(%)'],
            ev_results['平均']['MAPE(%)']
        ],
        'R²': [
            ev_results['9月16日']['R²'],
            ev_results['9月25日']['R²'],
            ev_results['平均']['R²']
        ]
    })
    info_df_ev = pd.DataFrame({
        '日期': ['说明'],
        'RMSE (kW)': [f"历史最大值: {ev_results['历史最大值']:.2f} kW"],
        'RMSE (%)': ['RMSE / 历史最大值 × 100%'],
        'MAPE (%)': ['平均绝对百分比误差'],
        'R²': ['决定系数（越接近1越好）']
    })
    final_df_ev = pd.concat([results_df_ev, info_df_ev], ignore_index=True)
    final_df_ev.to_excel(output_file_ev, index=False)
    print(f"✓ 保存结果: {output_file_ev}")
    
    # 创建综合对比表
    output_file_summary = os.path.join(output_dir, '预测评估指标综合对比.xlsx')
    
    with pd.ExcelWriter(output_file_summary, engine='openpyxl') as writer:
        # 光伏出力
        final_df_pv.to_excel(writer, sheet_name='光伏出力预测', index=False)
        # 台区总负荷
        final_df_load.to_excel(writer, sheet_name='台区总负荷预测', index=False)
        # EV充电负荷
        final_df_ev.to_excel(writer, sheet_name='EV充电负荷预测', index=False)
        
        # 综合对比表
        summary_df = pd.DataFrame({
            '预测类型': ['光伏出力', '台区总负荷', 'EV充电负荷'],
            'RMSE (kW)': [
                pv_results['平均']['RMSE'],
                load_results['平均']['RMSE'],
                ev_results['平均']['RMSE']
            ],
            'RMSE (%)': [
                pv_results['平均']['RMSE(%)'],
                load_results['平均']['RMSE(%)'],
                ev_results['平均']['RMSE(%)']
            ],
            'MAPE (%)': [
                pv_results['平均']['MAPE(%)'],
                load_results['平均']['MAPE(%)'],
                ev_results['平均']['MAPE(%)']
            ],
            'R²': [
                pv_results['平均']['R²'],
                load_results['平均']['R²'],
                ev_results['平均']['R²']
            ],
            '历史最大值 (kW)': [
                pv_results['历史最大值'],
                load_results['历史最大值'],
                ev_results['历史最大值']
            ]
        })
        summary_df.to_excel(writer, sheet_name='综合对比', index=False)
    
    print(f"✓ 保存综合对比: {output_file_summary}")
    
    print("\n" + "=" * 60)
    print("✅ 所有预测指标计算完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  1. {output_file_pv}")
    print(f"  2. {output_file_load}")
    print(f"  3. {output_file_ev}")
    print(f"  4. {output_file_summary}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
