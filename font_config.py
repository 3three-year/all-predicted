#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局字体配置文件
统一管理matplotlib中文字体设置，避免字体警告
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings

def setup_chinese_font():
    """设置中文字体，支持中文显示，禁用字体警告"""
    
    # 配置matplotlib字体设置，支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 设置plt的字体参数
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 禁用字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='.*missing from font.*')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
    
    # 禁用matplotlib的其他常见警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.backends')

# 自动执行字体设置
setup_chinese_font()
