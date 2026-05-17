#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版字体配置文件
使用FontProperties直接指定字体文件路径，确保中文字符正确显示
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import warnings
import os

def get_chinese_font_path():
    """获取系统中文字体文件路径"""
    # Windows系统中文字体路径
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',      # 黑体
        'C:/Windows/Fonts/msyh.ttc',        # 微软雅黑
        'C:/Windows/Fonts/msyhbd.ttc',      # 微软雅黑 Bold
        'C:/Windows/Fonts/simsun.ttc',      # 宋体
        'C:/Windows/Fonts/simkai.ttf',      # 楷体
    ]
    
    # 检查字体文件是否存在
    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"找到中文字体文件: {font_path}")
            return font_path
    
    print("警告: 未找到中文字体文件，将使用默认字体")
    return None

def create_chinese_font_properties(size=10, weight='normal'):
    """创建中文字体属性对象"""
    font_path = get_chinese_font_path()
    
    if font_path:
        try:
            # 使用FontProperties直接指定字体文件
            font_prop = FontProperties(fname=font_path, size=size, weight=weight)
            return font_prop
        except Exception as e:
            print(f"字体文件加载失败: {e}")
            return None
    else:
        return None

def setup_enhanced_chinese_font():
    """设置增强版中文字体配置"""
    
    # 1. 基础字体配置
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 2. 禁用字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='.*missing from font.*')
    warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.backends')
    
    # 3. 强制注册中文字体
    font_path = get_chinese_font_path()
    if font_path:
        try:
            # 注册字体到matplotlib
            fm.fontManager.addfont(font_path)
            matplotlib.font_manager._rebuild()
            print(f"中文字体已注册: {font_path}")
        except Exception as e:
            print(f"字体注册失败: {e}")
    
    # 4. 清除字体缓存
    try:
        fm._rebuild()
        print("字体缓存已清除")
    except:
        print("字体缓存清除失败，但不影响使用")

def get_font_properties(size=10, weight='normal'):
    """获取字体属性对象，用于绘图函数"""
    return create_chinese_font_properties(size, weight)

def test_chinese_font():
    """测试中文字体显示效果"""
    print("=== 测试中文字体显示效果 ===")
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 测试文本
    test_texts = [
        "光伏出力 (PV Output)",
        "基础负荷 (Base Load)", 
        "总负荷 (Total Load)",
        "柔性负荷 (Flexible Load)",
        "削峰填谷效果对比图",
        "基于真实算法结果的削峰填谷效果对比图"
    ]
    
    # 使用FontProperties设置字体
    font_prop = get_font_properties(size=12)
    
    if font_prop:
        # 使用FontProperties
        ax.text(0.1, 0.9, "使用FontProperties:", fontproperties=font_prop, transform=ax.transAxes)
        for i, text in enumerate(test_texts):
            ax.text(0.1, 0.8 - i*0.1, text, fontproperties=font_prop, transform=ax.transAxes)
    else:
        # 使用默认字体
        ax.text(0.1, 0.9, "使用默认字体:", transform=ax.transAxes)
        for i, text in enumerate(test_texts):
            ax.text(0.1, 0.8 - i*0.1, text, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("中文字体显示测试", fontproperties=get_font_properties(size=16, weight='bold'))
    
    # 保存测试图片
    test_filename = "test_chinese_font.png"
    plt.savefig(test_filename, dpi=300, bbox_inches='tight')
    print(f"测试图片已保存到: {test_filename}")
    
    plt.close()
    
    # 检查文件是否生成
    if os.path.exists(test_filename):
        print("中文字体测试完成")
        # 清理测试文件
        os.remove(test_filename)
        print("已清理测试文件")
    else:
        print("中文字体测试失败")

# 自动执行字体设置
setup_enhanced_chinese_font()

if __name__ == "__main__":
    test_chinese_font()
