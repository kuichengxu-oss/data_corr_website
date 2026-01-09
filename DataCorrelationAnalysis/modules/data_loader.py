"""
数据加载和基础配置模块
提供CSV和Excel文件读取、数据预处理和全局配置功能
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
import os


class DataConfig:
    """全局配置类"""
    
    @staticmethod
    def setup_page_config():
        """设置Streamlit页面配置"""
        st.set_page_config(page_title="数据相关性分析工具", layout="wide")
    
    @staticmethod
    def setup_visualization_config():
        """设置可视化配置"""
        sns.set_theme(style="whitegrid")
        
        # 处理中文显示和数学符号
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        # 设置数学字体，确保上标等符号正常显示
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        # 忽略卡方检验的小样本警告
        filterwarnings('ignore')


class DataLoader:
    """数据加载器类"""
    
    @staticmethod
    def detect_file_type(file):
        """检测文件类型"""
        if file is None:
            return None
        
        # 获取文件名
        filename = file.name.lower()
        
        # 根据文件扩展名判断类型
        if filename.endswith('.csv'):
            return 'csv'
        elif filename.endswith(('.xlsx', '.xls')):
            return 'excel'
        else:
            return None
    
    @staticmethod
    @st.cache_data
    def get_sheet_names(file):
        """获取 Excel 文件中所有的 Sheet 名称"""
        file_type = DataLoader.detect_file_type(file)
        if file_type != 'excel':
            return None
        
        xls = pd.ExcelFile(file)
        return xls.sheet_names
    
    @staticmethod
    @st.cache_data
    def load_data(file, sheet_name=None):
        """根据文件类型加载数据"""
        file_type = DataLoader.detect_file_type(file)
        
        if file_type == 'csv':
            # CSV文件直接读取
            return pd.read_csv(file)
        elif file_type == 'excel':
            # Excel文件需要指定sheet名称
            if sheet_name is None:
                # 如果没有指定sheet，使用第一个sheet
                xls = pd.ExcelFile(file)
                sheet_name = xls.sheet_names[0]
            return pd.read_excel(file, sheet_name=sheet_name)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    @staticmethod
    def get_column_info(df):
        """获取数据列信息"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        return {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'all_cols': all_cols,
            'total_rows': len(df),
            'total_cols': len(df.columns)
        }
    
    @staticmethod
    def validate_data(df):
        """验证数据是否适合分析"""
        if df.empty:
            return False, "数据为空"
        
        column_info = DataLoader.get_column_info(df)
        
        if not column_info['numeric_cols'] and not column_info['categorical_cols']:
            return False, "没有可分析的数据列"
        
        return True, "数据验证通过"


class DataPreview:
    """数据预览类"""
    
    @staticmethod
    def get_preview_settings(df):
        """获取数据预览设置"""
        max_rows = min(len(df), 1000)  # 最大1000行，避免性能问题
        min_rows = min(10, max_rows)    # 最小值不能超过最大值
        
        # 确保最小值严格小于最大值
        if min_rows == max_rows and max_rows > 1:
            min_rows = max_rows - 1
        
        # 如果只有1行数据，特殊处理
        if max_rows == 1:
            min_rows = 1
            default_value = 1
            step = 1
        else:
            default_value = min(min_rows, max_rows)
            step = 1 if max_rows < 20 else 5
        
        return {
            'min_rows': min_rows,
            'max_rows': max_rows,
            'default_value': default_value,
            'step': step,
            'total_rows': len(df)
        }
    
    @staticmethod
    def calculate_table_height(preview_rows):
        """计算动态表格高度"""
        base_height = 40  # 表头高度
        row_height = 35   # 每行高度
        max_height = 600  # 最大高度限制
        return min(base_height + preview_rows * row_height, max_height)
