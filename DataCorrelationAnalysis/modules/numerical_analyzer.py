"""
数值变量分析模块
提供数值变量相关性分析的核心功能
"""
import streamlit as st
import pandas as pd
import numpy as np
import math


class NumericalAnalyzer:
    """数值变量分析器"""
    
    def __init__(self, df):
        """初始化分析器"""
        self.df = df
        self.numeric_df = df.select_dtypes(include=[np.number])
    
    def validate_target_column(self, target_col):
        """验证目标列是否为数值型"""
        if target_col not in self.numeric_df.columns:
            return False, f"目标列 '{target_col}' 不是数值型数据，无法分析。"
        return True, "验证通过"
    
    def calculate_correlation(self, target_col, method='pearson'):
        """计算相关性矩阵"""
        corr_matrix = self.numeric_df.corr(method=method)
        target_corr = corr_matrix[target_col].drop(target_col)
        
        # 按绝对值排序，但保留原始值（以便区分正负相关）
        sorted_indices = target_corr.abs().sort_values(ascending=False).index
        target_corr_sorted = target_corr.loc[sorted_indices]
        
        return target_corr_sorted, corr_matrix
    
    def apply_correlation_threshold(self, target_corr_sorted, threshold=0.0):
        """应用相关性阈值筛选"""
        if threshold > 0:
            filtered_corr = target_corr_sorted[target_corr_sorted.abs() >= threshold]
            filtered_count = len(target_corr_sorted) - len(filtered_corr)
            
            info_msg = ""
            if filtered_count > 0:
                info_msg = f"🔍 相关性阈值筛选：已过滤掉 {filtered_count} 个相关性绝对值小于 {threshold} 的特征"
            
            if len(filtered_corr) == 0:
                return filtered_corr, f"⚠️ 没有特征的相关性绝对值大于等于 {threshold}，请降低阈值查看更多特征"
            
            return filtered_corr, info_msg
        
        return target_corr_sorted, ""
    
    def get_top_features(self, target_corr_sorted, n=10):
        """获取前N个特征"""
        return target_corr_sorted.head(n).index.tolist()
    
    def create_correlation_dataframe(self, target_corr_sorted, n=10):
        """创建相关性数据框用于显示"""
        corr_df = target_corr_sorted.head(n).reset_index()
        corr_df.columns = ['特征列', '相关系数']
        return corr_df


class NumericalAnalysisResult:
    """数值分析结果类"""
    
    def __init__(self, target_corr_sorted, corr_matrix, target_col, method):
        """初始化结果对象"""
        self.target_corr_sorted = target_corr_sorted
        self.corr_matrix = corr_matrix
        self.target_col = target_col
        self.method = method
        self.total_features = len(target_corr_sorted)
    
    def get_summary_stats(self):
        """获取摘要统计信息"""
        return {
            'total_features': self.total_features,
            'positive_corr': (self.target_corr_sorted > 0).sum(),
            'negative_corr': (self.target_corr_sorted < 0).sum(),
            'max_corr': self.target_corr_sorted.abs().max(),
            'min_corr': self.target_corr_sorted.abs().min(),
            'mean_corr': self.target_corr_sorted.abs().mean()
        }


class NumericalAnalysisConfig:
    """数值分析配置类"""
    
    @staticmethod
    def get_default_params():
        """获取默认参数"""
        return {
            'method': 'pearson',
            'top_n_plots': 6,
            'features_per_bar_plot': 15,
            'correlation_threshold': 0.0,
            'show_scatter_area': True,
            'show_confidence_interval': True
        }
    
    @staticmethod
    def get_correlation_methods():
        """获取可用的相关性计算方法"""
        return ['pearson', 'spearman', 'kendall']
    
    @staticmethod
    def validate_params(params):
        """验证参数有效性"""
        valid_methods = NumericalAnalysisConfig.get_correlation_methods()
        
        if params['method'] not in valid_methods:
            raise ValueError(f"不支持的相关性方法: {params['method']}")
        
        if not 0 <= params['correlation_threshold'] <= 1:
            raise ValueError("相关性阈值必须在0到1之间")
        
        if params['top_n_plots'] < 1 or params['top_n_plots'] > 50:
            raise ValueError("散点图数量必须在1到50之间")
        
        return True
