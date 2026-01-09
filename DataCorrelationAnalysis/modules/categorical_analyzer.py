"""
分类变量分析模块
提供分类变量相关性分析的核心功能，包括卡方检验、Cramer's V系数和对应分析
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


class CategoricalAnalyzer:
    """分类变量分析器"""
    
    def __init__(self, df):
        """初始化分析器"""
        self.df = df
    
    @staticmethod
    def cramers_v(confusion_matrix):
        """计算Cramer's V系数（量化分类变量相关强度）"""
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum() if isinstance(confusion_matrix, pd.DataFrame) else confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0  # 避免除以0
        return np.sqrt(chi2 / (n * min_dim))
    
    @staticmethod
    def detect_categorical_columns(df, threshold=0.1, include_numeric=False):
        """自动识别DataFrame中的类别列"""
        cat_cols = []
        total_rows = len(df)
        if total_rows == 0:
            return cat_cols

        for col in df.columns:
            # 规则1：直接识别字符串/分类类型列
            if df[col].dtype in ['object', 'category']:
                cat_cols.append(col)
            # 规则2：数值型但唯一值占比低的列（视为类别列）- 只有选择了包含数值列才执行
            elif include_numeric and pd.api.types.is_numeric_dtype(df[col]):
                unique_ratio = df[col].nunique() / total_rows
                if unique_ratio < threshold and df[col].nunique() > 1:  # 至少2个不同值
                    cat_cols.append(col)
        return cat_cols
    
    @staticmethod
    def validate_categorical_column(df, col, min_categories=2, min_samples=10):
        """验证分类列是否适合进行对应分析"""
        # 检查是否有足够的唯一值
        unique_values = df[col].nunique()
        if unique_values < min_categories:
            return False, f"唯一值数量不足（需要至少{min_categories}个，实际{unique_values}个）"
        
        # 检查是否有足够的有效样本
        valid_samples = df[col].notna().sum()
        if valid_samples < min_samples:
            return False, f"有效样本量不足（需要至少{min_samples}个，实际{valid_samples}个）"
        
        return True, "适合分析"
    
    def get_valid_target_columns(self, min_categories=2, min_samples=10):
        """获取所有适合作为目标列的分类列"""
        all_cols = self.df.columns.tolist()
        valid_cols = []
        invalid_info = {}
        
        for col in all_cols:
            is_valid, reason = self.validate_categorical_column(self.df, col, min_categories, min_samples)
            if is_valid:
                valid_cols.append(col)
            else:
                invalid_info[col] = reason
        
        return valid_cols, invalid_info
    
    def prepare_analysis_data(self, target_col, threshold=0.1, include_numeric=False):
        """准备分析数据"""
        # 自动识别类别列（排除目标列本身）
        cat_cols = self.detect_categorical_columns(self.df, threshold, include_numeric)
        cat_cols = [col for col in cat_cols if col != target_col]  # 排除目标列

        if not cat_cols:
            return None, None, "未识别到任何类别列，分析终止"

        # 筛选并保存「类别列+目标列」
        analysis_df = self.df[cat_cols + [target_col]].copy()
        # 去除含缺失值的行（避免统计检验出错）
        analysis_df = analysis_df.dropna(subset=cat_cols + [target_col])
        
        if len(analysis_df) == 0:
            return None, None, "筛选后数据为空（可能缺失值过多）"

        return analysis_df, cat_cols, f"自动识别出 {len(cat_cols)} 个类别列：{', '.join(cat_cols)}"
    
    def calculate_cramers_v_matrix(self, analysis_df, cat_cols, target_col):
        """计算Cramer's V相关性矩阵"""
        n_cols = len(cat_cols) + 1  # 类别列 + 目标列
        corr_matrix = np.eye(n_cols)  # 相关性矩阵
        cols_for_matrix = cat_cols + [target_col]
        
        # 计算所有列两两之间的Cramer's V
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col1, col2 = cols_for_matrix[i], cols_for_matrix[j]
                contingency_table = pd.crosstab(analysis_df[col1], analysis_df[col2])
                v_score = self.cramers_v(contingency_table)
                corr_matrix[i, j] = v_score
                corr_matrix[j, i] = v_score
        
        return corr_matrix, cols_for_matrix
    
    def analyze_categorical_correlation(self, analysis_df, cat_cols, target_col):
        """分析分类变量相关性"""
        analysis_results = []
        
        for cat_col in cat_cols:
            # 生成列联表
            contingency_table = pd.crosstab(analysis_df[cat_col], analysis_df[target_col])

            # 卡方检验（判断是否相关）
            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            # Cramer's V（量化相关强度）
            v_score = self.cramers_v(contingency_table)

            # 结果分类（方便解读）
            correlation_level = "无" if v_score < 0.1 else "弱" if v_score < 0.3 else "中" if v_score < 0.5 else "强"

            # 保存结果
            analysis_results.append({
                "类别列": cat_col,
                "卡方值": round(chi2, 4),
                "p值": round(p_value, 4),
                "Cramer's V": round(v_score, 4),
                "相关性强度": correlation_level,
                "是否显著（p<0.05）": "是" if p_value < 0.05 else "否"
            })
        
        return pd.DataFrame(analysis_results)


class CategoricalAnalysisResult:
    """分类分析结果类"""
    
    def __init__(self, analysis_results_df, corr_matrix, cols_for_matrix, target_col):
        """初始化结果对象"""
        self.analysis_results_df = analysis_results_df
        self.corr_matrix = corr_matrix
        self.cols_for_matrix = cols_for_matrix
        self.target_col = target_col
    
    def get_summary_stats(self):
        """获取摘要统计信息"""
        significant_count = (self.analysis_results_df['是否显著（p<0.05）'] == '是').sum()
        
        correlation_strength_counts = self.analysis_results_df['相关性强度'].value_counts()
        
        return {
            'total_categorical_vars': len(self.analysis_results_df),
            'significant_correlations': significant_count,
            'strong_correlations': correlation_strength_counts.get('强', 0),
            'moderate_correlations': correlation_strength_counts.get('中', 0),
            'weak_correlations': correlation_strength_counts.get('弱', 0),
            'no_correlations': correlation_strength_counts.get('无', 0)
        }
    
    def get_top_correlations(self, n=5):
        """获取相关性最强的前N个变量"""
        return self.analysis_results_df.nlargest(n, 'Cramer\'s V')


class CategoricalAnalysisConfig:
    """分类分析配置类"""
    
    @staticmethod
    def get_default_params():
        """获取默认参数"""
        return {
            'threshold': 0.1,
            'include_numeric': False,
            'show_point_values': True,
            'min_categories': 2,
            'min_samples': 10
        }
    
    @staticmethod
    def get_correlation_levels():
        """获取相关性强度分类标准"""
        return {
            'none': (0, 0.1),
            'weak': (0.1, 0.3),
            'moderate': (0.3, 0.5),
            'strong': (0.5, 1.0)
        }
    
    @staticmethod
    def classify_correlation_strength(v_score):
        """分类相关性强度"""
        if v_score < 0.1:
            return "无"
        elif v_score < 0.3:
            return "弱"
        elif v_score < 0.5:
            return "中"
        else:
            return "强"
