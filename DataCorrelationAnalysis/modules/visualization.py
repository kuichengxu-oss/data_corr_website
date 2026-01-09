"""
可视化绘图模块
提供数值分析和分类分析的可视化功能
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from prince import CA  # 对应分析核心库


class NumericalPlotter:
    """数值分析可视化类"""
    
    @staticmethod
    def plot_correlation_bar_chart(target_corr_sorted, target_col, method, features_per_bar_plot=15):
        """绘制相关性条形图（支持分页显示）"""
        total_features = len(target_corr_sorted)
        figures = []
        
        if total_features <= features_per_bar_plot:
            # 特征较少时，显示单个图
            fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
            colors = ['red' if x < 0 else 'blue' for x in target_corr_sorted.values]
            target_corr_sorted.plot(kind='bar', color=colors, alpha=0.7, ax=ax_bar)
            ax_bar.set_title(f'与 "{target_col}" 的相关性 ({method})', fontsize=14)
            ax_bar.set_ylabel('相关系数')
            ax_bar.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            figures.append(fig_bar)
        else:
            # 特征较多时，分多个图显示
            num_plots = math.ceil(total_features / features_per_bar_plot)
            
            for i in range(num_plots):
                start_idx = i * features_per_bar_plot
                end_idx = min((i + 1) * features_per_bar_plot, total_features)
                subset_data = target_corr_sorted.iloc[start_idx:end_idx]
                
                fig_bar, ax_bar = plt.subplots(figsize=(14, 8))
                colors = ['red' if x < 0 else 'blue' for x in subset_data.values]
                subset_data.plot(kind='bar', color=colors, alpha=0.7, ax=ax_bar)
                
                ax_bar.set_title(f'与 "{target_col}" 的相关性 ({method}) - 第 {i+1}/{num_plots} 部分 (特征 {start_idx+1}-{end_idx})', fontsize=12)
                ax_bar.set_ylabel('相关系数')
                ax_bar.axhline(0, color='black', linewidth=0.8)
                
                # 优化X轴标签显示
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                figures.append(fig_bar)
        
        return figures
    
    @staticmethod
    def plot_correlation_heatmap(numeric_df, target_col, top_features, method='pearson'):
        """绘制相关性热力图"""
        heatmap_cols = [target_col] + top_features
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df[heatmap_cols].corr(method=method), 
                    annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax_heat)
        ax_heat.set_title(f'Top {len(top_features)} 特征相关性热力图')
        plt.tight_layout()
        
        return fig_heat
    
    @staticmethod
    def plot_scatter_regression_grid(numeric_df, target_col, features_to_plot, target_corr_sorted, 
                                   show_scatter_area=True, show_confidence_interval=True):
        """绘制散点回归图网格"""
        cols_per_row = 3
        num_rows = math.ceil(len(features_to_plot) / cols_per_row)
        figures = []
        
        for row in range(num_rows):
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < len(features_to_plot):
                    feature = features_to_plot[idx]
                    
                    fig_scat, ax_scat = plt.subplots(figsize=(5, 4))
                    
                    # 根据开关设置散点透明度
                    if show_scatter_area:
                        scatter_kwargs = {'alpha': 0.4, 's': 30}
                    else:
                        scatter_kwargs = {'alpha': 1.0, 's': 20}
                    
                    # 根据开关设置置信区间
                    ci_param = None if not show_confidence_interval else 95
                    
                    sns.regplot(x=feature, y=target_col, data=numeric_df, ax=ax_scat,
                                scatter_kws=scatter_kwargs, 
                                line_kws={'color': 'red', 'linewidth': 1.5}, ci=ci_param)
                    ax_scat.set_title(f"{feature}\nCorr: {target_corr_sorted[feature]:.3f}")
                    ax_scat.set_xlabel(feature)
                    ax_scat.set_ylabel(target_col)
                    plt.tight_layout()
                    
                    figures.append(fig_scat)
        
        return figures


class CategoricalPlotter:
    """分类分析可视化类"""
    
    @staticmethod
    def plot_cramers_v_heatmap(corr_matrix, cols_for_matrix, target_col):
        """绘制Cramer's V相关性热力图"""
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="Blues",
            xticklabels=cols_for_matrix,
            yticklabels=cols_for_matrix,
            vmin=0,
            vmax=1,
            fmt=".4f",
            ax=ax_heat
        )
        ax_heat.set_title(f"类别列与目标列（{target_col}）相关性热力图（Cramer's V）", fontsize=12)
        plt.tight_layout()
        
        return fig_heat
    
    @staticmethod
    def plot_correspondence_analysis(contingency_table, col_name, target_col, show_point_values=True):
        """绘制对应分析图并返回matplotlib图形对象"""
        # 初始化对应分析模型并拟合
        ca = CA(n_components=2, random_state=42)
        ca.fit(contingency_table)

        # 提取行（类别列）和列（目标列）的坐标
        row_coords = ca.row_coordinates(contingency_table)
        col_coords = ca.column_coordinates(contingency_table)

        # 绘制对应分析图
        fig, ax = plt.subplots(figsize=(5, 4))

        # 1. 绘制行点（类别列）：蓝色
        ax.scatter(row_coords[0], row_coords[1], color='#2E86AB', s=80, label=f'{col_name}（类别列）')
        # 标注行类别名称（根据参数决定是否显示）
        if show_point_values:
            for idx, row in row_coords.iterrows():
                plt.text(row[0] + 0.02, row[1] + 0.02, str(idx), color='#2E86AB', fontsize=10)

        # 2. 绘制列点（目标列）：红色
        ax.scatter(col_coords[0], col_coords[1], color='#A23B72', s=80, label=f'{target_col}（目标列）')
        # 标注列类别名称（根据参数决定是否显示）
        if show_point_values:
            for idx, col in col_coords.iterrows():
                plt.text(col[0] + 0.02, col[1] + 0.02, str(idx), color='#A23B72', fontsize=10)

        # 3. 美化图表
        ax.set_title(f'对应分析：{col_name} vs {target_col}', fontsize=12, pad=15)
        
        # 兼容不同版本的prince库
        if hasattr(ca, 'explained_inertia_'):
            inertia1 = ca.explained_inertia_[0]
            inertia2 = ca.explained_inertia_[1]
        elif hasattr(ca, 'explained_inertia'):
            inertia1 = ca.explained_inertia[0]
            inertia2 = ca.explained_inertia[1]
        else:
            # 如果都没有，计算总惯量解释率
            total_inertia = ca.total_inertia_
            inertia1 = ca.eigenvalues_[0] / total_inertia
            inertia2 = ca.eigenvalues_[1] / total_inertia
        
        ax.set_xlabel(f'维度1（解释率：{inertia1:.2%}）')
        ax.set_ylabel(f'维度2（解释率：{inertia2:.2%}）')
        ax.grid(linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def validate_correspondence_analysis_data(contingency_table):
        """验证对应分析数据是否满足要求"""
        # 检查列联表维度是否满足对应分析要求
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return False, "维度不足，跳过对应分析（需要至少2行2列）"
        
        # 检查是否有足够的有效数据
        if contingency_table.sum().sum() < 10:
            return False, "样本量不足，跳过对应分析（需要至少10个有效样本）"
        
        return True, "适合进行对应分析"


class VisualizationHelper:
    """可视化辅助类"""
    
    @staticmethod
    def display_figures_in_grid(figures, cols_per_row=3):
        """在Streamlit中以网格形式显示图形"""
        if not figures:
            return
        
        num_rows = math.ceil(len(figures) / cols_per_row)
        
        for row in range(num_rows):
            st_cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < len(figures):
                    with st_cols[col]:
                        st.pyplot(figures[idx])
    
    @staticmethod
    def display_figures_with_separators(figures, separator="---"):
        """显示图形并添加分隔符"""
        for i, fig in enumerate(figures):
            st.pyplot(fig)
            if i < len(figures) - 1:
                st.markdown(separator)
    
    @staticmethod
    def get_plot_color_scheme():
        """获取绘图配色方案"""
        return {
            'positive_corr': 'blue',
            'negative_corr': 'red',
            'categorical_rows': '#2E86AB',
            'categorical_cols': '#A23B72',
            'heatmap_coolwarm': 'coolwarm',
            'heatmap_blues': 'Blues'
        }
    
    @staticmethod
    def get_default_figure_sizes():
        """获取默认图形尺寸"""
        return {
            'bar_chart_small': (12, 6),
            'bar_chart_large': (14, 8),
            'heatmap': (10, 8),
            'scatter': (5, 4),
            'correspondence': (5, 4)
        }
