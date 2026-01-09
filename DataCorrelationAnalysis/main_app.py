"""
重构后的数据相关性分析工具主应用
整合数值分析和分类分析功能，提供模块化、可复用的代码结构
"""
import streamlit as st
import pandas as pd
import numpy as np
import math

# 导入自定义模块
from modules.data_loader import DataConfig, DataLoader, DataPreview
from modules.numerical_analyzer import NumericalAnalyzer, NumericalAnalysisResult, NumericalAnalysisConfig
from modules.categorical_analyzer import CategoricalAnalyzer, CategoricalAnalysisResult, CategoricalAnalysisConfig
from modules.visualization import NumericalPlotter, CategoricalPlotter, VisualizationHelper


class StreamlitApp:
    """Streamlit应用主类"""
    
    def __init__(self):
        """初始化应用"""
        self.setup_app()
    
    def setup_app(self):
        """设置应用配置"""
        DataConfig.setup_page_config()
        DataConfig.setup_visualization_config()
    
    def run(self):
        """运行应用"""
        self.render_sidebar()
        self.render_main_content()
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.header("⚙️ 数据导入设置")
        
        # 文件上传
        uploaded_file = st.sidebar.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            self.handle_file_upload(uploaded_file)
        else:
            self.render_welcome_page()
    
    def handle_file_upload(self, uploaded_file):
        """处理文件上传"""
        # 检测文件类型
        file_type = DataLoader.detect_file_type(uploaded_file)
        
        if file_type == 'excel':
            # Excel文件需要选择Sheet
            sheets = DataLoader.get_sheet_names(uploaded_file)
            selected_sheet = st.sidebar.selectbox("选择要分析的 Sheet", options=sheets, index=0)
        else:
            # CSV文件没有Sheet概念
            selected_sheet = None
            st.sidebar.info(f"📄 检测到 {file_type.upper()} 文件")
        
        # 加载数据
        df = DataLoader.load_data(uploaded_file, selected_sheet)
        
        # 验证数据
        is_valid, validation_msg = DataLoader.validate_data(df)
        
        if not is_valid:
            st.error(validation_msg)
            return
        
        # 获取数据预览设置
        preview_settings = DataPreview.get_preview_settings(df)
        
        # 数据预览设置
        st.sidebar.markdown("**数据预览设置**")
        preview_rows = st.sidebar.slider(
            "首页显示数据行数",
            min_value=preview_settings['min_rows'],
            max_value=preview_settings['max_rows'],
            value=preview_settings['default_value'],
            step=preview_settings['step'],
            help=f"控制在首页显示的数据行数（当前数据共{preview_settings['total_rows']}行）"
        )
        
        # 计算动态表格高度
        calculated_height = DataPreview.calculate_table_height(preview_rows)
        
        # 获取列信息
        column_info = DataLoader.get_column_info(df)
        
        # 分析参数配置
        st.sidebar.markdown("---")
        st.sidebar.header("📊 分析参数")
        
        # 分析类型选择
        analysis_type = st.sidebar.radio(
            "选择分析类型",
            options=["数值变量分析", "分类变量分析"],
            index=0,
            key="analysis_type"
        )
        
        # 处理分析类型切换
        self.handle_analysis_type_change(analysis_type)
        
        if analysis_type == "数值变量分析":
            self.render_numerical_analysis_sidebar(df, column_info, selected_sheet, preview_rows, calculated_height, file_type)
        else:
            self.render_categorical_analysis_sidebar(df, column_info, selected_sheet, preview_rows, calculated_height, file_type)
    
    def handle_analysis_type_change(self, analysis_type):
        """处理分析类型改变"""
        if 'last_analysis_type' not in st.session_state:
            st.session_state.last_analysis_type = analysis_type
        elif st.session_state.last_analysis_type != analysis_type:
            # 重置页面显示状态到首页
            st.session_state.show_numeric_analysis = False
            st.session_state.show_categorical_analysis = False
            st.session_state.last_analysis_type = analysis_type
    
    def render_numerical_analysis_sidebar(self, df, column_info, selected_sheet, preview_rows, calculated_height, file_type):
        """渲染数值分析侧边栏"""
        if not column_info['numeric_cols']:
            st.error("没有数值列可供分析。")
            return
        
        # 参数设置
        target_col = st.sidebar.selectbox("目标列 (Y轴)", column_info['numeric_cols'], index=0)
        method = st.sidebar.selectbox("相关系数计算方法", 
                                      options=NumericalAnalysisConfig.get_correlation_methods(), 
                                      index=0)
        top_n = st.sidebar.slider("显示散点图数量", 1, 30, 6)
        features_per_bar = st.sidebar.slider("条形图每页特征数", 5, 30, 15, 
                                           help="控制每个条形图显示的特征数量，特征多时会分页显示")
        correlation_threshold = st.sidebar.slider("相关性阈值筛选", 0.0, 0.9, 0.0, 0.05,
                                                 help="只显示相关性绝对值大于等于此阈值的特征，0.0表示显示所有")
        show_scatter_area = st.sidebar.checkbox("显示散点透明区域", value=True, 
                                               help="开启后散点会有透明效果，便于观察数据密集程度")
        show_confidence_interval = st.sidebar.checkbox("显示置信区间阴影", value=True, 
                                                      help="开启后显示回归线的置信区间红色阴影区域")
        
        # 使用session state来跟踪页面显示状态
        if 'show_numeric_analysis' not in st.session_state:
            st.session_state.show_numeric_analysis = False
        
        # 动态按钮文本
        button_text = "生成数值分析报告" if not st.session_state.show_numeric_analysis else "返回首页"
        
        if st.sidebar.button(button_text, type="primary"):
            st.session_state.show_numeric_analysis = not st.session_state.show_numeric_analysis
            # 立即重新运行以反映状态变化
            st.rerun()
        
        if st.session_state.show_numeric_analysis:
            # 显示分析结果页面
            self.run_numerical_analysis(
                df, target_col, method, top_n, show_scatter_area, 
                show_confidence_interval, features_per_bar, correlation_threshold
            )
        else:
            # 显示首页
            # 显示当前文件信息
            if file_type == 'excel':
                st.info(f"当前 Sheet: **{selected_sheet}**，已加载 {len(df)} 行数据。请点击「生成数值分析报告」开始分析。")
            else:
                st.info(f"已加载 {len(df)} 行数据。请点击「生成数值分析报告」开始分析。")
            st.dataframe(df.head(preview_rows), use_container_width=True, height=calculated_height)
    
    def render_categorical_analysis_sidebar(self, df, column_info, selected_sheet, preview_rows, calculated_height, file_type):
        """渲染分类分析侧边栏"""
        if not column_info['all_cols']:
            st.error("没有数据列可供分析。")
            return
        
        # 获取适合作为目标列的选项
        analyzer = CategoricalAnalyzer(df)
        valid_target_cols, invalid_info = analyzer.get_valid_target_columns()
        
        if not valid_target_cols:
            st.error("❌ 没有适合作为目标列的数据列")
            st.markdown("**不合适的列及原因：**")
            for col, reason in invalid_info.items():
                st.markdown(f"- **{col}**: {reason}")
            return
        
        # 显示列验证信息
        with st.sidebar.expander("📋 列验证信息"):
            suitable_cols_html = ''.join([f'<p style="margin: 2px 0;">✅ {col}</p>' for col in valid_target_cols])
            unsuitable_cols_html = ''.join([f'<p style="margin: 2px 0;">❌ {col}: {reason}</p>' for col, reason in invalid_info.items()])
            
            st.markdown(f"""
            <div style="height: 300px; overflow-y: auto; padding: 0; border: none; background-color: transparent;">
                <p style="font-weight: bold; margin: 0;">适合分析的列：</p>
                {suitable_cols_html}
                {'<p style="font-weight: bold; margin: 5px 0 0 0;">不适合分析的列：</p>' if invalid_info else ''}
                {unsuitable_cols_html}
            </div>
            """, unsafe_allow_html=True)
        
        # 参数设置
        target_col = st.sidebar.selectbox("目标列", valid_target_cols, index=0)
        
        include_numeric = st.sidebar.checkbox(
            "包含数值列作为类别变量", 
            value=False,
            help="开启后可以将唯一值占比较低的数值列识别为类别变量"
        )
        
        threshold = 0.1  # 默认值
        if include_numeric:
            st.sidebar.markdown("**分类变量识别设置**")
            threshold = st.sidebar.slider(
                "数值型列唯一值占比阈值",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="数值型列的唯一值占比低于此值时，将被识别为类别列"
            )
        
        st.sidebar.markdown("**对应分析显示设置**")
        show_point_values = st.sidebar.checkbox(
            "显示散点图点值", 
            value=True,
            help="开启后在对应分析散点图中显示每个点的类别名称"
        )
        
        # 使用session state来跟踪页面显示状态
        if 'show_categorical_analysis' not in st.session_state:
            st.session_state.show_categorical_analysis = False
        
        # 动态按钮文本
        button_text = "生成分类分析报告" if not st.session_state.show_categorical_analysis else "返回首页"
        
        if st.sidebar.button(button_text, type="primary"):
            st.session_state.show_categorical_analysis = not st.session_state.show_categorical_analysis
            # 立即重新运行以反映状态变化
            st.rerun()
        
        if st.session_state.show_categorical_analysis:
            # 显示分析结果页面
            self.run_categorical_analysis(df, target_col, threshold, show_point_values, include_numeric)
        else:
            # 显示首页
            # 显示当前文件信息
            if file_type == 'excel':
                st.info(f"当前 Sheet: **{selected_sheet}**，已加载 {len(df)} 行数据。请点击「生成分类分析报告」开始分析。")
            else:
                st.info(f"已加载 {len(df)} 行数据。请点击「生成分类分析报告」开始分析。")
            st.dataframe(df.head(preview_rows), use_container_width=True, height=calculated_height)
    
    def run_numerical_analysis(self, df, target_col, method, top_n, show_scatter_area, 
                             show_confidence_interval, features_per_bar, correlation_threshold):
        """运行数值分析"""
        # 初始化分析器
        analyzer = NumericalAnalyzer(df)
        
        # 验证目标列
        is_valid, msg = analyzer.validate_target_column(target_col)
        if not is_valid:
            st.error(msg)
            return
        
        # 计算相关性
        target_corr_sorted, corr_matrix = analyzer.calculate_correlation(target_col, method)
        
        # 应用阈值筛选
        filtered_corr, info_msg = analyzer.apply_correlation_threshold(target_corr_sorted, correlation_threshold)
        
        if info_msg and "⚠️" in info_msg:
            st.warning(info_msg)
            return
        elif info_msg:
            st.info(info_msg)
        
        # 创建分析结果对象
        analysis_result = NumericalAnalysisResult(filtered_corr, corr_matrix, target_col, method)
        
        # 显示分析结果
        self.display_numerical_analysis_results(
            analyzer, analysis_result, method, top_n, show_scatter_area, 
            show_confidence_interval, features_per_bar
        )
    
    def display_numerical_analysis_results(self, analyzer, analysis_result, method, top_n, 
                                         show_scatter_area, show_confidence_interval, features_per_bar):
        """显示数值分析结果"""
        target_corr_sorted = analysis_result.target_corr_sorted
        target_col = analysis_result.target_col
        
        st.subheader(f"📊 分析结果: {target_col}")
        
        # 1. 显示相关性排名表格
        st.subheader("1. 相关性排名 (Top 10)")
        corr_df = analyzer.create_correlation_dataframe(target_corr_sorted, 10)
        st.dataframe(corr_df, use_container_width=True)
        
        # 2. 绘制相关性条形图
        st.subheader("2. 相关性条形图")
        total_features = len(target_corr_sorted)
        
        if total_features > features_per_bar:
            st.info(f"📊 共 {total_features} 个特征，将分为 {math.ceil(total_features / features_per_bar)} 个图表显示，每图最多 {features_per_bar} 个特征")
        
        bar_figures = NumericalPlotter.plot_correlation_bar_chart(
            target_corr_sorted, target_col, method, features_per_bar
        )
        VisualizationHelper.display_figures_with_separators(bar_figures)
        
        # 3. 绘制热力图
        st.subheader("3. 局部热力图 (Top Features)")
        top_features = analyzer.get_top_features(target_corr_sorted, 10)
        heatmap_fig = NumericalPlotter.plot_correlation_heatmap(
            analyzer.numeric_df, target_col, top_features, method
        )
        st.pyplot(heatmap_fig)
        
        # 4. 散点图网格
        st.markdown("---")
        st.subheader(f"📈 关键特征分布散点图 (Top {top_n})")
        features_to_plot = top_features[:top_n]
        
        scatter_figures = NumericalPlotter.plot_scatter_regression_grid(
            analyzer.numeric_df, target_col, features_to_plot, target_corr_sorted,
            show_scatter_area, show_confidence_interval
        )
        VisualizationHelper.display_figures_in_grid(scatter_figures, 3)
    
    def run_categorical_analysis(self, df, target_col, threshold, show_point_values, include_numeric):
        """运行分类分析"""
        # 初始化分析器
        analyzer = CategoricalAnalyzer(df)
        
        # 准备分析数据
        analysis_df, cat_cols, info_msg = analyzer.prepare_analysis_data(target_col, threshold, include_numeric)
        
        if analysis_df is None:
            st.warning(f"⚠️ {info_msg}")
            return
        
        st.info(f"🔍 {info_msg}")
        
        # 计算Cramer's V矩阵
        corr_matrix, cols_for_matrix = analyzer.calculate_cramers_v_matrix(analysis_df, cat_cols, target_col)
        
        # 分析分类变量相关性
        analysis_results_df = analyzer.analyze_categorical_correlation(analysis_df, cat_cols, target_col)
        
        # 创建分析结果对象
        analysis_result = CategoricalAnalysisResult(analysis_results_df, corr_matrix, cols_for_matrix, target_col)
        
        # 显示分析结果
        self.display_categorical_analysis_results(analyzer, analysis_result, analysis_df, show_point_values)
    
    def display_categorical_analysis_results(self, analyzer, analysis_result, analysis_df, show_point_values):
        """显示分类分析结果"""
        st.markdown("## 📊 分类变量相关性分析结果")
        
        # 显示分析结果表格
        st.subheader("📋 分类变量相关性分析结果")
        st.dataframe(analysis_result.analysis_results_df, use_container_width=True)
        
        # 绘制相关性热力图
        st.subheader("🎨 分类变量相关性热力图（Cramer's V）")
        heatmap_fig = CategoricalPlotter.plot_cramers_v_heatmap(
            analysis_result.corr_matrix, analysis_result.cols_for_matrix, analysis_result.target_col
        )
        st.pyplot(heatmap_fig)
        
        # 执行对应分析
        st.subheader("📈 对应分析图")
        self.display_correspondence_analysis(analyzer, analysis_result, analysis_df, show_point_values)
    
    def display_correspondence_analysis(self, analyzer, analysis_result, analysis_df, show_point_values):
        """显示对应分析图"""
        cat_cols = [col for col in analysis_result.cols_for_matrix if col != analysis_result.target_col]
        target_col = analysis_result.target_col
        
        cols_per_row = 3
        num_rows = math.ceil(len(cat_cols) / cols_per_row)
        
        for row in range(num_rows):
            st_cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < len(cat_cols):
                    cat_col = cat_cols[idx]
                    contingency_table = pd.crosstab(analysis_df[cat_col], analysis_df[target_col])
                    
                    with st_cols[col]:
                        # 验证对应分析数据
                        is_valid, msg = CategoricalPlotter.validate_correspondence_analysis_data(contingency_table)
                        if not is_valid:
                            st.warning(f"⚠️ {cat_col} {msg}")
                            continue
                        
                        try:
                            # 绘制对应分析图
                            fig_ca = CategoricalPlotter.plot_correspondence_analysis(
                                contingency_table, cat_col, target_col, show_point_values
                            )
                            st.pyplot(fig_ca)
                            st.write(f"**{cat_col} vs {target_col}**")
                        except Exception as e:
                            st.warning(f"⚠️ {cat_col} 的对应分析绘制失败：{str(e)}")
    
    def render_welcome_page(self):
        """渲染欢迎页面"""
        st.title("🚀 数据相关性自动化分析工具")
        st.markdown("### 支持数值变量和分类变量的相关性分析")
        st.info("支持 CSV 和 Excel 文件格式。请在左侧侧边栏上传您的数据文件开始使用。")
    
    def render_main_content(self):
        """渲染主要内容区域"""
        pass  # 主要内容在侧边栏处理中渲染


def main():
    """主函数"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
