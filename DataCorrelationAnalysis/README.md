# 数据相关性分析工具 - 重构版

## 项目概述

这是一个基于Streamlit的数据相关性分析工具，支持数值变量和分类变量的相关性分析。本项目采用模块化设计，提供了清晰、可复用的代码结构。

## 主要功能

### 数值变量分析
- 支持Pearson、Spearman、Kendall相关系数计算
- 相关性排名和可视化
- 相关性条形图（支持分页显示）
- 相关性热力图
- 散点回归图网格

### 分类变量分析
- 卡方检验
- Cramer's V系数计算
- 分类变量相关性热力图
- 对应分析（Correspondence Analysis）
- 自动识别分类变量

## 项目结构

```
refactored_code/
├── main_app.py                 # 主应用入口
├── modules/                    # 核心模块目录
│   ├── __init__.py            # 模块初始化文件
│   ├── data_loader.py         # 数据加载和配置模块
│   ├── numerical_analyzer.py  # 数值分析模块
│   ├── categorical_analyzer.py # 分类分析模块
│   └── visualization.py       # 可视化绘图模块
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明文档
```

## 模块说明

### 1. data_loader.py
- **DataConfig**: 全局配置类，负责页面配置和可视化设置
- **DataLoader**: 数据加载器，提供Excel文件读取和数据验证功能
- **DataPreview**: 数据预览类，处理数据预览相关设置

### 2. numerical_analyzer.py
- **NumericalAnalyzer**: 数值分析器，负责相关性计算和分析
- **NumericalAnalysisResult**: 分析结果类，封装分析结果和统计信息
- **NumericalAnalysisConfig**: 配置类，管理分析参数和验证

### 3. categorical_analyzer.py
- **CategoricalAnalyzer**: 分类分析器，负责分类变量分析
- **CategoricalAnalysisResult**: 分类分析结果类
- **CategoricalAnalysisConfig**: 分类分析配置类

### 4. visualization.py
- **NumericalPlotter**: 数值分析可视化类
- **CategoricalPlotter**: 分类分析可视化类
- **VisualizationHelper**: 可视化辅助类，提供通用绘图功能

### 5. main_app.py
- **StreamlitApp**: 主应用类，整合所有模块，提供完整的用户界面

## 安装和使用

### 1. 环境要求
- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行应用
```bash
streamlit run main_app.py
```

## 主要改进

### 1. 模块化设计
- 将原本单一文件的功能拆分为多个独立模块
- 每个模块负责特定的功能领域
- 提高代码的可维护性和可扩展性

### 2. 面向对象编程
- 使用类来组织相关功能
- 提供清晰的接口和封装
- 便于代码复用和测试

### 3. 职责分离
- 数据加载、分析计算、可视化渲染分别由不同模块负责
- 降低模块间的耦合度
- 便于独立开发和测试

### 4. 配置管理
- 集中管理配置参数
- 提供默认值和验证机制
- 便于参数调整和维护

### 5. 错误处理
- 改进了数据验证和错误处理机制
- 提供更友好的错误提示
- 增强应用的稳定性

### 6. 可扩展性
- 便于添加新的分析方法
- 易于扩展可视化功能
- 支持插件式功能扩展

## 使用示例

### 数值变量分析
1. 上传Excel文件
2. 选择"数值变量分析"
3. 选择目标列和分析参数
4. 查看相关性分析结果

### 分类变量分析
1. 上传Excel文件
2. 选择"分类变量分析"
3. 选择目标列和识别参数
4. 查看分类变量分析结果

## 技术特点

- **高性能**: 使用Streamlit缓存机制优化数据加载
- **用户友好**: 直观的界面设计和交互体验
- **功能完整**: 支持多种分析方法和可视化形式
- **代码质量**: 遵循Python编程规范，代码结构清晰
- **可维护**: 模块化设计便于维护和扩展

## 依赖包

- streamlit: Web应用框架
- pandas: 数据处理
- numpy: 数值计算
- matplotlib: 基础绘图
- seaborn: 统计可视化
- scipy: 科学计算
- prince: 对应分析
- openpyxl: Excel文件处理

## 注意事项

1. 确保上传的Excel文件格式正确
2. 分类变量分析需要足够的数据量
3. 对应分析需要至少2行2列的列联表
4. 建议使用现代浏览器以获得最佳体验

## 开发团队

本项目采用模块化重构，提高了代码的可维护性和可扩展性，为后续功能开发奠定了良好的基础。
