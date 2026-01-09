"""
重构后的数据相关性分析工具
模块化、可复用的代码结构

项目结构：
├── main_app.py                 # 主应用入口
├── modules/                    # 核心模块目录
│   ├── __init__.py            # 模块初始化文件
│   ├── data_loader.py         # 数据加载和配置模块
│   ├── numerical_analyzer.py  # 数值分析模块
│   ├── categorical_analyzer.py # 分类分析模块
│   └── visualization.py       # 可视化绘图模块
├── requirements.txt           # 依赖包列表
└── README.md                  # 项目说明文档

主要改进：
1. 模块化设计：将功能拆分为独立的模块，提高代码复用性
2. 面向对象：使用类来组织相关功能，提高代码结构清晰度
3. 职责分离：每个模块负责特定的功能领域
4. 配置集中：将配置参数集中管理，便于维护
5. 错误处理：改进了错误处理和验证机制
6. 可扩展性：便于添加新的分析方法和可视化功能

使用方法：
1. 安装依赖：pip install -r requirements.txt
2. 运行应用：streamlit run main_app.py
"""

# 模块初始化文件
