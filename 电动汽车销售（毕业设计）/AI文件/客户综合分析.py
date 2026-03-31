"""
电动汽车销售客户综合分析模块
整合客户年龄、性别、职业数据，进行多维度客户分析和预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
import tempfile
import shutil
import platform

# 解决多进程中文路径问题
# 设置临时文件夹为纯英文路径
TEMP_DIR = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\temp"
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

# 创建临时文件夹
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# 临时修改multiprocessing的临时目录
tempfile.tempdir = TEMP_DIR

warnings.filterwarnings('ignore')

# ========== 优先配置matplotlib中文显示（关键修复） ==========
# 清除字体缓存（兼容方式）
try:
    from matplotlib.font_manager import _rebuild
    _rebuild()
except:
    pass

# 设置全局字体配置（兼容Windows/macOS/Linux）
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10  # 设置默认字体大小

# 根据系统选择合适的中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']

# 验证字体是否加载成功
print("当前matplotlib字体配置:")
print(f"  字体家族: {plt.rcParams['font.family']}")
print(f"  无衬线字体: {plt.rcParams['font.sans-serif']}")

# 设置图片保存路径
IMAGE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\客户分析算法图片存储"

if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

print("=" * 80)
print("电动汽车销售客户综合分析系统")
print("=" * 80)

# 1. 数据加载和预处理
print("\n【1/7】正在加载客户数据...")


def load_customer_data():
    """
    加载并整合客户年龄、性别、职业数据
    返回合并后的DataFrame
    """
    # 加载三个客户数据文件
    age_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\客户年龄分析分析.xlsx")
    gender_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\客户性别分析.xlsx")
    occupation_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\客户职业分析.xlsx")

    print(f"  ✓ 年龄数据: {len(age_df)} 条记录")
    print(f"  ✓ 性别数据: {len(gender_df)} 条记录")
    print(f"  ✓ 职业数据: {len(occupation_df)} 条记录")

    # 数据清洗和预处理
    # 年龄数据处理
    age_df['age_group_encoded'] = LabelEncoder().fit_transform(age_df['age_group'])

    # 性别数据处理
    gender_df['gender_encoded'] = LabelEncoder().fit_transform(gender_df['user_gender'])

    # 职业数据处理
    occupation_df['occupation_encoded'] = LabelEncoder().fit_transform(occupation_df['user_occupation'])

    # 合并数据（按月份和统计维度合并）
    # 这里我们保留原始数据，分别进行分析，同时创建综合特征

    return age_df, gender_df, occupation_df


age_data, gender_data, occupation_data = load_customer_data()

# 2. 客户细分聚类分析
print("\n【2/7】执行客户细分聚类分析...")


def customer_segmentation_analysis(df, feature_cols, n_clusters=5):
    """
    使用K-Means聚类进行客户细分

    参数:
        df: DataFrame - 输入数据
        feature_cols: list - 用于聚类的特征列
        n_clusters: int - 聚类数量

    返回:
        df: DataFrame - 添加聚类标签的数据
        kmeans: KMeans模型
        scaler: StandardScaler模型
    """
    # 准备特征数据
    X = df[feature_cols].copy()

    # 处理缺失值
    X = X.fillna(X.mean())

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 执行K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 将聚类结果添加到原始数据
    df['cluster'] = cluster_labels

    # 计算轮廓系数评估聚类质量
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"  ✓ K-Means聚类完成 (n_clusters={n_clusters})")
    print(f"  ✓ 轮廓系数: {silhouette_avg:.3f}")

    # 分析每个聚类的特征
    print("\n  各聚类中心特征:")
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols
    )
    for i in range(n_clusters):
        print(f"    聚类 {i}: {cluster_centers.iloc[i].to_dict()}")

    return df, kmeans, scaler


# 对年龄数据进行聚类分析
age_features = ['customer_count', 'total_sales_volume', 'avg_purchase_price', 'age_group_encoded']
age_clustered, kmeans_age, scaler_age = customer_segmentation_analysis(
    age_data, age_features, n_clusters=4
)

# 3. 客户购买行为预测
print("\n【3/7】构建客户购买行为预测模型...")


def build_customer_prediction_model(df, target_col, feature_cols):
    """
    使用随机森林和梯度提升构建客户行为预测模型

    参数:
        df: DataFrame - 输入数据
        target_col: str - 目标变量列名
        feature_cols: list - 特征列名列表

    返回:
        best_model: 最佳预测模型
        X_test: 测试集特征
        y_test: 测试集目标变量
        label_encoder: 标签编码器（用于分类任务）
    """
    # 准备数据
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 处理缺失值
    X = X.fillna(X.mean())

    # 判断是分类任务还是回归任务
    if y.dtype == 'object' or len(y.unique()) < 10:
        # 分类任务
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        is_classification = True
        print(f"  ✓ 任务类型: 分类预测 (目标: {target_col})")
    else:
        # 回归任务
        label_encoder = None
        y_encoded = y
        is_classification = False
        print(f"  ✓ 任务类型: 回归预测 (目标: {target_col})")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    if is_classification:
        # 分类模型：随机森林 + 梯度提升
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        # 超参数调优
        param_grid = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    else:
        # 回归模型（这里简化处理，实际可以使用RandomForestRegressor）
        print("  ⚠ 回归任务将转换为分类任务处理")
        # 将连续值转换为类别
        y_binned = pd.cut(y_encoded, bins=5, labels=False)

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        param_grid = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
            }
        }

        # 重新划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binned, test_size=0.2, random_state=42
        )

    # 训练和评估模型
    best_score = 0
    best_model = None
    best_model_name = ""

    for model_name, model in models.items():
        print(f"\n  训练 {model_name}...")

        # 网格搜索 - 修改n_jobs=1避免多进程中文路径问题
        grid_search = GridSearchCV(
            model, param_grid[model_name], cv=5, scoring='accuracy' if is_classification else 'r2',
            n_jobs=1,  # 关键修改：使用单进程避免中文路径编码问题
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        # 评估
        score = grid_search.best_score_
        print(f"    最佳参数: {grid_search.best_params_}")
        print(f"    交叉验证得分: {score:.3f}")

        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    print(f"\n  ✓ 最佳模型: {best_model_name} (得分: {best_score:.3f})")

    # 在测试集上评估
    y_pred = best_model.predict(X_test)

    if is_classification:
        print("\n  分类报告:")
        print(classification_report(y_test, y_pred, digits=3))

    return best_model, X_test, y_test, label_encoder


# 预测客户偏好的品牌（基于年龄数据）
model_age, X_test_age, y_test_age, encoder_age = build_customer_prediction_model(
    age_clustered, 'preferred_brand',
    ['customer_count', 'total_sales_volume', 'avg_purchase_price', 'age_group_encoded', 'cluster']
)

# 4. 客户生命周期价值分析 (CLV)
print("\n【4/7】计算客户生命周期价值 (CLV)...")


def calculate_customer_lifetime_value(df, customer_id_col='age_group', time_col='stat_month'):
    """
    计算客户生命周期价值

    参数:
        df: DataFrame - 客户数据
        customer_id_col: str - 客户标识列
        time_col: str - 时间列

    返回:
        clv_df: DataFrame - CLV分析结果
    """
    # 转换日期格式
    df[time_col] = pd.to_datetime(df[time_col])

    # 按客户群体计算CLV指标
    clv_metrics = []

    for group in df[customer_id_col].unique():
        group_data = df[df[customer_id_col] == group]

        # 计算关键指标
        total_customers = group_data['customer_count'].sum()
        total_sales = group_data['total_sales_volume'].sum()
        avg_purchase = group_data['avg_purchase_price'].mean()

        # 计算月均购买频率
        months_active = group_data[time_col].nunique()
        purchase_frequency = total_customers / months_active if months_active > 0 else 0

        # 简单CLV计算 (CLV = 平均购买价值 × 购买频率 × 客户生命周期)
        # 假设客户生命周期为12个月
        customer_lifespan = 12
        clv = avg_purchase * purchase_frequency * customer_lifespan

        clv_metrics.append({
            'customer_segment': group,
            'total_customers': total_customers,
            'total_sales_volume': total_sales,
            'avg_purchase_value': avg_purchase,
            'purchase_frequency': purchase_frequency,
            'clv': clv
        })

    clv_df = pd.DataFrame(clv_metrics)
    clv_df = clv_df.sort_values('clv', ascending=False)

    print(f"  ✓ CLV计算完成，共 {len(clv_df)} 个客户群体")
    print(f"  ✓ 最高CLV: ¥{clv_df['clv'].max():,.2f}")
    print(f"  ✓ 最低CLV: ¥{clv_df['clv'].min():,.2f}")

    return clv_df


clv_results = calculate_customer_lifetime_value(age_data)

# 5. 数据可视化
print("\n【5/7】生成客户分析可视化图表...")


def plot_customer_analysis():
    """
    创建客户分析的可视化图表
    """
    # 设置图表样式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 重新应用字体配置，防止样式覆盖
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']

    # 图表1: 客户年龄群体销售趋势
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # 显式指定字体
    fig.suptitle('客户年龄群体销售趋势分析', fontsize=16, fontweight='bold', fontfamily='Microsoft YaHei')

    # 按年龄组统计客户数量趋势
    age_pivot = age_data.pivot_table(
        index='stat_month', columns='age_group',
        values='customer_count', aggfunc='sum'
    )
    age_pivot.plot(ax=axes[0, 0], marker='o', linewidth=2)
    axes[0, 0].set_title('各年龄组客户数量趋势', fontfamily='Microsoft YaHei')
    axes[0, 0].set_ylabel('客户数量', fontfamily='Microsoft YaHei')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(title='年龄组', prop={'family': 'Microsoft YaHei'})

    # 按年龄组统计销售额趋势
    sales_pivot = age_data.pivot_table(
        index='stat_month', columns='age_group',
        values='total_sales_volume', aggfunc='sum'
    )
    sales_pivot.plot(ax=axes[0, 1], marker='s', linewidth=2)
    axes[0, 1].set_title('各年龄组销售额趋势', fontfamily='Microsoft YaHei')
    axes[0, 1].set_ylabel('销售额', fontfamily='Microsoft YaHei')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title='年龄组', prop={'family': 'Microsoft YaHei'})

    # 客户聚类分布
    cluster_counts = age_clustered['cluster'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
    axes[1, 0].pie(cluster_counts.values, labels=cluster_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontfamily': 'Microsoft YaHei'})
    axes[1, 0].set_title('客户聚类分布', fontfamily='Microsoft YaHei')

    # CLV排名
    top_clv = clv_results.head(8)
    bars = axes[1, 1].barh(range(len(top_clv)), top_clv['clv'] / 10000,
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_clv))))
    axes[1, 1].set_yticks(range(len(top_clv)))
    axes[1, 1].set_yticklabels(top_clv['customer_segment'], fontfamily='Microsoft YaHei')
    axes[1, 1].set_xlabel('CLV (万元)', fontfamily='Microsoft YaHei')
    axes[1, 1].set_title('客户群体生命周期价值排名', fontfamily='Microsoft YaHei')
    axes[1, 1].invert_yaxis()

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 1].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                        f'¥{width:.1f}万', ha='left', va='center', fontsize=9, fontfamily='Microsoft YaHei')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\客户年龄分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {IMAGE_SAVE_PATH}\\客户年龄分析图表.png")
    plt.close()

    # 图表2: 性别和职业分析
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('客户性别与职业分析', fontsize=16, fontweight='bold', fontfamily='Microsoft YaHei')

    # 性别销售分布
    gender_sales = gender_data.groupby('user_gender')['total_sales_volume'].sum()
    colors = ['#FF6B6B', '#4ECDC4']
    axes[0, 0].pie(gender_sales.values, labels=gender_sales.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontfamily': 'Microsoft YaHei'})
    axes[0, 0].set_title('性别销售占比分布', fontfamily='Microsoft YaHei')

    # 性别客户数量对比
    gender_counts = gender_data.groupby('user_gender')['customer_count'].sum()
    bars = axes[0, 1].bar(gender_counts.index, gender_counts.values,
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('性别客户数量对比', fontfamily='Microsoft YaHei')
    axes[0, 1].set_ylabel('客户总数', fontfamily='Microsoft YaHei')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, height + 500,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontfamily='Microsoft YaHei')

    # 职业分布热力图
    occupation_pivot = occupation_data.pivot_table(
        index='user_occupation', values='total_sales_volume',
        aggfunc='sum'
    ).sort_values('total_sales_volume', ascending=False)

    # 创建热力图数据
    heatmap_data = occupation_pivot.values.reshape(-1, 1)
    im = axes[1, 0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_yticks(range(len(occupation_pivot)))
    axes[1, 0].set_yticklabels(occupation_pivot.index, fontsize=10, fontfamily='Microsoft YaHei')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_title('职业销售贡献热力图', fontfamily='Microsoft YaHei')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label('销售额', rotation=270, labelpad=15, fontfamily='Microsoft YaHei')

    # 购买力评分分布
    occ_with_score = occupation_data.dropna(subset=['purchase_ability_score'])
    axes[1, 1].scatter(
        occ_with_score['total_sales_volume'],
        occ_with_score['purchase_ability_score'],
        s=occ_with_score['customer_count'] / 10,
        alpha=0.6,
        c=occ_with_score['purchase_ability_score'],
        cmap='viridis'
    )
    axes[1, 1].set_xlabel('销售总量', fontfamily='Microsoft YaHei')
    axes[1, 1].set_ylabel('购买力评分', fontfamily='Microsoft YaHei')
    axes[1, 1].set_title('职业销售vs购买力评分\n(气泡大小=客户数量)', fontfamily='Microsoft YaHei')

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\客户性别职业分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {IMAGE_SAVE_PATH}\\客户性别职业分析图表.png")
    plt.close()


plot_customer_analysis()

# 6. 特征重要性分析
print("\n【6/7】分析特征重要性...")


def analyze_feature_importance(model, feature_cols):
    """
    分析模型中各特征的重要性

    参数:
        model: 训练好的模型
        feature_cols: list - 特征列名

    返回:
        importance_df: DataFrame - 特征重要性排序
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n  特征重要性排名:")
        for idx, row in importance_df.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")

        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance_df['feature'], importance_df['importance'],
                 color=plt.cm.viridis(np.linspace(0, 1, len(importance_df))))
        plt.xlabel('重要性得分', fontfamily='Microsoft YaHei')
        plt.title('客户行为预测特征重要性', fontfamily='Microsoft YaHei')
        plt.gca().invert_yaxis()
        plt.gca().set_yticklabels(importance_df['feature'], fontfamily='Microsoft YaHei')

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9, fontfamily='Microsoft YaHei')

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\特征重要性分析.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 特征重要性图表已保存")
        plt.close()

        return importance_df
    else:
        print("  ⚠ 模型不支持特征重要性分析")
        return None


feature_importance = analyze_feature_importance(
    model_age,
    ['customer_count', 'total_sales_volume', 'avg_purchase_price', 'age_group_encoded', 'cluster']
)

# 7. 生成分析报告
print("\n【7/7】生成客户综合分析报告...")


def generate_customer_report():
    """
    生成综合分析报告
    """
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售客户综合分析报告")
    report.append("=" * 80)
    report.append("")

    # 数据概览
    report.append("【数据概览】")
    report.append(f"  • 分析时间范围: {age_data['stat_month'].min()} 至 {age_data['stat_month'].max()}")
    report.append(f"  • 客户年龄群体: {', '.join(age_data['age_group'].unique())}")
    report.append(f"  • 性别分布: {', '.join(gender_data['user_gender'].unique())}")
    report.append(f"  • 职业类别: {len(occupation_data['user_occupation'].unique())} 种")
    report.append(f"  • 总客户数: {age_data['customer_count'].sum():,} 人")
    report.append(f"  • 总销售额: ¥{age_data['total_sales_volume'].sum():,.2f}")
    report.append("")

    # CLV分析结果
    report.append("【客户生命周期价值 (CLV) 分析】")
    report.append("  排名前三的高价值客户群体:")
    for i, row in clv_results.head(3).iterrows():
        report.append(f"    {i + 1}. {row['customer_segment']}")
        report.append(f"       CLV: ¥{row['clv']:,.2f}, 客户数: {row['total_customers']:,}")
        report.append(f"       平均购买: ¥{row['avg_purchase_value']:,.2f}")
        report.append("")

    # 聚类分析结果
    report.append("【客户聚类分析】")
    report.append(f"  • 最佳聚类数: 4")
    report.append(f"  • 各聚类客户分布:")
    cluster_dist = age_clustered['cluster'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        pct = count / len(age_clustered) * 100
        report.append(f"    聚类 {cluster}: {count} 条记录 ({pct:.1f}%)")
    report.append("")

    # 特征重要性
    if feature_importance is not None:
        report.append("【关键影响因素】")
        report.append("  对客户行为预测最重要的前3个特征:")
        for i, row in feature_importance.head(3).iterrows():
            report.append(f"    {i + 1}. {row['feature']}: {row['importance']:.3f}")
        report.append("")

    # 业务建议
    report.append("【业务建议】")
    report.append("  1. 高价值客户维护: 重点关注CLV排名前三的客户群体")
    report.append("  2. 精准营销: 根据聚类结果制定差异化营销策略")
    report.append("  3. 产品优化: 分析特征重要性，优化影响客户决策的关键因素")
    report.append("  4. 交叉销售: 利用客户细分结果，推荐相关产品")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)


report_content = generate_customer_report()
print(report_content)

# 保存报告到文件
with open(f"{IMAGE_SAVE_PATH}\\客户综合分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 分析报告已保存: {IMAGE_SAVE_PATH}\\客户综合分析报告.txt")

print("\n" + "=" * 80)
print("客户综合分析模块执行完成！")
print("=" * 80)

# 保存处理后的数据供后续模块使用
print("\n【附加】保存处理后的数据...")

# ========== 关键修改：将列名改为中文 ==========
# 1. 客户年龄聚类结果 - 列名映射
age_clustered_cn = age_clustered.copy()
age_column_mapping = {
    'stat_month': '统计月份',
    'age_group': '年龄组',
    'customer_count': '客户数量',
    'total_sales_volume': '销售总额',
    'avg_purchase_price': '平均购买价格',
    'preferred_brand': '偏好品牌',
    'age_group_encoded': '年龄组编码',
    'cluster': '聚类标签'
}
# 只替换存在的列名
age_clustered_cn.rename(columns={k: v for k, v in age_column_mapping.items() if k in age_clustered_cn.columns},
                       inplace=True)
age_clustered_cn.to_csv(f"{IMAGE_SAVE_PATH}\\客户年龄聚类结果.csv", index=False, encoding='utf-8-sig')

# 2. 客户生命周期价值 - 列名映射
clv_results_cn = clv_results.copy()
clv_column_mapping = {
    'customer_segment': '客户群体',
    'total_customers': '客户总数',
    'total_sales_volume': '销售总额',
    'avg_purchase_value': '平均购买价值',
    'purchase_frequency': '购买频率',
    'clv': '客户生命周期价值'
}
clv_results_cn.rename(columns=clv_column_mapping, inplace=True)
clv_results_cn.to_csv(f"{IMAGE_SAVE_PATH}\\客户生命周期价值.csv", index=False, encoding='utf-8-sig')

# 3. 特征重要性分析 - 列名映射
if feature_importance is not None:
    feature_importance_cn = feature_importance.copy()
    feature_column_mapping = {
        'feature': '特征名称',
        'importance': '重要性得分'
    }
    # 替换特征名称为中文
    feature_name_mapping = {
        'customer_count': '客户数量',
        'total_sales_volume': '销售总额',
        'avg_purchase_price': '平均购买价格',
        'age_group_encoded': '年龄组编码',
        'cluster': '聚类标签'
    }
    feature_importance_cn['feature'] = feature_importance_cn['feature'].map(feature_name_mapping).fillna(feature_importance_cn['feature'])
    feature_importance_cn.rename(columns=feature_column_mapping, inplace=True)
    feature_importance_cn.to_csv(f"{IMAGE_SAVE_PATH}\\特征重要性分析.csv", index=False, encoding='utf-8-sig')

# 清理临时文件夹
try:
    shutil.rmtree(TEMP_DIR)
    print(f"  ✓ 临时文件已清理")
except:
    pass

print(f"  ✓ 聚类结果、CLV数据和特征重要性数据已保存为CSV格式到: {IMAGE_SAVE_PATH}")
print("  ✨ 提示：CSV文件使用UTF-8-SIG编码，列名已改为中文，在Excel/WPS中打开不会出现中文乱码")