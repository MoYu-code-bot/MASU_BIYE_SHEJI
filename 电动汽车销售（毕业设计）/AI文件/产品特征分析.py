"""
电动汽车产品特征分析模块（优化版）
分析续航里程、电池容量、价格区间、驱动形式、智能化水平、充电效率等产品特征
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from scipy import stats
import warnings

# 关键修复：导入MultiIndex类
from pandas import MultiIndex

warnings.filterwarnings('ignore')

# ==================== 修复字体显示问题 ====================
# 优先选择支持中文和数字的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 确保所有文本使用相同的字体
plt.rcParams['font.family'] = 'sans-serif'
# 调整字体大小，避免拥挤（进一步调小）
plt.rcParams['font.size'] = 9
# 调整子图间距的全局参数
plt.rcParams['figure.subplot.left'] = 0.1
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.1
plt.rcParams['figure.subplot.top'] = 0.9
plt.rcParams['figure.subplot.wspace'] = 0.3
plt.rcParams['figure.subplot.hspace'] = 0.3

# ==================== 配置区域 ====================
IMAGE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\产品特征分析算法图片存储"
FILE_PATHS = {
    'range': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\续航里程分析.xlsx",
    'battery': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\电池容量分析.xlsx",
    'price': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\价格区间分析分析.xlsx",
    'drive': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\驱动形式分析分析.xlsx",
    'smart': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\智能化水平分析.xlsx",
    'charging': r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\AI文件\充电效率分析.xlsx"
}

# 列名映射配置
COLUMN_MAPPINGS = {
    'stat_month': ['stat_month', '统计月份', '月份', '时间', '日期', '统计时间'],
    'range_km': ['range_km', '续航里程', '续航(km)', '里程', '续航', '最大续航', '综合续航', 'avg_range_nedc'],
    'battery_capacity_kwh': ['battery_capacity_kwh', '电池容量', '容量(kwh)', '电池容量kwh', '电池功率', '电池大小', 'avg_range_nedc'],
    'price_range': ['price_range', '价格区间', '价格段', '价格档位', '价位'],
    'drive_type': ['drive_type', '驱动形式', '驱动类型', '驱动方式', '动力类型'],
    'smart_level': ['smart_level', '智能化水平', '智能等级', '智能评分', '智能程度', 'intelligence_level'],
    'charging_efficiency': ['charging_efficiency', '充电效率', '充电效能', '充电速度', '快充效率', 'charging_efficiency_range'],
    'sales_volume': ['sales_volume', '销量', '销售量', '销售数量', '销售台数'],
    'customer_preference': ['customer_preference', '客户偏好', '偏好度', '满意度', '受欢迎度', 'customer_satisfaction'],
    'efficiency_ratio': ['efficiency_ratio', '效率比', '能效比', '电池效率', '能量密度', 'customer_preference_score'],
    'avg_price': ['avg_price', '平均价格', '均价', '价格', '售价', '单价', 'avg_sale_price'],
    'market_share': ['market_share', '市场份额', '份额', '占比'],
    'performance_score': ['performance_score', '性能评分', '动力评分'],
    'energy_efficiency': ['energy_efficiency', '能效', '能源效率'],
    'tech_score': ['tech_score', '技术评分', '科技分', '智能评分', '技术分', 'future_potential'],
    'market_acceptance': ['market_acceptance', '市场接受度', '接受度评分']
}

# CSV中文列名映射
CSV_COLUMN_MAPPING = {
    'stat_month': '统计月份', 'range_km': '续航里程(km)', 'avg_range': '平均续航里程', 'range_sales': '续航相关销量',
    'range_preference': '续航客户偏好', 'customer_preference': '客户偏好', 'avg_battery_capacity': '平均电池容量',
    'battery_capacity_kwh': '电池容量(kWh)', 'avg_battery': '平均电池容量', 'battery_sales': '电池相关销量',
    'battery_efficiency': '电池效率', 'efficiency_ratio': '效率比', 'battery_avg_price': '电池均价',
    'avg_sale_price': '平均售价', 'price_range': '价格区间', 'dominant_price_range': '主导价格区间',
    'price_sales': '价格相关销量', 'avg_price': '平均价格', 'avg_profit_margin': '平均利润率',
    'customer_satisfaction_score': '客户满意度评分', 'drive_type': '驱动形式', 'dominant_drive': '主导驱动形式',
    'drive_sales': '驱动相关销量', 'drive_market_share': '驱动市场份额', 'performance_score': '性能评分',
    'energy_efficiency': '能源效率', 'market_share': '市场份额', 'smart_level': '智能化水平',
    'avg_smart_level': '平均智能化水平', 'smart_sales': '智能相关销量', 'tech_score': '技术评分',
    'avg_tech_score': '平均技术评分', 'charging_efficiency': '充电效率', 'charging_sales': '充电相关销量',
    'charging_time_hours': '充电时间(小时)', 'avg_market_acceptance': '平均市场接受度',
    'market_acceptance': '市场接受度', 'sales_volume': '销量', 'sample_size': '样本数量',
    'product_cluster': '产品聚类', 'feature': '特征', 'importance': '重要性', 'year': '年份'
}

# 聚合配置
AGG_CONFIGS = {
    'range': {'range_km': 'mean', 'sales_volume': 'sum', 'customer_preference': 'mean', 'avg_battery_capacity': 'mean'},
    'battery': {'battery_capacity_kwh': 'mean', 'sales_volume': 'sum', 'efficiency_ratio': 'mean', 'avg_sale_price': 'mean'},
    'price': {'sales_volume': 'sum', 'avg_price': 'mean', 'avg_profit_margin': 'mean'},
    'drive': {'sales_volume': 'sum', 'market_share': 'mean', 'performance_score': 'mean', 'energy_efficiency': 'mean'},
    'smart': {'smart_level': 'mean', 'sales_volume': 'sum', 'tech_score': 'mean'},
    'charging': {'sales_volume': 'sum', 'market_acceptance': 'mean'}
}

RENAME_RULES = {
    'range': {'range_km': 'avg_range', 'sales_volume': 'range_sales', 'customer_preference': 'range_preference'},
    'battery': {'battery_capacity_kwh': 'avg_battery', 'sales_volume': 'battery_sales', 'efficiency_ratio': 'battery_efficiency', 'avg_sale_price': 'battery_avg_price'},
    'price': {'sales_volume': 'price_sales'},
    'drive': {'sales_volume': 'drive_sales', 'market_share': 'drive_market_share'},
    'smart': {'smart_level': 'avg_smart_level', 'sales_volume': 'smart_sales', 'tech_score': 'avg_tech_score'},
    'charging': {'sales_volume': 'charging_sales', 'market_acceptance': 'avg_market_acceptance'}
}

os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
print("=" * 80)
print("电动汽车产品特征分析系统（优化版）")
print("=" * 80)

# ==================== 核心函数 ====================

def find_column(df, possible_names):
    """查找列名（支持模糊匹配）"""
    for name in possible_names:
        if name in df.columns:
            return name
    df_columns = [col.strip().lower() for col in df.columns]
    for target_name in possible_names:
        target_lower = target_name.strip().lower()
        for i, col in enumerate(df_columns):
            if target_lower in col or col in target_lower:
                return df.columns[i]
    return None

def standardize_columns(df):
    """标准化列名"""
    new_df = df.copy()
    rename_dict = {}
    for std_col, possible_names in COLUMN_MAPPINGS.items():
        found_col = find_column(new_df, possible_names)
        if found_col and found_col != std_col:
            rename_dict[found_col] = std_col
    return new_df.rename(columns=rename_dict)

def safe_agg(df, agg_dict, df_name):
    """安全聚合"""
    numeric_cols = {col: pd.to_numeric(df[col], errors='coerce') for col in agg_dict if col in df.columns}
    numeric_cols = {k: v for k, v in numeric_cols.items() if v.notna().sum() > 0}

    if not numeric_cols or 'stat_month' not in df.columns:
        print(f"  ⚠ {df_name} 无有效数据，跳过聚合")
        return pd.DataFrame()

    agg_df = pd.DataFrame(numeric_cols)
    agg_df['stat_month'] = df['stat_month']
    result = agg_df.groupby('stat_month').agg({k: agg_dict[k] for k in numeric_cols}).reset_index()

    if isinstance(result.columns, MultiIndex):
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        result.rename(columns={col: 'stat_month' for col in result.columns if 'stat_month' in col}, inplace=True)

    return result

def load_and_process_data():
    """加载并处理所有数据"""
    print("\n【1/9】加载产品特征数据...")
    data = {}
    for name, path in FILE_PATHS.items():
        try:
            df = pd.read_excel(path)
            df = standardize_columns(df)
            if 'stat_month' in df.columns:
                df['stat_month'] = pd.to_datetime(df['stat_month'], errors='ignore')
            data[name] = df
            print(f"  ✓ {name}: {len(df)}条记录，列数: {len(df.columns)}")
        except Exception as e:
            print(f"  ⚠ {name} 加载失败: {e}")
            data[name] = pd.DataFrame()
    return data

def merge_features(data):
    """合并所有特征"""
    print("\n【2/9】合并产品特征...")
    merged = pd.DataFrame()

    for name, df in data.items():
        if df.empty:
            continue

        agg_df = safe_agg(df, AGG_CONFIGS[name], name)
        if not agg_df.empty:
            agg_df = agg_df.rename(columns=RENAME_RULES.get(name, {}))

            if merged.empty:
                merged = agg_df
            else:
                merged = merged.merge(agg_df, on='stat_month', how='outer', suffixes=('', f'_{name}_dup'))

    # 清理重复列
    dup_cols = [col for col in merged.columns if '_dup' in col]
    for col in dup_cols:
        base = col.split('_')[0]
        if base in merged.columns:
            merged[base] = merged[base].fillna(merged[col])
            merged.drop(col, axis=1, inplace=True)

    # 填充缺失值
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged[col].isnull().any():
            merged[col] = merged[col].fillna(merged[col].mean())

    print(f"  ✓ 合并完成: {merged.shape}")
    return merged

def correlation_analysis(df):
    """特征相关性分析"""
    print("\n  计算特征相关性...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) <= 1:
        print("  ⚠ 数值特征不足")
        return pd.DataFrame()

    corr_matrix = df[numeric_cols].corr()

    # 增大图表尺寸，调整字体大小
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_filtered = corr_matrix.copy()
    corr_filtered[abs(corr_filtered) < 0.1] = 0

    # 调整热力图参数，避免文字重叠
    sns.heatmap(corr_filtered, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot_kws={"size": 8})  # 调小注释字体
    plt.title('产品特征相关性热力图', fontsize=14, fontweight='bold', pad=20)
    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\产品特征相关性热力图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 相关性热力图已保存")

    return corr_matrix

def clustering_analysis(df):
    """产品聚类分析"""
    print("\n【3/9】执行产品聚类分析...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if 'sales' not in col.lower() and '销量' not in col.lower()]

    if len(feature_cols) < 2:
        feature_cols = numeric_cols.tolist()

    available_features = [col for col in feature_cols if df[col].notna().sum() > 0]

    if len(available_features) < 2:
        print("  ⚠ 可用特征不足")
        return df, pd.DataFrame(), None

    print(f"  ✓ 使用 {len(available_features)} 个特征")

    X = df[available_features].fillna(df[available_features].mean())
    # 关键修复：保存拟合后的StandardScaler实例
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 用同一个scaler实例完成fit和transform

    models = {
        'KMeans': KMeans(n_clusters=min(5, len(X)), random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=min(5, len(X)))
    }

    best_score, best_model, best_labels = -1, None, None
    for name, model in models.items():
        try:
            labels = model.fit_predict(X_scaled)
            unique_labels = len(set(labels))
            if 1 < unique_labels < len(X):
                score = silhouette_score(X_scaled, labels)
                print(f"  • {name}: 轮廓系数={score:.3f}")
                if score > best_score:
                    best_score, best_model, best_labels = score, model, labels
        except Exception as e:
            print(f"  ⚠ {name}失败: {e}")

    if best_model is None:
        return df, pd.DataFrame(), None

    df['product_cluster'] = best_labels

    # 关键修复：使用已拟合的scaler实例进行逆变换
    if hasattr(best_model, 'cluster_centers_'):
        # 使用同一个scaler实例逆变换，而不是新建
        centers = pd.DataFrame(scaler.inverse_transform(best_model.cluster_centers_), columns=available_features)
    else:
        centers = pd.DataFrame(X.groupby(best_labels).mean(), columns=available_features)

    print(f"  ✓ 最佳模型: {type(best_model).__name__}, 轮廓系数: {best_score:.3f}")

    # 可视化 - 重点优化文字重叠问题
    if not centers.empty and len(centers) > 1:
        # 增大图表尺寸，增加子图间距
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('产品聚类分析', fontsize=16, fontweight='bold', y=0.98)  # 调整标题位置

        categories = centers.columns.tolist()
        # 特征名映射为中文，并简化过长的名称
        chinese_categories = []
        for cat in categories:
            cn_name = CSV_COLUMN_MAPPING.get(cat, cat)
            # 简化过长的名称
            if len(cn_name) > 6:
                if '续航' in cn_name:
                    cn_name = '续航'
                elif '电池' in cn_name:
                    cn_name = '电池'
                elif '价格' in cn_name:
                    cn_name = '价格'
                elif '智能' in cn_name:
                    cn_name = '智能'
                elif '充电' in cn_name:
                    cn_name = '充电'
                else:
                    cn_name = cn_name[:4] + '...'
            chinese_categories.append(cn_name)

        if len(categories) > 0 and len(centers) <= 8:
            angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
            angles += angles[:1]
            colors = plt.cm.Set3(np.linspace(0, 1, len(centers)))

            ax_polar = plt.subplot(1, 2, 1, projection='polar')
            # 调整极坐标图的边距
            ax_polar.set_theta_offset(np.pi / 2)
            ax_polar.set_theta_direction(-1)

            for i, row in centers.iterrows():
                values = [(v - min(row)) / (max(row) - min(row) + 1e-8) for v in row]
                values += values[:1]
                ax_polar.plot(angles, values, 'o-', linewidth=2, label=f'聚类 {i}', color=colors[i], markersize=4)
                ax_polar.fill(angles, values, alpha=0.25, color=colors[i])

            # 设置极坐标标签
            ax_polar.set_xticks(angles[:-1])
            ax_polar.set_xticklabels(chinese_categories, fontsize=9)
            # 隐藏径向刻度标签，减少视觉干扰
            ax_polar.set_yticklabels([])

            # 调整图例位置，避免重叠
            ax_polar.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
            ax_polar.set_title('聚类中心特征雷达图', y=1.05, fontsize=12)

        # 聚类分布柱状图优化
        cluster_counts = df['product_cluster'].value_counts().sort_index()
        axes[1].bar(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)], alpha=0.8,
                    edgecolor='black')
        axes[1].set_xlabel('聚类', fontsize=10)
        axes[1].set_ylabel('样本数量', fontsize=10)
        axes[1].set_title('聚类分布', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        # 调整刻度标签字体
        axes[1].tick_params(axis='both', labelsize=9)

        # 增加整体边距，避免文字被裁剪
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        plt.savefig(f"{IMAGE_SAVE_PATH}\\产品聚类分析图表.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 聚类图表已保存")

    return df, centers, best_model

def feature_importance_analysis(df):
    """特征重要性分析"""
    print("\n【4/9】分析特征重要性...")

    target_cols = [col for col in df.columns if 'sales' in col.lower() or '销量' in col.lower()]
    target_col = next((col for col in target_cols if df[col].notna().sum() > 5), None)

    if not target_col:
        print("  ⚠ 无可用目标变量")
        return pd.DataFrame(), None

    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in [target_col, 'stat_month', 'product_cluster'] and df[col].notna().sum() > 0]

    if len(feature_cols) < 1:
        print("  ⚠ 无可用特征")
        return pd.DataFrame(), None

    print(f"  ✓ 目标变量: {target_col}, 特征数: {len(feature_cols)}")

    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    y = y[(y >= y.quantile(0.01)) & (y <= y.quantile(0.99))]
    X = X.loc[y.index]

    if len(X) < 10:
        print("  ⚠ 样本不足")
        return pd.DataFrame(), None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
    }

    best_score, best_model = float('inf'), None
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"  • {name}: MSE={mse:.2f}, R²={r2:.3f}")
            if mse < best_score:
                best_score, best_model = mse, model
        except Exception as e:
            print(f"  ⚠ {name}失败: {e}")

    if not best_model:
        return pd.DataFrame(), None

    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)

    print("\n  特征重要性排名:")
    for i, row in importance_df.head(5).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")

    # 新增特征重要性可视化，只保留纵轴最上面5行
    plt.figure(figsize=(12, 6))
    # 过滤掉重要性为0的空值列，并只取前5行
    filtered_importance = importance_df[importance_df['importance'] > 0]
    top_features = filtered_importance.head(5)  # 只保留前5行
    # 映射为中文列名
    top_features['feature_cn'] = top_features['feature'].map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
    # 截断过长的名称
    top_features['feature_cn'] = top_features['feature_cn'].apply(lambda x: x[:8] + '...' if len(x) > 8 else x)

    # 绘制条形图
    ax = sns.barplot(x='importance', y='feature_cn', data=top_features, palette='viridis')
    plt.title('产品特征重要性排名 (Top 5)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('重要性得分', fontsize=10)
    plt.ylabel('特征名称', fontsize=10)
    plt.tick_params(axis='both', labelsize=9)
    plt.grid(axis='x', alpha=0.3)

    # 为每个条形图添加数值标注
    for i, v in enumerate(top_features['importance']):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\产品特征重要性排名.png", dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df, best_model

def price_optimization(df):
    """价格优化分析"""
    print("\n【5/9】执行价格优化分析...")

    price_col = next((col for col in df.columns if any(k in col.lower() for k in ['price', '价格', '均价', '售价'])), None)
    sales_col = next((col for col in df.columns if any(k in col.lower() for k in ['sales', '销量', '销售'])), None)

    if not price_col or not sales_col:
        print("  ⚠ 缺少价格或销量列")
        return {}

    price_data = df[[price_col, sales_col]].dropna()
    if len(price_data) < 5:
        print("  ⚠ 数据不足")
        return {}

    print(f"  ✓ 使用 {price_col} 和 {sales_col}（样本: {len(price_data)}）")

    try:
        for col in [price_col, sales_col]:
            q1, q3 = price_data[col].quantile(0.05), price_data[col].quantile(0.95)
            price_data = price_data[price_data[col].between(q1, q3)]

        if len(price_data) < 3:
            return {}

        log_price, log_sales = np.log(price_data[price_col]), np.log(price_data[sales_col])
        valid_idx = np.isfinite(log_price) & np.isfinite(log_sales)
        log_price, log_sales = log_price[valid_idx], log_sales[valid_idx]

        if len(log_price) < 3:
            return {}

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_sales)
        price_elasticity = slope

        price_data['sales_per_price'] = price_data[sales_col] / price_data[price_col]
        price_data_sorted = price_data.sort_values(price_col)
        window_size = min(3, len(price_data_sorted) // 3)
        price_data_sorted['rolling_avg'] = price_data_sorted['sales_per_price'].rolling(window=window_size, center=True).mean() if window_size >= 1 else price_data_sorted['sales_per_price']

        optimal_price = price_data_sorted.loc[price_data_sorted['rolling_avg'].idxmax(), price_col]
        optimal_range = (optimal_price * 0.9, optimal_price * 1.1)

        print(f"  ✓ 价格弹性: {price_elasticity:.3f} (R²={r_value**2:.3f})")
        print(f"  ✓ 最优价格: ¥{optimal_price:,.2f}")

        # 可视化 - 优化文字显示
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('价格优化分析', fontsize=16, fontweight='bold', y=0.98)

        # 价格-销量散点图
        axes[0].scatter(price_data[price_col], price_data[sales_col], alpha=0.6, s=50, color='blue')
        z = np.polyfit(price_data[price_col], price_data[sales_col], 1)
        axes[0].plot(price_data[price_col], np.poly1d(z)(price_data[price_col]), "r--", alpha=0.8, linewidth=2)
        axes[0].axvline(optimal_price, color='red', linestyle='--', label=f'最优价格: ¥{optimal_price:,.0f}')
        axes[0].axvspan(optimal_range[0], optimal_range[1], alpha=0.2, color='green', label='最优价格区间')
        axes[0].set_xlabel('平均价格 (¥)', fontsize=10)
        axes[0].set_ylabel('销量', fontsize=10)
        axes[0].set_title('价格-销量关系', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='both', labelsize=9)

        # 价格弹性分析文本框 - 优化布局
        elasticity_desc = "富有弹性" if price_elasticity < -1 else "缺乏弹性" if -1 <= price_elasticity < 0 else "吉芬商品" if price_elasticity > 0 else "无弹性"
        text_content = (
            f"价格弹性: {price_elasticity:.3f}\n"
            f"{elasticity_desc}\n\n"
            f"R² = {r_value**2:.3f}\n"
            f"最优价格: ¥{optimal_price:,.2f}\n"
            f"最优区间: ¥{optimal_range[0]:,.2f} - ¥{optimal_range[1]:,.2f}"
        )
        # 调整文本框位置和大小
        axes[1].text(0.05, 0.95, text_content,
                     transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     linespacing=1.2)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('价格弹性分析', fontsize=12, fontweight='bold')

        # 调整子图间距
        plt.subplots_adjust(wspace=0.2)
        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\价格优化分析图表.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 价格分析图表已保存")

        return {
            'price_elasticity': price_elasticity, 'r_squared': r_value**2, 'p_value': p_value,
            'optimal_price': optimal_price, 'optimal_range': optimal_range,
            'price_data': price_data_sorted, 'price_col': price_col, 'sales_col': sales_col
        }
    except Exception as e:
        print(f"  ⚠ 价格分析失败: {e}")
        return {}

def recommendation_system(df):
    """产品推荐系统"""
    print("\n【6/9】构建产品推荐系统...")

    if 'product_cluster' not in df.columns or len(df['product_cluster'].unique()) <= 1:
        print("  ⚠ 无有效聚类结果")
        return None, {}

    feature_keywords = ['range', 'battery', 'price', 'smart', 'charging', '续航', '电池', '价格', '智能', '充电']
    profile_features = [col for col in df.columns if any(k in col.lower() for k in feature_keywords) and df[col].dtype in [np.float64, np.int64] and df[col].notna().sum() > 0]

    if not profile_features:
        print("  ⚠ 无可用特征")
        return None, {}

    print(f"  ✓ 使用 {len(profile_features)} 个特征")

    profiles = {}
    for cluster in df['product_cluster'].unique():
        cluster_data = df[df['product_cluster'] == cluster]
        profiles[f'cluster_{cluster}'] = {**{f: cluster_data[f].mean() for f in profile_features}, 'sample_size': len(cluster_data)}

    def recommend(customer_preferences, top_n=3):
        similarities = {}
        for cluster_id, profile in profiles.items():
            try:
                profile_vec = np.array([profile[f] for f in profile_features])
                pref_values = [customer_preferences.get(f.replace('avg_', '').replace('_', '').lower(), profile[f]) for f in profile_features]
                pref_vec = np.array(pref_values)

                profile_norm = profile_vec / (np.linalg.norm(profile_vec) + 1e-8)
                pref_norm = pref_vec / (np.linalg.norm(pref_vec) + 1e-8)
                similarities[cluster_id] = np.dot(profile_norm, pref_norm)
            except:
                similarities[cluster_id] = 0

        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"  ✓ 创建 {len(profiles)} 个产品档案")
    return recommend, profiles

def trend_analysis(df):
    """技术趋势分析"""
    print("\n【7/9】执行技术趋势分析...")

    if 'stat_month' not in df.columns:
        print("  ⚠ 缺少时间列")
        return {}

    df_sorted = df.sort_values('stat_month').dropna(subset=['stat_month'])
    if len(df_sorted) < 5:
        print("  ⚠ 时间数据不足")
        return {}

    tech_keywords = ['range', 'battery', 'smart', 'charging', 'efficiency', '续航', '电池', '智能', '充电', '效率']
    tech_features = [col for col in df.columns if any(k in col.lower() for k in tech_keywords) and df[col].dtype in [np.float64, np.int64] and df[col].notna().sum() > len(df) * 0.1]

    if not tech_features:
        print("  ⚠ 无可用技术特征")
        return {}

    print(f"  ✓ 识别 {len(tech_features)} 个技术特征")

    insights = {}
    # 增大图表尺寸，优化布局
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('产品技术趋势分析', fontsize=16, fontweight='bold', y=0.98)

    for i, feature in enumerate(tech_features[:4]):
        try:
            feature_data = df_sorted[['stat_month', feature]].dropna()
            if len(feature_data) < 5:
                continue

            feature_data['year'] = feature_data['stat_month'].dt.year
            yearly_avg = feature_data.groupby('year')[feature].mean()

            if len(yearly_avg) < 2:
                continue

            growth_rates = yearly_avg.pct_change().dropna()
            avg_growth = growth_rates.mean() * 100

            x = np.arange(len(yearly_avg))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, yearly_avg.values)

            insights[feature] = {
                'avg_growth_rate': avg_growth, 'trend_slope': slope,
                'r_squared': r_value**2, 'p_value': p_value,
                'is_significant': p_value < 0.05, 'yearly_data': yearly_avg
            }

            row, col = i // 2, i % 2
            ax = axes[row, col]

            # 绘制趋势图
            ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6, color='blue')
            ax.plot(yearly_avg.index, np.poly1d(np.polyfit(yearly_avg.index, yearly_avg.values, 1))(yearly_avg.index), "r--", alpha=0.8, linewidth=2)

            # 简化特征名称
            feature_name = CSV_COLUMN_MAPPING.get(feature, feature)
            if len(feature_name) > 8:
                feature_name = feature_name[:6] + '...'
            ax.set_title(f'{feature_name} 年度趋势', fontsize=11, fontweight='bold')
            ax.set_xlabel('年份', fontsize=9)
            ax.set_ylabel('平均值', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # 调整文本框位置和大小
            significance = "（显著）" if p_value < 0.05 else ""
            text_box = ax.text(0.05, 0.95, f'年均增长率: {avg_growth:.2f}%{significance}',
                               transform=ax.transAxes, fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            # 调整文本框大小
            text_box.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        except Exception as e:
            print(f"  ⚠ {feature} 趋势分析失败: {e}")

    # 隐藏多余子图
    for i in range(len(tech_features), 4):
        axes[i // 2, i % 2].axis('off')

    # 调整子图间距
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\技术趋势分析图表.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 技术趋势图表已保存")

    return insights

def generate_report(df, corr_matrix, cluster_model, centers, importance, price_insights, trend_insights, profiles):
    """生成分析报告"""
    print("\n【8/9】生成可视化图表...")
    print("\n【9/9】生成综合分析报告...")

    report = []
    report.append("=" * 80)
    report.append("电动汽车产品特征分析报告")
    report.append("=" * 80)
    report.append("")

    # 数据概览
    report.append("【数据概览】")
    if 'stat_month' in df.columns:
        report.append(f"  • 分析时间范围: {df['stat_month'].min()} 至 {df['stat_month'].max()}")
    report.append(f"  • 总样本数: {len(df)} 个月度数据点")
    report.append(f"  • 有效数值特征数: {len(df.select_dtypes(include=[np.number]).columns)}")
    report.append("")

    # 聚类分析
    if cluster_model is not None and not centers.empty:
        report.append("【产品聚类分析】")
        report.append(f"  • 聚类数量: {len(centers)}")
        report.append(f"  • 最佳算法: {type(cluster_model).__name__}")
        report.append("")

    # 特征重要性
    if not importance.empty:
        report.append("【特征重要性分析】")
        for i, row in importance.head(5).iterrows():
            report.append(f"    {i+1}. {CSV_COLUMN_MAPPING.get(row['feature'], row['feature'])}: {row['importance']:.3f}")
        report.append("")

    # 价格分析
    if price_insights:
        report.append("【价格优化分析】")
        elasticity = price_insights['price_elasticity']
        desc = "富有弹性" if elasticity < -1 else "缺乏弹性" if -1 <= elasticity < 0 else "吉芬商品" if elasticity > 0 else "无弹性"
        report.append(f"  • 价格弹性: {elasticity:.3f} ({desc})")
        report.append(f"  • 最优价格: ¥{price_insights['optimal_price']:,.2f}")
        report.append(f"  • 最优区间: ¥{price_insights['optimal_range'][0]:,.2f} - ¥{price_insights['optimal_range'][1]:,.2f}")
        report.append("")

    # 技术趋势
    if trend_insights:
        report.append("【技术发展趋势】")
        for feature, insight in trend_insights.items():
            significance = "（显著）" if insight.get('is_significant', False) else ""
            report.append(f"    • {CSV_COLUMN_MAPPING.get(feature, feature)}: {insight['avg_growth_rate']:.2f}%{significance}")
        report.append("")

    # 业务建议
    report.append("【业务建议】")
    if cluster_model is not None:
        report.append("  1. 产品定位: 根据聚类结果将产品分为不同档次，满足不同客户群体需求")
    if price_insights:
        report.append(f"  2. 价格策略: 建议将产品定价控制在 ¥{price_insights['optimal_range'][0]:,.0f} - ¥{price_insights['optimal_range'][1]:,.0f} 区间")
    if not importance.empty:
        report.append(f"  3. 技术研发: 重点投入{CSV_COLUMN_MAPPING.get(importance.iloc[0]['feature'], importance.iloc[0]['feature'])}相关技术研发")
    if profiles:
        report.append("  4. 精准营销: 使用推荐系统为不同偏好的客户推荐匹配的产品")
    report.append("  5. 数据完善: 建议补充更多产品特征数据，提升分析准确性")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)

def save_data(df, corr_matrix, importance, centers):
    """保存数据"""
    print("\n【附加】保存处理后的数据...")

    try:
        if not df.empty:
            df_cn = df.rename(columns=CSV_COLUMN_MAPPING)
            df_cn.to_csv(f"{IMAGE_SAVE_PATH}\\产品特征汇总数据.csv", index=False, encoding='utf-8-sig')
            print(f"  ✓ 产品特征汇总数据已保存（中文列名）")

        if not corr_matrix.empty:
            corr_cn = corr_matrix.copy()
            corr_cn.index = corr_cn.index.map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
            corr_cn.columns = corr_cn.columns.map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
            corr_cn.to_csv(f"{IMAGE_SAVE_PATH}\\产品特征相关性矩阵.csv", encoding='utf-8-sig')
            print(f"  ✓ 产品特征相关性矩阵已保存（中文行列名）")

        if not importance.empty:
            importance_cn = importance.rename(columns=CSV_COLUMN_MAPPING)
            importance_cn.to_csv(f"{IMAGE_SAVE_PATH}\\产品特征重要性.csv", index=False, encoding='utf-8-sig')
            print(f"  ✓ 产品特征重要性已保存（中文列名）")

        if not centers.empty:
            centers_cn = centers.rename(columns=CSV_COLUMN_MAPPING)
            centers_cn.to_csv(f"{IMAGE_SAVE_PATH}\\产品特征聚类结果.csv", encoding='utf-8-sig')
            print(f"  ✓ 产品特征聚类结果已保存（中文列名）")

        print(f"  ✓ 所有结果已保存到: {IMAGE_SAVE_PATH}")
    except Exception as e:
        print(f"  ⚠ 保存数据失败: {e}")

# ==================== 主流程 ====================

def main():
    data = load_and_process_data()
    merged_df = merge_features(data)

    if merged_df.empty:
        print("\n⚠ 无有效数据，分析终止")
        return

    corr_matrix = correlation_analysis(merged_df)
    clustered_df, centers, cluster_model = clustering_analysis(merged_df)
    importance, pred_model = feature_importance_analysis(clustered_df)
    price_insights = price_optimization(clustered_df)
    recommend_func, profiles = recommendation_system(clustered_df)
    trend_insights = trend_analysis(clustered_df)

    report_content = generate_report(clustered_df, corr_matrix, cluster_model, centers, importance, price_insights, trend_insights, profiles)
    print("\n" + report_content)

    with open(f"{IMAGE_SAVE_PATH}\\产品特征分析报告.txt", 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\n✓ 分析报告已保存: {IMAGE_SAVE_PATH}\\产品特征分析报告.txt")

    save_data(clustered_df, corr_matrix, importance, centers)

    print("\n" + "=" * 80)
    print("产品特征分析模块执行完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()