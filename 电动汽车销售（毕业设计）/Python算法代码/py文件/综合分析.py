"""
电动汽车销售综合分析与关联分析模块
整合所有分析模块的结果，进行全局关联分析和洞察生成
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.transforms as mtransforms
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("警告: adjustText 库未安装，将使用简单标签。安装命令: pip install adjustText")

import platform
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# 终极修复：彻底解决字体渲染问题
# ==============================================
# 跨平台中文字体配置
system = platform.system()
if system == 'Windows':
    CHINESE_FONT = 'Microsoft YaHei'
elif system == 'Darwin':
    CHINESE_FONT = 'PingFang SC'
else:
    CHINESE_FONT = 'WenQuanYi Micro Hei'

# 1. 优先使用系统中最稳定的中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.formatter.use_mathtext'] = False

# ========== 关键修改：拆分图片和文档存储路径 ==========
# 基础存储路径
BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\综合分析"
# 图片存储路径
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
# 文档存储路径
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")

# 创建文件夹（如果不存在）
for path in [IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

print("=" * 80)
print("电动汽车销售综合分析与关联分析系统")
print("=" * 80)

# 1. 加载各模块的分析结果
print("\n【1/6】加载各模块分析结果...")

def load_analysis_results():
    """
    加载所有分析模块的结果数据

    返回:
        results: dict - 包含所有分析结果的字典
    """
    results = {}

    # 定义各模块CSV文件的基础路径
    BASE_CSV_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储"
    paths_config = {
        "customer": os.path.join(BASE_CSV_PATH, "客户综合分析", "文档分析"),
        "product": os.path.join(BASE_CSV_PATH, "产品特征分析", "文档分析"),
        "profit_transfer": os.path.join(BASE_CSV_PATH, "利润过户分析", "文档分析"),
        "sales_trend": os.path.join(BASE_CSV_PATH, "销售趋势分析", "文档分析")
    }

    # 客户分析结果
    try:
        customer_csv_path = paths_config["customer"]
        customer_cluster = pd.read_csv(os.path.join(customer_csv_path, "客户年龄聚类结果.csv"), encoding='utf-8-sig')
        customer_clv = pd.read_csv(os.path.join(customer_csv_path, "客户生命周期价值.csv"), encoding='utf-8-sig')
        results['customer'] = {
            'cluster_data': customer_cluster,
            'clv_data': customer_clv
        }
        print(f"  ✓ 客户分析结果已加载 ({len(customer_cluster)} 条记录)")
    except Exception as e:
        print(f"  ⚠ 客户分析结果加载失败: {e}")
        results['customer'] = None

    # 销售趋势分析结果
    try:
        sales_trend_csv_path = paths_config["sales_trend"]
        prophet_forecast = pd.read_csv(os.path.join(sales_trend_csv_path, "销售预测结果.csv"), encoding='utf-8-sig')
        regional_sales = pd.read_csv(os.path.join(sales_trend_csv_path, "地区销售汇总.csv"), encoding='utf-8-sig')
        results['sales_trend'] = {
            'forecast': prophet_forecast,
            'regional_sales': regional_sales
        }
        print(f"  ✓ 销售趋势分析结果已加载")
    except Exception as e:
        print(f"  ⚠ 销售趋势分析结果加载失败: {e}")
        results['sales_trend'] = None

    # 产品特征分析结果
    try:
        product_csv_path = paths_config["product"]
        product_cluster = pd.read_csv(os.path.join(product_csv_path, "产品特征聚类结果.csv"), encoding='utf-8-sig')
        feature_correlation = pd.read_csv(os.path.join(product_csv_path, "产品特征相关性矩阵.csv"), index_col=0, encoding='utf-8-sig')
        feature_importance = pd.read_csv(os.path.join(product_csv_path, "产品特征重要性.csv"), encoding='utf-8-sig')

        # 将产品特征的英文名称映射为中文
        feature_name_map = {
            'drive_sales': '续航里程',
            'battery_sales': '电池容量',
            'price_sales': '销售价格',
            'smart_sales': '智能配置',
            'charging_sales': '充电速度',
            'avg_battery': '平均续航'
        }
        if '特征' in feature_importance.columns:
            feature_importance['特征'] = feature_importance['特征'].map(feature_name_map).fillna(feature_importance['特征'])
        elif '特征名称' in feature_importance.columns:
            feature_importance['特征名称'] = feature_importance['特征名称'].map(feature_name_map).fillna(feature_importance['特征名称'])

        results['product'] = {
            'cluster_data': product_cluster,
            'correlation': feature_correlation,
            'importance': feature_importance
        }
        print(f"  ✓ 产品特征分析结果已加载 ({len(product_cluster)} 条记录)")
    except Exception as e:
        print(f"  ⚠ 产品特征分析结果加载失败: {e}")
        results['product'] = None

    # 利润过户分析结果
    try:
        profit_csv_path = paths_config["profit_transfer"]
        brand_profit = pd.read_csv(os.path.join(profit_csv_path, "品牌利润分析.csv"), index_col=0, encoding='utf-8-sig')
        brand_transfer = pd.read_csv(os.path.join(profit_csv_path, "品牌过户分析.csv"), index_col=0, encoding='utf-8-sig')
        profit_importance = pd.read_csv(os.path.join(profit_csv_path, "利润预测特征重要性.csv"), encoding='utf-8-sig')
        results['profit_transfer'] = {
            'brand_profit': brand_profit,
            'brand_transfer': brand_transfer,
            'profit_importance': profit_importance
        }
        print(f"  ✓ 利润过户分析结果已加载")
    except Exception as e:
        print(f"  ⚠ 利润过户分析结果加载失败: {e}")
        results['profit_transfer'] = None

    return results

analysis_results = load_analysis_results()

# 2. 跨模块关联分析
print("\n【2/6】执行跨模块关联分析...")

def cross_module_correlation_analysis():
    """
    分析不同模块之间的关联关系

    返回:
        correlations: dict - 跨模块关联分析结果
    """
    correlations = {}

    # 地区维度关联分析（原品牌维度，但CSV中是地区数据）
    if (analysis_results['sales_trend'] is not None and
        analysis_results['profit_transfer'] is not None):

        try:
            # 从地区销售数据中提取地区销售数据
            regional_sales = analysis_results['sales_trend']['regional_sales']
            region_sales = regional_sales.groupby('地区')['总销售量'].sum()

            # 品牌利润数据
            brand_profit = analysis_results['profit_transfer']['brand_profit']

            # 品牌过户数据
            brand_transfer = analysis_results['profit_transfer']['brand_transfer']

            # 安全合并数据 - 检查索引是否匹配（地区 vs 品牌维度可能不同）
            common_index = region_sales.index.intersection(brand_profit.index)
            if len(common_index) >= 2:
                # 索引有交集，使用匹配的数据
                transfer_reindexed = brand_transfer.reindex(common_index)
                region_analysis = pd.DataFrame({
                    '销售总量': region_sales.loc[common_index],
                    '总利润': brand_profit.loc[common_index, '总利润'],
                    '平均利润率': brand_profit.loc[common_index, '平均利润率'],
                    '总过户数量': transfer_reindexed['总过户数量'] if '总过户数量' in transfer_reindexed.columns else 0
                }).fillna(0)
            else:
                # 索引不匹配，使用位置对齐作为降级方案
                print(f"  ⚠ 地区销售索引与利润索引不完全匹配（交集: {len(common_index)}），使用位置对齐")
                min_len = min(len(region_sales), len(brand_profit), len(brand_transfer))
                if min_len < 2:
                    raise ValueError(f"可用数据不足（仅 {min_len} 条），无法进行关联分析")
                region_analysis = pd.DataFrame({
                    '销售总量': region_sales.values[:min_len],
                    '总利润': brand_profit['总利润'].values[:min_len],
                    '平均利润率': brand_profit['平均利润率'].values[:min_len],
                    '总过户数量': brand_transfer['总过户数量'].values[:min_len]
                }, index=region_sales.index[:min_len]).fillna(0)

            # 计算相关性
            region_correlation = region_analysis.corr()
            correlations['region'] = {
                'data': region_analysis,
                'correlation_matrix': region_correlation
            }

            print(f"  ✓ 地区维度关联分析完成 ({len(region_analysis)} 个地区)")

            # 找出重要关联
            important_correlations = []
            for i in range(len(region_correlation.columns)):
                for j in range(i+1, len(region_correlation.columns)):
                    corr_val = region_correlation.iloc[i, j]
                    if abs(corr_val) > 0.5:  # 只关注强相关
                        important_correlations.append({
                            'feature1': region_correlation.columns[i],
                            'feature2': region_correlation.columns[j],
                            'correlation': corr_val
                        })

            if important_correlations:
                correlations['region']['important_correlations'] = important_correlations
                print(f"  ✓ 识别 {len(important_correlations)} 个强关联关系")

        except Exception as e:
            print(f"  ⚠ 地区关联分析失败: {e}")

    # 客户-产品关联分析
    if (analysis_results['customer'] is not None and
        analysis_results['product'] is not None):

        try:
            # 计算客户聚类和购买偏好的关系
            customer_data = analysis_results['customer']['cluster_data']
            product_data = analysis_results['product']['cluster_data']

            # 按时间对齐数据 - 使用实际列名
            customer_monthly = customer_data.groupby('统计月份')['客户数量'].sum()

            # 产品数据没有月份列，使用索引作为替代
            if '统计月份' in product_data.columns:
                product_monthly = product_data.groupby('统计月份').size()
            else:
                # 如果没有月份列，创建一个简单的序列
                product_monthly = pd.Series(range(len(customer_monthly)),
                                          index=customer_monthly.index[:len(customer_monthly)])

            # 确保两个序列长度相同
            min_len = min(len(customer_monthly), len(product_monthly))
            customer_monthly = customer_monthly.iloc[:min_len]
            product_monthly = product_monthly.iloc[:min_len]

            # 合并数据
            customer_product = pd.DataFrame({
                '客户数量': customer_monthly,
                '产品销量': product_monthly
            }).fillna(0)

            # 计算相关性
            if len(customer_product) > 1:
                cp_correlation = customer_product.corr().iloc[0, 1]
                correlations['customer_product'] = {
                    'data': customer_product,
                    'correlation': cp_correlation
                }
                print(f"  ✓ 客户-产品关联分析完成 (相关性: {cp_correlation:.3f})")
            else:
                print(f"  ⚠ 客户-产品数据不足，无法计算相关性")

        except Exception as e:
            print(f"  ⚠ 客户-产品关联分析失败: {e}")

    # 时间维度关联分析
    if analysis_results['sales_trend'] is not None:
        try:
            # 预测结果分析
            forecast_data = analysis_results['sales_trend']['forecast']

            # 提取趋势和季节性成分 - 使用实际列名
            if '趋势' in forecast_data.columns and '年度季节性' in forecast_data.columns:
                trend_seasonal_corr = forecast_data['趋势'].corr(forecast_data['年度季节性'])
                correlations['time_components'] = {
                    '趋势_季节性相关性': trend_seasonal_corr
                }
                print(f"  ✓ 时间成分关联分析完成 (趋势-季节性相关性: {trend_seasonal_corr:.3f})")

        except Exception as e:
            print(f"  ⚠ 时间维度关联分析失败: {e}")

    return correlations

correlations = cross_module_correlation_analysis()

# 3. 综合聚类分析
print("\n【3/6】执行综合聚类分析...")

def comprehensive_clustering_analysis():
    """
    基于多维度数据进行综合聚类分析

    返回:
        clustering_results: dict - 聚类分析结果
    """
    clustering_results = {}

    # 使用地区数据代替品牌数据
    if correlations.get('region') is not None and len(correlations['region']['data']) > 1:
        region_data = correlations['region']['data']

        # 标准化数据
        scaler = StandardScaler()
        region_scaled = scaler.fit_transform(region_data)

        # 确定最佳聚类数（确保K范围有效）
        max_k = min(6, len(region_data))
        if max_k >= 2:
            silhouette_scores = []
            K_range = range(2, max_k)

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(region_scaled)
                score = silhouette_score(region_scaled, labels)
                silhouette_scores.append(score)
                print(f"  • K={k}: 轮廓系数 = {score:.3f}")

            # 选择最佳K值
            best_k = K_range[np.argmax(silhouette_scores)]
            print(f"  ✓ 最佳聚类数: {best_k}")

            # 执行最终聚类
            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(region_scaled)

            # 添加聚类标签 - 使用副本避免污染原始关联数据
            region_data = region_data.copy()
            region_data['聚类标签'] = final_labels

            clustering_results['region'] = {
                'data': region_data,
                'model': final_kmeans,
                'scaler': scaler,
                '最佳聚类数': best_k,
                '轮廓系数': silhouette_scores
            }

            print(f"  ✓ 地区综合聚类完成 ({best_k} 个聚类)")
            print(f"  ✓ 聚类分布: {pd.Series(final_labels).value_counts().to_dict()}")
        else:
            print(f"  ⚠ 地区数据量不足，无法进行聚类分析 (仅{len(region_data)}个地区)")
    else:
        print(f"  ⚠ 无可用的地区关联数据，跳过聚类分析")

    return clustering_results

clustering_results = comprehensive_clustering_analysis()

# 4. 主成分分析 (PCA)
print("\n【4/6】执行主成分分析...")

def principal_component_analysis():
    """
    对多维度数据进行主成分分析

    返回:
        pca_results: dict - PCA分析结果
    """
    pca_results = {}

    # 使用地区数据
    if correlations.get('region') is not None and len(correlations['region']['data']) > 1:
        try:
            # 准备地区数据 - 排除非数值列和聚类标签
            region_data = correlations['region']['data']
            numeric_cols = region_data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['聚类标签']
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            region_numeric = region_data[feature_cols]

            if len(feature_cols) < 2:
                print(f"  ⚠ 数值特征不足（{len(feature_cols)}个），跳过PCA分析")
                return pca_results

            # 执行PCA
            pca = PCA()
            pca_result = pca.fit_transform(StandardScaler().fit_transform(region_numeric))

            # 计算解释的方差比
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            # 确定需要的主成分数量（解释80%方差）
            if np.any(cumulative_variance >= 0.8):
                n_components = np.argmax(cumulative_variance >= 0.8) + 1
            else:
                n_components = len(cumulative_variance)  # 所有成分都不足80%，使用全部

            pca_results['region'] = {
                'pca': pca,
                '解释方差比': explained_variance,
                '累积解释方差': cumulative_variance,
                '主成分数量': n_components,
                'pca_result': pca_result,
                'feature_cols': feature_cols
            }

            print(f"  ✓ PCA分析完成")
            print(f"  ✓ 前{n_components}个主成分解释 {cumulative_variance[n_components-1]:.1%} 方差")

            # 输出主成分负荷
            print(f"\n  前{n_components}个主成分的重要特征负荷:")
            for i in range(min(n_components, 3)):  # 显示前3个主成分
                component_loadings = pd.DataFrame({
                    '特征': feature_cols,
                    '负荷系数': pca.components_[i]
                }).sort_values('负荷系数', key=abs, ascending=False)

                print(f"\n  主成分 {i+1} (解释 {explained_variance[i]:.1%} 方差):")
                for idx, row in component_loadings.head(3).iterrows():
                    print(f"    {row['特征']}: {row['负荷系数']:.3f}")

        except Exception as e:
            print(f"  ⚠ PCA分析失败: {e}")
    else:
        print(f"  ⚠ 无可用的地区数据，跳过PCA分析")

    return pca_results

pca_results = principal_component_analysis()

# 5. 生成综合洞察
print("\n【5/6】生成综合业务洞察...")

def generate_comprehensive_insights():
    """
    基于所有分析结果生成综合业务洞察

    返回:
        insights: dict - 综合业务洞察
    """
    insights = {}

    # 地区综合洞察
    if clustering_results.get('region') is not None:
        region_clusters = clustering_results['region']['data']

        # 分析每个聚类的特征
        cluster_profiles = {}
        for cluster_id in sorted(region_clusters['聚类标签'].unique()):
            cluster_data = region_clusters[region_clusters['聚类标签'] == cluster_id]

            profile = {
                '聚类大小': len(cluster_data),
                '平均销量': cluster_data['销售总量'].mean(),
                '平均利润': cluster_data['总利润'].mean(),
                '平均利润率': cluster_data['平均利润率'].mean(),
                '平均过户数量': cluster_data['总过户数量'].mean(),
                '包含地区': cluster_data.index.tolist()
            }

            cluster_profiles[f'聚类_{cluster_id}'] = profile

        insights['地区聚类'] = cluster_profiles

        print(f"  ✓ 识别 {len(cluster_profiles)} 个地区聚类类型")

        # 识别明星地区和问题地区
        if '平均利润率' in region_clusters.columns and '销售总量' in region_clusters.columns:
            # 高销售 + 高利润
            star_regions = region_clusters[
                (region_clusters['销售总量'] > region_clusters['销售总量'].median()) &
                (region_clusters['平均利润率'] > region_clusters['平均利润率'].median())
            ]

            # 低销售 + 低利润
            problem_regions = region_clusters[
                (region_clusters['销售总量'] < region_clusters['销售总量'].median()) &
                (region_clusters['平均利润率'] < region_clusters['平均利润率'].median())
            ]

            insights['明星地区'] = star_regions.index.tolist()
            insights['问题地区'] = problem_regions.index.tolist()

            print(f"  ✓ 明星地区: {len(star_regions)} 个")
            print(f"  ✓ 问题地区: {len(problem_regions)} 个")

    # 客户价值洞察
    if analysis_results['customer'] is not None:
        clv_data = analysis_results['customer']['clv_data']

        # 高价值客户群体 - 使用实际列名
        top_clv = clv_data.head(3)
        insights['高价值客户群体'] = [
            {
                '客户群体': row['客户群体'],
                '客户生命周期价值': row['客户生命周期价值'],
                '客户数量': row['客户总数']
            }
            for _, row in top_clv.iterrows()
        ]

        print(f"  ✓ 识别 {len(insights['高价值客户群体'])} 个高价值客户群体")

    # 产品趋势洞察
    if analysis_results['product'] is not None:
        importance_data = analysis_results['product']['importance']

        if not importance_data.empty:
            # 最重要的产品特征 - 使用实际列名
            top_features = importance_data.head(3)
            insights['关键产品特征'] = [
                {
                    '特征名称': row['特征'] if '特征' in row else row['特征名称'],
                    '重要性得分': row['重要性']
                }
                for _, row in top_features.iterrows()
            ]

            print(f"  ✓ 识别 {len(insights['关键产品特征'])} 个关键产品特征")

    # 时间趋势洞察
    if analysis_results['sales_trend'] is not None:
        forecast_data = analysis_results['sales_trend']['forecast']

        if '预测值' in forecast_data.columns:
            # 未来预测
            future_forecast = forecast_data.tail(12)['预测值'].values
            total_forecast = future_forecast.sum()
            avg_forecast = future_forecast.mean()

            insights['销售预测'] = {
                '未来12个月总销售额': total_forecast,
                '未来12个月月均销售额': avg_forecast,
                '增长趋势': '上升' if future_forecast[-1] > future_forecast[0] else '下降'
            }

            print(f"  ✓ 未来12个月预测销售额: ¥{total_forecast:,.2f}")

    return insights

comprehensive_insights = generate_comprehensive_insights()

# 6. 可视化综合仪表盘
print("\n【6/6】生成综合分析可视化仪表盘...")

def plot_comprehensive_dashboard():
    """
    创建综合分析仪表盘
    """
    # 增大画布尺寸，为每个子图提供更多空间
    fig = plt.figure(figsize=(40, 30))
    # 调整子图间距，防止标签挤压
    gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.6)

    fig.suptitle('电动汽车销售综合分析仪表盘', fontsize=24, fontweight='bold', y=0.98)

    # 1. 地区聚类分析 (左上)
    if clustering_results.get('region') is not None:
        ax1 = fig.add_subplot(gs[0, 0])
        region_clusters = clustering_results['region']['data']

        cluster_counts = region_clusters['聚类标签'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
        bars = ax1.bar(cluster_counts.index, cluster_counts.values,
                       color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('聚类', fontsize=12)
        ax1.set_ylabel('地区数量', fontsize=12)
        ax1.set_title('地区聚类分布', fontsize=15, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. 客户生命周期价值 (上中)
    if analysis_results['customer'] is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        clv_data = analysis_results['customer']['clv_data'].head(8)

        bars = ax2.barh(range(len(clv_data)), clv_data['客户生命周期价值']/10000,
                        color=plt.cm.viridis(np.linspace(0, 1, len(clv_data))))
        ax2.set_yticks(range(len(clv_data)))
        ax2.set_yticklabels(clv_data['客户群体'], fontsize=11)
        ax2.set_xlabel('CLV (万元)', fontsize=12)
        ax2.set_title('客户生命周期价值', fontsize=15, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=11)
        ax2.invert_yaxis()

        # 修复CLV数值标签显示，确保数字清晰
        for i, bar in enumerate(bars):
            width = bar.get_width()
            # 明确指定字体和大小，避免使用默认的monospace导致乱码
            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                     f'¥{width:.1f}万', ha='left', va='center', fontsize=10, fontfamily=CHINESE_FONT)

    # 3. 产品特征重要性 (上右)
    if analysis_results['product'] is not None:
        ax3 = fig.add_subplot(gs[0, 2])
        # 只取前5行，删除最后一行“平均续航”
        importance_data = analysis_results['product']['importance'].head(5)

        # 使用正确的列名（已在加载时映射为中文）
        feature_col = '特征' if '特征' in importance_data.columns else '特征名称'
        bars = ax3.barh(importance_data[feature_col], importance_data['重要性'],
                        color=plt.cm.plasma(np.linspace(0, 1, len(importance_data))))
        ax3.set_xlabel('重要性得分', fontsize=12)
        ax3.set_title('产品特征重要性', fontsize=15, fontweight='bold')
        ax3.tick_params(axis='both', labelsize=11)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)

        # 在每个条形上标注具体数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', ha='left', va='center', fontsize=10, fontfamily=CHINESE_FONT)

    # 4. 销售预测 (右上)
    if analysis_results['sales_trend'] is not None:
        ax4 = fig.add_subplot(gs[0, 3])
        forecast_data = analysis_results['sales_trend']['forecast']

        if '预测值' in forecast_data.columns:
            # 显示最近12个月的预测
            recent_forecast = forecast_data.tail(12)
            ax4.plot(range(len(recent_forecast)), recent_forecast['预测值'],
                     marker='o', linewidth=2, markersize=4, color='blue')
            if '预测下限' in recent_forecast.columns and '预测上限' in recent_forecast.columns:
                ax4.fill_between(range(len(recent_forecast)),
                               recent_forecast['预测下限'],
                               recent_forecast['预测上限'],
                               alpha=0.2, color='blue')
            ax4.set_xlabel('未来月份', fontsize=12)
            ax4.set_ylabel('预测销售额', fontsize=12)
            ax4.set_title('销售预测 (未来12个月)', fontsize=15, fontweight='bold')
            ax4.tick_params(axis='both', labelsize=11)
            ax4.grid(True, alpha=0.3)

    # 5. 地区销售vs利润散点图 (中左，跨2列)
    if correlations.get('region') is not None:
        ax5 = fig.add_subplot(gs[1, :2])
        region_data = correlations['region']['data']

        scatter = ax5.scatter(region_data['销售总量'], region_data['总利润'],
                           s=120, alpha=0.8, c=region_data['平均利润率'],
                           cmap='RdYlGn', edgecolors='black', linewidth=0.8)

        # --- 使用adjust_text自动调整重叠的标签（如已安装） ---
        texts = []
        for i, (region, row) in enumerate(region_data.iterrows()):
            # 获取数据点的坐标
            x, y = row['销售总量'], row['总利润']
            # 添加标签到列表
            texts.append(ax5.text(x, y, region, fontsize=9, fontfamily=CHINESE_FONT, ha='center', va='center'))

        # 自动调整文本位置，避免重叠
        if HAS_ADJUST_TEXT:
            adjust_text(texts, ax=ax5, expand_text=(1.2, 1.2), force_text=(0.5, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

        ax5.set_xlabel('销售总量', fontsize=12)
        ax5.set_ylabel('总利润', fontsize=12)
        ax5.set_title('地区销售-利润分析\n(颜色=利润率)', fontsize=15, fontweight='bold')
        ax5.tick_params(axis='both', labelsize=11)
        ax5.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('利润率 (%)', fontsize=12)
        cbar.ax.tick_params(labelsize=11)

    # 6. 相关矩阵热力图 (中右，跨2列)
    if correlations.get('region') is not None:
        ax6 = fig.add_subplot(gs[1, 2:])
        region_corr = correlations['region']['correlation_matrix']

        # 选择主要特征
        key_features = ['销售总量', '总利润', '平均利润率', '总过户数量']
        available_features = [f for f in key_features if f in region_corr.columns]

        if len(available_features) >= 3:
            corr_subset = region_corr.loc[available_features, available_features]
            mask = np.triu(np.ones_like(corr_subset, dtype=bool))

            sns.heatmap(corr_subset, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax6,
                       annot_kws={"size": 12})
            ax6.set_title('地区特征相关性热力图', fontsize=15, fontweight='bold')
            ax6.tick_params(axis='both', labelsize=11)
            # 旋转Y轴标签，防止文字重叠
            ax6.set_yticklabels(ax6.get_yticklabels(), rotation=90, ha='right')

    # 7. PCA结果 (左下)
    if pca_results.get('region') is not None:
        ax7 = fig.add_subplot(gs[2, 0])
        pca_data = pca_results['region']

        # 绘制碎石图
        ax7.plot(range(1, len(pca_data['解释方差比']) + 1),
                 pca_data['累积解释方差'], 'bo-', linewidth=2, markersize=6)
        ax7.axhline(y=0.8, color='r', linestyle='--', label='80%方差')
        ax7.axvline(x=pca_data['主成分数量'], color='g', linestyle='--',
                   label=f"最佳维度: {pca_data['主成分数量']}")
        ax7.set_xlabel('主成分数量', fontsize=12)
        ax7.set_ylabel('累积解释方差', fontsize=12)
        ax7.set_title('PCA碎石图', fontsize=15, fontweight='bold')
        ax7.tick_params(axis='both', labelsize=11)
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)

    # 8. 综合洞察摘要 (中下，跨3列)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    # 准备洞察文本（移除了Emoji，确保兼容性）
    insight_text = "综合业务洞察与建议\n\n"

    insight_text += "【核心发现】\n"
    if '明星地区' in comprehensive_insights:
        star_regions = comprehensive_insights['明星地区']
        insight_text += f"• 明星地区 ({len(star_regions)}个): {', '.join(star_regions[:3])}{'...' if len(star_regions) > 3 else ''}\n"

    if '问题地区' in comprehensive_insights:
        problem_regions = comprehensive_insights['问题地区']
        insight_text += f"• 问题地区 ({len(problem_regions)}个): {', '.join(problem_regions[:3])}{'...' if len(problem_regions) > 3 else ''}\n"

    if '高价值客户群体' in comprehensive_insights:
        hv_segments = comprehensive_insights['高价值客户群体']
        insight_text += f"• 高价值客户群体: {len(hv_segments)} 个\n"
        for seg in hv_segments[:2]:
            insight_text += f"  - {seg['客户群体']}: CLV ¥{seg['客户生命周期价值']:,.0f}，客户数 {seg['客户数量']}人\n"

    if '关键产品特征' in comprehensive_insights:
        key_features = comprehensive_insights['关键产品特征']
        insight_text += f"• 关键产品特征: {', '.join([f['特征名称'] for f in key_features])}\n"

    if '销售预测' in comprehensive_insights:
        forecast = comprehensive_insights['销售预测']
        insight_text += f"• 销售预测: 未来12个月 ¥{forecast['未来12个月总销售额']:,.0f}，趋势: {forecast['增长趋势']}\n"

    insight_text += "\n【战略建议】\n"
    insight_text += "1. 地区策略：\n"
    insight_text += "   - 资源倾斜：加大明星地区的市场投入，复制成功经验\n"
    insight_text += "   - 问题诊断：深入分析问题地区的根本原因，制定针对性改进方案\n"
    insight_text += "   - 组合优化：考虑淘汰或重组低价值地区，聚焦高潜力市场\n\n"

    insight_text += "2. 客户策略：\n"
    insight_text += "   - 精准运营：重点维护高价值客户群体，提升客户生命周期价值\n"
    insight_text += "   - 个性化服务：开发针对不同客户群体的个性化产品和服务\n"
    insight_text += "   - 体验提升：优化客户触点，提升整体客户体验\n\n"

    insight_text += "3. 产品策略：\n"
    insight_text += "   - 研发聚焦：重点投资续航里程、电池容量等关键产品特征\n"
    insight_text += "   - 组合优化：根据聚类结果优化产品线，提升整体竞争力\n"
    insight_text += "   - 交叉销售：利用产品推荐系统提升客单价\n\n"

    insight_text += "4. 销售策略：\n"
    insight_text += "   - 库存管理：基于销售预测优化库存周转，降低成本\n"
    insight_text += "   - 市场投入：加强高增长地区的市场推广，抢占先机\n"
    insight_text += "   - 季节性营销：制定针对性的季节性营销策略，提升销量\n\n"

    insight_text += "5. 利润优化：\n"
    insight_text += "   - 成本控制：分析成本结构，识别降本机会\n"
    insight_text += "   - 定价策略：优化定价策略，提升利润率\n"
    insight_text += "   - 产品组合：加强高利润产品的推广，优化利润结构\n"

    # 修复综合洞察文本框的字体，确保中文正常显示
    ax8.text(0.02, 0.98, insight_text, transform=ax8.transAxes, fontsize=13,
             verticalalignment='top', fontfamily=CHINESE_FONT,  # 明确指定为微软雅黑
             bbox=dict(boxstyle='round,pad=1', facecolor='#F8F9FA', alpha=0.9, edgecolor='#2E86AB', linewidth=2))
    ax8.set_title('综合洞察与战略建议', fontsize=20, fontweight='bold', pad=20)

    plt.savefig(f"{IMAGE_SAVE_PATH}\\综合分析仪表盘.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 综合分析仪表盘已保存: {IMAGE_SAVE_PATH}\\综合分析仪表盘.png")
    plt.close()

plot_comprehensive_dashboard()

# 7. 生成综合分析报告
print("\n【附加】生成综合分析报告...")

def generate_comprehensive_report():
    """
    生成综合分析报告
    """
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售综合分析报告")
    report.append("=" * 80)
    report.append("")

    # 执行摘要
    report.append("【执行摘要】")
    report.append("  本报告整合了客户分析、销售趋势、产品特征、利润过户四大模块的分析结果，")
    report.append("  通过机器学习算法和统计分析方法，提供全面的业务洞察和决策建议。")
    report.append("")

    # 关键发现
    report.append("【关键发现】")

    if '明星地区' in comprehensive_insights:
        report.append(f"  1. 明星地区识别: 发现 {len(comprehensive_insights['明星地区'])} 个高销售高利润地区")

    if '高价值客户群体' in comprehensive_insights:
        hv_segments = comprehensive_insights['高价值客户群体']
        total_clv = sum([seg['客户生命周期价值'] for seg in hv_segments])
        report.append(f"  2. 高价值客户: 识别 {len(hv_segments)} 个高价值客户群体，总CLV达 ¥{total_clv:,.2f}")

    if '销售预测' in comprehensive_insights:
        forecast = comprehensive_insights['销售预测']
        report.append(f"  3. 销售预测: 未来12个月预计销售额 ¥{forecast['未来12个月总销售额']:,.2f}")
        report.append(f"     增长趋势: {forecast['增长趋势']}")

    if '关键产品特征' in comprehensive_insights:
        report.append(f"  4. 产品特征: 识别出影响销售的关键产品特征")

    report.append("")

    # 跨模块关联洞察
    if correlations.get('region') is not None and 'important_correlations' in correlations['region']:
        report.append("【跨模块关联洞察】")
        important_corrs = correlations['region']['important_correlations'][:3]  # 前3个
        for corr in important_corrs:
            report.append(f"  • {corr['feature1']} 与 {corr['feature2']}: {corr['correlation']:.3f}")
        report.append("")

    # 地区聚类分析
    if clustering_results.get('region') is not None:

        report.append("【地区聚类分析】")
        cluster_profiles = comprehensive_insights['地区聚类']
        for cluster_id, profile in cluster_profiles.items():
            report.append(f"  {cluster_id}:")
            report.append(f"    • 地区数量: {profile['聚类大小']}")
            report.append(f"    • 平均销售: ¥{profile['平均销量']:,.2f}")
            report.append(f"    • 平均利润: ¥{profile['平均利润']:,.2f}")
            report.append(f"    • 平均利润率: {profile['平均利润率']:.2f}%")
        report.append("")

    # 战略建议
    report.append("【战略建议】")
    report.append("  1. 地区策略:")
    report.append("     • 加大对明星地区的资源投入")
    report.append("     • 分析问题地区的根本原因并制定改进计划")
    report.append("     • 考虑地区组合优化，淘汰表现不佳的地区")
    report.append("")
    report.append("  2. 客户策略:")
    report.append("     • 制定高价值客户群体的保留策略")
    report.append("     • 开发针对不同客户群体的个性化产品")
    report.append("     • 提升客户体验，增加客户生命周期价值")
    report.append("")
    report.append("  3. 产品策略:")
    report.append("     • 重点投资关键产品特征的研发")
    report.append("     • 根据产品聚类结果优化产品组合")
    report.append("     • 利用产品推荐系统提升交叉销售")
    report.append("")
    report.append("  4. 销售策略:")
    report.append("     • 基于销售预测优化库存管理")
    report.append("     • 加强高增长地区的市场投入")
    report.append("     • 制定季节性营销策略")
    report.append("")
    report.append("  5. 利润优化:")
    report.append("     • 分析成本结构，识别降本机会")
    report.append("     • 优化定价策略，提升利润率")
    report.append("     • 加强高利润产品的推广")
    report.append("")

    # 风险提示
    report.append("【风险提示】")
    report.append("  • 市场波动可能影响预测准确性")
    report.append("  • 竞争加剧可能导致利润率下降")
    report.append("  • 技术变革可能影响产品竞争力")
    report.append("  • 政策变化可能影响市场需求")
    report.append("")

    # 下一步行动
    report.append("【下一步行动】")
    report.append("  1. 建立数据驱动的决策机制")
    report.append("  2. 定期更新分析模型")
    report.append("  3. 监控关键指标变化")
    report.append("  4. 执行建议的优化措施")
    report.append("  5. 评估改进效果")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)

report_content = generate_comprehensive_report()
print(report_content)

# ========== 关键修改：保存报告到文档文件夹 ==========
# 保存报告到文件
with open(f"{DOCUMENT_SAVE_PATH}\\综合分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 综合分析报告已保存: {DOCUMENT_SAVE_PATH}\\综合分析报告.txt")

# 保存关键数据
print("\n【附加】保存综合分析结果...")

# ========== 关键修改：所有CSV文件保存到文档文件夹 ==========
# 保存地区聚类结果
if clustering_results.get('region') is not None:
    clustering_results['region']['data'].to_csv(f"{DOCUMENT_SAVE_PATH}\\地区综合聚类结果.csv", index=True, encoding='utf-8-sig')
    print(f"  ✓ 地区综合聚类结果已保存")

# 保存跨模块关联数据
if correlations.get('region') is not None:
    correlations['region']['data'].to_csv(f"{DOCUMENT_SAVE_PATH}\\地区跨模块关联数据.csv", index=True, encoding='utf-8-sig')
    correlations['region']['correlation_matrix'].to_csv(f"{DOCUMENT_SAVE_PATH}\\地区跨模块相关性矩阵.csv", index=True, encoding='utf-8-sig')
    print(f"  ✓ 跨模块关联数据已保存")

print(f"\n✓ 所有分析结果已保存到:")
print(f"  • 图片文件: {IMAGE_SAVE_PATH}")
print(f"  • 文档文件: {DOCUMENT_SAVE_PATH}")

print("\n" + "=" * 80)
print("电动汽车销售综合分析系统执行完成！")
print("=" * 80)
print("\n生成的分析文件:")
print("  • 客户综合分析.py - 客户细分与行为分析")
print("  • 销售趋势分析.py - 时间序列预测与趋势分析")
print("  • 产品特征分析.py - 产品特征关联与推荐")
print("  • 利润过户分析.py - 利润构成与过户分析")
print("  • 综合分析.py - 跨模块关联与综合洞察")
print("  • requirements.txt - Python依赖库清单")
print("\n可视化图表和分析报告已保存至:")
print(f"  • 图片: {IMAGE_SAVE_PATH}")
print(f"  • 文档: {DOCUMENT_SAVE_PATH}")
print("\n建议执行步骤:")
print("  1. 安装依赖库: pip install -r requirements.txt")
print("  2. 运行各分析模块: python 客户综合分析.py")
print("  3. 查看生成的图表和报告")
print("  4. 根据分析结果制定业务策略")
print("=" * 80)