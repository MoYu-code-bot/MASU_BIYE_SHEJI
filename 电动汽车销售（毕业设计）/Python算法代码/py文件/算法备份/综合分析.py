# =============================================================================
# 文件：综合分析.py
# -----------------------------------------------------------------------------
# 总体说明
#   不直接读原始 Excel，而是聚合各子模块已导出的 CSV（客户、产品、利润过户、销售趋势），做跨模块
#   皮尔逊相关、地区维度的 K-Means（轮廓系数选 k）、PCA 降维解释，并生成大屏仪表盘与综合报告。
#
# 核心算法
#   ① 跨模块关联（cross_module_correlation_analysis）：地区销售 vs 利润/过户等指标对齐后求相关矩阵。
#   ② K-Means + 轮廓系数扫描（comprehensive_clustering_analysis）：对地区综合指标表选最优簇数并打标签。
#   ③ PCA（principal_component_analysis）：标准化后全成分拟合，按累积方差≥80% 截取主成分并读负荷。
#
# 辅助：generate_comprehensive_insights / plot_comprehensive_dashboard / generate_comprehensive_report 为规则汇总与可视化；
#       adjustText 用于散点标签防重叠（可选）。
# =============================================================================

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





system = platform.system()
if system == 'Windows':
    CHINESE_FONT = 'Microsoft YaHei'
elif system == 'Darwin':
    CHINESE_FONT = 'PingFang SC'
else:
    CHINESE_FONT = 'WenQuanYi Micro Hei'


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.formatter.use_mathtext'] = False



BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\综合分析"

IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")

DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")


for path in [IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

print("=" * 80)
print("电动汽车销售综合分析与关联分析系统")
print("=" * 80)


print("\n【1/6】加载各模块分析结果...")


# =============================================================================
# [数据准备] load_analysis_results：从固定目录读取各子模块 CSV，失败时对应键为 None，后续算法需判空。
# =============================================================================

def load_analysis_results():
    results = {}


    BASE_CSV_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储"
    paths_config = {
        "customer": os.path.join(BASE_CSV_PATH, "客户综合分析", "文档分析"),
        "product": os.path.join(BASE_CSV_PATH, "产品特征分析", "文档分析"),
        "profit_transfer": os.path.join(BASE_CSV_PATH, "利润过户分析", "文档分析"),
        "sales_trend": os.path.join(BASE_CSV_PATH, "销售趋势分析", "文档分析")
    }


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


    try:
        product_csv_path = paths_config["product"]
        product_cluster = pd.read_csv(os.path.join(product_csv_path, "产品特征聚类结果.csv"), encoding='utf-8-sig')
        feature_correlation = pd.read_csv(os.path.join(product_csv_path, "产品特征相关性矩阵.csv"), index_col=0, encoding='utf-8-sig')
        feature_importance = pd.read_csv(os.path.join(product_csv_path, "产品特征重要性.csv"), encoding='utf-8-sig')


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


print("\n【2/6】执行跨模块关联分析...")


# =============================================================================
# [算法 ①] 跨模块皮尔逊相关（cross_module_correlation_analysis）
# 用处：在地区、客户-产品、预测序列内部趋势与季节性等维度上构造 DataFrame 后调用 .corr()，提取强相关对（|r|>0.5）。
# =============================================================================

def cross_module_correlation_analysis():
    correlations = {}


    if (analysis_results['sales_trend'] is not None and
        analysis_results['profit_transfer'] is not None):

        try:

            regional_sales = analysis_results['sales_trend']['regional_sales']
            region_sales = regional_sales.groupby('地区')['总销售量'].sum()


            brand_profit = analysis_results['profit_transfer']['brand_profit']


            brand_transfer = analysis_results['profit_transfer']['brand_transfer']




            normalize_index = lambda idx: pd.Index([
                str(x).strip().replace('省', '').replace('市', '').replace('自治区', '').replace('特别行政区', '')
                for x in idx
            ])
            region_sales_norm = region_sales.copy()
            region_sales_norm.index = normalize_index(region_sales_norm.index)
            brand_profit_norm = brand_profit.copy()
            brand_profit_norm.index = normalize_index(brand_profit_norm.index)
            brand_transfer_norm = brand_transfer.copy()
            brand_transfer_norm.index = normalize_index(brand_transfer_norm.index)

            common_index = region_sales_norm.index.intersection(brand_profit_norm.index)
            if len(common_index) >= 2:

                transfer_reindexed = brand_transfer_norm.reindex(common_index)
                region_analysis = pd.DataFrame({
                    '销售总量': region_sales_norm.loc[common_index],
                    '总利润': brand_profit_norm.loc[common_index, '总利润'],
                    '平均利润率': brand_profit_norm.loc[common_index, '平均利润率'],
                    '总过户数量': transfer_reindexed['总过户数量'] if '总过户数量' in transfer_reindexed.columns else 0
                }).fillna(0)
            else:


                print(f"  ℹ 地区销售与利润/过户维度不同（交集: {len(common_index)}），改用排序对齐")
                transfer_sort_col = '总过户数量' if '总过户数量' in brand_transfer_norm.columns else brand_transfer_norm.columns[0]
                region_sales_ranked = region_sales.sort_values(ascending=False)
                profit_ranked = brand_profit_norm.sort_values(by='总利润', ascending=False)
                transfer_ranked = brand_transfer_norm.sort_values(by=transfer_sort_col, ascending=False)

                min_len = min(len(region_sales_ranked), len(profit_ranked), len(transfer_ranked))
                if min_len < 2:
                    raise ValueError(f"可用数据不足（仅 {min_len} 条），无法进行关联分析")
                region_analysis = pd.DataFrame({
                    '销售总量': region_sales_ranked.values[:min_len],
                    '总利润': profit_ranked['总利润'].values[:min_len],
                    '平均利润率': profit_ranked['平均利润率'].values[:min_len],
                    '总过户数量': transfer_ranked[transfer_sort_col].values[:min_len]
                }, index=region_sales_ranked.index[:min_len]).fillna(0)


            region_correlation = region_analysis.corr()
            correlations['region'] = {
                'data': region_analysis,
                'correlation_matrix': region_correlation
            }

            print(f"  ✓ 地区维度关联分析完成 ({len(region_analysis)} 个地区)")


            important_correlations = []
            for i in range(len(region_correlation.columns)):
                for j in range(i+1, len(region_correlation.columns)):
                    corr_val = region_correlation.iloc[i, j]
                    if abs(corr_val) > 0.5:
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


    if (analysis_results['customer'] is not None and
        analysis_results['product'] is not None):

        try:

            customer_data = analysis_results['customer']['cluster_data']
            product_data = analysis_results['product']['cluster_data']


            customer_monthly = customer_data.groupby('统计月份')['客户数量'].sum()


            if '统计月份' in product_data.columns:
                product_monthly = product_data.groupby('统计月份').size()
            else:

                product_monthly = pd.Series(range(len(customer_monthly)),
                                          index=customer_monthly.index[:len(customer_monthly)])


            min_len = min(len(customer_monthly), len(product_monthly))
            customer_monthly = customer_monthly.iloc[:min_len]
            product_monthly = product_monthly.iloc[:min_len]


            customer_product = pd.DataFrame({
                '客户数量': customer_monthly,
                '产品销量': product_monthly
            }).fillna(0)


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


    if analysis_results['sales_trend'] is not None:
        try:

            forecast_data = analysis_results['sales_trend']['forecast']


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


print("\n【3/6】执行综合聚类分析...")


# =============================================================================
# [算法 ②] 地区综合指标 K-Means + 轮廓系数（comprehensive_clustering_analysis）
# 用处：在「销售+利润+过户」等地区级数值矩阵上标准化，对 k=2..Kmax-1 扫描轮廓系数，取最优 k 再 fit 一次输出标签。
# =============================================================================

def comprehensive_clustering_analysis():
    clustering_results = {}


    if correlations.get('region') is not None and len(correlations['region']['data']) > 1:
        region_data = correlations['region']['data']


        scaler = StandardScaler()
        region_scaled = scaler.fit_transform(region_data)


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


            best_k = K_range[np.argmax(silhouette_scores)]
            print(f"  ✓ 最佳聚类数: {best_k}")


            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(region_scaled)


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


print("\n【4/6】执行主成分分析...")


# =============================================================================
# [算法 ③] PCA 主成分分析（principal_component_analysis）
# 用处：对地区数值特征（可含聚类标签列但排除）做标准化后 PCA，查看解释方差比与前几主成分的负荷，辅助降维解读。
# =============================================================================

def principal_component_analysis():
    pca_results = {}


    if correlations.get('region') is not None and len(correlations['region']['data']) > 1:
        try:

            region_data = correlations['region']['data']
            numeric_cols = region_data.select_dtypes(include=[np.number]).columns
            exclude_cols = ['聚类标签']
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            region_numeric = region_data[feature_cols]

            if len(feature_cols) < 2:
                print(f"  ⚠ 数值特征不足（{len(feature_cols)}个），跳过PCA分析")
                return pca_results


            pca = PCA()
            pca_result = pca.fit_transform(StandardScaler().fit_transform(region_numeric))


            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)


            if np.any(cumulative_variance >= 0.8):
                n_components = np.argmax(cumulative_variance >= 0.8) + 1
            else:
                n_components = len(cumulative_variance)

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


            print(f"\n  前{n_components}个主成分的重要特征负荷:")
            for i in range(min(n_components, 3)):
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


print("\n【5/6】生成综合业务洞察...")


# =============================================================================
# [规则汇总] generate_comprehensive_insights：基于聚类结果做四分位象限划分（明星/问题地区），合并 CLV、特征重要性、预测尾窗等，无新模型。
# =============================================================================

def generate_comprehensive_insights():
    insights = {}


    if clustering_results.get('region') is not None:
        region_clusters = clustering_results['region']['data']


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


        if '平均利润率' in region_clusters.columns and '销售总量' in region_clusters.columns:

            star_regions = region_clusters[
                (region_clusters['销售总量'] > region_clusters['销售总量'].median()) &
                (region_clusters['平均利润率'] > region_clusters['平均利润率'].median())
            ]


            problem_regions = region_clusters[
                (region_clusters['销售总量'] < region_clusters['销售总量'].median()) &
                (region_clusters['平均利润率'] < region_clusters['平均利润率'].median())
            ]

            insights['明星地区'] = star_regions.index.tolist()
            insights['问题地区'] = problem_regions.index.tolist()

            print(f"  ✓ 明星地区: {len(star_regions)} 个")
            print(f"  ✓ 问题地区: {len(problem_regions)} 个")


    if analysis_results['customer'] is not None:
        clv_data = analysis_results['customer']['clv_data']


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


    if analysis_results['product'] is not None:
        importance_data = analysis_results['product']['importance']

        if not importance_data.empty:

            top_features = importance_data.head(3)
            insights['关键产品特征'] = [
                {
                    '特征名称': row['特征'] if '特征' in row else row['特征名称'],
                    '重要性得分': row['重要性']
                }
                for _, row in top_features.iterrows()
            ]

            print(f"  ✓ 识别 {len(insights['关键产品特征'])} 个关键产品特征")


    if analysis_results['sales_trend'] is not None:
        forecast_data = analysis_results['sales_trend']['forecast']

        if '预测值' in forecast_data.columns:

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


print("\n【6/6】生成综合分析可视化仪表盘...")


# =============================================================================
# [可视化] plot_comprehensive_dashboard：多子图 GridSpec 展示相关热力图、聚类、PCA、预测、客户/产品条形图等。
# =============================================================================

def plot_comprehensive_dashboard():

    fig = plt.figure(figsize=(40, 30))

    gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.6)

    fig.suptitle('电动汽车销售综合分析仪表盘', fontsize=24, fontweight='bold', y=0.98)


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


        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')


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


        for i, bar in enumerate(bars):
            width = bar.get_width()

            ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                     f'¥{width:.1f}万', ha='left', va='center', fontsize=10, fontfamily=CHINESE_FONT)


    if analysis_results['product'] is not None:
        ax3 = fig.add_subplot(gs[0, 2])

        importance_data = analysis_results['product']['importance'].head(5)


        feature_col = '特征' if '特征' in importance_data.columns else '特征名称'
        bars = ax3.barh(importance_data[feature_col], importance_data['重要性'],
                        color=plt.cm.plasma(np.linspace(0, 1, len(importance_data))))
        ax3.set_xlabel('重要性得分', fontsize=12)
        ax3.set_title('产品特征重要性', fontsize=15, fontweight='bold')
        ax3.tick_params(axis='both', labelsize=11)
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)


        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', ha='left', va='center', fontsize=10, fontfamily=CHINESE_FONT)


    if analysis_results['sales_trend'] is not None:
        ax4 = fig.add_subplot(gs[0, 3])
        forecast_data = analysis_results['sales_trend']['forecast']

        if '预测值' in forecast_data.columns:

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


    if correlations.get('region') is not None:
        ax5 = fig.add_subplot(gs[1, :2])
        region_data = correlations['region']['data']

        scatter = ax5.scatter(region_data['销售总量'], region_data['总利润'],
                           s=120, alpha=0.8, c=region_data['平均利润率'],
                           cmap='RdYlGn', edgecolors='black', linewidth=0.8)


        texts = []
        for i, (region, row) in enumerate(region_data.iterrows()):

            x, y = row['销售总量'], row['总利润']

            texts.append(ax5.text(x, y, region, fontsize=9, fontfamily=CHINESE_FONT, ha='center', va='center'))


        if HAS_ADJUST_TEXT:
            adjust_text(texts, ax=ax5, expand_text=(1.2, 1.2), force_text=(0.5, 0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

        ax5.set_xlabel('销售总量', fontsize=12)
        ax5.set_ylabel('总利润', fontsize=12)
        ax5.set_title('地区销售-利润分析\n(颜色=利润率)', fontsize=15, fontweight='bold')
        ax5.tick_params(axis='both', labelsize=11)
        ax5.grid(True, alpha=0.3)


        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('利润率 (%)', fontsize=12)
        cbar.ax.tick_params(labelsize=11)


    if correlations.get('region') is not None:
        ax6 = fig.add_subplot(gs[1, 2:])
        region_corr = correlations['region']['correlation_matrix']


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

            ax6.set_yticklabels(ax6.get_yticklabels(), rotation=90, ha='right')


    if pca_results.get('region') is not None:
        ax7 = fig.add_subplot(gs[2, 0])
        pca_data = pca_results['region']


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



    plt.savefig(f"{IMAGE_SAVE_PATH}\\综合分析仪表盘.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 综合分析仪表盘已保存: {IMAGE_SAVE_PATH}\\综合分析仪表盘.png")
    plt.close()

plot_comprehensive_dashboard()


print("\n【附加】生成综合分析报告...")


# =============================================================================
# [输出] generate_comprehensive_report：将相关、聚类、PCA、洞察等写入 TXT，并视情况导出补充 CSV。
# =============================================================================

def generate_comprehensive_report():
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售综合分析报告")
    report.append("=" * 80)
    report.append("")


    report.append("【执行摘要】")
    report.append("  本报告整合了客户分析、销售趋势、产品特征、利润过户四大模块的分析结果，")
    report.append("  通过机器学习算法和统计分析方法，提供全面的业务洞察和决策建议。")
    report.append("")


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


    if correlations.get('region') is not None and 'important_correlations' in correlations['region']:
        report.append("【跨模块关联洞察】")
        important_corrs = correlations['region']['important_correlations'][:3]
        for corr in important_corrs:
            report.append(f"  • {corr['feature1']} 与 {corr['feature2']}: {corr['correlation']:.3f}")
        report.append("")


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


    report.append("【风险提示】")
    report.append("  • 市场波动可能影响预测准确性")
    report.append("  • 竞争加剧可能导致利润率下降")
    report.append("  • 技术变革可能影响产品竞争力")
    report.append("  • 政策变化可能影响市场需求")
    report.append("")


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



with open(f"{DOCUMENT_SAVE_PATH}\\综合分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 综合分析报告已保存: {DOCUMENT_SAVE_PATH}\\综合分析报告.txt")


print("\n【附加】保存综合分析结果...")



if clustering_results.get('region') is not None:
    clustering_results['region']['data'].to_csv(f"{DOCUMENT_SAVE_PATH}\\地区综合聚类结果.csv", index=True, encoding='utf-8-sig')
    print(f"  ✓ 地区综合聚类结果已保存")


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