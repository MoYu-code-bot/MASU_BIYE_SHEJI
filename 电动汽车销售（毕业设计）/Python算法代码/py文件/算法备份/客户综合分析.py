# =============================================================================
# 文件：客户综合分析.py
# -----------------------------------------------------------------------------
# 总体说明
#   读取年龄、性别、职业三类客户 Excel；对类别列 LabelEncoder 数值化；在年龄维度上做 K-Means 细分，
#   估算分群 CLV，最后出图与文本/CSV 报告。
#
# 核心算法
#   ① K-Means + StandardScaler + 轮廓系数：年龄相关数值特征上的客户分群。
#   ② CLV 启发式：按客群汇总客单价、频次与假定生命周期月数估算生命周期价值（非机器学习）。
#
# 说明：文件头部另导入 DBSCAN、PCA、TSNE，当前主流程未调用，可留作降维或密度聚类扩展。
# 环境：TEMP_DIR 等环境变量将临时文件定向到指定盘符，减轻系统盘压力。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
import tempfile
import shutil
import platform



TEMP_DIR = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\temp"
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR


if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


tempfile.tempdir = TEMP_DIR

warnings.filterwarnings('ignore')



plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    CHINESE_FONT = 'Microsoft YaHei'
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
    CHINESE_FONT = 'PingFang SC'
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']
    CHINESE_FONT = 'WenQuanYi Micro Hei'


print("当前matplotlib字体配置:")
print(f"  字体家族: {plt.rcParams['font.family']}")
print(f"  无衬线字体: {plt.rcParams['font.sans-serif']}")


BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\客户综合分析"
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")


if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)
if not os.path.exists(DOCUMENT_SAVE_PATH):
    os.makedirs(DOCUMENT_SAVE_PATH)

print("=" * 80)
print("电动汽车销售客户综合分析系统")
print("=" * 80)


print("\n【1/5】正在加载客户数据...")


# =============================================================================
# [数据准备] load_customer_data：读三张表，并对 age_group / gender / occupation 做 LabelEncoder，供聚类与分类使用。
# =============================================================================

def load_customer_data():
    age_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\客户年龄分析.xlsx")
    gender_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\客户性别分析.xlsx")
    occupation_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\客户职业分析.xlsx")

    print(f"  ✓ 年龄数据: {len(age_df)} 条记录")
    print(f"  ✓ 性别数据: {len(gender_df)} 条记录")
    print(f"  ✓ 职业数据: {len(occupation_df)} 条记录")



    le_age_group = LabelEncoder()
    age_df['age_group_encoded'] = le_age_group.fit_transform(age_df['age_group'])


    le_gender = LabelEncoder()
    gender_df['gender_encoded'] = le_gender.fit_transform(gender_df['user_gender'])


    le_occupation = LabelEncoder()
    occupation_df['occupation_encoded'] = le_occupation.fit_transform(occupation_df['user_occupation'])




    return age_df, gender_df, occupation_df


age_data, gender_data, occupation_data = load_customer_data()


print("\n【2/5】执行客户细分聚类分析...")


# =============================================================================
# [算法 ①] K-Means 客户细分（customer_segmentation_analysis）
# 用处：在标准化后的多维客户行为/属性空间划分簇，输出簇标签与逆变换后的聚类中心，便于解读各群画像。
# =============================================================================

def customer_segmentation_analysis(df, feature_cols, n_clusters=5):
    X = df[feature_cols].copy()


    X = X.fillna(X.mean())


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)


    df['cluster'] = cluster_labels


    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"  ✓ K-Means聚类完成 (n_clusters={n_clusters})")
    print(f"  ✓ 轮廓系数: {silhouette_avg:.3f}")


    print("\n  各聚类中心特征:")
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols
    )
    for i in range(n_clusters):
        print(f"    聚类 {i}: {cluster_centers.iloc[i].to_dict()}")

    return df, kmeans, scaler



age_features = ['customer_count', 'total_sales_volume', 'avg_purchase_price', 'age_group_encoded']
age_clustered, kmeans_age, scaler_age = customer_segmentation_analysis(
    age_data, age_features, n_clusters=4
)


print("\n【3/5】计算客户生命周期价值 (CLV)...")


# =============================================================================
# [业务指标 ③] 简化 CLV（calculate_customer_lifetime_value）
# 用处：按客群汇总人数、销量、均价，用「月均客户数 × 客单价 × 假定寿命月数」得到可排序的 CLV 表（经验公式）。
# =============================================================================

def calculate_customer_lifetime_value(df, customer_id_col='age_group', time_col='stat_month'):
    df[time_col] = pd.to_datetime(df[time_col])


    clv_metrics = []

    for group in df[customer_id_col].unique():
        group_data = df[df[customer_id_col] == group]


        total_customers = group_data['customer_count'].sum()
        total_sales = group_data['total_sales_volume'].sum()
        avg_purchase = group_data['avg_purchase_price'].mean()


        months_active = group_data[time_col].nunique()
        avg_monthly_customers = total_customers / months_active if months_active > 0 else 0



        customer_lifespan = 12
        clv = avg_purchase * avg_monthly_customers * customer_lifespan

        clv_metrics.append({
            'customer_segment': group,
            'total_customers': total_customers,
            'total_sales_volume': total_sales,
            'avg_purchase_value': avg_purchase,
            'purchase_frequency': avg_monthly_customers,
            'clv': clv
        })

    clv_df = pd.DataFrame(clv_metrics)
    clv_df = clv_df.sort_values('clv', ascending=False)

    print(f"  ✓ CLV计算完成，共 {len(clv_df)} 个客户群体")
    print(f"  ✓ 最高CLV: ¥{clv_df['clv'].max():,.2f}")
    print(f"  ✓ 最低CLV: ¥{clv_df['clv'].min():,.2f}")

    return clv_df


clv_results = calculate_customer_lifetime_value(age_data)


print("\n【4/5】生成客户分析可视化图表...")


# =============================================================================
# [可视化] plot_customer_analysis：透视表、饼图、条形图、热力图、散点气泡 — 汇总聚类与 CLV 等结果，无新模型训练。
# =============================================================================

def plot_customer_analysis():
    plt.style.use('seaborn-v0_8-darkgrid')


    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle('客户年龄群体销售趋势分析', fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)


    age_pivot = age_data.pivot_table(
        index='stat_month', columns='age_group',
        values='customer_count', aggfunc='sum'
    )
    age_pivot.plot(ax=axes[0, 0], marker='o', linewidth=2)
    axes[0, 0].set_title('各年龄组客户数量趋势', fontfamily=CHINESE_FONT)
    axes[0, 0].set_ylabel('客户数量', fontfamily=CHINESE_FONT)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(title='年龄组', prop={'family': CHINESE_FONT})


    sales_pivot = age_data.pivot_table(
        index='stat_month', columns='age_group',
        values='total_sales_volume', aggfunc='sum'
    )
    sales_pivot.plot(ax=axes[0, 1], marker='s', linewidth=2)
    axes[0, 1].set_title('各年龄组销售额趋势', fontfamily=CHINESE_FONT)
    axes[0, 1].set_ylabel('销售额', fontfamily=CHINESE_FONT)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title='年龄组', prop={'family': CHINESE_FONT})


    cluster_counts = age_clustered['cluster'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
    axes[1, 0].pie(cluster_counts.values, labels=cluster_counts.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontfamily': CHINESE_FONT})
    axes[1, 0].set_title('客户聚类分布', fontfamily=CHINESE_FONT)


    top_clv = clv_results.head(8)
    bars = axes[1, 1].barh(range(len(top_clv)), top_clv['clv'] / 10000,
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_clv))))
    axes[1, 1].set_yticks(range(len(top_clv)))
    axes[1, 1].set_yticklabels(top_clv['customer_segment'], fontfamily=CHINESE_FONT)
    axes[1, 1].set_xlabel('CLV (万元)', fontfamily=CHINESE_FONT)
    axes[1, 1].set_title('客户群体生命周期价值排名', fontfamily=CHINESE_FONT)
    axes[1, 1].invert_yaxis()


    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1, 1].text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                        f'¥{width:.1f}万', ha='left', va='center', fontsize=9, fontfamily=CHINESE_FONT)

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\客户年龄分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {IMAGE_SAVE_PATH}\\客户年龄分析图表.png")
    plt.close()


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('客户性别与职业分析', fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)


    gender_sales = gender_data.groupby('user_gender')['total_sales_volume'].sum()
    colors = ['#FF6B6B', '#4ECDC4']
    axes[0, 0].pie(gender_sales.values, labels=gender_sales.index,
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontfamily': CHINESE_FONT})
    axes[0, 0].set_title('性别销售占比分布', fontfamily=CHINESE_FONT)


    gender_counts = gender_data.groupby('user_gender')['customer_count'].sum()
    bars = axes[0, 1].bar(gender_counts.index, gender_counts.values,
                          color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('性别客户数量对比', fontfamily=CHINESE_FONT)
    axes[0, 1].set_ylabel('客户总数', fontfamily=CHINESE_FONT)


    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, height + 500,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontfamily=CHINESE_FONT)


    occupation_pivot = occupation_data.pivot_table(
        index='user_occupation', values='total_sales_volume',
        aggfunc='sum'
    ).sort_values('total_sales_volume', ascending=False)


    heatmap_data = occupation_pivot.values.reshape(-1, 1)
    im = axes[1, 0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_yticks(range(len(occupation_pivot)))
    axes[1, 0].set_yticklabels(occupation_pivot.index, fontsize=10, fontfamily=CHINESE_FONT)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_title('职业销售贡献热力图', fontfamily=CHINESE_FONT)


    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label('销售额', rotation=270, labelpad=15, fontfamily=CHINESE_FONT)


    occ_with_score = occupation_data.dropna(subset=['purchase_ability_score'])
    axes[1, 1].scatter(
        occ_with_score['total_sales_volume'],
        occ_with_score['purchase_ability_score'],
        s=occ_with_score['customer_count'] / 10,
        alpha=0.6,
        c=occ_with_score['purchase_ability_score'],
        cmap='viridis'
    )
    axes[1, 1].set_xlabel('销售总量', fontfamily=CHINESE_FONT)
    axes[1, 1].set_ylabel('购买力评分', fontfamily=CHINESE_FONT)
    axes[1, 1].set_title('职业销售vs购买力评分\n(气泡大小=客户数量)', fontfamily=CHINESE_FONT)

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\客户性别职业分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {IMAGE_SAVE_PATH}\\客户性别职业分析图表.png")
    plt.close()


plot_customer_analysis()


print("\n【5/5】生成客户综合分析报告...")


# =============================================================================
# [输出] generate_customer_report：将聚类、CLV 等结果写入 TXT 与 CSV。
# =============================================================================

def generate_customer_report():
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售客户综合分析报告")
    report.append("=" * 80)
    report.append("")


    report.append("【数据概览】")
    report.append(f"  • 分析时间范围: {age_data['stat_month'].min()} 至 {age_data['stat_month'].max()}")
    report.append(f"  • 客户年龄群体: {', '.join(age_data['age_group'].unique())}")
    report.append(f"  • 性别分布: {', '.join(gender_data['user_gender'].unique())}")
    report.append(f"  • 职业类别: {len(occupation_data['user_occupation'].unique())} 种")
    report.append(f"  • 总客户数: {age_data['customer_count'].sum():,} 人")
    report.append(f"  • 总销售额: ¥{age_data['total_sales_volume'].sum():,.2f}")
    report.append("")


    report.append("【客户生命周期价值 (CLV) 分析】")
    report.append("  排名前三的高价值客户群体:")
    for rank, (i, row) in enumerate(clv_results.head(3).iterrows(), 1):
        report.append(f"    {rank}. {row['customer_segment']}")
        report.append(f"       CLV: ¥{row['clv']:,.2f}, 客户数: {row['total_customers']:,}")
        report.append(f"       平均购买: ¥{row['avg_purchase_value']:,.2f}")
        report.append("")


    report.append("【客户聚类分析】")
    report.append(f"  • 最佳聚类数: 4")
    report.append(f"  • 各聚类客户分布:")
    cluster_dist = age_clustered['cluster'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        pct = count / len(age_clustered) * 100
        report.append(f"    聚类 {cluster}: {count} 条记录 ({pct:.1f}%)")
    report.append("")


    report.append("【业务建议】")
    report.append("  1. 高价值客户维护: 重点关注CLV排名前三的客户群体")
    report.append("  2. 精准营销: 根据聚类结果制定差异化营销策略")
    report.append("  3. 交叉销售: 利用客户细分结果，推荐相关产品")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)


report_content = generate_customer_report()
print(report_content)


with open(f"{DOCUMENT_SAVE_PATH}\\客户综合分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 分析报告已保存: {DOCUMENT_SAVE_PATH}\\客户综合分析报告.txt")

print("\n" + "=" * 80)
print("客户综合分析模块执行完成！")
print("=" * 80)


print("\n【附加】保存处理后的数据...")



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

age_clustered_cn.rename(columns={k: v for k, v in age_column_mapping.items() if k in age_clustered_cn.columns},
                       inplace=True)
age_clustered_cn.to_csv(f"{DOCUMENT_SAVE_PATH}\\客户年龄聚类结果.csv", index=False, encoding='utf-8-sig')


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
clv_results_cn.to_csv(f"{DOCUMENT_SAVE_PATH}\\客户生命周期价值.csv", index=False, encoding='utf-8-sig')


try:
    shutil.rmtree(TEMP_DIR)
    print(f"  ✓ 临时文件已清理")
except Exception as e:
    print(f"  ⚠ 临时文件清理失败: {e}")

print(f"  ✓ 聚类结果与 CLV 数据已保存为 CSV: {DOCUMENT_SAVE_PATH}")
print(f"  ✓ 所有分析图片已保存到: {IMAGE_SAVE_PATH}")
print("  ✨ 提示：CSV文件使用UTF-8-SIG编码，列名已改为中文，在Excel/WPS中打开不会出现中文乱码")