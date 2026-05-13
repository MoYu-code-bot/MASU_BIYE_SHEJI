"""
读取客户年龄相关 Excel，用 K-Means 把客户分成几类，计算轮廓系数与简化的客户生命周期价值（CLV），并导出表格和饼图。
"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_SAVE = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\客户综合分析"
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE, "文档分析")
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(DOCUMENT_SAVE_PATH, exist_ok=True)

DATA_DIR = r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表"
AGE_XLSX = os.path.join(DATA_DIR, "客户年龄分析.xlsx")


def load_customer_data():
    """
    作用：从磁盘读取客户年龄分析Excel，并把文字形式的年龄组（age_group）转换成数字编码 age_group_encoded。
    """
    age_df = pd.read_excel(AGE_XLSX)
    age_df["age_group_encoded"] = LabelEncoder().fit_transform(age_df["age_group"])
    print(f"  ✓ 年龄表 {len(age_df)} 条")
    return age_df


def build_cluster_name_map(centers: pd.DataFrame) -> dict:
    """
    作用：根据每个簇在聚类中心上的数值高低。
    """
    med = centers.median()
    raw = {}
    for i in range(len(centers)):
        r = centers.iloc[i]
        parts = [
            "高销售额" if r["total_sales_volume"] >= med["total_sales_volume"] else "低销售额",
            "多客户量" if r["customer_count"] >= med["customer_count"] else "少客户量",
            "高客单价" if r["avg_purchase_price"] >= med["avg_purchase_price"] else "低客单价",
            "年龄偏成熟段" if r["age_group_encoded"] >= med["age_group_encoded"] else "年龄偏年轻段",
        ]
        raw[i] = "".join(parts[:3]) + "客群"
    buckets = defaultdict(list)
    for k, v in raw.items():
        buckets[v].append(k)
    out = {}
    for v, keys in buckets.items():
        if len(keys) == 1:
            out[keys[0]] = v
        else:
            for k in keys:
                out[k] = f"{v}（{k}）"
    return out


def kmeans_segment(df, feature_cols, n_clusters=4):
    """
    作用：对选定的数值列先做标准化（消除量纲差异），再运行 K-Means，把每一行客户数据打上簇标签 cluster。
    """
    X = df[feature_cols].fillna(df[feature_cols].mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df = df.copy()
    df["cluster"] = labels
    sil = silhouette_score(Xs, labels)
    print(f"  ✓ K-Means k={n_clusters}  轮廓系数={sil:.3f}")
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feature_cols)
    return df, km, centers, sil


def cluster_to_clv_table(clustered_age: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    """
    作用：按聚类结果汇总每一类的客户总数、销售总额、平均购买价格，并用简化公式估算 CLV（客户生命周期价值）。
    CLV ≈ 平均购买价值 × 购买频率 × 12。结果按 CLV 从高到低排序，便于看出哪类客户更值得长期经营。
    """
    months = clustered_age["stat_month"].nunique() if "stat_month" in clustered_age.columns else 1
    rows = []
    for c in sorted(clustered_age["cluster"].unique()):
        g = clustered_age[clustered_age["cluster"] == c]
        tc = g["customer_count"].sum()
        ts = g["total_sales_volume"].sum()
        ap = g["avg_purchase_price"].mean()
        freq = tc / months if months else 0
        lifespan = 12
        clv = ap * freq * lifespan
        rows.append(
            {
                "customer_segment": id_to_name.get(c, str(c)),
                "total_customers": tc,
                "total_sales_volume": ts,
                "avg_purchase_value": ap,
                "purchase_frequency": freq,
                "clv": clv,
            }
        )
    return pd.DataFrame(rows).sort_values("clv", ascending=False)


def plot_kmeans(age_clustered, id_to_name: dict):
    """
    作用：统计每个簇有多少条记录，算占比，画饼图并用中文簇名做图例，保存为 PNG 图片。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cc = age_clustered["cluster"].value_counts().sort_index()
    pie_labels = [id_to_name.get(i, str(i)) for i in cc.index]
    ax.pie(cc.values, labels=pie_labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("客户 K-Means 聚类占比")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_SAVE_PATH, "客户年龄分析图表.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {IMAGE_SAVE_PATH}\\客户年龄分析图表.png")


def main():
    """
    作用：脚本总流程——读数据、四列特征 K-Means（默认 4 类）、生成中文簇名、算 CLV 表、画饼图、
    把聚类结果和生命周期价值导出为 CSV，并写一份简短文字报告。
    """
    print("=" * 50 + "\n客户综合分析（K-Means）\n" + "=" * 50)
    age_data = load_customer_data()
    feats = ["customer_count", "total_sales_volume", "avg_purchase_price", "age_group_encoded"]
    age_clustered, _km, centers, sil = kmeans_segment(age_data, feats, n_clusters=4)
    id_to_name = build_cluster_name_map(centers)
    print("  聚类中心（原始量纲）:\n", centers.round(2).to_string())
    print("  簇标签映射:", id_to_name)
    age_clustered = age_clustered.copy()
    age_clustered["cluster_name"] = age_clustered["cluster"].map(id_to_name)
    clv_df = cluster_to_clv_table(age_clustered, id_to_name)
    plot_kmeans(age_clustered, id_to_name)

    col_age = {
        "stat_month": "统计月份",
        "age_group": "年龄组",
        "customer_count": "客户数量",
        "total_sales_volume": "销售总额",
        "avg_purchase_price": "平均购买价格",
        "preferred_brand": "偏好品牌",
        "age_group_encoded": "年龄组编码",
        "cluster": "聚类标签",
        "cluster_name": "聚类名称",
    }
    age_cn = age_clustered.rename(columns={k: v for k, v in col_age.items() if k in age_clustered.columns})
    age_cn.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "客户年龄聚类结果.csv"), index=False, encoding="utf-8-sig")

    col_clv = {
        "customer_segment": "客户群体",
        "total_customers": "客户总数",
        "total_sales_volume": "销售总额",
        "avg_purchase_value": "平均购买价值",
        "purchase_frequency": "购买频率",
        "clv": "客户生命周期价值",
    }
    clv_df.rename(columns=col_clv).round(2).to_csv(
        os.path.join(DOCUMENT_SAVE_PATH, "客户生命周期价值.csv"), index=False, encoding="utf-8-sig"
    )

    lines = [
        "=" * 50,
        "客户综合分析报告（K-Means）",
        f"轮廓系数: {sil:.3f}",
        "各客群样本量: "
        + ", ".join(
            f"{id_to_name.get(k, str(k))}={v}"
            for k, v in sorted(age_clustered["cluster"].value_counts().items())
        ),
        "=" * 50,
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]
    rep = "\n".join(lines)
    print("\n" + rep)
    with open(os.path.join(DOCUMENT_SAVE_PATH, "客户综合分析报告.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"\n✓ CSV/报告 → {DOCUMENT_SAVE_PATH}")


if __name__ == "__main__":
    main()
