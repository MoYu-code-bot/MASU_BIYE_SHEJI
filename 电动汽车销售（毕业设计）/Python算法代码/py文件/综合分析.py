"""
读取各子分析模块已导出的 CSV，把「地区」维度的销售、利润、过户等指标对齐成一张表，再做主成分分析（PCA），输出解释方差、负荷矩阵和示意图。
依赖各模块「文档分析」目录下约定文件名的结果文件。
"""
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\综合分析"
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")
for _p in (IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH):
    os.makedirs(_p, exist_ok=True)
BASE_CSV = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储"
CFG = {
    "customer": os.path.join(BASE_CSV, "客户综合分析", "文档分析"),
    "product": os.path.join(BASE_CSV, "产品特征分析", "文档分析"),
    "profit_transfer": os.path.join(BASE_CSV, "利润过户分析", "文档分析"),
    "sales_trend": os.path.join(BASE_CSV, "销售趋势分析", "文档分析"),
}


def load_analysis_results():
    """
    作用：按固定路径尝试读取客户、销售趋势、产品特征、利润过户四个子模块生成的 CSV，装进一个字典里。
    """
    out = {}
    try:
        c = CFG["customer"]
        out["customer"] = {"cluster_data": pd.read_csv(os.path.join(c, "客户年龄聚类结果.csv"), encoding="utf-8-sig"), "clv_data": pd.read_csv(os.path.join(c, "客户生命周期价值.csv"), encoding="utf-8-sig")}
        print(f"  OK 客户 {len(out['customer']['cluster_data'])} 行")
    except Exception as e:
        print(f"  WARN 客户: {e}")
        out["customer"] = None
    try:
        s = CFG["sales_trend"]
        out["sales_trend"] = {"forecast": pd.read_csv(os.path.join(s, "销售预测结果.csv"), encoding="utf-8-sig"), "regional_sales": pd.read_csv(os.path.join(s, "地区销售汇总.csv"), encoding="utf-8-sig")}
        print("  OK 销售趋势")
    except Exception as e:
        print(f"  WARN 销售趋势: {e}")
        out["sales_trend"] = None
    try:
        p = CFG["product"]
        imp = pd.read_csv(os.path.join(p, "产品特征重要性.csv"), encoding="utf-8-sig")
        cn = "特征" if "特征" in imp.columns else "特征名称"
        fm = {"drive_sales": "续航里程", "battery_sales": "电池容量", "price_sales": "销售价格", "smart_sales": "智能配置", "charging_sales": "充电速度", "avg_battery": "平均续航"}
        if cn in imp.columns:
            imp = imp.copy()
            imp[cn] = imp[cn].map(fm).fillna(imp[cn])
        out["product"] = {"cluster_data": pd.read_csv(os.path.join(p, "产品特征聚类结果.csv"), encoding="utf-8-sig"), "correlation": pd.read_csv(os.path.join(p, "产品特征相关性矩阵.csv"), index_col=0, encoding="utf-8-sig"), "importance": imp}
        print("  OK 产品特征")
    except Exception as e:
        print(f"  WARN 产品特征: {e}")
        out["product"] = None
    try:
        pt = CFG["profit_transfer"]
        out["profit_transfer"] = {"brand_profit": pd.read_csv(os.path.join(pt, "品牌利润分析.csv"), index_col=0, encoding="utf-8-sig"), "brand_transfer": pd.read_csv(os.path.join(pt, "品牌过户分析.csv"), index_col=0, encoding="utf-8-sig"), "profit_importance": pd.read_csv(os.path.join(pt, "利润预测特征重要性.csv"), encoding="utf-8-sig")}
        print("  OK 利润过户")
    except Exception as e:
        print(f"  WARN 利润过户: {e}")
        out["profit_transfer"] = None
    return out


def norm_index(idx):
    """
    作用：把索引里的地区名字符串做简单规范化——去掉首尾空格，并去掉省、市、自治区等后缀。
    """
    return pd.Index([str(x).strip().replace("省", "").replace("市", "").replace("自治区", "").replace("特别行政区", "") for x in idx])


def build_region_table(ar):
    """
    作用：在销售趋势模块的地区销量表与利润过户模块的品牌（此处索引当作地区对齐键）表之间，构造一张地区 × 综合指标矩阵。
    """
    if ar.get("sales_trend") is None or ar.get("profit_transfer") is None:
        return None
    region_sales = ar["sales_trend"]["regional_sales"].groupby("地区")["总销售量"].sum()
    bp, bt = ar["profit_transfer"]["brand_profit"].copy(), ar["profit_transfer"]["brand_transfer"].copy()
    rs, bp.index, bt.index = region_sales.copy(), norm_index(bp.index), norm_index(bt.index)
    rs.index = norm_index(rs.index)
    common = rs.index.intersection(bp.index)
    if len(common) >= 2:
        tr = bt.reindex(common)
        return pd.DataFrame({"销售总量": rs.loc[common], "总利润": bp.loc[common, "总利润"], "平均利润率": bp.loc[common, "平均利润率"], "总过户数量": tr["总过户数量"] if "总过户数量" in tr.columns else 0}).fillna(0)
    rsr, pr = region_sales.sort_values(ascending=False), bp.sort_values(by="总利润", ascending=False)
    tcol = "总过户数量" if "总过户数量" in bt.columns else bt.columns[0]
    trr = bt.sort_values(by=tcol, ascending=False)
    n = min(len(rsr), len(pr), len(trr))
    if n < 2:
        return None
    return pd.DataFrame({"销售总量": rsr.values[:n], "总利润": pr["总利润"].values[:n], "平均利润率": pr["平均利润率"].values[:n], "总过户数量": trr[tcol].values[:n]}, index=rsr.index[:n]).fillna(0)


def principal_component_analysis(region_df):
    """
    作用：对地区综合指标表先做标准化（消除量纲），再执行 PCA，把多个相关指标压缩成少数几个主成分方向。
    """
    feats = list(region_df.select_dtypes(include=[np.number]).columns)
    if len(feats) < 2:
        return None
    Xs = StandardScaler().fit_transform(region_df[feats])
    pca = PCA()
    coord = pca.fit_transform(Xs)
    ev, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    k = int(np.argmax(cum >= 0.8) + 1) if np.any(cum >= 0.8) else len(cum)
    load = pd.DataFrame(pca.components_[:k].T, index=feats, columns=[f"PC{i+1}" for i in range(k)])
    return {"pca": pca, "coord": coord, "ev": ev, "cum": cum, "k": k, "feats": feats, "loadings": load, "table": region_df}


def export_pca(res):
    """
    作用：根据 PCA 结果画图——横轴为主成分个数，纵轴为累积解释方差比，并标出 80% 阈值与选中的 k 值。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(res["cum"]) + 1), res["cum"], "bo-", lw=2, ms=5)
    ax.axhline(0.8, color="r", ls="--", label="80%")
    ax.axvline(res["k"], color="g", ls="--", label=f"k={res['k']}")
    ax.set_xlabel("主成分数")
    ax.set_ylabel("累积解释方差比")
    ax.set_title("PCA 累积解释方差（综合分析）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_SAVE_PATH, "综合分析_PCA.png"), dpi=300, bbox_inches="tight")
    plt.close()
    res["table"].to_csv(os.path.join(DOCUMENT_SAVE_PATH, "地区PCA输入特征.csv"), encoding="utf-8-sig")
    pd.DataFrame({"主成分": [f"PC{i+1}" for i in range(len(res["ev"]))], "解释方差比": res["ev"], "累积方差比": res["cum"]}).to_csv(os.path.join(DOCUMENT_SAVE_PATH, "PCA解释方差.csv"), index=False, encoding="utf-8-sig")
    res["loadings"].to_csv(os.path.join(DOCUMENT_SAVE_PATH, "PCA负荷矩阵.csv"), encoding="utf-8-sig")
    with open(os.path.join(DOCUMENT_SAVE_PATH, "综合分析报告.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(["综合分析（仅 PCA）", f"保留主成分数(累积>=80%): {res['k']}", f"累积解释方差: {float(res['cum'][res['k'] - 1]):.4f}", f"特征列: {', '.join(res['feats'])}"]))


print("=" * 60 + "\n综合分析 — 主成分分析 PCA\n" + "=" * 60 + "\n加载子模块 CSV...")
AR = load_analysis_results()
print("构造地区特征矩阵...")
REG = build_region_table(AR)
if REG is None or len(REG) < 2:
    print("WARN 无法进行 PCA：缺少销售/利润数据或有效行数不足。")
else:
    print(f"  OK 特征表 {len(REG)} 行")
    R = principal_component_analysis(REG)
    if R:
        export_pca(R)
        print(f"  OK k={R['k']} cum={float(R['cum'][R['k'] - 1]):.4f} -> {DOCUMENT_SAVE_PATH}")
print("完成.")
