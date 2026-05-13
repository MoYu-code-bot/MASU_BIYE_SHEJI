"""
读取利润表与过户表，先做按品牌的描述性汇总，再用梯度提升树（配合网格搜索）分别拟合利润与过户均价，并导出 CSV 与图表。
"""
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BASE = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\利润过户分析"
IMAGE_SAVE_PATH = os.path.join(BASE, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE, "文档分析")
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(DOCUMENT_SAVE_PATH, exist_ok=True)
DATA_DIR = r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表"
COLUMN_MAPPING = {
    "profit": ["total_profit", "利润", "净利润", "收益", "profit"],
    "revenue": ["total_revenue", "收入", "销售额", "营收", "revenue"],
    "total_cost": ["total_cost", "总成本", "成本", "总费用"],
    "brand": ["brand", "品牌", "品牌名称"],
    "stat_month": ["stat_month", "统计月份", "月份", "时间"],
    "transfer_count": ["transfer_count", "过户数量", "交易数量"],
    "avg_transfer_price": ["avg_transfer_price", "平均过户价格", "平均交易价格"],
}


def get_column_name(df, keys):
    """
    作用：在一张表里，从候选列名列表 keys 里找出第一个真实存在的列名。
    """
    for k in keys:
        if k in df.columns:
            return k
    return None


def encode_brand_series(d, br_c):
    """
    作用：把品牌这一列用 LabelEncoder 变成整数列 brand_encoded，供树模型作为数值特征使用。
    """
    if br_c is not None and d[br_c].nunique() > 1:
        d["brand_encoded"] = LabelEncoder().fit_transform(d[br_c].astype(str))
    else:
        d["brand_encoded"] = 0


def load_tables():
    """
    作用：从指标表目录读取利润分析，过户状态分析两个Excel，把统计月份转成日期类型，
    并对利润、收入、过户量、均价等关键列做删除空值、去掉明显不合理行等基础清洗。
    """
    profit_df = pd.read_excel(os.path.join(DATA_DIR, "利润分析.xlsx"))
    transfer_df = pd.read_excel(os.path.join(DATA_DIR, "过户状态分析.xlsx"))
    for df in (profit_df, transfer_df):
        sm = get_column_name(df, COLUMN_MAPPING["stat_month"])
        if sm:
            df[sm] = pd.to_datetime(df[sm], errors="coerce")
    pc, rv = get_column_name(profit_df, COLUMN_MAPPING["profit"]), get_column_name(profit_df, COLUMN_MAPPING["revenue"])
    if pc and rv:
        profit_df = profit_df.dropna(subset=[pc, rv])
        profit_df = profit_df[profit_df[rv] > 0]
    tc, ap = get_column_name(transfer_df, COLUMN_MAPPING["transfer_count"]), get_column_name(transfer_df, COLUMN_MAPPING["avg_transfer_price"])
    if tc and ap:
        transfer_df = transfer_df.dropna(subset=[tc, ap])
        transfer_df = transfer_df[transfer_df[tc] >= 0]
    print(f"  OK 利润 {len(profit_df)} 条  过户 {len(transfer_df)} 条")
    return profit_df, transfer_df


def profit_and_transfer_insights(profit_df, transfer_df):
    """
    作用：做描述性统计——按品牌汇总利润、利润率，以及过户量、均价、过户金额等。
    """
    pi, ti = {}, {}
    pc, rv, br = get_column_name(profit_df, COLUMN_MAPPING["profit"]), get_column_name(profit_df, COLUMN_MAPPING["revenue"]), get_column_name(profit_df, COLUMN_MAPPING["brand"])
    if pc and rv and not profit_df.empty:
        g = profit_df.copy()
        g["profit_margin"] = (g[pc] / g[rv].replace(0, np.nan)) * 100
        pi["overall_metrics"] = {"record_count": len(g), "total_profit": g[pc].sum(), "total_revenue": g[rv].sum(), "avg_margin": g["profit_margin"].mean(), "margin_std": g["profit_margin"].std()}
        pi["brand_profit"] = g.groupby(br).agg({pc: "sum", "profit_margin": "mean"}).rename(columns={pc: "总利润", "profit_margin": "平均利润率"}).sort_values("总利润", ascending=False) if br else pd.DataFrame()
    tc, ap, br2 = get_column_name(transfer_df, COLUMN_MAPPING["transfer_count"]), get_column_name(transfer_df, COLUMN_MAPPING["avg_transfer_price"]), get_column_name(transfer_df, COLUMN_MAPPING["brand"])
    if tc and ap and not transfer_df.empty:
        h = transfer_df.copy()
        h["transfer_amount"] = h[tc] * h[ap]
        ti["overall_metrics"] = {"total_records": len(h), "total_transfer_count": h[tc].sum(), "avg_monthly_transfers": h[tc].mean(), "avg_transfer_price": h[ap].mean(), "total_transfer_amount": h["transfer_amount"].sum()}
        ti["brand_transfer"] = h.groupby(br2).agg({tc: "sum", ap: "mean", "transfer_amount": "sum"}).rename(columns={tc: "总过户数量", ap: "平均过户价格", "transfer_amount": "过户总金额"}).sort_values("总过户数量", ascending=False) if br2 else pd.DataFrame()
    return pi, ti


def fit_gbr(X_train, X_test, y_train, y_test, feature_names):
    """
    作用：在已经标准化好的训练集、测试集上，用网格搜索（GridSearchCV）挑选梯度提升回归的超参数，
    """
    gs = GridSearchCV(GradientBoostingRegressor(random_state=42), {"n_estimators": [80, 120], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}, cv=3, scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X_train, y_train)
    pred = gs.best_estimator_.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    fi = pd.DataFrame({"特征名称": feature_names, "重要性": gs.best_estimator_.feature_importances_}).sort_values("重要性", ascending=False)
    return {"GradientBoosting": {"rmse": rmse, "r2": r2_score(y_test, pred)}}, fi


def train_profit_gbr(df):
    """
    作用：以利润为预测目标，构造特征：年份、月份、季度、月份的正余弦（刻画季节性）、收入、品牌编码、可选成本等；
    按时间顺序取前 80% 做训练、后 20% 做测试（若测试集太小则改为随机划分），标准化后调用 fit_gbr。
    """
    pc, rv, sm = get_column_name(df, COLUMN_MAPPING["profit"]), get_column_name(df, COLUMN_MAPPING["revenue"]), get_column_name(df, COLUMN_MAPPING["stat_month"])
    br_c, cc = get_column_name(df, COLUMN_MAPPING["brand"]), get_column_name(df, COLUMN_MAPPING["total_cost"])
    if not all([pc, rv, sm]) or len(df) < 20:
        return {}, pd.DataFrame()
    d = df.dropna(subset=[sm]).copy()
    d["year"], d["month"], d["quarter"] = d[sm].dt.year, d[sm].dt.month, d[sm].dt.quarter
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * d["month"] / 12), np.cos(2 * np.pi * d["month"] / 12)
    encode_brand_series(d, br_c)
    feats = ["year", "month", "quarter", "month_sin", "month_cos", rv, "brand_encoded"] + ([cc] if cc and cc in d.columns else [])
    feats = [c for c in feats if c in d.columns]
    X, y = d[feats].fillna(0), d[pc]
    si = int(len(X) * 0.8)
    X_tr, X_te, y_tr, y_te = X[:si], X[si:], y[:si], y[si:]
    if len(X_te) < 5:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    return fit_gbr(sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te, feats)


def train_transfer_gbr(df):
    """
    作用：以平均过户价格为预测目标，特征包括过户量、品牌编码，以及若存在统计月份则加入年、月、季度。
    过滤掉均价不大于 0 的行；样本不足则不建模。随机划分训练测试集后同样走 fit_gbr。
    """
    tc, ap, sm = get_column_name(df, COLUMN_MAPPING["transfer_count"]), get_column_name(df, COLUMN_MAPPING["avg_transfer_price"]), get_column_name(df, COLUMN_MAPPING["stat_month"])
    br_c = get_column_name(df, COLUMN_MAPPING["brand"])
    if not tc or not ap or len(df.dropna(subset=[tc, ap])) < 20:
        return None
    d = df.dropna(subset=[tc, ap]).copy()
    d = d[d[ap] > 0]
    encode_brand_series(d, br_c)
    feats = [tc, "brand_encoded"]
    if sm and sm in d.columns:
        d[sm] = pd.to_datetime(d[sm], errors="coerce")
        d = d.dropna(subset=[sm])
        d["year"], d["month"], d["quarter"] = d[sm].dt.year, d[sm].dt.month, d[sm].dt.quarter
        feats.extend(["year", "month", "quarter"])
    feats = [c for c in feats if c in d.columns]
    X, y = d[feats].fillna(0), d[ap]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    out, _ = fit_gbr(sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te, feats)
    o = out["GradientBoosting"]
    return {"rmse": o["rmse"], "r2": o["r2"], "model_name": "GradientBoosting"}


def main():
    """
    作用：串联整个流程——读表、描述性汇总、训练利润与过户两个梯度提升模型、绘制品牌利润 Top10 图、
    """
    print("=" * 50 + "\n利润过户分析（梯度提升回归）\n" + "=" * 50)
    profit_df, transfer_df = load_tables()
    pi, ti = profit_and_transfer_insights(profit_df, transfer_df)
    profit_results, profit_fi = train_profit_gbr(profit_df)
    transfer_results = train_transfer_gbr(transfer_df)
    bp = pi.get("brand_profit") if pi else None
    if bp is not None and not bp.empty:
        top = bp.head(10)
        vals = top["总利润"].values[::-1]
        ma = float(np.nanmax(np.abs(vals))) if len(vals) else 0.0
        if ma <= 0:
            lbl_scale = 1.0
        else:
            lbl_scale = 10 ** int(np.ceil(np.log10(ma)))
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(range(len(top)), vals, color="steelblue")
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            txt = f"{v / lbl_scale:.2f}"
            if w >= 0:
                ax.annotate(txt, xy=(w, y), xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=9)
            else:
                ax.annotate(txt, xy=(w, y), xytext=(-4, 0), textcoords="offset points", va="center", ha="right", fontsize=9)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index.astype(str)[::-1])
        ax.set_xlabel("总利润")
        ax.set_title("品牌利润 Top10")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_SAVE_PATH, "利润分析图表.png"), dpi=300, bbox_inches="tight")
        plt.close()
    if pi and not pi.get("brand_profit", pd.DataFrame()).empty:
        df_csv = pi["brand_profit"].copy()
        df_csv.index.name = "品牌"
        df_csv.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "品牌利润分析.csv"), encoding="utf-8-sig")
    if ti and not ti.get("brand_transfer", pd.DataFrame()).empty:
        df_t = ti["brand_transfer"].copy()
        df_t.index.name = "品牌"
        df_t.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "品牌过户分析.csv"), encoding="utf-8-sig")
    if not profit_fi.empty:
        profit_fi.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "利润预测特征重要性.csv"), index=False, encoding="utf-8-sig")
    tr_s = "未拟合"
    if transfer_results:
        tr = transfer_results
        tr_s = f"rmse={float(tr['rmse']):.4f}, r2={float(tr['r2']):.4f}, {tr['model_name']}"
    pr2 = profit_results.get("GradientBoosting", {}).get("r2")
    pr2_s = f"{float(pr2):.6f}" if pr2 is not None else "N/A"
    rep = "\n".join(["=" * 50, "利润过户分析报告（仅 GBR）", f"利润模型 R2: {pr2_s}", f"过户价模型: {tr_s}", "=" * 50])
    with open(os.path.join(DOCUMENT_SAVE_PATH, "利润过户分析报告.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep + f"\n输出目录: {DOCUMENT_SAVE_PATH}")


if __name__ == "__main__":
    main()
