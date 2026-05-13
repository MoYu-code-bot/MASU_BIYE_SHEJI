"""
把多份产品相关 Excel 合并成按月份对齐的宽表，用随机森林看哪些指标与销量关系更密切，并导出表格与图表。
"""
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import MultiIndex
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\产品特征分析"
IMAGE_SAVE_PATH = os.path.join(ROOT_SAVE_PATH, "图片分析")
DOC_SAVE_PATH = os.path.join(ROOT_SAVE_PATH, "文档分析")
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
os.makedirs(DOC_SAVE_PATH, exist_ok=True)
BASE = r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表"
FILE_PATHS = {k: os.path.join(BASE, v) for k, v in [("range", "续航里程分析.xlsx"), ("battery", "电池容量分析.xlsx"), ("price", "价格区间分析.xlsx"), ("drive", "驱动形式分析.xlsx"), ("smart", "智能化水平分析.xlsx"), ("charging", "充电效率分析.xlsx")]}
COLUMN_MAPPINGS = {"stat_month": ["stat_month", "统计月份", "月份", "时间", "日期", "统计时间"], "range_km": ["range_km", "续航里程", "续航(km)", "里程", "续航", "最大续航", "综合续航", "avg_range_nedc"], "battery_capacity_kwh": ["battery_capacity_kwh", "电池容量", "容量(kwh)", "电池容量kwh", "电池功率", "电池大小", "avg_range_nedc"], "price_range": ["price_range", "价格区间", "价格段", "价格档位", "价位"], "drive_type": ["drive_type", "驱动形式", "驱动类型", "驱动方式", "动力类型"], "smart_level": ["smart_level", "智能化水平", "智能等级", "智能评分", "智能程度", "intelligence_level"], "charging_efficiency": ["charging_efficiency", "充电效率", "充电效能", "充电速度", "快充效率", "charging_efficiency_range"], "sales_volume": ["sales_volume", "销量", "销售量", "销售数量", "销售台数"], "customer_preference": ["customer_preference", "客户偏好", "偏好度", "满意度", "受欢迎度", "customer_satisfaction"], "efficiency_ratio": ["efficiency_ratio", "效率比", "能效比", "电池效率", "能量密度", "customer_preference_score"], "avg_price": ["avg_price", "平均价格", "均价", "价格", "售价", "单价", "avg_sale_price"], "market_share": ["market_share", "市场份额", "份额", "占比"], "performance_score": ["performance_score", "性能评分", "动力评分"], "energy_efficiency": ["energy_efficiency", "能效", "能源效率"], "tech_score": ["tech_score", "技术评分", "科技分", "技术分", "future_potential"], "market_acceptance": ["market_acceptance", "市场接受度", "接受度评分"]}
CSV_COLUMN_MAPPING = {"stat_month": "统计月份", "range_km": "续航里程(km)", "avg_range": "平均续航里程", "range_sales": "续航相关销量", "range_preference": "续航客户偏好", "customer_preference": "客户偏好", "avg_battery_capacity": "平均电池容量", "battery_capacity_kwh": "电池容量(kWh)", "avg_battery": "平均电池容量", "battery_sales": "电池相关销量", "battery_efficiency": "电池效率", "efficiency_ratio": "效率比", "battery_avg_price": "电池均价", "avg_sale_price": "平均售价", "price_range": "价格区间", "dominant_price_range": "主导价格区间", "price_sales": "价格相关销量", "avg_price": "平均价格", "avg_profit_margin": "平均利润率", "customer_satisfaction_score": "客户满意度评分", "drive_type": "驱动形式", "dominant_drive": "主导驱动形式", "drive_sales": "驱动相关销量", "drive_market_share": "驱动市场份额", "performance_score": "性能评分", "energy_efficiency": "能源效率", "market_share": "市场份额", "smart_level": "智能化水平", "avg_smart_level": "平均智能化水平", "smart_sales": "智能相关销量", "tech_score": "技术评分", "avg_tech_score": "平均技术评分", "charging_efficiency": "充电效率", "charging_sales": "充电相关销量", "charging_time_hours": "充电时间(小时)", "avg_market_acceptance": "平均市场接受度", "market_acceptance": "市场接受度", "sales_volume": "销量", "sample_size": "样本数量", "product_cluster": "产品聚类", "feature": "特征", "importance": "重要性", "year": "年份"}
AGG_CONFIGS = {"range": {"range_km": "mean", "sales_volume": "sum", "customer_preference": "mean", "avg_battery_capacity": "mean"}, "battery": {"battery_capacity_kwh": "mean", "sales_volume": "sum", "efficiency_ratio": "mean", "avg_sale_price": "mean"}, "price": {"sales_volume": "sum", "avg_price": "mean", "avg_profit_margin": "mean"}, "drive": {"sales_volume": "sum", "market_share": "mean", "performance_score": "mean", "energy_efficiency": "mean"}, "smart": {"smart_level": "mean", "sales_volume": "sum", "tech_score": "mean"}, "charging": {"sales_volume": "sum", "market_acceptance": "mean"}}
RENAME_RULES = {"range": {"range_km": "avg_range", "sales_volume": "range_sales", "customer_preference": "range_preference"}, "battery": {"battery_capacity_kwh": "avg_battery", "sales_volume": "battery_sales", "efficiency_ratio": "battery_efficiency", "avg_sale_price": "battery_avg_price"}, "price": {"sales_volume": "price_sales"}, "drive": {"sales_volume": "drive_sales", "market_share": "drive_market_share"}, "smart": {"smart_level": "avg_smart_level", "sales_volume": "smart_sales", "tech_score": "avg_tech_score"}, "charging": {"sales_volume": "charging_sales", "market_acceptance": "avg_market_acceptance"}}


def _find_col(df, names):
    """
    作用：在一张表里，找出实际存在列名。
    """
    for n in names:
        if n in df.columns:
            return n
    low = [c.strip().lower() for c in df.columns]
    for t in names:
        tl = t.strip().lower()
        for i, cl in enumerate(low):
            if tl in cl or cl in tl:
                return df.columns[i]
    return None


def _std_cols(df):
    """
    作用：统一改成程序内部约定的标准名字。
    """
    rd = {}
    for std, names in COLUMN_MAPPINGS.items():
        f = _find_col(df, names)
        if f and f != std:
            rd[f] = std
    return df.rename(columns=rd)


def _agg(df, agg_dict, name):
    """
    作用：对单张产品维度表，按统计月份分组，把多行明细聚合成每个月一行的汇总行。
    例如：销量类字段通常按月求和，续航、偏好等水平类指标按月求平均。
    """
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in agg_dict if c in df.columns}
    num = {k: v for k, v in num.items() if v.notna().sum() > 0}
    if not num or "stat_month" not in df.columns:
        print(f"  ⚠ {name} 跳过聚合")
        return pd.DataFrame()
    ad = pd.DataFrame(num)
    ad["stat_month"] = df["stat_month"]
    r = ad.groupby("stat_month").agg({k: agg_dict[k] for k in num}).reset_index()
    if isinstance(r.columns, MultiIndex):
        r.columns = ["_".join(c).strip() for c in r.columns.values]
        for c in list(r.columns):
            if "stat_month" in c:
                r.rename(columns={c: "stat_month"}, inplace=True)
                break
    return r


def _load_merge():
    """
    作用：读取六类产品特征 Excel，标准化列名、按月汇总后，按月份把多张表横向拼成一张宽表。
    宽表里每一行是一个月，列来自续航、电池、价格、驱动、智能、充电等不同维度；重复含义的列会做合并与缺失填充。
    """
    print("\n【1/2】加载并合并…")
    data = {}
    for k, path in FILE_PATHS.items():
        try:
            d = pd.read_excel(path)
            d = _std_cols(d)
            if "stat_month" in d.columns:
                d["stat_month"] = pd.to_datetime(d["stat_month"], errors="coerce")
            data[k] = d
            print(f"  ✓ {k}: {len(d)}")
        except Exception as e:
            print(f"  ⚠ {k}: {e}")
            data[k] = pd.DataFrame()
    merged = pd.DataFrame()
    for name, df in data.items():
        if df.empty:
            continue
        a = _agg(df, AGG_CONFIGS[name], name)
        if a.empty:
            continue
        a = a.rename(columns=RENAME_RULES.get(name, {}))
        merged = a if merged.empty else merged.merge(a, on="stat_month", how="outer", suffixes=("", f"_{name}_dup"))
    for col in [c for c in merged.columns if "_dup" in c]:
        b = col.rsplit("_", 2)[0] if col.count("_") >= 2 else col.split("_")[0]
        if b in merged.columns:
            merged[b] = merged[b].fillna(merged[col])
        merged.drop(col, axis=1, inplace=True)
    for c in merged.select_dtypes(include=[np.number]).columns:
        merged[c] = merged[c].fillna(merged[c].mean())
    print(f"  ✓ 合并 {merged.shape}")
    return merged


def _round_numeric_2(df):
    """
    作用：把表中所有数值列统一保留两位小数，让导出到 CSV 的数字更整齐、便于阅读。
    """
    d = df.copy()
    for c in d.select_dtypes(include=[np.number]).columns:
        d[c] = d[c].round(2)
    return d


def random_forest_importance(df):
    """
    作用：在合并好的月度数据上，用随机森林回归估计哪些数值指标对销量类目标帮助更大。
    """
    print("\n【2/2】随机森林回归…")
    targets = [c for c in df.columns if "sales" in c.lower() or "销量" in c]
    tgt = next((c for c in targets if df[c].notna().sum() > 5), None)
    if not tgt:
        print("  ⚠ 无销量目标列")
        return pd.DataFrame(), None, None
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (tgt, "stat_month") and df[c].notna().sum() > 0]
    if len(feats) < 1:
        print("  ⚠ 无特征列")
        return pd.DataFrame(), None, None
    X = df[feats].fillna(df[feats].mean())
    y = df[tgt].fillna(df[tgt].mean())
    y = y[(y >= y.quantile(0.01)) & (y <= y.quantile(0.99))]
    X = X.loc[y.index]
    if len(X) < 10:
        print("  ⚠ 样本不足")
        return pd.DataFrame(), None, None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(Xtr, ytr)
    pr = rf.predict(Xte)
    print(f"  ✓ 目标={tgt}  MSE={mean_squared_error(yte, pr):.2f}  R²={r2_score(yte, pr):.3f}")
    imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp = imp.loc[imp["importance"].round(2) > 0].reset_index(drop=True)
    imp["importance"] = imp["importance"].round(2)
    top = imp.head(10).copy()
    top["cn"] = top["feature"].map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
    top["cn"] = top["cn"].apply(lambda s: (str(s)[:10] + "…") if len(str(s)) > 11 else s)
    _, ax = plt.subplots(figsize=(10, 5))
    if top.empty:
        ax.text(0.5, 0.5, "无非零重要性特征", ha="center", va="center", transform=ax.transAxes)
    else:
        y_lab = top["cn"][::-1]
        x_val = top["importance"][::-1]
        bars = ax.barh(y_lab, x_val, color="steelblue")
        xmax = float(x_val.max()) if len(x_val) else 0.0
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.008 * xmax + 1e-6,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1.0)
    ax.set_xlabel("重要性")
    ax.set_title(f"随机森林特征重要性 Top{len(top)}" if len(top) else "随机森林特征重要性")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_SAVE_PATH, "产品特征重要性排名.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return imp, tgt, feats


def main():
    """
    作用：程序入口，依次执行合并数据 → 随机森林 → 导出多种 CSV 和文字报告。
    """
    print("=" * 50 + "\n产品特征分析（随机森林）\n" + "=" * 50)
    merged = _load_merge()
    if merged.empty:
        print("无数据，终止。")
        return
    imp, tgt, feats = random_forest_importance(merged)
    num = merged.select_dtypes(include=[np.number])
    corr = num.corr() if num.shape[1] > 1 else pd.DataFrame()
    summary = _round_numeric_2(merged.rename(columns=CSV_COLUMN_MAPPING))
    summary.to_csv(os.path.join(DOC_SAVE_PATH, "产品特征汇总数据.csv"), index=False, encoding="utf-8-sig")
    if not corr.empty:
        cc = _round_numeric_2(corr.copy())
        cc.index = cc.index.map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
        cc.columns = cc.columns.map(lambda x: CSV_COLUMN_MAPPING.get(x, x))
        cc.to_csv(os.path.join(DOC_SAVE_PATH, "产品特征相关性矩阵.csv"), encoding="utf-8-sig")
    if not imp.empty:
        _round_numeric_2(imp.rename(columns=CSV_COLUMN_MAPPING)).to_csv(
            os.path.join(DOC_SAVE_PATH, "产品特征重要性.csv"), index=False, encoding="utf-8-sig"
        )
    if feats:
        row = _round_numeric_2(merged[feats].mean().to_frame().T.rename(columns=CSV_COLUMN_MAPPING))
        row.to_csv(os.path.join(DOC_SAVE_PATH, "产品特征聚类结果.csv"), encoding="utf-8-sig")
    print(f"  ✓ CSV → {DOC_SAVE_PATH}")
    lines = ["=" * 50, "产品特征分析报告（随机森林）", f"样本: {len(merged)} 月  目标: {tgt or '无'}"]
    if not imp.empty:
        lines += ["Top5 重要性:"] + [f"  • {CSV_COLUMN_MAPPING.get(r['feature'], r['feature'])}: {r['importance']:.2f}" for _, r in imp.head(5).iterrows()]
    lines.append("=" * 50)
    rep = "\n".join(lines)
    print("\n" + rep)
    with open(os.path.join(DOC_SAVE_PATH, "产品特征分析报告.txt"), "w", encoding="utf-8") as f:
        f.write(rep)


if __name__ == "__main__":
    main()
