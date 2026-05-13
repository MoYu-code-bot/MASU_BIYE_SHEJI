"""
读取月度销量明细，整理成固定时间范围内的月度序列，用 ARIMA 预测未来若干个月，并导出预测表、趋势图与地区汇总。
"""
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\销售趋势分析"
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")
for path in (IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH):
    os.makedirs(path, exist_ok=True)

DATA_DIR = (
    r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表"
)
MONTHLY_XLSX = os.path.join(DATA_DIR, "月度销售趋势.xlsx")
REGIONAL_XLSX = os.path.join(DATA_DIR, "地区销售趋势.xlsx")

SAMPLE_START = pd.Timestamp("2015-04-01")
SAMPLE_END = pd.Timestamp("2025-04-30")
SAMPLE_LABEL = "2015-04—2025-04"


def load_monthly_sales():
    """
    作用：从月度销售趋势Excel 读入数据，把程序内部用的英文列名统一改成中文列名。
    删除日期或销量为空的行，保证后面按月汇总时数据完整。
    """
    df = pd.read_excel(MONTHLY_XLSX)
    cn_map = {
        "stat_month": "统计月份",
        "brand": "品牌",
        "total_sales_volume": "总销售量",
        "avg_sale_price": "平均销售价格",
        "total_production_volume": "总生产量",
    }
    df = df.rename(columns={k: v for k, v in cn_map.items() if k in df.columns})
    if "统计月份" not in df.columns:
        raise KeyError("月度表缺少「统计月份」列。")
    if "总销售量" not in df.columns:
        raise KeyError("月度表缺少「总销售量」列。")
    df["统计月份"] = pd.to_datetime(df["统计月份"], errors="coerce")
    df["总销售量"] = pd.to_numeric(df["总销售量"], errors="coerce")
    df = df.dropna(subset=["统计月份", "总销售量"])
    print(f"  OK 月度原始行数: {len(df)} 条（将按 {SAMPLE_LABEL} 汇总为月度序列）")
    return df


def monthly_series(df, date_col="统计月份", value_col="总销售量"):
    """
    作用：把明细按月末重采样，把多行加总成每个月一个总销量，再只保留 SAMPLE_START 到 SAMPLE_END 之间的区间。
    """
    s = df.sort_values(date_col).set_index(date_col)[value_col].resample("M").sum()
    if s.isna().all():
        raise ValueError("月度销量序列为空，无法拟合 ARIMA。")
    s = s.loc[SAMPLE_START:SAMPLE_END]
    if len(s) < 12:
        raise ValueError(f"样本区间内有效月份不足（当前 {len(s)} 个月），请检查 Excel 是否在 {SAMPLE_LABEL} 范围内。")
    print(f"  OK 用于建模的月度序列: {s.index.min().date()} ~ {s.index.max().date()}，共 {len(s)} 个月")
    return s.fillna(s.mean())


def arima_forecast(ts, periods=12, order=(1, 1, 1)):
    """
    作用：对历史月度销量序列拟合 ARIMA，向前预测 periods 个月，并整理成带日期、预测均值、置信区间的表格。
    """
    fit = ARIMA(ts, order=order).fit()
    frame = fit.get_forecast(steps=periods).summary_frame()
    if "mean_ci_lower" not in frame.columns:
        frame = frame.copy()
        frame["mean_ci_lower"] = frame["mean_lower"]
        frame["mean_ci_upper"] = frame["mean_upper"]
    last = ts.index[-1]
    fut = pd.date_range(start=last, periods=periods + 1, freq="M")[1:]
    print(f"  OK ARIMA{order} 完成，未来 {periods} 月；预测均值 {frame['mean'].min():.2f}~{frame['mean'].max():.2f}")
    forecast_only = pd.DataFrame(
        {
            "日期": fut,
            "预测值": frame["mean"].values,
            "预测下限": frame["mean_ci_lower"].values,
            "预测上限": frame["mean_ci_upper"].values,
        }
    )
    return ts, frame, fut, forecast_only


def plot_arima(ts, fut, frame, out_png, sample_label=SAMPLE_LABEL):
    """
    作用：画两张叠在一起的内容——历史销量折线、未来预测折线与置信带；竖线标出历史与预测的分界点。
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts.index, ts.values, label=f"历史销量（样本 {sample_label}）", linewidth=2, color="C0")
    ax.axvline(ts.index[-1], color="gray", ls=":", lw=1.2, label="历史终点 / 预测起点")
    lo = np.maximum(frame["mean_ci_lower"].values, 0)
    ax.plot(fut, frame["mean"], "--", color="C3", label="ARIMA 预测（未来12个月）", linewidth=2)
    ax.fill_between(fut, lo, frame["mean_ci_upper"].values, color="C3", alpha=0.2, label="95% 区间（下限截断于0）")
    ax.set_title(f"月度总销量 ARIMA 预测\n历史样本：{sample_label}；预测自 {ts.index[-1].strftime('%Y-%m')} 次月起")
    ax.set_xlabel("日期")
    ax.set_ylabel("总销售量")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.subplots_adjust(bottom=0.14, right=0.78)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  OK 图: {out_png}")


def export_regional_groupby():
    """
    作用：读取地区销售趋势Excel，按地区分组汇总总销售量；若表中有市场份额、销售排名等列则一并做简单聚合（如求均值）。
    """
    try:
        r = pd.read_excel(REGIONAL_XLSX)
        m = {
            "origin": "地区",
            "total_sales_volume": "总销售量",
            "market_share": "市场份额",
            "sales_rank": "销售排名",
        }
        r = r.rename(columns={k: v for k, v in m.items() if k in r.columns})
        if "地区" not in r.columns or "总销售量" not in r.columns:
            print("  WARN 地区表缺少「地区」或「总销售量」，跳过地区 CSV")
            return
        r["总销售量"] = pd.to_numeric(r["总销售量"], errors="coerce").fillna(0)
        agg = {"总销售量": "sum"}
        if "市场份额" in r.columns:
            agg["市场份额"] = "mean"
        if "销售排名" in r.columns:
            agg["销售排名"] = "mean"
        out = r.groupby("地区", as_index=False).agg(agg).sort_values("总销售量", ascending=False)
        out.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "地区销售汇总.csv"), index=False, encoding="utf-8-sig")
        print("  OK 地区销售汇总.csv（按地区分组汇总）")
    except Exception as e:
        print(f"  WARN 地区 CSV 未导出: {e}")


def build_report(ts, frame):
    """
    作用：把样本时间范围、历史销量合计、未来 12 月预测合计与月均、以及当前运行时间拼成多行文字。
    """
    return "\n".join(
        [
            "=" * 60,
            "电动汽车销售趋势分析（ARIMA）",
            "=" * 60,
            f"历史样本区间: {SAMPLE_LABEL}",
            f"实际序列: {ts.index.min().date()} ~ {ts.index.max().date()}  合计 {ts.sum():,.2f}",
            f"预测段(未来12月)合计 {frame['mean'].sum():,.2f}  月均 {frame['mean'].mean():,.2f}",
            "=" * 60,
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
    )


if __name__ == "__main__":
    print("=" * 60 + "\n销售趋势分析（ARIMA）\n" + "=" * 60)
    monthly = load_monthly_sales()
    ts, fc_frame, fut_idx, forecast_csv = arima_forecast(monthly_series(monthly))
    plot_arima(ts, fut_idx, fc_frame, os.path.join(IMAGE_SAVE_PATH, "销售预测图表.png"))
    forecast_csv.to_csv(os.path.join(DOCUMENT_SAVE_PATH, "销售预测结果.csv"), index=False, encoding="utf-8-sig")
    print(f"  OK 销售预测结果.csv（共 {len(forecast_csv)} 行，均为 {SAMPLE_LABEL} 末月之后的预测月）")
    export_regional_groupby()
    rep = build_report(ts, fc_frame)
    with open(os.path.join(DOCUMENT_SAVE_PATH, "销售趋势分析报告.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep + "\n  OK 报告已写入文档分析目录\n" + "=" * 60 + "\n完成。\n" + "=" * 60)
