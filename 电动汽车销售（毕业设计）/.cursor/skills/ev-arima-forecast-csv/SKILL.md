---
name: ev-arima-forecast-csv
description: >-
  Clarifies ARIMA-based sales-trend outputs for the EV graduation project: what
  is being predicted, whether 销售预测结果.csv should list only future months
  versus history+forecast, and how that interacts with 综合分析.py. Use when the
  user asks what ARIMA predicts, why historical years appear in a forecast CSV,
  Prophet-style merged exports, or how to keep prediction files future-only.
---

# EV 毕业设计：ARIMA 销售预测与 CSV 约定

## 何时读此 Skill

用户讨论 **销售趋势分析.py**、**ARIMA**、**销售预测结果.csv**、历史数据混进预测文件、或 **综合分析** 读取预测表时出现，先按本节回答或改代码。

## ARIMA 在本项目里「预测了什么」

- 数据来自 `电动汽车销售指标表/月度销售趋势.xlsx`。
- 对 **「统计月份 + 总销售量」** 按自然月 `resample("M").sum()`：**该月表内所有行的销量加总**（多品牌、多记录在表层面先加总），得到 **一条单变量月度销量序列**。
- **ARIMA(p,d,q)**（如 (1,1,1)）拟合这条序列，**向前外推 `periods` 个月**（常见为 12），输出点预测与置信区间。
- **不是**按品牌、按地区分别预测；论文表述建议写清：**「基于月度汇总总销量的 ARIMA 外推」**。

## 「销售预测结果」CSV 应只含未来还是可含历史？

**用户语义上的「预测结果」= 仅未来若干个月**，列建议固定为：`日期`、`预测值`、`预测下限`、`预测上限`。

**易混点（历史为何会出现在「预测」CSV 里）**：

- 早期 **Prophet / 对齐下游** 的写法会把 **历史真实销量** 与 **未来预测** **纵向拼接** 到同一 DataFrame 再存盘，历史段用真实值填「预测值」、上下限用 NaN 或同值——便于一张图画全长曲线，或沿用 `tail(12)` 取「最后 12 行」当未来。
- 这会导致 CSV 里出现 **2015–2025 等已发生月份**，与文件名「预测结果」直觉冲突。

**推荐约定（与本项目当前实现一致）**：

- **`销售预测结果.csv`：只写入未来 `periods` 行**（全部为历史样本终点之后的月份）。
- **历史 + 预测的可视化**：只在 **PNG**（如 `销售预测图表.png`）中绘制，不强制写进「预测结果」CSV。

## 与 `综合分析.py` 的衔接

- 综合分析读取 `文档分析/销售预测结果.csv`，并用 **`forecast_data.tail(12)`** 汇总「未来 12 个月」等指标。
- 当 CSV **恰好为 12 行且均为预测月** 时，`tail(12)` 仍表示 **全部预测**，行为正确。
- 若将来把 `periods` 改为 24 等，需同步检查综合分析是否仍假设「最后 12 行」代表业务上的「未来一年」，必要时改为显式筛选日期或增加参数。

## 回答用户时的简短模板

1. **预测对象**：月度汇总总销量序列，外推未来 N 个月（默认 12）。
2. **为何曾含历史**：历史与预测拼接导出是旧格式习惯；**预测专用 CSV 建议只保留未来行**。
3. **改完后**：重新运行 `销售趋势分析.py` 覆盖生成 CSV；图仍可看全长趋势。

## 改代码时的检查清单

- [ ] `销售预测结果.csv` 是否仅包含 `get_forecast(steps=periods)` 对应日期行？
- [ ] 列名仍为 `日期` / `预测值` / `预测下限` / `预测上限`（与综合分析、旧 Prophet 列兼容）？
- [ ] 修改 `periods` 后是否检查 `综合分析.py` 中 `tail(12)` 的语义是否仍成立？
