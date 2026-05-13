# 字段映射参考

本文件用于在修复“代码字段”和“CSV真实字段”不一致时快速对照。

## 常见字段归一建议

- 时间类：`stat_month` / `month` / `统计月份` / `月份`
- 品牌类：`brand` / `品牌`
- 区域类：`region` / `地区` / `区域`
- 销量类：`sales_volume` / `total_sales_volume` / `销量`
- 销售额类：`sales_amount` / `total_sales_amount` / `销售额`
- 利润类：`profit` / `total_profit` / `利润`
- 利润率类：`profit_margin` / `avg_margin` / `利润率`
- 客户分群：`customer_segment` / `segment` / `客户分群`
- 客户数量：`customer_count` / `total_customers` / `客户数`

## 强制校验模板

在读取 CSV 后立刻校验关键列：

```python
required = ["brand", "total_sales_volume"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"CSV 缺少关键列: {missing}; 当前列: {list(df.columns)}"
    )
```

## 文件缺失模板

```python
from pathlib import Path

path = Path(file_path)
if not path.exists():
    raise FileNotFoundError(f"未找到文件: {path}")
```

## 禁止事项

- 不得使用 `np.random` 生成业务数据替代真实数据
- 不得在 `except` 里悄悄造模拟表继续执行
- 不得假设 CSV 一定有英文列名
