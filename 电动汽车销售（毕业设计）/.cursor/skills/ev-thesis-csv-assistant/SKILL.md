---
name: ev-thesis-csv-assistant
description: Repair Python analysis code strictly against real CSV files without mock data, and enforce graduation-thesis formatting requirements. Use when the user mentions CSV mismatch errors, KeyError, column mapping, machine learning analysis for EV sales, or thesis formatting/checking.
---

# EV Thesis CSV Assistant

## 适用场景

- 用户要求根据真实 CSV 修复 Python 报错（如 `KeyError`、字段名不一致、编码问题）
- 用户明确要求不要模拟数据、不要模板数据
- 用户要求按现有表数据做机器学习分析
- 用户要求输出毕业论文内容并符合学院格式规范

## 核心原则

1. 严格以用户给定 CSV/Excel 为准，不生成模拟数据。
2. 先读取实际列名，再改代码，不允许猜字段。
3. 报错修复优先保证“可运行 + 可复现 + 可解释”。
4. 给初学者可读结果：步骤清楚、术语简单、输出明确。
5. 涉及论文内容时，遵循项目中的论文格式规则文件。

## 代码修复工作流（CSV 场景）

复制以下清单并执行：

```text
Task Progress:
- [ ] Step 1: 收集报错栈与目标脚本
- [ ] Step 2: 读取所有相关 CSV 的真实列名与样例行
- [ ] Step 3: 建立“代码字段 -> CSV真实字段”映射
- [ ] Step 4: 移除模拟数据分支，改为真实数据校验与报错提示
- [ ] Step 5: 统一路径、编码、缺失值处理策略
- [ ] Step 6: 运行脚本验证并修复剩余异常
- [ ] Step 7: 输出完整可运行代码与依赖清单
```

### Step 1: 收集上下文

- 必须拿到：
  - 目标 Python 文件路径
  - 报错全文
  - 用户提供的 CSV 文件列表

### Step 2: 读取真实数据结构

- 对每个 CSV：
  - 读取列名
  - 检查关键字段是否存在（如时间、品牌、销量、利润、客户分群）
  - 检查编码（优先 `utf-8-sig`）

### Step 3: 字段映射

- 建立显式映射表，例如：

```text
代码字段 expected_col -> 真实字段 actual_col
customer_segment -> 客户分群
stat_month -> 统计月份
brand -> 品牌
```

- 如果找不到等价字段：停止并向用户说明缺失字段，不强行继续。

### Step 4: 禁止模拟数据

- 删除/禁用以下模式：
  - `except: 生成模拟数据`
  - `np.random.*` 伪造业务数据
  - 文件不存在时自动构造 DataFrame

- 替代方案：
  - 文件缺失时抛出可读异常并列出缺失文件
  - 字段缺失时给出“发现列名列表 + 需要列名列表”

### Step 5: 稳定化处理

- 统一路径：优先使用配置常量或参数化路径
- 时间字段：统一转 `datetime` 后再按月/季度/年聚合
- 数值字段：`to_numeric(errors="coerce")` 后处理缺失值
- 聚类/预测前，明确筛除空值并记录样本数

### Step 6: 验证

- 至少执行一次完整运行
- 若失败，继续迭代修复，直到主流程跑通
- 输出“已修复问题清单 + 尚存风险”

## 论文与文档工作流

1. 先读取项目内论文规范文档（如 `paper_config.md`）。
2. 生成内容时确保：
   - 结构完整（摘要、关键词、正文、参考文献、致谢等）
   - 术语前后一致（如 Hive、Spark SQL、Prophet）
   - 引用与参考文献编号一致
3. 若用户要求，更新 `README.md`：
   - 功能用途
   - 使用方法
   - 参数说明
   - 输出说明

## 输出格式要求

- 修代码时必须提供：
  - 修改后的完整代码文件
  - 依赖文件（如 `requirements.txt`）
  - 关键修复点说明（简洁）
- 解释面向初学者，避免过度抽象术语。

## 质量门槛

- 不使用模拟数据
- 不忽略用户给定文件
- 不凭空创建不存在字段
- 结果可运行、可定位、可复核

## 附加资源

- 详细字段映射与异常模板见 [reference.md](reference.md)
- 常见请求示例见 [examples.md](examples.md)
