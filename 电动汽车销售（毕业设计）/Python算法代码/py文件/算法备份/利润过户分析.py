# =============================================================================
# 文件：利润过户分析.py
# -----------------------------------------------------------------------------
# 总体说明
#   读取利润分析、过户状态两张 Excel，通过 COLUMN_MAPPING 与 get_column_name 兼容列名；先做利润结构、
#   品牌/月度过户等描述统计与聚合，再分别构建「利润预测」与「平均过户价预测」两套树模型（GridSearchCV），
#   最后绘图并导出 CSV/报告。
#
# 核心算法
#   ① 利润侧回归：时间周期特征（年/月/季度/正余弦月）+ 收入（及可选成本、品牌编码）→ 随机森林 / 梯度提升 + GridSearchCV。
#   ② 过户价回归：过户量、品牌编码、时间特征 → 同上集成模型选优，输出 RMSE/MAE/R²。
#
# 描述分析（非学习）：profit_composition_analysis、transfer_status_analysis 中的分组聚合、分布统计、金额=量×价等。
# 说明：sklearn.cluster.KMeans 已导入，当前脚本未使用，可扩展对品牌或月份模式的无监督分群。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
import platform

warnings.filterwarnings('ignore')


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


BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\利润过户分析"
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")


for path in [IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

print("=" * 80)
print("电动汽车利润与过户状态分析系统")
print("=" * 80)


COLUMN_MAPPING = {

    'profit': ['total_profit', '利润', '净利润', '收益', 'profit'],
    'revenue': ['total_revenue', '收入', '销售额', '营收', 'revenue'],
    'total_cost': ['total_cost', '总成本', '成本', '总费用', 'total_cost'],
    'profit_margin': ['profit_margin', '利润率', '毛利率'],
    'profit_per_unit': ['profit_per_unit', '单位利润', '单车利润'],
    'brand': ['brand', '品牌', '品牌名称'],
    'stat_month': ['stat_month', '统计月份', '月份', '时间'],


    'transfer_status': ['transfer_status', '过户状态', '交易状态'],
    'transfer_count': ['transfer_count', '过户数量', '交易数量'],
    'avg_transfer_price': ['avg_transfer_price', '平均过户价格', '平均交易价格'],
    'transfer_reason_analysis': ['transfer_reason_analysis', '过户原因分析', '交易原因分析']
}


# =============================================================================
# [数据准备] get_column_name：在候选列名列表中返回 DataFrame 中第一个存在的列名，用于多源表头映射。
# =============================================================================

def get_column_name(df, target_columns):
    for col in target_columns:
        if col in df.columns:
            return col
    return None



print("\n【1/7】正在加载利润和过户数据...")


# =============================================================================
# [数据准备] load_profit_transfer_data：读两张表，按关键列 dropna / 过滤异常值，统一解析统计月份。
# =============================================================================

def load_profit_transfer_data():
    try:
        profit_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\利润分析.xlsx")
        print(f"  ✓ 利润数据: {len(profit_df)} 条记录")
        print(f"    利润数据列名: {list(profit_df.columns)}")


        profit_col = get_column_name(profit_df, COLUMN_MAPPING['profit'])
        revenue_col = get_column_name(profit_df, COLUMN_MAPPING['revenue'])
        drop_subset = [c for c in [profit_col, revenue_col] if c is not None]
        if drop_subset:
            profit_df = profit_df.dropna(subset=drop_subset)
        if revenue_col and revenue_col in profit_df.columns:
            profit_df = profit_df[profit_df[revenue_col] > 0]

    except Exception as e:
        print(f"  ✗ 加载利润数据失败: {e}")
        profit_df = pd.DataFrame()

    try:
        transfer_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\过户状态分析.xlsx")
        print(f"  ✓ 过户数据: {len(transfer_df)} 条记录")
        print(f"    过户数据列名: {list(transfer_df.columns)}")


        tc_col = get_column_name(transfer_df, COLUMN_MAPPING['transfer_count'])
        tp_col = get_column_name(transfer_df, COLUMN_MAPPING['avg_transfer_price'])
        drop_subset = [c for c in [tc_col, tp_col] if c is not None]
        if drop_subset:
            transfer_df = transfer_df.dropna(subset=drop_subset)
        if tc_col and tc_col in transfer_df.columns:
            transfer_df = transfer_df[transfer_df[tc_col] >= 0]

    except Exception as e:
        print(f"  ✗ 加载过户数据失败: {e}")
        transfer_df = pd.DataFrame()



    for df in [profit_df, transfer_df]:
        if not df.empty:

            stat_month_col = get_column_name(df, COLUMN_MAPPING['stat_month'])
            if stat_month_col:
                df[stat_month_col] = pd.to_datetime(df[stat_month_col], errors='coerce')

    return profit_df, transfer_df


profit_data, transfer_data = load_profit_transfer_data()


print("\n【2/7】执行利润构成分析...")


# =============================================================================
# [描述分析] profit_composition_analysis：按品牌/时间聚合利润、收入、利润率等，输出 insights 字典供图表与下游模型参考。
# =============================================================================

def profit_composition_analysis(df):
    if df.empty:
        print("  ✗ 利润数据为空，跳过利润分析")
        return {}


    profit_col = get_column_name(df, COLUMN_MAPPING['profit'])
    revenue_col = get_column_name(df, COLUMN_MAPPING['revenue'])
    total_cost_col = get_column_name(df, COLUMN_MAPPING['total_cost'])
    profit_margin_col = get_column_name(df, COLUMN_MAPPING['profit_margin'])
    profit_per_unit_col = get_column_name(df, COLUMN_MAPPING['profit_per_unit'])
    brand_col = get_column_name(df, COLUMN_MAPPING['brand'])
    stat_month_col = get_column_name(df, COLUMN_MAPPING['stat_month'])


    if not all([profit_col, revenue_col]):
        missing_cols = []
        if not profit_col:
            missing_cols.append('总利润(total_profit)')
        if not revenue_col:
            missing_cols.append('总收入(total_revenue)')
        print(f"  ✗ 缺少关键列: {', '.join(missing_cols)}")
        return {}


    if profit_margin_col is None:
        df['profit_margin'] = (df[profit_col] / df[revenue_col].replace(0, np.nan)) * 100
    else:
        df['profit_margin'] = df[profit_margin_col]

    if total_cost_col:
        df['cost_ratio'] = (df[total_cost_col] / df[revenue_col].replace(0, np.nan)) * 100
        df['cost_profit_ratio'] = df[total_cost_col] / df[profit_col].replace(0, np.nan)
    else:
        df['cost_ratio'] = np.nan
        df['cost_profit_ratio'] = np.nan


    brand_profit = pd.DataFrame()
    if brand_col:
        agg_dict = {
            profit_col: ['sum', 'mean', 'max', 'min'],
            revenue_col: ['sum', 'mean'],
            'profit_margin': ['mean', 'std'],
        }

        if total_cost_col:
            agg_dict[total_cost_col] = ['sum', 'mean']
            agg_dict['cost_ratio'] = 'mean'
        if profit_per_unit_col:
            agg_dict[profit_per_unit_col] = 'mean'

        brand_profit = df.groupby(brand_col).agg(agg_dict).round(2)



        brand_profit.columns = ['_'.join(col).strip() for col in brand_profit.columns.values]


        column_mapping = {
            f'{profit_col}_sum': '总利润',
            f'{profit_col}_mean': '平均利润',
            f'{profit_col}_max': '最高利润',
            f'{profit_col}_min': '最低利润',
            f'{revenue_col}_sum': '总收入',
            f'{revenue_col}_mean': '平均收入',
            'profit_margin_mean': '平均利润率',
            'profit_margin_std': '利润率标准差'
        }


        if total_cost_col:
            column_mapping[f'{total_cost_col}_sum'] = '总成本'
            column_mapping[f'{total_cost_col}_mean'] = '平均成本'
            column_mapping['cost_ratio_mean'] = '平均成本率'


        if profit_per_unit_col:
            column_mapping[f'{profit_per_unit_col}_mean'] = '平均单位利润'


        brand_profit = brand_profit.rename(columns=column_mapping)
        brand_profit = brand_profit.sort_values('总利润', ascending=False)
    else:
        brand_profit = pd.DataFrame()
        print("  ⚠ 未找到品牌列，跳过品牌利润分析")


    monthly_profit = pd.DataFrame()
    if stat_month_col and not df[stat_month_col].isnull().all():
        df['year'] = df[stat_month_col].dt.year
        df['month'] = df[stat_month_col].dt.month

        monthly_profit = df.groupby(['year', 'month']).agg({
            profit_col: 'sum',
            revenue_col: 'sum',
            'profit_margin': 'mean'
        }).reset_index()
        monthly_profit.columns = ['年份', '月份', '月度总利润', '月度总收入', '月度平均利润率']


    high_profit_threshold = df['profit_margin'].quantile(0.75)
    low_profit_threshold = df['profit_margin'].quantile(0.25)

    high_profit_records = df[df['profit_margin'] >= high_profit_threshold]
    low_profit_records = df[df['profit_margin'] <= low_profit_threshold]

    print(f"  ✓ 数据量: {len(df)} 条有效记录")
    print(f"  ✓ 利润率范围: {df['profit_margin'].min():.2f}% - {df['profit_margin'].max():.2f}%")
    print(f"  ✓ 平均利润率: {df['profit_margin'].mean():.2f}%")
    print(f"  ✓ 总利润: ¥{df[profit_col].sum():,.2f}")
    print(f"  ✓ 总收入: ¥{df[revenue_col].sum():,.2f}")

    if not brand_profit.empty:
        top_brand = brand_profit.index[0]
        top_profit = brand_profit.iloc[0]['总利润']
        print(f"  ✓ 最高利润品牌: {top_brand} (¥{top_profit:,.2f})")

    insights = {
        'brand_profit': brand_profit,
        'monthly_profit': monthly_profit,
        'high_profit_threshold': high_profit_threshold,
        'low_profit_threshold': low_profit_threshold,
        'high_profit_records': high_profit_records,
        'low_profit_records': low_profit_records,
        'overall_metrics': {
            'total_profit': df[profit_col].sum(),
            'total_revenue': df[revenue_col].sum(),
            'avg_margin': df['profit_margin'].mean(),
            'margin_std': df['profit_margin'].std(),
            'record_count': len(df)
        }
    }

    return insights


profit_insights = profit_composition_analysis(profit_data)


print("\n【3/7】执行过户状态分析...")


# =============================================================================
# [描述分析] transfer_status_analysis：过户量×均价得到金额，按品牌/状态/原因/年月聚合，形成过户侧 insights。
# =============================================================================

def transfer_status_analysis(df):
    if df.empty:
        print("  ✗ 过户数据为空，跳过过户分析")
        return {}


    transfer_status_col = get_column_name(df, COLUMN_MAPPING['transfer_status'])
    transfer_count_col = get_column_name(df, COLUMN_MAPPING['transfer_count'])
    avg_transfer_price_col = get_column_name(df, COLUMN_MAPPING['avg_transfer_price'])
    transfer_reason_col = get_column_name(df, COLUMN_MAPPING['transfer_reason_analysis'])
    brand_col = get_column_name(df, COLUMN_MAPPING['brand'])
    stat_month_col = get_column_name(df, COLUMN_MAPPING['stat_month'])


    required_cols = [transfer_count_col, avg_transfer_price_col]
    if None in required_cols:
        missing = []
        if not transfer_count_col:
            missing.append('过户数量(transfer_count)')
        if not avg_transfer_price_col:
            missing.append('平均过户价格(avg_transfer_price)')
        print(f"  ✗ 缺少关键列: {', '.join(missing)}")
        return {}


    df['transfer_amount'] = df[transfer_count_col] * df[avg_transfer_price_col]


    brand_transfer = pd.DataFrame()
    if brand_col:
        agg_dict = {
            transfer_count_col: ['sum', 'mean'],
            avg_transfer_price_col: 'mean',
            'transfer_amount': 'sum'
        }

        brand_transfer = df.groupby(brand_col).agg(agg_dict).round(2)


        brand_transfer.columns = ['_'.join(col).strip() for col in brand_transfer.columns.values]


        column_mapping = {
            f'{transfer_count_col}_sum': '总过户数量',
            f'{transfer_count_col}_mean': '平均月度过户量',
            f'{avg_transfer_price_col}_mean': '平均过户价格',
            'transfer_amount_sum': '过户总金额'
        }

        brand_transfer = brand_transfer.rename(columns=column_mapping)
        brand_transfer = brand_transfer.sort_values('总过户数量', ascending=False)


    transfer_status_dist = pd.Series()
    if transfer_status_col:
        transfer_status_dist = df[transfer_status_col].value_counts()


    transfer_reasons = pd.Series()
    if transfer_reason_col:
        transfer_reasons = df[transfer_reason_col].value_counts()


    monthly_transfers = pd.DataFrame()
    if stat_month_col and not df[stat_month_col].isnull().all():
        df['year'] = df[stat_month_col].dt.year
        df['month'] = df[stat_month_col].dt.month

        monthly_transfers = df.groupby(['year', 'month']).agg({
            transfer_count_col: 'sum',
            avg_transfer_price_col: 'mean',
            'transfer_amount': 'sum'
        }).reset_index()
        monthly_transfers.columns = ['年份', '月份', '月度过户总量', '月度平均价格', '月度过户总金额']


    total_transfers = df[transfer_count_col].sum()
    avg_monthly_transfers = df[transfer_count_col].mean()
    avg_transfer_price = df[avg_transfer_price_col].mean()
    total_transfer_amount = df['transfer_amount'].sum()

    print(f"  ✓ 总过户记录数: {len(df):,} 条")
    print(f"  ✓ 累计过户数量: {total_transfers:,.0f} 辆")
    print(f"  ✓ 平均月度过户量: {avg_monthly_transfers:.1f} 辆/月")
    print(f"  ✓ 平均过户价格: ¥{avg_transfer_price:,.2f}")
    print(f"  ✓ 过户交易总金额: ¥{total_transfer_amount:,.2f}")

    insights = {
        'brand_transfer': brand_transfer,
        'transfer_status_dist': transfer_status_dist,
        'transfer_reasons': transfer_reasons,
        'monthly_transfers': monthly_transfers,
        'overall_metrics': {
            'total_records': len(df),
            'total_transfer_count': total_transfers,
            'avg_monthly_transfers': avg_monthly_transfers,
            'avg_transfer_price': avg_transfer_price,
            'total_transfer_amount': total_transfer_amount
        }
    }

    return insights


transfer_insights = transfer_status_analysis(transfer_data)


print("\n【4/7】构建利润预测模型...")


# =============================================================================
# [算法 ①] 利润回归（build_profit_prediction_model）：RandomForestRegressor / GradientBoostingRegressor + GridSearchCV
# 用处：以时间编码与经营指标预测利润列；StandardScaler 特征缩放；时间序优先 80/20 切分，测试集过小时改 random split。
# =============================================================================

def build_profit_prediction_model(df, insights=None):
    if df.empty or not insights:
        print("  ✗ 利润数据不足，跳过利润预测模型构建")
        return None, {}, None, None, pd.DataFrame()


    profit_col = get_column_name(df, COLUMN_MAPPING['profit'])
    revenue_col = get_column_name(df, COLUMN_MAPPING['revenue'])
    total_cost_col = get_column_name(df, COLUMN_MAPPING['total_cost'])
    stat_month_col = get_column_name(df, COLUMN_MAPPING['stat_month'])
    brand_col = get_column_name(df, COLUMN_MAPPING['brand'])

    if not all([profit_col, revenue_col, stat_month_col]):
        print("  ✗ 缺少构建模型所需的关键列")
        return None, {}, None, None, pd.DataFrame()


    df[stat_month_col] = pd.to_datetime(df[stat_month_col], errors='coerce')
    df = df.dropna(subset=[stat_month_col])

    if len(df) < 20:
        print("  ✗ 数据量不足（少于20条），无法构建预测模型")
        return None, {}, None, None, pd.DataFrame()

    df['year'] = df[stat_month_col].dt.year
    df['month'] = df[stat_month_col].dt.month
    df['quarter'] = df[stat_month_col].dt.quarter


    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


    le_brand = LabelEncoder()
    if brand_col and df[brand_col].nunique() > 1:
        df['brand_encoded'] = le_brand.fit_transform(df[brand_col].astype(str))
    else:
        df['brand_encoded'] = 0
        le_brand = None


    feature_cols = ['year', 'month', 'quarter', 'month_sin', 'month_cos',
                    revenue_col, 'brand_encoded']


    if total_cost_col and total_cost_col in df.columns:
        feature_cols.append(total_cost_col)


    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].fillna(0)
    y = df[profit_col]


    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_test) < 5:
        print("  ✗ 测试集数据量不足，调整为随机划分")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }


    param_grids = {
        'RandomForest': {
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    best_score = float('inf')
    best_model = None
    best_model_name = ""
    results = {}

    for model_name, model in models.items():
        print(f"  训练 {model_name}...")

        try:

            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=3,
                scoring='neg_mean_squared_error', n_jobs=1
            )
            grid_search.fit(X_train_scaled, y_train)


            y_pred = grid_search.best_estimator_.predict(X_test_scaled)


            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"    最佳参数: {grid_search.best_params_}")
            print(f"    测试集 RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")


            results[model_name] = {
                'model': grid_search.best_estimator_,
                'predictions': y_pred,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'best_params': grid_search.best_params_
            }

            if mse < best_score:
                best_score = mse
                best_model = grid_search.best_estimator_
                best_model_name = model_name
        except Exception as e:
            print(f"    训练 {model_name} 失败: {e}")
            continue

    if best_model:
        print(f"  ✓ 最佳模型: {best_model_name}")
        print(f"  ✓ 最佳测试集 RMSE: {np.sqrt(best_score):.2f}")
    else:
        print("  ✗ 所有模型训练失败")
        return None, {}, scaler, le_brand, pd.DataFrame()


    feature_importance = pd.DataFrame()
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            '特征名称': feature_cols,
            '重要性': best_model.feature_importances_
        }).sort_values('重要性', ascending=False)

        print("\n  特征重要性排名:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['特征名称']}: {row['重要性']:.3f}")

    return best_model, results, scaler, le_brand, feature_importance


profit_model, profit_results, profit_scaler, brand_encoder, profit_feature_importance = build_profit_prediction_model(
    profit_data, insights=profit_insights)


print("\n【5/7】构建过户价格预测模型...")


# =============================================================================
# [算法 ②] 平均过户价回归（build_transfer_price_model）：树模型 + GridSearchCV
# 用处：用过户量、品牌、年月季节等解释均价水平，评估拟合误差，用于辅助监测异常均价或压力测试（非定价因果模型）。
# =============================================================================

def build_transfer_price_model(df, insights=None):
    if df.empty or not insights:
        print("  ✗ 过户数据不足，跳过过户价格预测模型构建")
        return None, {}, None, None, None


    transfer_count_col = get_column_name(df, COLUMN_MAPPING['transfer_count'])
    avg_transfer_price_col = get_column_name(df, COLUMN_MAPPING['avg_transfer_price'])
    brand_col = get_column_name(df, COLUMN_MAPPING['brand'])
    stat_month_col = get_column_name(df, COLUMN_MAPPING['stat_month'])

    required_cols = [transfer_count_col, avg_transfer_price_col]
    if None in required_cols:
        print("  ✗ 缺少构建模型所需的关键列")
        return None, {}, None, None, None


    df = df.dropna(subset=[transfer_count_col, avg_transfer_price_col])
    df = df[df[avg_transfer_price_col] > 0]

    if len(df) < 20:
        print("  ✗ 有效数据量不足（少于20条），无法构建模型")
        return None, {}, None, None, None


    if stat_month_col:
        df[stat_month_col] = pd.to_datetime(df[stat_month_col], errors='coerce')
        df = df.dropna(subset=[stat_month_col])
        df['year'] = df[stat_month_col].dt.year
        df['month'] = df[stat_month_col].dt.month
        df['quarter'] = df[stat_month_col].dt.quarter


    le_brand = LabelEncoder()
    if brand_col and df[brand_col].nunique() > 1:
        df['brand_encoded'] = le_brand.fit_transform(df[brand_col].astype(str))
    else:
        df['brand_encoded'] = 0
        le_brand = None


    feature_cols = [transfer_count_col, 'brand_encoded']
    if stat_month_col in df.columns:
        feature_cols.extend(['year', 'month', 'quarter'])


    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].fillna(0)
    y = df[avg_transfer_price_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }


    param_grids = {
        'RandomForest': {
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    best_score = float('inf')
    best_model = None
    best_model_name = ""
    final_mae = 0
    final_r2 = 0

    for model_name, model in models.items():
        print(f"  训练 {model_name}...")

        try:

            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=3,
                scoring='neg_mean_squared_error', n_jobs=1
            )
            grid_search.fit(X_train_scaled, y_train)


            y_pred = grid_search.best_estimator_.predict(X_test_scaled)


            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"    测试集 RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

            if mse < best_score:
                best_score = mse
                best_model = grid_search.best_estimator_
                best_model_name = model_name
                final_mae = mae
                final_r2 = r2
        except Exception as e:
            print(f"    训练 {model_name} 失败: {e}")
            continue

    if best_model:
        print(f"  ✓ 最佳模型: {best_model_name}")
        print(f"  ✓ 最佳测试集 RMSE: {np.sqrt(best_score):.2f}")
    else:
        print("  ✗ 所有模型训练失败")
        return None, {}, None, None, None

    return best_model, {
        'rmse': np.sqrt(best_score),
        'mae': final_mae,
        'r2': final_r2,
        'model_name': best_model_name
    }, scaler, le_brand, None


transfer_model, transfer_results, transfer_scaler, transfer_brand_encoder, transfer_reason_encoder = build_transfer_price_model(
    transfer_data, insights=transfer_insights)


print("\n【6/7】生成利润与过户分析可视化图表...")


# =============================================================================
# [可视化] plot_profit_transfer_analysis：利润条形/趋势、过户分布、预测对比等子图，依赖 insights 与训练好的模型。
# =============================================================================

def plot_profit_transfer_analysis():

    if profit_insights and not profit_insights.get('brand_profit', pd.DataFrame()).empty:

        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        elif system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        fig.suptitle('利润分析图表', fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)


        brand_profit = profit_insights['brand_profit']
        top_10_brands = brand_profit.head(10)
        bars = axes[0, 0].barh(range(len(top_10_brands)), top_10_brands['总利润'],
                               color=plt.cm.viridis(np.linspace(0, 1, len(top_10_brands))))
        axes[0, 0].set_yticks(range(len(top_10_brands)))
        axes[0, 0].set_yticklabels(top_10_brands.index, fontfamily=CHINESE_FONT)
        axes[0, 0].set_xlabel('总利润 (¥)', fontfamily=CHINESE_FONT)
        axes[0, 0].set_title('品牌利润排名（前10）', fontfamily=CHINESE_FONT)
        axes[0, 0].invert_yaxis()


        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 0].text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'¥{width / 1e6:.1f}M', ha='left', va='center', fontsize=9, fontfamily=CHINESE_FONT)


        axes[0, 1].hist(profit_data['profit_margin'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(profit_data['profit_margin'].mean(), color='red', linestyle='--',
                           label=f'均值: {profit_data["profit_margin"].mean():.2f}%')
        axes[0, 1].axvline(profit_data['profit_margin'].quantile(0.75), color='green', linestyle='--',
                           label=f'高利润阈值: {profit_data["profit_margin"].quantile(0.75):.2f}%')
        axes[0, 1].set_xlabel('利润率 (%)', fontfamily=CHINESE_FONT)
        axes[0, 1].set_ylabel('频数', fontfamily=CHINESE_FONT)
        axes[0, 1].set_title('利润率分布', fontfamily=CHINESE_FONT)
        axes[0, 1].legend(prop={'family': CHINESE_FONT})
        axes[0, 1].grid(True, alpha=0.3)


        if 'cost_ratio' in profit_data.columns and not profit_data['cost_ratio'].isna().all():
            axes[1, 0].scatter(profit_data['cost_ratio'], profit_data['profit_margin'],
                               alpha=0.6, c='orange', edgecolors='black', s=50)
            axes[1, 0].set_xlabel('成本率 (%)', fontfamily=CHINESE_FONT)
            axes[1, 0].set_ylabel('利润率 (%)', fontfamily=CHINESE_FONT)
            axes[1, 0].set_title('成本率 vs 利润率', fontfamily=CHINESE_FONT)
            axes[1, 0].grid(True, alpha=0.3)
        else:

            axes[1, 0].bar(range(len(top_10_brands)), top_10_brands['平均利润率'],
                           color=plt.cm.plasma(np.linspace(0, 1, len(top_10_brands))))
            axes[1, 0].set_xticks(range(len(top_10_brands)))
            axes[1, 0].set_xticklabels(top_10_brands.index, rotation=45, ha='right', fontfamily=CHINESE_FONT)
            axes[1, 0].set_ylabel('平均利润率 (%)', fontfamily=CHINESE_FONT)
            axes[1, 0].set_title('品牌平均利润率对比', fontfamily=CHINESE_FONT)
            axes[1, 0].grid(True, alpha=0.3)


        monthly_profit = profit_insights['monthly_profit']
        if not monthly_profit.empty:

            if '年份' in monthly_profit.columns and '月份' in monthly_profit.columns:

                monthly_profit['date_str'] = monthly_profit['年份'].astype(str) + '-' + monthly_profit['月份'].astype(
                    str).str.zfill(2) + '-01'
                dates = pd.to_datetime(monthly_profit['date_str'], format='%Y-%m-%d')

                axes[1, 1].plot(dates, monthly_profit['月度总利润'], marker='o', linewidth=2, markersize=4,
                                color='darkblue')
                axes[1, 1].set_xlabel('日期', fontfamily=CHINESE_FONT)
                axes[1, 1].set_ylabel('月度总利润 (¥)', fontfamily=CHINESE_FONT)
                axes[1, 1].set_title('月度利润趋势', fontfamily=CHINESE_FONT)
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '月度数据\n格式错误', ha='center', va='center',
                                transform=axes[1, 1].transAxes, fontsize=14, fontfamily=CHINESE_FONT)
                axes[1, 1].set_title('月度利润趋势', fontfamily=CHINESE_FONT)
        else:
            axes[1, 1].text(0.5, 0.5, '月度数据\n不可用', ha='center', va='center',
                            transform=axes[1, 1].transAxes, fontsize=14, fontfamily=CHINESE_FONT)
            axes[1, 1].set_title('月度利润趋势', fontfamily=CHINESE_FONT)

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\利润分析图表.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 利润分析图表已保存到图片分析文件夹")
        plt.close()
    else:
        print("  ⚠ 利润数据不足，跳过利润图表生成")


    if transfer_insights and not transfer_insights.get('brand_transfer', pd.DataFrame()).empty:

        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        elif system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('过户状态分析图表', fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)


        brand_transfer = transfer_insights['brand_transfer']
        top_10_brands = brand_transfer.head(10)
        bars = axes[0, 0].barh(range(len(top_10_brands)), top_10_brands['总过户数量'],
                               color=plt.cm.viridis(np.linspace(0, 1, len(top_10_brands))))
        axes[0, 0].set_yticks(range(len(top_10_brands)))
        axes[0, 0].set_yticklabels(top_10_brands.index, fontfamily=CHINESE_FONT)
        axes[0, 0].set_xlabel('总过户数量 (辆)', fontfamily=CHINESE_FONT)
        axes[0, 0].set_title('品牌过户数量排名（前10）', fontfamily=CHINESE_FONT)
        axes[0, 0].invert_yaxis()


        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 0].text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{int(width):,}', ha='left', va='center', fontsize=9, fontfamily=CHINESE_FONT)


        transfer_status = transfer_insights['transfer_status_dist']
        if not transfer_status.empty:
            axes[0, 1].pie(transfer_status.values, labels=transfer_status.index,
                           autopct='%1.1f%%', startangle=90, textprops={'fontfamily': CHINESE_FONT})
            axes[0, 1].set_title('过户状态分布', fontfamily=CHINESE_FONT)
        else:

            axes[0, 1].bar(range(len(top_10_brands)), top_10_brands['平均过户价格'] / 1e4,
                           color=plt.cm.plasma(np.linspace(0, 1, len(top_10_brands))))
            axes[0, 1].set_xticks(range(len(top_10_brands)))
            axes[0, 1].set_xticklabels(top_10_brands.index, rotation=45, ha='right', fontfamily=CHINESE_FONT)
            axes[0, 1].set_ylabel('平均过户价格 (万元)', fontfamily=CHINESE_FONT)
            axes[0, 1].set_title('品牌平均过户价格对比', fontfamily=CHINESE_FONT)


        transfer_reasons = transfer_insights['transfer_reasons']
        if not transfer_reasons.empty:
            axes[1, 0].bar(transfer_reasons.index, transfer_reasons.values,
                           color='coral', alpha=0.8, edgecolor='black')
            axes[1, 0].set_xlabel('过户原因', fontfamily=CHINESE_FONT)
            axes[1, 0].set_ylabel('数量', fontfamily=CHINESE_FONT)
            axes[1, 0].set_title('过户原因分布', fontfamily=CHINESE_FONT)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '过户原因数据\n不可用', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=14, fontfamily=CHINESE_FONT)
            axes[1, 0].set_title('过户原因分布', fontfamily=CHINESE_FONT)


        monthly_transfers = transfer_insights['monthly_transfers']
        if not monthly_transfers.empty:
            if '年份' in monthly_transfers.columns and '月份' in monthly_transfers.columns:

                monthly_transfers['date_str'] = monthly_transfers['年份'].astype(str) + '-' + monthly_transfers[
                    '月份'].astype(str).str.zfill(2) + '-01'
                dates = pd.to_datetime(monthly_transfers['date_str'], format='%Y-%m-%d')

                axes[1, 1].plot(dates, monthly_transfers['月度过户总量'],
                                marker='o', linewidth=2, markersize=4, color='darkred')
                axes[1, 1].set_xlabel('日期', fontfamily=CHINESE_FONT)
                axes[1, 1].set_ylabel('月度过户总量 (辆)', fontfamily=CHINESE_FONT)
                axes[1, 1].set_title('月度过户量趋势', fontfamily=CHINESE_FONT)
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '月度数据\n格式错误', ha='center', va='center',
                                transform=axes[1, 1].transAxes, fontsize=14, fontfamily=CHINESE_FONT)
                axes[1, 1].set_title('月度过户量趋势', fontfamily=CHINESE_FONT)
        else:
            axes[1, 1].text(0.5, 0.5, '月度数据\n不可用', ha='center', va='center',
                            transform=axes[1, 1].transAxes, fontsize=14, fontfamily=CHINESE_FONT)
            axes[1, 1].set_title('月度过户量趋势', fontfamily=CHINESE_FONT)

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\过户状态分析图表.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 过户状态分析图表已保存到图片分析文件夹")
        plt.close()
    else:
        print("  ⚠ 过户数据不足，跳过过户图表生成")


plot_profit_transfer_analysis()


print("\n【7/7】生成利润与过户分析报告...")


# =============================================================================
# [输出] generate_profit_transfer_report：汇总利润与过户指标、模型误差与业务要点到 TXT，并配合其他落盘逻辑使用。
# =============================================================================

def generate_profit_transfer_report():
    report = []
    report.append("=" * 80)
    report.append("电动汽车利润与过户状态分析报告")
    report.append("=" * 80)
    report.append("")


    report.append("【利润分析】")
    if profit_insights:
        overall = profit_insights['overall_metrics']
        report.append(f"  • 有效分析记录: {overall['record_count']:,} 条")
        report.append(f"  • 总利润: ¥{overall['total_profit']:,.2f}")
        report.append(f"  • 总收入: ¥{overall['total_revenue']:,.2f}")
        report.append(f"  • 平均利润率: {overall['avg_margin']:.2f}%")
        report.append(f"  • 利润率标准差: {overall['margin_std']:.2f}%")
        report.append("")


        brand_profit = profit_insights['brand_profit']
        if not brand_profit.empty:
            report.append("  利润排名前三的品牌:")
            for i, (brand, row) in enumerate(brand_profit.head(3).iterrows()):
                report.append(f"    {i + 1}. {brand}")
                report.append(f"       总利润: ¥{row['总利润']:,.2f}")
                report.append(f"       平均利润率: {row['平均利润率']:.2f}%")
                if '平均成本率' in row and not pd.isna(row['平均成本率']):
                    report.append(f"       平均成本率: {row['平均成本率']:.2f}%")
            report.append("")
    else:
        report.append("  • 利润数据不足，无法生成详细分析")
        report.append("")


    report.append("【过户状态分析】")
    if transfer_insights:
        transfer_overall = transfer_insights['overall_metrics']
        report.append(f"  • 有效分析记录: {transfer_overall['total_records']:,} 条")
        report.append(f"  • 累计过户数量: {transfer_overall['total_transfer_count']:,.0f} 辆")
        report.append(f"  • 平均月度过户量: {transfer_overall['avg_monthly_transfers']:.1f} 辆/月")
        report.append(f"  • 平均过户价格: ¥{transfer_overall['avg_transfer_price']:,.2f}")
        report.append(f"  • 过户交易总金额: ¥{transfer_overall['total_transfer_amount']:,.2f}")
        report.append("")


        brand_transfer = transfer_insights['brand_transfer']
        if not brand_transfer.empty:
            report.append("  过户数量排名前三的品牌:")
            for i, (brand, row) in enumerate(brand_transfer.head(3).iterrows()):
                report.append(f"    {i + 1}. {brand}: {row['总过户数量']:,.0f} 辆")
                report.append(f"       平均过户价格: ¥{row['平均过户价格']:,.2f}")
                report.append(f"       过户总金额: ¥{row['过户总金额']:,.2f}")
            report.append("")


        transfer_reasons = transfer_insights['transfer_reasons']
        if not transfer_reasons.empty:
            report.append("  主要过户原因:")
            for reason, count in transfer_reasons.head(3).items():
                pct = count / transfer_reasons.sum() * 100
                report.append(f"    • {reason}: {count} 条 ({pct:.1f}%)")
            report.append("")
    else:
        report.append("  • 过户数据不足，无法生成详细分析")
        report.append("")


    report.append("【模型性能】")
    if profit_results:
        best_profit_model = min(profit_results.keys(), key=lambda x: profit_results[x]['rmse'])
        report.append(f"  利润预测模型:")
        report.append(f"    最佳模型: {best_profit_model}")
        report.append(f"    测试集 RMSE: {profit_results[best_profit_model]['rmse']:.2f}")
        report.append(f"    测试集 R²: {profit_results[best_profit_model]['r2']:.3f}")
    else:
        report.append(f"  利润预测模型: 未构建（数据不足或训练失败）")

    if transfer_results and isinstance(transfer_results, dict) and 'rmse' in transfer_results:
        report.append("\n  过户价格预测模型:")
        report.append(f"    最佳模型: {transfer_results.get('model_name', '未知')}")
        report.append(f"    测试集 RMSE: {transfer_results['rmse']:.2f}")
        report.append(f"    测试集 R²: {transfer_results['r2']:.3f}")
    else:
        report.append("\n  过户价格预测模型: 未构建（数据不足或训练失败）")
    report.append("")


    report.append("【业务建议】")
    report.append("  1. 利润优化: 重点关注高利润品牌和产品线，提升整体利润率")
    report.append("  2. 成本控制: 分析成本结构，优化高成本环节的费用支出")
    report.append("  3. 库存管理: 根据过户活跃度和品牌热度调整库存策略")
    report.append("  4. 二手车定价: 利用过户价格数据优化二手车定价模型")
    report.append("  5. 市场策略: 针对高过户量品牌加大市场推广力度")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)


report_content = generate_profit_transfer_report()
print(report_content)


try:
    with open(f"{DOCUMENT_SAVE_PATH}\\利润过户分析报告.txt", 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"\n✓ 分析报告已保存到文档分析文件夹: {DOCUMENT_SAVE_PATH}\\利润过户分析报告.txt")
except Exception as e:
    print(f"\n✗ 保存报告失败: {e}")


print("\n【附加】保存处理后的数据...")
try:
    if profit_insights and not profit_insights.get('brand_profit', pd.DataFrame()).empty:
        brand_profit_df = profit_insights['brand_profit'].copy()
        brand_profit_df.index.name = "品牌"
        brand_profit_df.to_csv(f"{DOCUMENT_SAVE_PATH}\\品牌利润分析.csv", encoding='utf-8-sig')
        print("  ✓ 品牌利润分析数据已保存到文档分析文件夹")

    if transfer_insights and not transfer_insights.get('brand_transfer', pd.DataFrame()).empty:
        brand_transfer_df = transfer_insights['brand_transfer'].copy()
        brand_transfer_df.index.name = "品牌"
        brand_transfer_df.to_csv(f"{DOCUMENT_SAVE_PATH}\\品牌过户分析.csv", encoding='utf-8-sig')
        print("  ✓ 品牌过户分析数据已保存到文档分析文件夹")

    if not profit_feature_importance.empty:
        profit_feature_importance.to_csv(f"{DOCUMENT_SAVE_PATH}\\利润预测特征重要性.csv", index=False, encoding='utf-8-sig')
        print("  ✓ 利润预测特征重要性数据已保存到文档分析文件夹")

    print(f"  ✓ 利润过户分析结果已保存到: ")
    print(f"     图片文件: {IMAGE_SAVE_PATH}")
    print(f"     文档文件: {DOCUMENT_SAVE_PATH}")
except Exception as e:
    print(f"  ✗ 保存数据失败: {e}")


print("\n" + "=" * 80)
print("利润与过户状态分析模块执行完成！")
print("=" * 80)