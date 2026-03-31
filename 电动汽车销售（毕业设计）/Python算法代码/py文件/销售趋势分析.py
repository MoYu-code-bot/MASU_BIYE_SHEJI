"""
电动汽车销售趋势分析模块
整合月度、季度、年度、地区销售数据，进行时间序列预测和趋势分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import os
import tempfile
import shutil
import platform

# ========== 关键修复：解决中文路径/用户名导致的编码问题 ==========
# 设置临时文件夹为纯英文路径
TEMP_DIR = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\temp"
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR

# 创建临时文件夹
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# 临时修改tempfile的临时目录
tempfile.tempdir = TEMP_DIR

# 尝试导入Prophet，失败则使用备选方案
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("  ⚠ Prophet库未安装，将使用ARIMA模型替代")

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 关键修改1：拆分图片和文档存储路径 ==========
# 基础存储路径
BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\销售趋势分析"
# 图片存储路径
IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")
# 文档存储路径
DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")

# 创建文件夹（如果不存在）
for path in [IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

print("=" * 80)
print("电动汽车销售趋势分析系统")
print("=" * 80)

# 1. 数据加载和预处理
print("\n【1/8】正在加载销售趋势数据...")


def load_sales_trend_data():
    """
    加载并整合月度、季度、年度、地区销售数据
    返回合并后的DataFrame
    """
    # ========== 关键修改2：更新Excel文件读取路径 ==========
    # 加载销售数据（修改后的路径）
    monthly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\电动汽车销售指标表\月度销售趋势.xlsx")
    quarterly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\电动汽车销售指标表\季度销售趋势.xlsx")
    yearly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\电动汽车销售指标表\年度销售趋势.xlsx")
    regional_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售\电动汽车销售指标表\地区销售趋势.xlsx")

    print(f"  ✓ 月度数据: {len(monthly_df)} 条记录")
    print(f"  ✓ 季度数据: {len(quarterly_df)} 条记录")
    print(f"  ✓ 年度数据: {len(yearly_df)} 条记录")
    print(f"  ✓ 地区数据: {len(regional_df)} 条记录")

    # 数据清洗和预处理：统一转换为中文列名（避免后续列名混乱）
    # 月度数据列名映射（英文→中文）
    monthly_cn_map = {
        'stat_month': '统计月份',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'avg_sale_price': '平均销售价格',
        'total_production_volume': '总生产量'
    }
    monthly_df = monthly_df.rename(columns={k: v for k, v in monthly_cn_map.items() if k in monthly_df.columns})

    # 季度数据列名映射
    quarterly_cn_map = {
        'stat_quarter': '统计季度',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'avg_sale_price': '平均销售价格'
    }
    quarterly_df = quarterly_df.rename(columns={k: v for k, v in quarterly_cn_map.items() if k in quarterly_df.columns})

    # 年度数据列名映射
    yearly_cn_map = {
        'stat_year': '统计年度',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'market_share': '市场份额'
    }
    yearly_df = yearly_df.rename(columns={k: v for k, v in yearly_cn_map.items() if k in yearly_df.columns})

    # 地区数据列名映射
    regional_cn_map = {
        'origin': '地区',
        'stat_month': '统计月份',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'market_share': '市场份额',
        'sales_rank': '销售排名'
    }
    regional_df = regional_df.rename(columns={k: v for k, v in regional_cn_map.items() if k in regional_df.columns})

    # 转换日期格式
    for df in [monthly_df, quarterly_df, yearly_df, regional_df]:
        if '统计月份' in df.columns:
            df['统计月份'] = pd.to_datetime(df['统计月份'], errors='coerce')
        elif '统计季度' in df.columns:
            df['统计季度'] = pd.to_datetime(df['统计季度'], errors='coerce')
        elif '统计年度' in df.columns:
            df['统计年度'] = pd.to_datetime(df['统计年度'], format='%Y', errors='coerce')

    return monthly_df, quarterly_df, yearly_df, regional_df


# 加载数据（此时返回的DataFrame已全部为中文列名）
monthly_data, quarterly_data, yearly_data, regional_data = load_sales_trend_data()

# 2. 时间序列分解分析
print("\n【2/8】执行时间序列分解分析...")


def time_series_decomposition_analysis(df, date_col, value_col, period=12):
    """
    对时间序列进行季节性分解分析（使用中文列名）

    参数:
        df: DataFrame - 输入数据（中文列名）
        date_col: str - 日期列名（中文，如"统计月份"）
        value_col: str - 数值列名（中文，如"总销售量"）
        period: int - 季节周期

    返回:
        decomposition: 分解结果
        df_decomposed: 包含分解成分的DataFrame（中文列名）
    """
    # 按日期排序
    df_sorted = df.sort_values(date_col).copy()

    # 设置日期为索引
    df_sorted.set_index(date_col, inplace=True)

    # 重采样为月度数据（如果原始数据不是月度）
    if not isinstance(df_sorted.index, pd.DatetimeIndex):
        df_sorted.index = pd.to_datetime(df_sorted.index)

    # 对数值列进行重采样
    ts_data = df_sorted[value_col].resample('M').sum()

    # 填充缺失值
    ts_data = ts_data.fillna(ts_data.mean())

    # 执行季节性分解
    try:
        decomposition = seasonal_decompose(ts_data, model='additive', period=period, extrapolate_trend='freq')

        print(f"  ✓ 时间序列分解完成")
        print(f"  ✓ 趋势成分范围: {decomposition.trend.min():.2f} 到 {decomposition.trend.max():.2f}")
        print(f"  ✓ 季节成分方差: {decomposition.seasonal.var():.2f}")
        print(f"  ✓ 残差标准差: {decomposition.resid.std():.2f}")

        # 创建包含所有成分的DataFrame（中文列名）
        df_decomposed = pd.DataFrame({
            '原始数据': ts_data,
            '趋势': decomposition.trend,
            '季节性': decomposition.seasonal,
            '残差': decomposition.resid
        })

        return decomposition, df_decomposed

    except Exception as e:
        print(f"  ⚠ 分解失败: {e}")
        return None, pd.DataFrame()


# 对月度销售数据进行分解（使用中文列名调用）
monthly_decomp, monthly_decomposed = time_series_decomposition_analysis(
    monthly_data, '统计月份', '总销售量', period=12
)

# 3. 时间序列预测（修复版：确保输出中文列名 + 删除无用列）
print("\n【3/8】进行时间序列预测...")


def arima_forecast(df, date_col, value_col, periods=12, order=(1, 1, 1)):
    """
    使用ARIMA进行时间序列预测（Prophet备选方案），输出中文列名

    参数:
        df: DataFrame - 输入数据（中文列名）
        date_col: str - 日期列名（中文）
        value_col: str - 数值列名（中文）
        periods: int - 预测期数
        order: tuple - ARIMA模型阶数

    返回:
        model: ARIMA模型
        forecast: 预测结果（中文列名）
        future_dates: 未来日期列表
    """
    # 准备数据
    df_sorted = df.sort_values(date_col).copy()
    df_sorted.set_index(date_col, inplace=True)

    # 重采样为月度数据
    ts_data = df_sorted[value_col].resample('M').sum()
    ts_data = ts_data.fillna(ts_data.mean())

    # 训练ARIMA模型
    model = ARIMA(ts_data, order=order)
    results = model.fit()

    # 预测
    forecast = results.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()

    # 创建未来日期
    last_date = ts_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]

    print(f"  ✓ ARIMA模型训练完成")
    print(f"  ✓ 预测未来 {periods} 个月")
    print(f"  ✓ 预测区间: {forecast_df['mean'].min():.2f} - {forecast_df['mean'].max():.2f}")

    # 构造类似Prophet的返回结构
    class FakeProphet:
        def __init__(self, forecast_df, future_dates, ts_data):
            self.forecast_df = forecast_df
            self.future_dates = future_dates
            self.ts_data = ts_data

        def plot(self, forecast, ax=None):
            if ax is None:
                ax = plt.gca()
            # 绘制历史数据
            ax.plot(self.ts_data.index, self.ts_data.values, label='历史数据', color='blue')
            # 绘制预测数据
            ax.plot(self.future_dates, self.forecast_df['mean'].values, label='预测', color='red', linestyle='--')
            # 添加置信区间
            ax.fill_between(self.future_dates,
                            self.forecast_df['mean_lower'].values,
                            self.forecast_df['mean_upper'].values,
                            alpha=0.2, color='red')
            ax.legend()
            return ax

        def plot_components(self, forecast):
            # 创建新的图窗
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'ARIMA模型无成分分解', ha='center', va='center', transform=ax.transAxes)
            return fig

    # 构造预测结果DataFrame（纯中文列名，无英文列）
    prophet_style_forecast = pd.DataFrame({
        '日期': list(ts_data.index) + list(future_dates),
        '预测值': list(ts_data.values) + list(forecast_df['mean'].values),
        '预测下限': list(ts_data.values) + list(forecast_df['mean_lower'].values),
        '预测上限': list(ts_data.values) + list(forecast_df['mean_upper'].values)
    })

    fake_prophet = FakeProphet(forecast_df, future_dates, ts_data)

    return fake_prophet, prophet_style_forecast, future_dates


def prophet_forecast_fixed(df, date_col, value_col, periods=12):
    """
    修复版Prophet预测：确保输出纯中文列名 + 删除无用的叠加项列
    """
    try:
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")

        # 准备Prophet所需的数据格式（仅临时使用英文列，最终输出中文）
        prophet_df = df[[date_col, value_col]].copy()
        # 临时重命名为Prophet要求的英文列（ds/y），仅用于训练
        prophet_df_train = prophet_df.rename(columns={date_col: 'ds', value_col: 'y'})

        # 创建并训练Prophet模型
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        # 添加月度季节性
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # 训练模型（使用临时英文列）
        model.fit(prophet_df_train)

        # 创建未来日期（先生成英文列，后续转换为中文）
        future = model.make_future_dataframe(periods=periods, freq='M')

        # 进行预测（此时结果为英文列）
        forecast_en = model.predict(future)

        print(f"  ✓ Prophet模型训练完成")
        print(f"  ✓ 预测未来 {periods} 个月")
        print(f"  ✓ 预测区间: {forecast_en['yhat'].tail(periods).min():.2f} - {forecast_en['yhat'].tail(periods).max():.2f}")

        # 关键：英文列→中文列映射（完整覆盖所有Prophet输出列）
        en_to_cn_map = {
            'ds': '日期',
            'yhat': '预测值',
            'yhat_lower': '预测下限',
            'yhat_upper': '预测上限',
            'trend': '趋势',
            'trend_lower': '趋势下限',
            'trend_upper': '趋势上限',
            'yearly': '年度季节性',
            'yearly_lower': '年度季节性下限',
            'yearly_upper': '年度季节性上限',
            'monthly': '月度季节性',
            'monthly_lower': '月度季节性下限',
            'monthly_upper': '月度季节性上限',
            'additive_terms': '叠加项',
            'additive_terms_lower': '叠加项下限',
            'additive_terms_upper': '叠加项上限'
        }

        # 转换为纯中文列名（仅保留中文列，删除所有英文列）
        forecast_cn = forecast_en.rename(columns=en_to_cn_map)
        # 过滤掉可能未映射的英文列（确保无残留）
        forecast_cn = forecast_cn[[v for k, v in en_to_cn_map.items() if v in forecast_cn.columns]]

        # ========== 核心修改：删除值全为0的无用列 ==========
        drop_cols = ['叠加项', '叠加项下限', '叠加项上限']
        # 只删除存在的列（避免KeyError）
        drop_cols = [col for col in drop_cols if col in forecast_cn.columns]
        if drop_cols:
            forecast_cn = forecast_cn.drop(columns=drop_cols)
            print(f"  ✓ 已删除无用列: {', '.join(drop_cols)}")

        # 未来日期DataFrame也转换为中文列
        future_cn = future.rename(columns={'ds': '日期'})

        return model, forecast_cn, future_cn

    except Exception as e:
        print(f"  ⚠ Prophet模型失败: {str(e)[:100]}")
        print(f"  ✓ 切换到ARIMA模型进行预测")
        # 使用ARIMA作为备选方案（已确保中文列名）
        return arima_forecast(df, date_col, value_col, periods=periods)


# 对月度数据进行预测（输出纯中文列名的结果）
prophet_model, prophet_forecast, prophet_future = prophet_forecast_fixed(
    monthly_data, '统计月份', '总销售量', periods=12
)

# 4. 多维度销售预测模型（全程使用中文列名）
print("\n【4/8】构建多维度销售预测模型...")


def build_multidimensional_sales_predictor(df, feature_cols, target_col):
    """
    构建多维度销售预测模型（全程使用中文列名）

    参数:
        df: DataFrame - 输入数据（中文列名）
        feature_cols: list - 特征列名列表（中文）
        target_col: str - 目标变量列名（中文）

    返回:
        best_model: 最佳预测模型
        results: dict - 包含预测结果和评估指标（中文键名）
    """
    # 准备数据
    df_ml = df.copy()
    feature_cols = feature_cols.copy()  # 避免修改调用方的列表

    # 处理分类变量（中文列名）
    le_dict = {}
    for col in list(feature_cols):  # 遍历副本，安全修改原列表
        if col in df_ml.columns and df_ml[col].dtype == 'object':
            le = LabelEncoder()
            # 编码列名使用中文后缀（如"品牌_编码"）
            encoded_col = f"{col}_编码"
            df_ml[encoded_col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le
            # 替换原始列为编码列
            feature_cols[feature_cols.index(col)] = encoded_col

    # 特征工程：提取时间特征（中文列名）
    if '统计月份' in df_ml.columns:
        df_ml['年份'] = df_ml['统计月份'].dt.year
        df_ml['月份'] = df_ml['统计月份'].dt.month
        df_ml['季度'] = df_ml['统计月份'].dt.quarter

        # 添加周期性特征（中文列名）
        df_ml['月份正弦值'] = np.sin(2 * np.pi * df_ml['月份'] / 12)
        df_ml['月份余弦值'] = np.cos(2 * np.pi * df_ml['月份'] / 12)

        # 添加中文时间特征列
        feature_cols.extend(['年份', '月份', '季度', '月份正弦值', '月份余弦值'])

    # 过滤存在的特征列
    feature_cols = [col for col in feature_cols if col in df_ml.columns]

    # 准备特征和目标变量（中文列名）
    # 确保按时间排序，保证时间序列分割的正确性
    if '统计月份' in df_ml.columns:
        df_ml = df_ml.sort_values('统计月份').reset_index(drop=True)
    X = df_ml[feature_cols].fillna(0)
    y = df_ml[target_col]

    # 划分训练集和测试集（按时间顺序）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  ✓ 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型（中文模型名）
    models = {
        '随机森林': RandomForestRegressor(random_state=42, n_estimators=200),
        '梯度提升': GradientBoostingRegressor(random_state=42, n_estimators=200)
    }

    # 超参数网格（中文键名）
    param_grids = {
        '随机森林': {
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        '梯度提升': {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    }

    # 训练和评估模型
    best_score = float('inf')
    best_model = None
    best_model_name = ""
    results = {}

    for model_name, model in models.items():
        print(f"\n  训练 {model_name}...")

        try:
            # 网格搜索 - 使用单进程避免中文路径问题
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=TimeSeriesSplit(n_splits=5),
                scoring='neg_mean_squared_error', n_jobs=1  # 单进程
            )
            grid_search.fit(X_train_scaled, y_train)

            # 最佳模型
            best_model_current = grid_search.best_estimator_

            # 预测
            y_pred = best_model_current.predict(X_test_scaled)

            # 评估（中文指标名）
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"    最佳参数: {grid_search.best_params_}")
            print(f"    测试集 均方误差(MSE): {mse:.2f}, 平均绝对误差(MAE): {mae:.2f}, 决定系数(R²): {r2:.3f}")

            # 保存结果（中文键名）
            results[model_name] = {
                '模型': best_model_current,
                '预测值': y_pred,
                '均方误差(MSE)': mse,
                '平均绝对误差(MAE)': mae,
                '决定系数(R²)': r2,
                '最佳参数': grid_search.best_params_
            }

            if mse < best_score:
                best_score = mse
                best_model = best_model_current
                best_model_name = model_name
        except Exception as e:
            print(f"    ⚠ {model_name}训练失败: {e}")
            continue

    if best_model:
        print(f"\n  ✓ 最佳模型: {best_model_name}")
        print(f"  ✓ 最佳测试集 均方误差(MSE): {best_score:.2f}")
    else:
        print(f"\n  ⚠ 所有模型训练失败")
        best_model = RandomForestRegressor(random_state=42, n_estimators=100)
        best_model.fit(X_train_scaled, y_train)

    return best_model, results, scaler, le_dict


# 构建月度销售预测模型（全程中文列名）
# 特征列和目标列均为中文
monthly_features = ['品牌', '平均销售价格', '总生产量']
best_monthly_model, monthly_results, monthly_scaler, monthly_encoders = build_multidimensional_sales_predictor(
    monthly_data,  # 已为中文列名
    monthly_features,
    '总销售量'  # 中文目标列
)

# 5. 地区销售分析（全程中文列名）
print("\n【5/8】执行地区销售分析...")


def regional_sales_analysis(df):
    """
    分析地区销售模式和区域特征（输入输出均为中文列名）

    参数:
        df: DataFrame - 地区销售数据（中文列名）

    返回:
        regional_insights: dict - 地区分析洞察（中文键名+中文列名）
    """
    # 按地区统计销售总量（中文列名聚合）
    regional_sales = df.groupby('地区').agg({
        '总销售量': 'sum',
        '市场份额': 'mean',
        '销售排名': 'mean'
    }).sort_values('总销售量', ascending=False)

    # 索引列名设置为中文
    regional_sales.index.name = '地区'

    # 按地区统计品牌表现（中文列名）
    brand_regional = df.groupby(['地区', '品牌'])['总销售量'].sum().unstack(fill_value=0)
    brand_regional.index.name = '地区'
    brand_regional.columns.name = '品牌'

    # 计算地区集中度（赫芬达尔指数）
    hhi_by_region = {}
    for region in df['地区'].unique():
        region_data = df[df['地区'] == region]
        brand_shares = region_data.groupby('品牌')['总销售量'].sum()
        if brand_shares.sum() > 0:
            hhi = (brand_shares / brand_shares.sum()).pow(2).sum()
            hhi_by_region[region] = hhi
        else:
            hhi_by_region[region] = 0

    # 识别高增长地区
    df_sorted = df.sort_values(['地区', '统计月份'])

    growth_by_region = {}
    for region in df['地区'].unique():
        region_data = df_sorted[df_sorted['地区'] == region]
        if len(region_data) >= 6:  # 至少需要6个月数据
            # 计算增长率（中文列名）
            early_sales = region_data['总销售量'].iloc[:3].mean()
            recent_sales = region_data['总销售量'].iloc[-3:].mean()
            if early_sales > 0:
                growth_rate = (recent_sales - early_sales) / early_sales * 100
                growth_by_region[region] = growth_rate

    print(f"  ✓ 地区分析完成")
    print(f"  ✓ 识别 {len(regional_sales)} 个地区")
    print(f"  ✓ 最高销售地区: {regional_sales.index[0]} ({regional_sales.iloc[0]['总销售量']:,.2f})")

    # 结果字典（中文键名）
    insights = {
        '地区销售汇总': regional_sales,
        '品牌地区销售': brand_regional,
        '地区赫芬达尔指数': hhi_by_region,
        '地区销售增长率': growth_by_region
    }

    return insights


# 执行地区分析（输入已为中文列名）
regional_insights = regional_sales_analysis(regional_data)

# 6. 可视化分析（使用中文列名和中文标签）
print("\n【6/8】生成销售趋势可视化图表...")


def plot_sales_trend_analysis():
    """
    创建销售趋势分析的可视化图表（全程中文标签）
    """
    # 图表1: 时间序列分解和预测
    if monthly_decomp is not None:
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        fig.suptitle('月度销售时间序列分解分析', fontsize=16, fontweight='bold')

        # 原始数据（中文列名）
        axes[0].plot(monthly_decomposed.index, monthly_decomposed['原始数据'],
                     color='blue', linewidth=2, label='原始数据')
        axes[0].set_title('原始销售数据')
        axes[0].set_ylabel('销售额')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 趋势成分（中文列名）
        axes[1].plot(monthly_decomposed.index, monthly_decomposed['趋势'],
                     color='red', linewidth=2, label='趋势')
        axes[1].set_title('趋势成分')
        axes[1].set_ylabel('趋势值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 季节成分（中文列名）
        axes[2].plot(monthly_decomposed.index, monthly_decomposed['季节性'],
                     color='green', linewidth=2, label='季节性')
        axes[2].set_title('季节成分')
        axes[2].set_ylabel('季节值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 残差成分（中文列名）
        axes[3].plot(monthly_decomposed.index, monthly_decomposed['残差'],
                     color='orange', linewidth=1, label='残差')
        axes[3].set_title('残差成分')
        axes[3].set_ylabel('残差值')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\时间序列分解图表.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 时间序列分解图表已保存")
        plt.close()

    # 图表2: 预测结果（中文列名）
    if prophet_forecast is not None:
        # 创建预测结果图
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('销售预测分析', fontsize=16, fontweight='bold')

        # 绘制预测结果（中文列名）
        ax.plot(prophet_forecast['日期'], prophet_forecast['预测值'], label='预测值', color='red', linewidth=2)
        ax.fill_between(prophet_forecast['日期'],
                        prophet_forecast['预测下限'],
                        prophet_forecast['预测上限'],
                        alpha=0.2, color='red', label='预测区间')

        # 添加趋势线（如果存在中文趋势列）
        if '趋势' in prophet_forecast.columns:
            ax.plot(prophet_forecast['日期'], prophet_forecast['趋势'],
                    label='趋势线', color='blue', linestyle='--', linewidth=1.5)

        ax.set_title('销售预测 (包含趋势和季节性)')
        ax.set_xlabel('日期')
        ax.set_ylabel('销售额')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\销售预测图表.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 销售预测图表已保存")
        plt.close()

    # 图表3: 地区销售分析（中文列名）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('地区销售分析', fontsize=16, fontweight='bold')

    # 地区销售排名（中文列名）
    regional_sales = regional_insights['地区销售汇总']
    top_15_regions = regional_sales.head(15)
    bars = axes[0, 0].barh(range(len(top_15_regions)), top_15_regions['总销售量'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_15_regions))))
    axes[0, 0].set_yticks(range(len(top_15_regions)))
    axes[0, 0].set_yticklabels(top_15_regions.index)
    axes[0, 0].set_xlabel('销售总量')
    axes[0, 0].set_title('地区销售排名 (前15)')
    axes[0, 0].invert_yaxis()

    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            axes[0, 0].text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{width / 1e6:.1f}M', ha='left', va='center', fontsize=8)

    # 地区增长率（中文列名）
    growth_data = pd.Series(regional_insights['地区销售增长率']).sort_values(ascending=False)
    top_10_growth = growth_data.head(10)
    colors = ['green' if x > 0 else 'red' for x in top_10_growth.values]
    bars = axes[0, 1].bar(range(len(top_10_growth)), top_10_growth.values, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(top_10_growth)))
    axes[0, 1].set_xticklabels(top_10_growth.index, rotation=45, ha='right', fontsize=9)
    axes[0, 1].set_ylabel('增长率 (%)')
    axes[0, 1].set_title('地区销售增长率 (前10)')
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 1].grid(True, alpha=0.3)

    # 品牌地区热力图（中文列名）
    brand_regional = regional_insights['品牌地区销售']
    top_regions = brand_regional.sum(axis=1).nlargest(8).index
    top_brands = brand_regional.sum(axis=0).nlargest(6).index
    heatmap_data = brand_regional.loc[top_regions, top_brands]
    im = axes[1, 0].imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xticks(range(len(top_brands)))
    axes[1, 0].set_xticklabels(top_brands, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_yticks(range(len(top_regions)))
    axes[1, 0].set_yticklabels(top_regions, fontsize=9)
    axes[1, 0].set_title('品牌-地区销售热力图')

    # 添加颜色条（中文标签）
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label('销售额', rotation=270, labelpad=15)

    # 地区市场集中度（HHI）（中文列名）
    hhi_data = pd.Series(regional_insights['地区赫芬达尔指数']).sort_values()
    top_10_hhi = hhi_data.head(10)
    axes[1, 1].scatter(range(len(top_10_hhi)), top_10_hhi.values,
                       s=100, c=top_10_hhi.values, cmap='RdYlBu_r', alpha=0.7)
    axes[1, 1].set_xticks(range(len(top_10_hhi)))
    axes[1, 1].set_xticklabels(top_10_hhi.index, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('赫芬达尔-赫希曼指数 (HHI)')
    axes[1, 1].set_title('地区市场集中度 (前10)')
    axes[1, 1].grid(True, alpha=0.3)

    # 添加HHI解释文本（中文）
    axes[1, 1].text(0.02, 0.98, 'HHI说明:\n低(<0.1): 竞争激烈\n中(0.1-0.2): 适度集中\n高(>0.2): 高度集中',
                    transform=axes[1, 1].transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\地区销售分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 地区销售分析图表已保存")
    plt.close()

    # 图表4: 模型性能对比（中文列名）
    if monthly_results and len(monthly_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('销售预测模型性能对比', fontsize=16, fontweight='bold')

        # 模型性能指标对比（中文列名）- 使用双Y轴避免量纲差异
        model_names = list(monthly_results.keys())
        mse_scores = [monthly_results[name]['均方误差(MSE)'] for name in model_names]
        mae_scores = [monthly_results[name]['平均绝对误差(MAE)'] for name in model_names]
        r2_scores = [monthly_results[name]['决定系数(R²)'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.3

        # 左Y轴：MSE 和 MAE（误差指标）
        bars1 = axes[0].bar(x - width/2, mse_scores, width, label='均方误差(MSE)', color='red', alpha=0.7)
        bars2 = axes[0].bar(x + width/2, mae_scores, width, label='平均绝对误差(MAE)', color='blue', alpha=0.7)
        axes[0].set_xlabel('模型')
        axes[0].set_ylabel('误差值')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 右Y轴：R²（0-1范围）
        ax0_right = axes[0].twinx()
        ax0_right.plot(x, r2_scores, 'go-', linewidth=2, markersize=8, label='决定系数(R²)')
        ax0_right.set_ylabel('R² 值')
        ax0_right.set_ylim(0, 1.1)
        ax0_right.legend(loc='upper right')

        axes[0].set_title('模型评估指标对比')

        # 预测结果可视化（中文列名）
        best_model_name = min(model_names, key=lambda x: monthly_results[x]['均方误差(MSE)'])
        best_predictions = monthly_results[best_model_name]['预测值']

        # 获取测试集的实际值（中文列名）
        split_idx = int(len(monthly_data) * 0.8)
        actual_values = monthly_data['总销售量'].iloc[split_idx:].values

        # 只显示前50个点
        display_points = min(50, len(actual_values))
        axes[1].plot(actual_values[:display_points], label='实际值', color='black', linewidth=2)
        axes[1].plot(best_predictions[:display_points], label=f'{best_model_name}预测',
                     color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('销售额')
        axes[1].set_title(f'{best_model_name}预测效果 (前{display_points}个点)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{IMAGE_SAVE_PATH}\\模型性能对比图表.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ 模型性能对比图表已保存")
        plt.close()


plot_sales_trend_analysis()

# 7. 生成销售分析报告（中文）
print("\n【7/8】生成销售趋势分析报告...")


def generate_sales_trend_report():
    """
    生成综合分析报告（纯中文）
    """
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售趋势分析报告")
    report.append("=" * 80)
    report.append("")

    # 数据概览（中文描述）
    report.append("【数据概览】")
    report.append(f"  • 分析时间范围: {monthly_data['统计月份'].min()} 至 {monthly_data['统计月份'].max()}")
    report.append(f"  • 涉及品牌: {len(monthly_data['品牌'].unique())} 个")
    report.append(f"  • 涉及地区: {len(regional_data['地区'].unique())} 个")
    report.append(f"  • 总销售额: ¥{monthly_data['总销售量'].sum():,.2f}")
    report.append("")

    # 时间序列分析结果（中文描述）
    if monthly_decomp is not None:
        report.append("【时间序列分析】")
        report.append(f"  • 季节性强度: {monthly_decomposed['季节性'].var():.2f}")
        report.append(f"  • 残差标准差: {monthly_decomposed['残差'].std():.2f}")
        trend_series = monthly_decomposed['趋势'].dropna()
        if len(trend_series) > 1:
            trend_slope = (trend_series.iloc[-1] - trend_series.iloc[0]) / len(trend_series)
            report.append(f"  • 趋势斜率: {trend_slope:.2f} (每月份)")
    report.append("")

    # 预测结果（中文描述）
    report.append("【销售预测结果】")
    if '预测值' in prophet_forecast.columns:
        future_forecast = prophet_forecast['预测值'].tail(12).values
        report.append(f"  • 未来12个月预测销售额: ¥{future_forecast.sum():,.2f}")
        report.append(f"  • 预测均值: ¥{future_forecast.mean():,.2f}")
        report.append(f"  • 预测区间: ¥{future_forecast.min():,.2f} - ¥{future_forecast.max():,.2f}")
    report.append("")

    # 地区分析结果（中文描述）
    report.append("【地区销售分析】")
    top_3_regions = regional_insights['地区销售汇总'].head(3)
    report.append("  销售排名前三的地区:")
    for i, (region, row) in enumerate(top_3_regions.iterrows()):
        report.append(f"    {i + 1}. {region}: ¥{row['总销售量']:,.2f}")
        report.append(f"       平均市场份额: {row['市场份额']:.3f}")
        if not pd.isna(row['销售排名']):
            report.append(f"       平均排名: {row['销售排名']:.1f}")
    report.append("")

    # 高增长地区（中文描述）
    growth_data = pd.Series(regional_insights['地区销售增长率']).sort_values(ascending=False)
    if len(growth_data) > 0:
        report.append("  增长率最高的地区:")
        for i, (region, growth) in enumerate(growth_data.head(3).items()):
            report.append(f"    {i + 1}. {region}: {growth:.1f}%")
        report.append("")

    # 模型性能（中文描述）
    if monthly_results and len(monthly_results) > 0:
        report.append("【预测模型性能】")
        best_model_name = min(monthly_results.keys(), key=lambda x: monthly_results[x]['均方误差(MSE)'])
        best_result = monthly_results[best_model_name]
        report.append(f"  最佳模型: {best_model_name}")
        report.append(f"  • 测试集 均方误差(MSE): {best_result['均方误差(MSE)']:.2f}")
        report.append(f"  • 测试集 平均绝对误差(MAE): {best_result['平均绝对误差(MAE)']:.2f}")
        report.append(f"  • 测试集 决定系数(R²): {best_result['决定系数(R²)']:.3f}")
    report.append("")

    # 业务建议（中文描述）
    report.append("【业务建议】")
    report.append("  1. 库存管理: 根据季节性分解结果，优化库存配置")
    report.append("  2. 区域策略: 重点投入高增长地区，巩固优势地区")
    report.append("  3. 销售预测: 使用时间序列模型进行月度销售计划")
    report.append("  4. 市场扩张: 分析低渗透率地区的市场机会")
    report.append("  5. 品牌策略: 根据地区品牌表现调整营销策略")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    report.append("=" * 80)

    return "\n".join(report)


report_content = generate_sales_trend_report()
print(report_content)

# ========== 关键修改3：报告保存到文档文件夹 ==========
# 保存报告到文件（UTF-8编码确保中文正常）
with open(f"{DOCUMENT_SAVE_PATH}\\销售趋势分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 分析报告已保存: {DOCUMENT_SAVE_PATH}\\销售趋势分析报告.txt")

# 8. 保存预测结果（全部CSV文件使用纯中文列名）
print("\n【8/8】保存预测结果...")

# ========== 关键修改4：所有CSV文件保存到文档文件夹 ==========
# 保存销售预测结果（纯中文列名，无英文残留）
if prophet_forecast is not None:
    # 保存为CSV文件，使用utf-8-sig编码（解决Excel中文乱码）
    prophet_forecast.to_csv(
        f"{DOCUMENT_SAVE_PATH}\\销售预测结果.csv",
        index=False,
        encoding='utf-8-sig'  # 关键：确保Excel打开中文正常
    )
    print(f"  ✓ 销售预测结果已保存为CSV文件（纯中文列名，已删除叠加项相关列）")

# 保存地区销售汇总（中文列名）
regional_sales_df = regional_insights['地区销售汇总'].reset_index()
regional_sales_df.to_csv(
    f"{DOCUMENT_SAVE_PATH}\\地区销售汇总.csv",
    index=False,
    encoding='utf-8-sig'
)
print(f"  ✓ 地区销售汇总已保存为CSV文件（纯中文列名）")

# 保存模型性能对比（中文列名）
if monthly_results and len(monthly_results) > 0:
    # 构建中文列名的性能DataFrame
    performance_data = []
    for model_name, result in monthly_results.items():
        performance_data.append({
            '模型名称': model_name,
            '均方误差(MSE)': result['均方误差(MSE)'],
            '平均绝对误差(MAE)': result['平均绝对误差(MAE)'],
            '决定系数(R²)': result['决定系数(R²)']
        })
    performance_df = pd.DataFrame(performance_data)

    # 保存CSV
    performance_df.to_csv(
        f"{DOCUMENT_SAVE_PATH}\\模型性能对比.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 模型性能对比已保存为CSV文件（纯中文列名）")

# 保存地区销售增长率（中文列名）
growth_data = regional_insights['地区销售增长率']
growth_df = pd.DataFrame({
    '地区': list(growth_data.keys()),
    '增长率(%)': list(growth_data.values())
}).sort_values('增长率(%)', ascending=False)
growth_df.to_csv(
    f"{DOCUMENT_SAVE_PATH}\\地区销售增长率.csv",
    index=False,
    encoding='utf-8-sig'
)
print(f"  ✓ 地区销售增长率已保存为CSV文件（纯中文列名）")

# 保存地区市场集中度（中文列名）
hhi_data = regional_insights['地区赫芬达尔指数']
hhi_df = pd.DataFrame({
    '地区': list(hhi_data.keys()),
    '赫芬达尔指数': list(hhi_data.values())
}).sort_values('赫芬达尔指数', ascending=False)
hhi_df.to_csv(
    f"{DOCUMENT_SAVE_PATH}\\地区市场集中度.csv",
    index=False,
    encoding='utf-8-sig'
)
print(f"  ✓ 地区市场集中度已保存为CSV文件（纯中文列名）")

# 清理临时文件夹
try:
    shutil.rmtree(TEMP_DIR)
    print(f"  ✓ 临时文件已清理")
except Exception:
    pass

print("\n" + "=" * 80)
print("销售趋势分析模块执行完成！")
print("=" * 80)