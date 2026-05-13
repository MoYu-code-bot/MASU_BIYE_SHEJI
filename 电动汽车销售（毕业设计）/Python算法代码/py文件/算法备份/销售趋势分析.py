# =============================================================================
# 文件：销售趋势分析.py
# -----------------------------------------------------------------------------
# 总体说明
#   读入月/季/年/地区销售 Excel，统一中文列名与时间类型；对月度销量做季节分解与预测，并用多特征回归
#   拟合销量，辅以地区层面的汇总、HHI 集中度与增长率描述，最后绘图与导出 CSV/报告。
#
# 核心算法
#   ① 经典季节分解（seasonal_decompose）：把月度销量拆成趋势、季节、残差（加性模型）。
#   ② ARIMA：单变量时间序列外推；Prophet 不可用时回退到本分支（statsmodels）。
#   ③ Prophet（可选）：Facebook 加法型趋势 + 年/月季节性；失败则回退 ARIMA。
#   ④ 多变量回归：随机森林 / 梯度提升 + GridSearchCV + TimeSeriesSplit，在品牌编码与时间特征上预测销量。
#   ⑤ 地区分析：分组聚合 + 赫芬达尔指数（HHI）+ 首尾窗口增长率（描述性，非学习模型）。
#
# 说明：LinearRegression、ExponentialSmoothing 已导入，当前主流程未使用，可扩展基线或 Holt-Winters。
# =============================================================================

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



TEMP_DIR = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\temp"
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR


if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


tempfile.tempdir = TEMP_DIR


try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("  ⚠ Prophet库未安装，将使用ARIMA模型替代")

warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



BASE_SAVE_PATH = r"D:\Masu_BiYe_sheji\Python_SuanFa\electric_car_sale\算法图片存储\销售趋势分析"

IMAGE_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "图片分析")

DOCUMENT_SAVE_PATH = os.path.join(BASE_SAVE_PATH, "文档分析")


for path in [IMAGE_SAVE_PATH, DOCUMENT_SAVE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

print("=" * 80)
print("电动汽车销售趋势分析系统")
print("=" * 80)


print("\n【1/8】正在加载销售趋势数据...")


# =============================================================================
# [数据准备] load_sales_trend_data：读四张表并按约定映射为中文列名，解析各表时间列。
# =============================================================================

def load_sales_trend_data():
    monthly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\月度销售趋势.xlsx")
    quarterly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\季度销售趋势.xlsx")
    yearly_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\年度销售趋势.xlsx")
    regional_df = pd.read_excel(r"D:\HuaweiMoveData\Users\江磊\Desktop\电动汽车销售（毕业设计）\电动汽车销售指标表\地区销售趋势.xlsx")

    print(f"  ✓ 月度数据: {len(monthly_df)} 条记录")
    print(f"  ✓ 季度数据: {len(quarterly_df)} 条记录")
    print(f"  ✓ 年度数据: {len(yearly_df)} 条记录")
    print(f"  ✓ 地区数据: {len(regional_df)} 条记录")



    monthly_cn_map = {
        'stat_month': '统计月份',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'avg_sale_price': '平均销售价格',
        'total_production_volume': '总生产量'
    }
    monthly_df = monthly_df.rename(columns={k: v for k, v in monthly_cn_map.items() if k in monthly_df.columns})


    quarterly_cn_map = {
        'stat_quarter': '统计季度',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'avg_sale_price': '平均销售价格'
    }
    quarterly_df = quarterly_df.rename(columns={k: v for k, v in quarterly_cn_map.items() if k in quarterly_df.columns})


    yearly_cn_map = {
        'stat_year': '统计年度',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'market_share': '市场份额'
    }
    yearly_df = yearly_df.rename(columns={k: v for k, v in yearly_cn_map.items() if k in yearly_df.columns})


    regional_cn_map = {
        'origin': '地区',
        'stat_month': '统计月份',
        'brand': '品牌',
        'total_sales_volume': '总销售量',
        'market_share': '市场份额',
        'sales_rank': '销售排名'
    }
    regional_df = regional_df.rename(columns={k: v for k, v in regional_cn_map.items() if k in regional_df.columns})


    for df in [monthly_df, quarterly_df, yearly_df, regional_df]:
        if '统计月份' in df.columns:
            df['统计月份'] = pd.to_datetime(df['统计月份'], errors='coerce')
        elif '统计季度' in df.columns:
            df['统计季度'] = pd.to_datetime(df['统计季度'], errors='coerce')
        elif '统计年度' in df.columns:
            df['统计年度'] = pd.to_datetime(df['统计年度'], format='%Y', errors='coerce')

    return monthly_df, quarterly_df, yearly_df, regional_df



monthly_data, quarterly_data, yearly_data, regional_data = load_sales_trend_data()


print("\n【2/8】执行时间序列分解分析...")


# =============================================================================
# [算法 ①] 季节分解 time_series_decomposition_analysis（statsmodels.seasonal_decompose）
# 用处：将重采样后的月度销量分解为趋势、季节项与残差，用于观察周期强度与不规则波动。
# =============================================================================

def time_series_decomposition_analysis(df, date_col, value_col, period=12):
    df_sorted = df.sort_values(date_col).copy()


    df_sorted.set_index(date_col, inplace=True)


    if not isinstance(df_sorted.index, pd.DatetimeIndex):
        df_sorted.index = pd.to_datetime(df_sorted.index)


    ts_data = df_sorted[value_col].resample('M').sum()


    ts_data = ts_data.fillna(ts_data.mean())


    try:
        decomposition = seasonal_decompose(ts_data, model='additive', period=period, extrapolate_trend='freq')

        print(f"  ✓ 时间序列分解完成")
        print(f"  ✓ 趋势成分范围: {decomposition.trend.min():.2f} 到 {decomposition.trend.max():.2f}")
        print(f"  ✓ 季节成分方差: {decomposition.seasonal.var():.2f}")
        print(f"  ✓ 残差标准差: {decomposition.resid.std():.2f}")


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



monthly_decomp, monthly_decomposed = time_series_decomposition_analysis(
    monthly_data, '统计月份', '总销售量', period=12
)


print("\n【3/8】进行时间序列预测...")


# =============================================================================
# [算法 ②] ARIMA 单步外推（arima_forecast）
# 用处：指定 (p,d,q) 拟合 ARIMA，对未来 periods 个月给出均值与置信区间；输出包装成与 Prophet 类似的结构便于同一套绘图逻辑复用。
# =============================================================================

def arima_forecast(df, date_col, value_col, periods=12, order=(1, 1, 1)):
    df_sorted = df.sort_values(date_col).copy()
    df_sorted.set_index(date_col, inplace=True)


    ts_data = df_sorted[value_col].resample('M').sum()
    ts_data = ts_data.fillna(ts_data.mean())


    model = ARIMA(ts_data, order=order)
    results = model.fit()


    forecast = results.get_forecast(steps=periods)
    forecast_df = forecast.summary_frame()


    last_date = ts_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]

    print(f"  ✓ ARIMA模型训练完成")
    print(f"  ✓ 预测未来 {periods} 个月")
    print(f"  ✓ 预测区间: {forecast_df['mean'].min():.2f} - {forecast_df['mean'].max():.2f}")


    class FakeProphet:
        def __init__(self, forecast_df, future_dates, ts_data):
            self.forecast_df = forecast_df
            self.future_dates = future_dates
            self.ts_data = ts_data

        def plot(self, forecast, ax=None):
            if ax is None:
                ax = plt.gca()

            ax.plot(self.ts_data.index, self.ts_data.values, label='历史数据', color='blue')

            ax.plot(self.future_dates, self.forecast_df['mean'].values, label='预测', color='red', linestyle='--')

            ax.fill_between(self.future_dates,
                            self.forecast_df['mean_lower'].values,
                            self.forecast_df['mean_upper'].values,
                            alpha=0.2, color='red')
            ax.legend()
            return ax

        def plot_components(self, forecast):

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'ARIMA模型无成分分解', ha='center', va='center', transform=ax.transAxes)
            return fig


    prophet_style_forecast = pd.DataFrame({
        '日期': list(ts_data.index) + list(future_dates),
        '预测值': list(ts_data.values) + list(forecast_df['mean'].values),
        '预测下限': list(ts_data.values) + list(forecast_df['mean_lower'].values),
        '预测上限': list(ts_data.values) + list(forecast_df['mean_upper'].values)
    })

    fake_prophet = FakeProphet(forecast_df, future_dates, ts_data)

    return fake_prophet, prophet_style_forecast, future_dates


# =============================================================================
# [算法 ③] Prophet 预测（prophet_forecast_fixed），失败则调用算法 ②
# 用处：捕捉年周期与自定义月周期（fourier_order）；多plicative 季节模式；输出中文列名预测表供下游综合分析读取。
# =============================================================================

def prophet_forecast_fixed(df, date_col, value_col, periods=12):
    try:
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")


        prophet_df = df[[date_col, value_col]].copy()

        prophet_df_train = prophet_df.rename(columns={date_col: 'ds', value_col: 'y'})


        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )


        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)


        model.fit(prophet_df_train)


        future = model.make_future_dataframe(periods=periods, freq='M')


        forecast_en = model.predict(future)

        print(f"  ✓ Prophet模型训练完成")
        print(f"  ✓ 预测未来 {periods} 个月")
        print(f"  ✓ 预测区间: {forecast_en['yhat'].tail(periods).min():.2f} - {forecast_en['yhat'].tail(periods).max():.2f}")


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


        forecast_cn = forecast_en.rename(columns=en_to_cn_map)

        forecast_cn = forecast_cn[[v for k, v in en_to_cn_map.items() if v in forecast_cn.columns]]


        drop_cols = ['叠加项', '叠加项下限', '叠加项上限']

        drop_cols = [col for col in drop_cols if col in forecast_cn.columns]
        if drop_cols:
            forecast_cn = forecast_cn.drop(columns=drop_cols)
            print(f"  ✓ 已删除无用列: {', '.join(drop_cols)}")


        future_cn = future.rename(columns={'ds': '日期'})

        return model, forecast_cn, future_cn

    except Exception as e:
        print(f"  ⚠ Prophet模型失败: {str(e)[:100]}")
        print(f"  ✓ 切换到ARIMA模型进行预测")

        return arima_forecast(df, date_col, value_col, periods=periods)



prophet_model, prophet_forecast, prophet_future = prophet_forecast_fixed(
    monthly_data, '统计月份', '总销售量', periods=12
)


print("\n【4/8】构建多维度销售预测模型...")


# =============================================================================
# [算法 ④] 多特征销量回归 + GridSearchCV + TimeSeriesSplit（build_multidimensional_sales_predictor）
# 用处：对类别列 LabelEncoder，补充年月季度与正余弦月份特征；时间顺序划分下用 RF/GB 网格搜索最小化 MSE，得到可解释的多因素销量拟合。
# =============================================================================

def build_multidimensional_sales_predictor(df, feature_cols, target_col):
    df_ml = df.copy()
    feature_cols = feature_cols.copy()


    le_dict = {}
    for col in list(feature_cols):
        if col in df_ml.columns and df_ml[col].dtype == 'object':
            le = LabelEncoder()

            encoded_col = f"{col}_编码"
            df_ml[encoded_col] = le.fit_transform(df_ml[col].astype(str))
            le_dict[col] = le

            feature_cols[feature_cols.index(col)] = encoded_col


    if '统计月份' in df_ml.columns:
        df_ml['年份'] = df_ml['统计月份'].dt.year
        df_ml['月份'] = df_ml['统计月份'].dt.month
        df_ml['季度'] = df_ml['统计月份'].dt.quarter


        df_ml['月份正弦值'] = np.sin(2 * np.pi * df_ml['月份'] / 12)
        df_ml['月份余弦值'] = np.cos(2 * np.pi * df_ml['月份'] / 12)


        feature_cols.extend(['年份', '月份', '季度', '月份正弦值', '月份余弦值'])


    feature_cols = [col for col in feature_cols if col in df_ml.columns]



    if '统计月份' in df_ml.columns:
        df_ml = df_ml.sort_values('统计月份').reset_index(drop=True)
    X = df_ml[feature_cols].fillna(0)
    y = df_ml[target_col]


    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  ✓ 训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    models = {
        '随机森林': RandomForestRegressor(random_state=42, n_estimators=200),
        '梯度提升': GradientBoostingRegressor(random_state=42, n_estimators=200)
    }


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


    best_score = float('inf')
    best_model = None
    best_model_name = ""
    results = {}

    for model_name, model in models.items():
        print(f"\n  训练 {model_name}...")

        try:

            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=TimeSeriesSplit(n_splits=5),
                scoring='neg_mean_squared_error', n_jobs=1
            )
            grid_search.fit(X_train_scaled, y_train)


            best_model_current = grid_search.best_estimator_


            y_pred = best_model_current.predict(X_test_scaled)


            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"    最佳参数: {grid_search.best_params_}")
            print(f"    测试集 均方误差(MSE): {mse:.2f}, 平均绝对误差(MAE): {mae:.2f}, 决定系数(R²): {r2:.3f}")


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




monthly_features = ['品牌', '平均销售价格', '总生产量']
best_monthly_model, monthly_results, monthly_scaler, monthly_encoders = build_multidimensional_sales_predictor(
    monthly_data,
    monthly_features,
    '总销售量'
)


print("\n【5/8】执行地区销售分析...")


# =============================================================================
# [描述分析 ⑤] 地区销售与集中度（regional_sales_analysis）
# 用处：按地区/地区×品牌聚合；计算各地区品牌销量份额平方和（HHI）；用时间序列首尾窗口均值近似增长率。
# =============================================================================

def regional_sales_analysis(df):
    regional_sales = df.groupby('地区').agg({
        '总销售量': 'sum',
        '市场份额': 'mean',
        '销售排名': 'mean'
    }).sort_values('总销售量', ascending=False)


    regional_sales.index.name = '地区'


    brand_regional = df.groupby(['地区', '品牌'])['总销售量'].sum().unstack(fill_value=0)
    brand_regional.index.name = '地区'
    brand_regional.columns.name = '品牌'


    hhi_by_region = {}
    for region in df['地区'].unique():
        region_data = df[df['地区'] == region]
        brand_shares = region_data.groupby('品牌')['总销售量'].sum()
        if brand_shares.sum() > 0:
            hhi = (brand_shares / brand_shares.sum()).pow(2).sum()
            hhi_by_region[region] = hhi
        else:
            hhi_by_region[region] = 0


    df_sorted = df.sort_values(['地区', '统计月份'])

    growth_by_region = {}
    for region in df['地区'].unique():
        region_data = df_sorted[df_sorted['地区'] == region]
        if len(region_data) >= 6:

            early_sales = region_data['总销售量'].iloc[:3].mean()
            recent_sales = region_data['总销售量'].iloc[-3:].mean()
            if early_sales > 0:
                growth_rate = (recent_sales - early_sales) / early_sales * 100
                growth_by_region[region] = growth_rate

    print(f"  ✓ 地区分析完成")
    print(f"  ✓ 识别 {len(regional_sales)} 个地区")
    print(f"  ✓ 最高销售地区: {regional_sales.index[0]} ({regional_sales.iloc[0]['总销售量']:,.2f})")


    insights = {
        '地区销售汇总': regional_sales,
        '品牌地区销售': brand_regional,
        '地区赫芬达尔指数': hhi_by_region,
        '地区销售增长率': growth_by_region
    }

    return insights



regional_insights = regional_sales_analysis(regional_data)


print("\n【6/8】生成销售趋势可视化图表...")


# =============================================================================
# [可视化] plot_sales_trend_analysis：分解图、预测曲线、地区图、模型对比等，依赖前述算法产出的全局变量。
# =============================================================================

def plot_sales_trend_analysis():
    if monthly_decomp is not None:
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        fig.suptitle('月度销售时间序列分解分析', fontsize=16, fontweight='bold')


        axes[0].plot(monthly_decomposed.index, monthly_decomposed['原始数据'],
                     color='blue', linewidth=2, label='原始数据')
        axes[0].set_title('原始销售数据')
        axes[0].set_ylabel('销售额')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)


        axes[1].plot(monthly_decomposed.index, monthly_decomposed['趋势'],
                     color='red', linewidth=2, label='趋势')
        axes[1].set_title('趋势成分')
        axes[1].set_ylabel('趋势值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)


        axes[2].plot(monthly_decomposed.index, monthly_decomposed['季节性'],
                     color='green', linewidth=2, label='季节性')
        axes[2].set_title('季节成分')
        axes[2].set_ylabel('季节值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)


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


    if prophet_forecast is not None:

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('销售预测分析', fontsize=16, fontweight='bold')


        ax.plot(prophet_forecast['日期'], prophet_forecast['预测值'], label='预测值', color='red', linewidth=2)
        ax.fill_between(prophet_forecast['日期'],
                        prophet_forecast['预测下限'],
                        prophet_forecast['预测上限'],
                        alpha=0.2, color='red', label='预测区间')


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


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('地区销售分析', fontsize=16, fontweight='bold')


    regional_sales = regional_insights['地区销售汇总']
    top_15_regions = regional_sales.head(15)
    bars = axes[0, 0].barh(range(len(top_15_regions)), top_15_regions['总销售量'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(top_15_regions))))
    axes[0, 0].set_yticks(range(len(top_15_regions)))
    axes[0, 0].set_yticklabels(top_15_regions.index)
    axes[0, 0].set_xlabel('销售总量')
    axes[0, 0].set_title('地区销售排名 (前15)')
    axes[0, 0].invert_yaxis()


    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            axes[0, 0].text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{width / 1e6:.1f}M', ha='left', va='center', fontsize=8)


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


    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label('销售额', rotation=270, labelpad=15)


    hhi_data = pd.Series(regional_insights['地区赫芬达尔指数']).sort_values()
    top_10_hhi = hhi_data.head(10)
    axes[1, 1].scatter(range(len(top_10_hhi)), top_10_hhi.values,
                       s=100, c=top_10_hhi.values, cmap='RdYlBu_r', alpha=0.7)
    axes[1, 1].set_xticks(range(len(top_10_hhi)))
    axes[1, 1].set_xticklabels(top_10_hhi.index, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('赫芬达尔-赫希曼指数 (HHI)')
    axes[1, 1].set_title('地区市场集中度 (前10)')
    axes[1, 1].grid(True, alpha=0.3)


    axes[1, 1].text(0.02, 0.98, 'HHI说明:\n低(<0.1): 竞争激烈\n中(0.1-0.2): 适度集中\n高(>0.2): 高度集中',
                    transform=axes[1, 1].transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{IMAGE_SAVE_PATH}\\地区销售分析图表.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 地区销售分析图表已保存")
    plt.close()


    if monthly_results and len(monthly_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('销售预测模型性能对比', fontsize=16, fontweight='bold')


        model_names = list(monthly_results.keys())
        mse_scores = [monthly_results[name]['均方误差(MSE)'] for name in model_names]
        mae_scores = [monthly_results[name]['平均绝对误差(MAE)'] for name in model_names]
        r2_scores = [monthly_results[name]['决定系数(R²)'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.3


        bars1 = axes[0].bar(x - width/2, mse_scores, width, label='均方误差(MSE)', color='red', alpha=0.7)
        bars2 = axes[0].bar(x + width/2, mae_scores, width, label='平均绝对误差(MAE)', color='blue', alpha=0.7)
        axes[0].set_xlabel('模型')
        axes[0].set_ylabel('误差值')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)


        ax0_right = axes[0].twinx()
        ax0_right.plot(x, r2_scores, 'go-', linewidth=2, markersize=8, label='决定系数(R²)')
        ax0_right.set_ylabel('R² 值')
        ax0_right.set_ylim(0, 1.1)
        ax0_right.legend(loc='upper right')

        axes[0].set_title('模型评估指标对比')


        best_model_name = min(model_names, key=lambda x: monthly_results[x]['均方误差(MSE)'])
        best_predictions = monthly_results[best_model_name]['预测值']


        split_idx = int(len(monthly_data) * 0.8)
        actual_values = monthly_data['总销售量'].iloc[split_idx:].values


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


print("\n【7/8】生成销售趋势分析报告...")


# =============================================================================
# [输出] generate_sales_trend_report：拼接分解指标、Prophet/ARIMA 预测摘要、地区排名与多变量回归 MSE 等文本段落。
# =============================================================================

def generate_sales_trend_report():
    report = []
    report.append("=" * 80)
    report.append("电动汽车销售趋势分析报告")
    report.append("=" * 80)
    report.append("")


    report.append("【数据概览】")
    report.append(f"  • 分析时间范围: {monthly_data['统计月份'].min()} 至 {monthly_data['统计月份'].max()}")
    report.append(f"  • 涉及品牌: {len(monthly_data['品牌'].unique())} 个")
    report.append(f"  • 涉及地区: {len(regional_data['地区'].unique())} 个")
    report.append(f"  • 总销售额: ¥{monthly_data['总销售量'].sum():,.2f}")
    report.append("")


    if monthly_decomp is not None:
        report.append("【时间序列分析】")
        report.append(f"  • 季节性强度: {monthly_decomposed['季节性'].var():.2f}")
        report.append(f"  • 残差标准差: {monthly_decomposed['残差'].std():.2f}")
        trend_series = monthly_decomposed['趋势'].dropna()
        if len(trend_series) > 1:
            trend_slope = (trend_series.iloc[-1] - trend_series.iloc[0]) / len(trend_series)
            report.append(f"  • 趋势斜率: {trend_slope:.2f} (每月份)")
    report.append("")


    report.append("【销售预测结果】")
    if '预测值' in prophet_forecast.columns:
        future_forecast = prophet_forecast['预测值'].tail(12).values
        report.append(f"  • 未来12个月预测销售额: ¥{future_forecast.sum():,.2f}")
        report.append(f"  • 预测均值: ¥{future_forecast.mean():,.2f}")
        report.append(f"  • 预测区间: ¥{future_forecast.min():,.2f} - ¥{future_forecast.max():,.2f}")
    report.append("")


    report.append("【地区销售分析】")
    top_3_regions = regional_insights['地区销售汇总'].head(3)
    report.append("  销售排名前三的地区:")
    for i, (region, row) in enumerate(top_3_regions.iterrows()):
        report.append(f"    {i + 1}. {region}: ¥{row['总销售量']:,.2f}")
        report.append(f"       平均市场份额: {row['市场份额']:.3f}")
        if not pd.isna(row['销售排名']):
            report.append(f"       平均排名: {row['销售排名']:.1f}")
    report.append("")


    growth_data = pd.Series(regional_insights['地区销售增长率']).sort_values(ascending=False)
    if len(growth_data) > 0:
        report.append("  增长率最高的地区:")
        for i, (region, growth) in enumerate(growth_data.head(3).items()):
            report.append(f"    {i + 1}. {region}: {growth:.1f}%")
        report.append("")


    if monthly_results and len(monthly_results) > 0:
        report.append("【预测模型性能】")
        best_model_name = min(monthly_results.keys(), key=lambda x: monthly_results[x]['均方误差(MSE)'])
        best_result = monthly_results[best_model_name]
        report.append(f"  最佳模型: {best_model_name}")
        report.append(f"  • 测试集 均方误差(MSE): {best_result['均方误差(MSE)']:.2f}")
        report.append(f"  • 测试集 平均绝对误差(MAE): {best_result['平均绝对误差(MAE)']:.2f}")
        report.append(f"  • 测试集 决定系数(R²): {best_result['决定系数(R²)']:.3f}")
    report.append("")


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



with open(f"{DOCUMENT_SAVE_PATH}\\销售趋势分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"\n✓ 分析报告已保存: {DOCUMENT_SAVE_PATH}\\销售趋势分析报告.txt")


print("\n【8/8】保存预测结果...")

# 将 prophet_forecast、地区汇总、模型指标、增长率、HHI 等写入 DOCUMENT_SAVE_PATH，供「综合分析」等模块读取。

if prophet_forecast is not None:

    prophet_forecast.to_csv(
        f"{DOCUMENT_SAVE_PATH}\\销售预测结果.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 销售预测结果已保存为CSV文件（纯中文列名，已删除叠加项相关列）")


regional_sales_df = regional_insights['地区销售汇总'].reset_index()
regional_sales_df.to_csv(
    f"{DOCUMENT_SAVE_PATH}\\地区销售汇总.csv",
    index=False,
    encoding='utf-8-sig'
)
print(f"  ✓ 地区销售汇总已保存为CSV文件（纯中文列名）")


if monthly_results and len(monthly_results) > 0:

    performance_data = []
    for model_name, result in monthly_results.items():
        performance_data.append({
            '模型名称': model_name,
            '均方误差(MSE)': result['均方误差(MSE)'],
            '平均绝对误差(MAE)': result['平均绝对误差(MAE)'],
            '决定系数(R²)': result['决定系数(R²)']
        })
    performance_df = pd.DataFrame(performance_data)


    performance_df.to_csv(
        f"{DOCUMENT_SAVE_PATH}\\模型性能对比.csv",
        index=False,
        encoding='utf-8-sig'
    )
    print(f"  ✓ 模型性能对比已保存为CSV文件（纯中文列名）")


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


try:
    shutil.rmtree(TEMP_DIR)
    print(f"  ✓ 临时文件已清理")
except Exception:
    pass

print("\n" + "=" * 80)
print("销售趋势分析模块执行完成！")
print("=" * 80)