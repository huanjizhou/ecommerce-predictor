#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧚‍♀️ 电商用户行为预测模型 - 自动进化脚本

功能：
1. 滚动式训练：每次用所有历史数据训练
2. 自动验证：用下个月数据验证准确性
3. 结果保存：保留每次验证的历史记录
4. 持续进化：随着新数据导入，模型越来越强

使用方法：
    # 阶段 1：10 月训练，11 月验证
    python ecommerce_predictor_auto.py --train-end 2019-11-01 --val-end 2019-12-01
    
    # 阶段 2：10-11 月训练，12 月验证（导入 12 月数据后）
    python ecommerce_predictor_auto.py --train-end 2019-12-01 --val-end 2020-01-01
    
    # 阶段 3：10-12 月训练，1 月验证（导入 1 月数据后）
    python ecommerce_predictor_auto.py --train-end 2020-01-01 --val-end 2020-02-01

作者：E-Commerce Predictor Team
版本：v5.0-Auto
"""

import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import argparse
import json
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# 加载 .env 文件（如果存在）
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    HAS_PLOT = True
except:
    HAS_PLOT = False
    print("⚠️ 警告：matplotlib/seaborn 不可用，将跳过图表生成")

# 机器学习库
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("⚠️ XGBoost 不可用")

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 兼容不同 sklearn 版本
try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if mask.sum() == 0:
            return 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ==================== 数据库配置 ====================
DB_CONFIG = {
    'host': os.getenv('RDS_HOST', 'your-rds-host.mysql.rds.aliyuncs.com'),
    'port': int(os.getenv('RDS_PORT', '3306')),
    'user': os.getenv('RDS_USER', 'paimon'),
    'password': os.getenv('RDS_PASSWORD', 'YOUR_PASSWORD'),
    'database': os.getenv('RDS_DATABASE', 'ecommerce_demo')
}

# 验证必要的环境变量
REQUIRED_ENV = ['RDS_HOST', 'RDS_USER', 'RDS_PASSWORD', 'RDS_DATABASE']
missing = [env for env in REQUIRED_ENV if env not in os.environ and env not in os.getenv.__dict__]
if missing and os.getenv('RDS_HOST') is None:
    print(f"⚠️ 警告：未配置数据库环境变量 {missing}")
    print("请设置环境变量或创建 .env 文件")
    print("示例：export RDS_HOST=your-host.com")

def connect_db():
    """连接数据库"""
    return pymysql.connect(**DB_CONFIG)

# ==================== 数据查询 ====================
def query_daily_metrics(conn, start_date, end_date):
    """查询每日核心指标"""
    query = f"""
    SELECT 
        DATE(event_time) as date,
        event_type,
        COUNT(*) as cnt,
        COUNT(DISTINCT user_id) as uv
    FROM ecommerce_events
    WHERE event_time >= '{start_date}' AND event_time < '{end_date}'
    GROUP BY DATE(event_time), event_type
    ORDER BY date, event_type
    """
    return pd.read_sql(query, conn)

# ==================== 特征工程 ====================
def create_features(df):
    """
    创建核心特征（20 个精简特征）
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # === 1. 时间特征（3 个） ===
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    # === 2. 美国节假日（4 个） ===
    df['is_veterans_day'] = (df['date'] == pd.Timestamp('2019-11-11')).astype(int)
    df['is_thanksgiving'] = (df['date'] == pd.Timestamp('2019-11-28')).astype(int)
    df['is_black_friday'] = (df['date'] == pd.Timestamp('2019-11-29')).astype(int)
    df['is_cyber_monday'] = (df['date'] == pd.Timestamp('2019-12-02')).astype(int)
    
    # === 3. 滞后特征（4 个） ===
    for lag_col in ['lag_1_pv', 'lag_7_pv']:
        target = 'pv'
        lag = int(lag_col.split('_')[1])
        df[lag_col] = df[target].shift(lag)
    
    for lag_col in ['lag_1_uv', 'lag_7_uv']:
        target = 'pv_uv'  # 修正：实际列名是 pv_uv
        lag = int(lag_col.split('_')[1])
        df[lag_col] = df[target].shift(lag)
    
    # === 4. 滚动统计（4 个） ===
    for window in [7]:
        df[f'ma_{window}_pv'] = df['pv'].rolling(window=window).mean()
        df[f'ma_{window}_uv'] = df['pv_uv'].rolling(window=window).mean()  # 修正
        df[f'median_{window}_pv'] = df['pv'].rolling(window=window).median()
        df[f'std_{window}_pv'] = df['pv'].rolling(window=window).std()
    
    # === 5. 增长率（2 个） ===
    df['mom_growth_pv'] = df['pv'].pct_change(periods=1)
    df['wow_growth_pv'] = df['pv'].pct_change(periods=7)
    
    # === 6. 转化率（2 个） ===
    df['uv_purchase_rate'] = df['purchase_uv'] / df['pv_uv'].replace(0, 1)  # 修正列名
    df['cart_to_purchase_rate'] = df['purchase_uv'] / df['cart_uv'].replace(0, 1)
    
    # === 7. 购物季（1 个） ===
    df['is_christmas_season'] = (df['date'] >= pd.Timestamp('2019-11-15')).astype(int)
    
    return df

def prepare_data(df, train_df_last=None, is_train=True):
    """
    准备训练/验证数据
    
    Args:
        df: 当前数据集
        train_df_last: 训练集的最后一行（用于填充验证集的滞后特征）
        is_train: 是否是训练集
    """
    df = create_features(df)
    
    if is_train:
        # 训练集：用前向填充
        df = df.fillna(method='ffill').fillna(0)
        return df, df.iloc[-1]  # 返回最后一行用于验证集
    else:
        # 验证集：用训练集的最后一行填充第一行的 NaN
        if train_df_last is not None:
            # 将训练集最后一行添加到验证集开头，计算特征后再去掉
            combined = pd.concat([train_df_last.to_frame().T, df], ignore_index=True)
            combined = combined.fillna(method='ffill').fillna(0)
            df = combined.iloc[1:].reset_index(drop=True)  # 去掉添加的行
        else:
            df = df.fillna(method='ffill').fillna(0)
        return df, None

# ==================== 模型训练 ====================
def train_models(X_train, y_pv_train, y_purchase_train):
    """训练多个模型"""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        # 训练 PV 预测模型
        model_pv = model.__class__(**model.get_params())
        model_pv.fit(X_train, y_pv_train)
        
        # 训练购买量预测模型
        model_purchase = model.__class__(**model.get_params())
        model_purchase.fit(X_train, y_purchase_train)
        
        trained_models[name] = {'pv': model_pv, 'purchase': model_purchase}
        
        # 交叉验证（使用兼容的 scoring）
        loo = LeaveOneOut()
        try:
            cv_scores_pv = cross_val_score(model_pv, X_train, y_pv_train, cv=loo, scoring='neg_mean_absolute_percentage_error')
        except:
            # sklearn 旧版本不支持，手动计算
            cv_scores_pv = []
            for train_idx, test_idx in loo.split(X_train):
                model_tmp = model_pv.__class__(**model_pv.get_params())
                model_tmp.fit(X_train.iloc[train_idx], y_pv_train.iloc[train_idx])
                y_pred = model_tmp.predict(X_train.iloc[test_idx])
                y_true = y_pv_train.iloc[test_idx]
                mape = mean_absolute_percentage_error(y_true, y_pred)
                cv_scores_pv.append(-mape)
            cv_scores_pv = np.array(cv_scores_pv)
        
        results[name] = {
            'cv_mape': -cv_scores_pv.mean() * 100,
            'cv_std': cv_scores_pv.std() * 100,
        }
    
    return trained_models, results

def evaluate_models(trained_models, X_val, y_pv_val, y_purchase_val):
    """评估模型"""
    results = {}
    
    for name, models in trained_models.items():
        # PV 预测
        pv_pred = models['pv'].predict(X_val)
        pv_mape = mean_absolute_percentage_error(y_pv_val, pv_pred) * 100
        pv_rmse = np.sqrt(mean_squared_error(y_pv_val, pv_pred))
        pv_r2 = r2_score(y_pv_val, pv_pred)
        
        # 购买量预测
        purchase_pred = models['purchase'].predict(X_val)
        purchase_mape = mean_absolute_percentage_error(y_purchase_val, purchase_pred) * 100
        purchase_rmse = np.sqrt(mean_squared_error(y_purchase_val, purchase_pred))
        purchase_r2 = r2_score(y_purchase_val, purchase_pred)
        
        results[name] = {
            'pv_mape': pv_mape,
            'pv_rmse': pv_rmse,
            'pv_r2': pv_r2,
            'purchase_mape': purchase_mape,
            'purchase_rmse': purchase_rmse,
            'purchase_r2': purchase_r2,
            'pv_predictions': pv_pred,
            'purchase_predictions': purchase_pred,
        }
    
    return results

# ==================== 黑五分析 ====================
def analyze_black_friday(df, predictions, actuals, dates):
    """分析黑五期间预测效果"""
    bf_mask = (dates >= pd.Timestamp('2019-11-22')) & (dates <= pd.Timestamp('2019-11-30'))
    
    if bf_mask.sum() == 0:
        # 尝试圣诞节期间
        bf_mask = (dates >= pd.Timestamp('2019-12-20')) & (dates <= pd.Timestamp('2019-12-31'))
        event_name = "圣诞节"
    else:
        event_name = "黑五"
    
    if bf_mask.sum() == 0:
        return {'event': event_name, 'bf_mape': 0, 'bf_days': 0}
    
    bf_pred = predictions[bf_mask]
    bf_actual = actuals[bf_mask]
    bf_mape = mean_absolute_percentage_error(bf_actual, bf_pred) * 100
    
    return {
        'event': event_name,
        'bf_mape': bf_mape,
        'bf_days': bf_mask.sum(),
        'bf_pred_avg': bf_pred.mean(),
        'bf_actual_avg': bf_actual.mean(),
    }

# ==================== 结果保存 ====================
def save_validation_history(validation_result):
    """保存验证历史到 JSON 文件"""
    history_file = '/root/.openclaw/workspace/skills/ecommerce-predictor/validation_history.json'
    
    # 读取现有历史
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = {'versions': []}
    
    # 添加新结果
    history['versions'].append(validation_result)
    
    # 保存
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"✅ 验证结果已保存到：{history_file}")

def generate_report(validation_result, output_path):
    """生成验证报告"""
    from datetime import datetime
    
    version = validation_result['version']
    train_start = validation_result['train_start']
    train_end = validation_result['train_end']
    val_start = validation_result['val_start']
    val_end = validation_result['val_end']
    train_days = validation_result['train_days']
    val_days = validation_result['val_days']
    best_model = validation_result['best_model']
    best_metrics = validation_result['best_metrics']
    bf_analysis = validation_result['black_friday_analysis']
    
    report = f"""# 🧚‍♀️ 电商用户行为预测模型 - 验证报告 {version}

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> **版本**: {version}
> **作者**: E-Commerce Predictor Team

---

## 📊 数据划分

| 数据集 | 时间范围 | 天数 |
|--------|---------|------|
| **训练集** | {train_start} ~ {train_end} | {train_days} 天 |
| **验证集** | {val_start} ~ {val_end} | {val_days} 天 |

---

## 🏆 最佳模型：{best_model}

### PV 预测性能
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| MAPE | {best_metrics['pv_mape']:.2f}% | <20% | {"✅" if best_metrics['pv_mape'] < 20 else "⚠️"} |
| RMSE | {best_metrics['pv_rmse']:.2f} | - | - |
| R² | {best_metrics['pv_r2']:.4f} | >0.5 | {"✅" if best_metrics['pv_r2'] > 0.5 else "⚠️"} |

### 购买量预测性能
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| MAPE | {best_metrics['purchase_mape']:.2f}% | <15% | {"✅" if best_metrics['purchase_mape'] < 15 else "⚠️"} |
| RMSE | {best_metrics['purchase_rmse']:.2f} | - | - |
| R² | {best_metrics['purchase_r2']:.4f} | >0.5 | {"✅" if best_metrics['purchase_r2'] > 0.5 else "⚠️"} |

---

## 🎉 节假日专项分析

**分析期间**: {bf_analysis.get('event', '未知')} ({bf_analysis.get('bf_days', 0)} 天)
- **预测误差 (MAPE)**: {bf_analysis.get('bf_mape', 0):.2f}%
- **预测均值**: {bf_analysis.get('bf_pred_avg', 0):.2f}
- **实际均值**: {bf_analysis.get('bf_actual_avg', 0):.2f}

---

## 📈 模型进化历史

| 版本 | 训练数据 | 验证数据 | PV MAPE | 购买量 MAPE | 节假日误差 |
|------|---------|---------|---------|------------|-----------|
"""
    
    # 读取历史数据
    history_file = '/root/.openclaw/workspace/skills/ecommerce-predictor/validation_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        for v in history['versions']:
            report += f"| {v['version']} | {v['train_start']}~{v['train_end']} | {v['val_start']}~{v['val_end']} | {v['best_metrics']['pv_mape']:.2f}% | {v['best_metrics']['purchase_mape']:.2f}% | {v['black_friday_analysis'].get('bf_mape', 0):.2f}% |\n"
    
    report += f"""
---

## 💡 经验总结

### 本次验证的洞察
1. 训练数据增加到 {train_days} 天，模型稳定性{"提升" if train_days > 31 else "待观察"}
2. {bf_analysis.get('event', '节假日')}期间预测误差为 {bf_analysis.get('bf_mape', 0):.2f}%
3. 最佳模型是 **{best_model}**，说明数据特征{"适合线性模型" if 'Lasso' in best_model or 'Ridge' in best_model else "需要非线性模型"}

### 下一步建议
- [ ] 继续导入新月份数据
- [ ] 观察模型性能是否持续提升
- [ ] 如有过拟合迹象，考虑增加正则化

---

*自动进化报告 - E-Commerce Predictor Team*
"""
    
    # 保存报告
    report_path = os.path.join(output_path, f'validation_report_{version}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 验证报告已保存到：{report_path}")
    return report_path

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='🧚‍♀️ 电商预测模型 - 自动进化脚本')
    parser.add_argument('--train-end', type=str, required=True, help='训练集结束日期 (如：2019-11-01)')
    parser.add_argument('--val-end', type=str, required=True, help='验证集结束日期 (如：2019-12-01)')
    parser.add_argument('--train-start', type=str, default='2019-10-01', help='训练集开始日期 (默认：2019-10-01)')
    parser.add_argument('--output', type=str, default='/root/.openclaw/workspace/skills/ecommerce-predictor/results', help='输出目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧚‍♀️ 电商用户行为预测模型 - 自动进化 v5.0")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 版本号
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 连接数据库
    print("\n📊 步骤 1: 连接数据库...")
    conn = connect_db()
    print("✅ 数据库连接成功")
    
    # 提取数据
    print(f"\n📊 步骤 2: 提取数据...")
    print(f"   - 训练集：{args.train_start} ~ {args.train_end} (不含)")
    print(f"   - 验证集：{args.train_end} ~ {args.val_end} (不含)")
    
    train_data = query_daily_metrics(conn, args.train_start, args.train_end)
    val_data = query_daily_metrics(conn, args.train_end, args.val_end)
    
    print(f"✅ 训练集：{len(train_data)} 条记录")
    print(f"✅ 验证集：{len(val_data)} 条记录")
    
    # 数据预处理
    print("\n📊 步骤 3: 数据预处理...")
    
    # 转换为宽表
    train_pivot = train_data.pivot_table(
        index='date', 
        columns='event_type',
        values=['cnt', 'uv'],
        aggfunc='sum'
    ).fillna(0)
    train_pivot.columns = ['_'.join(col).strip() for col in train_pivot.columns]
    train_pivot = train_pivot.reset_index()
    
    val_pivot = val_data.pivot_table(
        index='date',
        columns='event_type', 
        values=['cnt', 'uv'],
        aggfunc='sum'
    ).fillna(0)
    val_pivot.columns = ['_'.join(col).strip() for col in val_pivot.columns]
    val_pivot = val_pivot.reset_index()
    
    # 重命名列（适配实际数据库结构）
    for df in [train_pivot, val_pivot]:
        if 'cnt_purchase' in df.columns:
            df['purchase'] = df['cnt_purchase']
        if 'cnt_view' in df.columns:
            df['pv'] = df['cnt_view']  # view = pv
        if 'cnt_cart' in df.columns:
            df['cart'] = df['cnt_cart']
        if 'uv_purchase' in df.columns:
            df['purchase_uv'] = df['uv_purchase']
        if 'uv_view' in df.columns:
            df['pv_uv'] = df['uv_view']  # view = pv
        if 'uv_cart' in df.columns:
            df['cart_uv'] = df['uv_cart']
    
    # 填充缺失列
    for col in ['purchase', 'pv', 'cart', 'pv_uv', 'cart_uv']:
        if col not in train_pivot.columns:
            train_pivot[col] = 0
        if col not in val_pivot.columns:
            val_pivot[col] = 0
    
    # 准备特征（关键：先合并再计算特征，避免数据泄露）
    # 1. 合并训练集和验证集
    train_pivot['is_val'] = False
    val_pivot['is_val'] = True
    combined = pd.concat([train_pivot, val_pivot], ignore_index=True)
    
    # 2. 重命名列
    if 'cnt_purchase' in combined.columns: combined['purchase'] = combined['cnt_purchase']
    if 'cnt_view' in combined.columns: combined['pv'] = combined['cnt_view']
    if 'cnt_cart' in combined.columns: combined['cart'] = combined['cnt_cart']
    if 'uv_purchase' in combined.columns: combined['purchase_uv'] = combined['uv_purchase']
    if 'uv_view' in combined.columns: combined['pv_uv'] = combined['uv_view']
    if 'uv_cart' in combined.columns: combined['cart_uv'] = combined['uv_cart']
    
    # 3. 填充缺失列
    for col in ['purchase', 'pv', 'cart', 'purchase_uv', 'pv_uv', 'cart_uv']:
        if col not in combined.columns:
            combined[col] = 0
    
    # 4. 计算特征（用整个数据集，这样滞后特征是正确的）
    combined_featured = create_features(combined)
    combined_featured = combined_featured.fillna(method='ffill').fillna(0)
    
    # 5. 拆分回训练集和验证集
    train_featured = combined_featured[combined_featured['is_val'] == False].copy()
    val_featured = combined_featured[combined_featured['is_val'] == True].copy()
    
    # 6. 提取特征和标签（排除原始列和目标列）
    exclude_cols = ['date', 'pv', 'purchase', 'uv', 'cart', 'is_val', 
                    'cnt_cart', 'cnt_purchase', 'cnt_view',  # 原始计数列
                    'uv_cart', 'uv_purchase', 'uv_view',  # 原始 UV 列
                    'purchase_uv', 'pv_uv', 'cart_uv']  # 重命名后的 UV 列
    feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
    X_train = train_featured[feature_cols]
    y_pv_train = train_featured['pv']
    y_purchase_train = train_featured['purchase']
    
    X_val = val_featured[feature_cols]
    y_pv_val = val_featured['pv']
    y_purchase_val = val_featured['purchase']
    
    val_df = val_featured  # 保存用于后续分析
    
    print(f"✅ 特征数量：{len(feature_cols)}")
    print(f"✅ 训练样本：{len(X_train)}")
    print(f"✅ 验证样本：{len(X_val)}")
    
    # 训练模型
    print("\n📊 步骤 4: 训练模型...")
    trained_models, cv_results = train_models(X_train, y_pv_train, y_purchase_train)
    
    print("\n📊 交叉验证结果 (LOO):")
    for name, metrics in cv_results.items():
        print(f"   {name}: MAPE={metrics['cv_mape']:.2f}% (±{metrics['cv_std']:.2f}%)")
    
    # 评估模型
    print("\n📊 步骤 5: 验证集评估...")
    val_results = evaluate_models(trained_models, X_val, y_pv_val, y_purchase_val)
    
    print("\n验证集性能:")
    for name, metrics in val_results.items():
        print(f"   {name}: PV MAPE={metrics['pv_mape']:.2f}%, Purchase MAPE={metrics['purchase_mape']:.2f}%")
    
    # 选择最佳模型
    best_model = min(val_results.keys(), key=lambda x: val_results[x]['pv_mape'])
    best_metrics = val_results[best_model]
    
    print(f"\n🏆 最佳模型：{best_model}")
    print(f"   PV MAPE: {best_metrics['pv_mape']:.2f}%")
    print(f"   Purchase MAPE: {best_metrics['purchase_mape']:.2f}%")
    
    # 黑五/节假日分析
    print("\n📊 步骤 6: 节假日专项分析...")
    bf_analysis = analyze_black_friday(
        val_df,
        best_metrics['pv_predictions'],
        y_pv_val.values,
        val_df['date'].values
    )
    print(f"   {bf_analysis['event']}期间误差：{bf_analysis['bf_mape']:.2f}%")
    
    # 保存结果
    print("\n📊 步骤 7: 保存结果...")
    
    validation_result = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'train_start': args.train_start,
        'train_end': args.train_end,
        'val_start': args.train_end,  # 修正：验证集开始 = 训练集结束
        'val_end': args.val_end,
        'train_days': len(train_pivot),
        'val_days': len(val_pivot),
        'best_model': best_model,
        'best_metrics': {
            'pv_mape': float(best_metrics['pv_mape']),
            'pv_rmse': float(best_metrics['pv_rmse']),
            'pv_r2': float(best_metrics['pv_r2']),
            'purchase_mape': float(best_metrics['purchase_mape']),
            'purchase_rmse': float(best_metrics['purchase_rmse']),
            'purchase_r2': float(best_metrics['purchase_r2']),
        },
        'black_friday_analysis': bf_analysis,
        'all_models': {k: {'pv_mape': v['pv_mape'], 'purchase_mape': v['purchase_mape']} 
                      for k, v in val_results.items()}
    }
    
    # 保存验证历史
    save_validation_history(validation_result)
    
    # 生成报告
    generate_report(validation_result, args.output)
    
    # 生成图表
    if HAS_PLOT and len(val_df) > 0:
        print("\n📊 步骤 8: 生成图表...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # PV 预测对比
        dates = val_df['date'].values
        axes[0].plot(dates, y_pv_val.values, 'b-', label='实际 PV', linewidth=2)
        axes[0].plot(dates, best_metrics['pv_predictions'], 'r--', label=f'{best_model} 预测', linewidth=2)
        axes[0].set_title(f'PV 预测对比 - {version}', fontsize=14)
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('PV')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # 购买量预测对比
        axes[1].plot(dates, y_purchase_val.values, 'b-', label='实际购买量', linewidth=2)
        axes[1].plot(dates, best_metrics['purchase_predictions'], 'r--', label=f'{best_model} 预测', linewidth=2)
        axes[1].set_title('购买量预测对比', fontsize=14)
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('购买量')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        chart_path = os.path.join(args.output, f'prediction_{version}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 图表已保存到：{chart_path}")
    
    # 关闭数据库连接
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ 自动进化完成！")
    print("=" * 60)
    print(f"\n📁 输出文件:")
    print(f"   - 验证历史：/root/.openclaw/workspace/skills/ecommerce-predictor/validation_history.json")
    print(f"   - 验证报告：{args.output}/validation_report_{version}.md")
    if HAS_PLOT:
        print(f"   - 预测图表：{args.output}/prediction_{version}.png")
    
    print(f"\n💡 提示：下次导入新数据后，运行:")
    print(f"   python ecommerce_predictor_auto.py --train-end {args.val_end} --val-end <下个月>")

if __name__ == '__main__':
    main()
