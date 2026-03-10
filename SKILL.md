---
name: ecommerce-predictor
description: "电商用户行为时间序列预测。GradientBoosting/Lasso 预测 PV、UV、购买量。Use when: 预测、时间序列、趋势分析、销量预测、电商预测。NOT for: 实时风控、中国电商双 11/618、非时序分类问题。"
metadata:
  openclaw:
    emoji: "📈"
    requires:
      bins: ["python3"]
---

# 电商预测 Skill

基于美国电商历史数据进行时间序列预测，支持滚动式训练和持续验证。

---

## 🚀 快速开始

### 最常用命令（滚动预测）

```bash
# 标准滚动预测 - 推荐
python3 {baseDir}/ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01

# 小样本快速测试（<100 天）
python3 {baseDir}/ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2019-10-31 \
  --val-end 2019-11-30
```

### 购买量专项优化

```bash
# 购买量预测需要更长训练窗口（150 天+）
python3 {baseDir}/optimize_purchase_v3.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

---

## 📊 预测流程

### Step 1: 配置数据库连接

**方式一：系统环境变量（推荐）**
```bash
export RDS_HOST=<只读实例外网地址>
export RDS_USER=<数据库账号>
export RDS_PASSWORD=<密码>
export RDS_DATABASE=<数据库名>
```

**方式二：`.env` 文件**

创建 `{baseDir}/.env` 文件，脚本会自动加载：
```bash
RDS_HOST=<只读实例外网地址>
RDS_PORT=3306
RDS_USER=<数据库账号>
RDS_PASSWORD=<密码>
RDS_DATABASE=<数据库名>
```

**验证配置：**
```bash
# 检查环境变量是否生效
echo $RDS_HOST

# 或运行测试查询
python3 {baseDir}/ecommerce_predictor_auto.py --help
```

### Step 2: 提取数据

```python
import pymysql
import pandas as pd
from dotenv import load_dotenv
import os

# 加载 .env 文件（如果存在）
load_dotenv('{baseDir}/.env')

conn = pymysql.connect(
    host=os.getenv('RDS_HOST'),
    port=3306,
    user=os.getenv('RDS_USER'),
    password=os.getenv('RDS_PASSWORD'),
    database=os.getenv('RDS_DATABASE')
)
```

**依赖安装：**
```bash
pip install python-dotenv
```

query = """
SELECT 
    DATE(event_time) as date,
    event_type,
    COUNT(*) as cnt,
    COUNT(DISTINCT user_id) as uv
FROM ecommerce_events
WHERE event_time >= %s AND event_time < %s
GROUP BY DATE(event_time), event_type
ORDER BY date, event_type
"""

df = pd.read_sql(query, conn, params=[train_start, train_end])
```

### Step 3: 建立预测模型

**模型选择规则：**
- 训练天数 < 100 → 用 `Lasso(alpha=0.01)`
- 训练天数 ≥ 100 → 用 `GradientBoostingRegressor(n_estimators=50, max_depth=3)`

**代码示例：**
```python
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor

if len(X_train) < 100:
    model = Lasso(alpha=0.01, max_iter=10000, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

model.fit(X_train, y_train)
```

**对比所有模型：**
```bash
python3 {baseDir}/ecommerce_predictor_v4.py --compare-models
```

### Step 4: 验证与评估

```python
from sklearn.metrics import mean_absolute_percentage_error, r2_score

y_pred = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred) * 100
r2 = r2_score(y_val, y_pred)

print(f'PV MAPE: {mape:.2f}%')
print(f'R²: {r2:.4f}')
```

**LOO 交叉验证（小样本必备）：**
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_absolute_percentage_error')
print(f'LOO CV MAPE: {(-scores.mean()):.2f}% (+/- {scores.std()*2:.2f}%)')
```

### Step 5: 生成报告

自动进化脚本会自动生成：
- `results/validation_report_v*.md` - Markdown 验证报告
- `results/prediction_v*.png` - 预测趋势图
- `validation_history.json` - 版本性能历史记录

---

## 🔧 特征工程（20 个核心特征）

**时间特征（3 个）：** `day_of_week`, `day_of_month`, `is_weekend`

**美国节假日（4 个）：** `is_veterans_day`, `is_thanksgiving`, `is_black_friday`, `is_cyber_monday`

**滞后特征（4 个）：** `lag_1_pv`, `lag_7_pv`, `lag_1_uv`, `lag_7_uv`

**滚动统计（4 个）：** `ma_7_pv`, `ma_7_uv`, `median_7_pv`, `std_7_pv`

**增长率（2 个）：** `mom_growth_pv`, `wow_growth_pv`

**转化率（2 个）：** `uv_purchase`, `cart_to_purchase_rate`

**购物季（1 个）：** `is_christmas_season`

---

## 🏆 当前最佳配置

| 预测目标 | 推荐版本 | 训练窗口 | 模型 | 预期 MAPE |
|---------|---------|---------|------|----------|
| **PV 预测** | v6.0/v7.0 | 92-123 天 | GradientBoosting | **2.73%** |
| **购买量预测** | v9.0 | 183 天 | GradientBoosting | **12.65%** |
| **小样本** | v4.0 | 31 天 | Lasso | 17.50% |

---

## ⚠️ 注意事项

1. **地区匹配** - 这是🇺🇸美国电商数据，节假日用黑五/感恩节，**不是中国双 11**
2. **特殊时期** - 2020 年 3 月起为美国疫情期，数据模式突变，建议单独标注
3. **特征/样本比** - 保持 < 1.0（如 31 样本用 20 特征）
4. **模型选择** - <100 天用 Lasso，≥100 天用 GradientBoosting
5. **滚动验证** - 每月导入新数据后重新验证

---

## 🔍 故障排查

| 问题 | 解决方案 |
|------|---------|
| **预测误差过大** | 检查特征/样本比是否 > 1.0；增加训练数据；用 LOO CV 验证稳定性 |
| **模型不收敛** | `Lasso(max_iter=10000)` 增加迭代次数 |
| **节假日因子无效** | 确认节假日日期正确；检查训练集是否包含节假日数据 |
| **数据库连接超时** | 检查白名单配置；增加 `connect_timeout=30` |
| **性能突然下降** | 检查验证集是否包含特殊时期数据（疫情/大促） |
| **环境变量未生效** | 运行 `echo $RDS_HOST` 检查；或确认 `.env` 文件路径正确 |
| **找不到脚本** | 使用 `{baseDir}/脚本名.py` 确保路径正确 |

---

## ⚠️ 环境变量说明

**重要：** 本 Skill 的 `requires.env` 未设置，因为配置方式灵活：

| 方式 | 配置方法 | 适用场景 |
|------|---------|---------|
| **系统环境变量** | `export RDS_HOST=xxx` | 服务器部署、CI/CD |
| **`.env` 文件** | `{baseDir}/.env` | 本地开发、多环境切换 |

脚本会自动按以下优先级加载：
1. 系统环境变量（`process.env`）
2. `{baseDir}/.env` 文件（通过 `python-dotenv`）

**验证配置：**
```bash
# 检查环境变量
echo $RDS_HOST

# 测试数据库连接
python3 {baseDir}/ecommerce_predictor_auto.py --help
```

---

## 📁 参考文件

- **版本历史：** `{baseDir}/HISTORY.md` - v1.0→v9.0 完整演进记录
- **数据集说明：** `{baseDir}/DATASET.md` - 数据来源、节假日、特殊时期
- **自动进化指南：** `{baseDir}/AUTO_EVOLUTION_GUIDE.md` - 滚动训练教程

---

## 📚 输出指标说明

| 指标 | 说明 | 优秀标准 |
|------|------|---------|
| **MAPE** | 平均绝对百分比误差 | <20% 达标，<10% 优秀 |
| **R²** | 决定系数 | >0.5 达标，>0.7 优秀 |
| **LOO CV MAPE** | 交叉验证 MAPE | 稳定性指标，越低越稳 |

---

**版本：** v9.0  
**最后更新：** 2026-03-08  
**数据周期：** 2019-10-01 ~ 2020-04-30（7 个月）
