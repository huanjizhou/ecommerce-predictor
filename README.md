# Ecommerce Predictor

[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-blue?logo=openclaw)](https://openclaw.com)
[![Python Version](https://img.shields.io/badge/Python-3.6+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MAPE](https://img.shields.io/badge/PV%20MAPE-2.73%25-brightgreen)]()
[![Data](https://img.shields.io/badge/Data-REES46%20(US)-orange)]()

> **Time series forecasting for ecommerce user behavior** — Predict PV, UV, and purchase volume using GradientBoosting/Lasso models with rolling training and continuous validation.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Configuration](#-configuration)
- [Feature Engineering](#-feature-engineering)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **🎯 Multi-Metric Prediction** — Forecast daily PV (page views), UV (unique visitors), and purchase volume
- **🤖 Auto Model Selection** — Intelligently chooses between Lasso (<100 days) and GradientBoosting (≥100 days)
- **📈 Rolling Training** — Continuous validation with expandable training windows
- **🇺🇸 US Holiday Integration** — Veterans Day, Thanksgiving, Black Friday, Cyber Monday, Christmas
- **🔍 LOO Cross-Validation** — Leave-One-Out CV ensures model stability for small samples
- **📊 Auto Reporting** — Generates Markdown reports, prediction charts, and performance history
- **🚀 DuckDB Acceleration** — 1000x+ query speedup with DuckDB backend
- **🔧 Flexible Configuration** — Support for environment variables or `.env` files

---

## 🚀 Quick Start

### Standard Rolling Prediction (Recommended)

```bash
python3 ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

### Quick Test (Small Sample <100 Days)

```bash
python3 ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2019-10-31 \
  --val-end 2019-11-30
```

### Purchase Volume Optimization

```bash
# Purchase prediction requires longer training window (150+ days)
python3 optimize_purchase_v3.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

---

## 📦 Installation

### Prerequisites

- Python 3.6+
- MySQL/MariaDB or DuckDB
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost pymysql python-dotenv matplotlib
```

### Clone or Download

```bash
# If using git
git clone <repository-url>
cd ecommerce-predictor

# Or copy scripts to your workspace
cp *.py /your/workspace/
```

---

## 🏆 Model Performance

### Current Best Configurations

| Target | Recommended Version | Training Window | Model | MAPE | Use Case |
|--------|-------------------|-----------------|-------|------|----------|
| **PV Prediction** | v3.0 / v4.0 | 92-123 days | GradientBoosting | **2.73%** | Normal periods |
| **Purchase Volume** | v6.0 | 183 days | GradientBoosting | **12.65%** | Long-term training |
| **Holiday Prediction** | v1.0 | 31 days | Exponential Smoothing | 11.39% (Black Friday) | Small samples |
| **Quick Baseline** | v1.0 | 31 days | Exponential Smoothing | 26.92% | Insufficient data |

### Version Performance History

| 版本 | 训练天数 | 最佳模型 | PV 误差 | 购买量误差 | 发生了什么 |
|------|---------|---------|---------|-----------|-----------|
| v1.0 | 31 天 | 指数平滑 | 26.92% | 11.39% | 基线模型，黑五误差 45% |
| v2.0 | 61 天 | RandomForest | 8.15% | 23.17% | ✅ 训练数据翻倍 |
| v3.0 | 92 天 | GradientBoosting | 2.73% | 42.28% | ✅ PV 预测历史最优 |
| v4.0 | 123 天 | GradientBoosting | 2.73% | 42.28% | ✅ 正式 v1 |
| v5.0 | 152 天 | Ridge | 29.99% | 36.99% | ⚠️ 撞上美国疫情爆发 |
| v6.0 | 183 天 | GradientBoosting | 10.03% | 12.65% | ✅ 购买量预测最优 |

---

## 📊 Dataset

### Source Information

- **Dataset Name:** eCommerce behavior data from multi category store
- **Platform:** [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- **Provider:** REES46 Marketing Platform (New York, USA)
- **Company:** Founded 2013, Founder: Michael Kechinov
- **Website:** [rees46.com](https://rees46.com/)

### Data Statistics

| Metric | Value |
|--------|-------|
| **Total Events** | ~110 million |
| **Unique Users** | ~5.3 million |
| **Unique Products** | ~200,000 |
| **Categories** | ~700 |
| **Region** | 🇺🇸 United States |
| **Time Range** | 2019-10-01 ~ 2020-04-30 (7 months) |

### Event Types

| Type | Description | Ratio |
|------|-------------|-------|
| `view` | Product view | ~90% |
| `cart` | Add to cart | ~7% |
| `purchase` | Purchase | ~3% |

### US Holiday Factors

| Date | Holiday | Impact |
|------|---------|--------|
| November 11 | Veterans Day | Federal holiday promotions |
| November 28 | Thanksgiving | Shopping season starts |
| November 29 | Black Friday | Biggest shopping event |
| December 2 | Cyber Monday | E-commerce exclusive deals |
| December 25 | Christmas | Christmas shopping season |
| January 1 | New Year's Day | New Year promotions |

> **Important:** This is 🇺🇸 **US e-commerce data**. Use Black Friday/Thanksgiving, **NOT China's Double 11/618**.

### Special Periods

| Period | Event | Impact |
|--------|-------|--------|
| 2020-03 (mid-late) | 🦠 US Pandemic Outbreak | Stay-at-home orders, e-commerce surge, pattern shift |
| 2020-04 | Full Lockdown | Online shopping peak |

**Recommendation:** Data from March 2020 onwards represents pandemic period — mark separately or exclude from normal-period models.

---

## ⚙️ Configuration

### Database Connection

#### Method 1: Environment Variables (Recommended)

```bash
export RDS_HOST=<read-replica-public-endpoint>
export RDS_USER=<db-username>
export RDS_PASSWORD=<password>
export RDS_DATABASE=<db-name>
export RDS_PORT=3306
```

#### Method 2: `.env` File

Create `{baseDir}/.env` file (automatically loaded by scripts):

```bash
RDS_HOST=<read-replica-public-endpoint>
RDS_PORT=3306
RDS_USER=<db-username>
RDS_PASSWORD=<password>
RDS_DATABASE=<db-name>
```

**Dependencies:**
```bash
pip install python-dotenv
```

### Verification

```bash
# Check environment variables
echo $RDS_HOST

# Test database connection
python3 ecommerce_predictor_auto.py --help
```

### Database Schema

```sql
CREATE TABLE ecommerce_events (
    event_id BIGINT PRIMARY KEY,
    event_time DATETIME,
    event_type ENUM('view', 'cart', 'purchase'),
    user_id BIGINT,
    product_id BIGINT,
    category_id INT,
    category_code VARCHAR(50),
    brand VARCHAR(50),
    price DECIMAL(10,2),
    user_session VARCHAR(50)
);
```

### Daily Aggregation Query

```sql
SELECT 
    DATE(event_time) as date,
    event_type,
    COUNT(*) as cnt,
    COUNT(DISTINCT user_id) as uv
FROM ecommerce_events
WHERE event_time >= %s AND event_time < %s
GROUP BY DATE(event_time), event_type
ORDER BY date, event_type;
```

---

## 🔧 Feature Engineering

### 20 Core Features

#### Time Features (3)
- `day_of_week` — Day of week (0-6)
- `day_of_month` — Day of month (1-31)
- `is_weekend` — Weekend flag (Saturday/Sunday)

#### US Holidays (4)
- `is_veterans_day` — Veterans Day (Nov 11)
- `is_thanksgiving` — Thanksgiving (4th Thursday of November)
- `is_black_friday` — Black Friday (day after Thanksgiving)
- `is_cyber_monday` — Cyber Monday (Monday after Thanksgiving)

#### Lag Features (4)
- `lag_1_pv` — PV lag 1 day
- `lag_7_pv` — PV lag 7 days
- `lag_1_uv` — UV lag 1 day
- `lag_7_uv` — UV lag 7 days

#### Rolling Statistics (4)
- `ma_7_pv` — 7-day moving average PV
- `ma_7_uv` — 7-day moving average UV
- `median_7_pv` — 7-day median PV
- `std_7_pv` — 7-day standard deviation PV

#### Growth Rates (2)
- `mom_growth_pv` — Month-over-month PV growth
- `wow_growth_pv` — Week-over-week PV growth

#### Conversion Rates (2)
- `uv_purchase` — Purchase/UV conversion rate
- `cart_to_purchase_rate` — Purchase/Cart conversion rate

#### Shopping Season (1)
- `is_christmas_season` — Christmas season flag (Dec 1-25)

### Feature Selection Guidelines

- **Feature/Sample Ratio < 1.0** — e.g., 31 samples use 20 features (0.65 ratio)
- **Avoid Overfitting** — v3.0 had 51 features/31 samples (1.65 ratio) → R²=-0.30
- **Start Simple** — Begin with core features, add complexity only if needed

---

## 📚 Best Practices

### Recommended Model (v6.0 for purchase, v3.0/v4.0 for PV)

| Training Days | Recommended Model | Reason |
|---------------|------------------|--------|
| < 100 | Lasso (alpha=0.01) | Prevents overfitting on small samples |
| ≥ 100 | GradientBoosting (n_estimators=50, max_depth=3) | Captures non-linear patterns |

### Validation Workflow

1. **Configure database connection** (env vars or `.env`)
2. **Extract data** with proper date ranges
3. **Build model** (auto-selected based on training days)
4. **Validate** with MAPE and R² metrics
5. **Cross-validate** with LOO CV for small samples
6. **Generate reports** (Markdown + charts + history)

### Performance Metrics

| Metric | Description | Good Threshold | Excellent |
|--------|-------------|----------------|-----------|
| **MAPE** | Mean Absolute Percentage Error | <20% | <10% |
| **R²** | Coefficient of Determination | >0.5 | >0.7 |
| **LOO CV MAPE** | Cross-validation stability | Lower is better | <5% |

### Lessons Learned

#### ✅ Success Patterns

1. **Features ≠ Better** — v3.0's 51 features caused severe overfitting
2. **Simple Models for Small Samples** — v4.0's Lasso worked best on 31 samples
3. **Cross-Validation is Essential** — LOO CV confirms model stability
4. **Match Holidays to Region** — US uses Black Friday/Thanksgiving, not Double 11
5. **More Data = More Stability** — Purchase MAPE dropped from 42% to 13% with more data
6. **Rolling Validation Catches Anomalies** — v8.0's pandemic data caused performance drop
7. **Model Selection Matters** — <100 days: Lasso, ≥100 days: GradientBoosting

#### ❌ Common Pitfalls

1. **Assumption Bias** — Assuming November = Double 11 (it's US data!)
2. **Too Many Features** — 51 features / 31 samples = 1.65 ratio → overfitting
3. **Complex Models Prematurely** — XGBoost underperformed Lasso on small samples
4. **Geographic Confusion** — US pandemic (2020-03) vs China pandemic (2020-02)

---

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| **High prediction error** | Check feature/sample ratio (<1.0); increase training data; use LOO CV for stability |
| **Model not converging** | Increase iterations: `Lasso(max_iter=10000)` |
| **Holiday factors ineffective** | Verify holiday dates; ensure training set includes holiday data |
| **Database connection timeout** | Check whitelist; add `connect_timeout=30` |
| **Sudden performance drop** | Check if validation set includes special periods (pandemic/promotions) |
| **Environment variables not working** | Run `echo $RDS_HOST` to verify; check `.env` file path |
| **Script not found** | Use `{baseDir}/script_name.py` for correct path |
| **Overfitting detected** | Reduce features; use Lasso; apply LOO CV; collect more data |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive
- Use semantic versioning for releases

### Reporting Issues

- Use GitHub Issues for bug reports
- Include Python version, OS, and error logs
- Provide minimal reproducible examples
- Label issues appropriately (bug, enhancement, question)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Free to use for commercial and non-commercial purposes
- ✅ Free to modify and distribute
- ✅ No warranty provided
- ⚠️ Include copyright notice in copies

---

## 📞 Support

- **Documentation:** See `SKILL.md`, `DATASET.md`, `HISTORY.md`, `CHANGELOG.md`
- **Issues:** Open a GitHub issue for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas

---

## 🙏 Acknowledgments

- **Dataset Provider:** REES46 Marketing Platform (New York, USA)
- **Data Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- **Framework:** OpenClaw Skill System

---

**Version:** v6.0  
**Last Updated:** 2026-03-09  
**Data Period:** 2019-10-01 ~ 2020-04-30 (7 months)  
**Maintained by:** E-Commerce Predictor Team

---

<div align="center">

**Made with ❤️ for the OpenClaw Community**

[⬆ Back to Top](#ecommerce-predictor)

</div>
