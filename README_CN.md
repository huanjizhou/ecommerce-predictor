# Ecommerce Predictor

[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-blue?logo=openclaw)](https://openclaw.com)
[![Python Version](https://img.shields.io/badge/Python-3.6+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MAPE](https://img.shields.io/badge/PV%20MAPE-2.73%25-brightgreen)]()
[![Data](https://img.shields.io/badge/Data-REES46%20(US)-orange)]()

> **电商用户行为时间序列预测** — 使用 GradientBoosting/Lasso 模型预测 PV、UV 和购买量，支持滚动训练和持续验证。

---

## 📋 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [安装指南](#-安装指南)
- [模型性能](#-模型性能)
- [数据集](#-数据集)
- [配置指南](#-配置指南)
- [特征工程](#-特征工程)
- [最佳实践](#-最佳实践)
- [故障排查](#-故障排查)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

---

## ✨ 功能特性

- **🎯 多指标预测** — 预测每日 PV（页面浏览量）、UV（独立访客数）和购买量
- **🤖 自动模型选择** — 智能选择 Lasso（<100 天）或 GradientBoosting（≥100 天）
- **📈 滚动训练** — 支持可扩展训练窗口的持续验证
- **🇺🇸 美国节假日集成** — Veterans Day、Thanksgiving、Black Friday、Cyber Monday、Christmas
- **🔍 LOO 交叉验证** — Leave-One-Out CV 确保小样本模型稳定性
- **📊 自动报告生成** — 生成 Markdown 报告、预测图表和性能历史记录
- **🚀 DuckDB 加速** — DuckDB 后端实现 1000 倍 + 查询加速
- **🔧 灵活配置** — 支持环境变量或 `.env` 文件配置

---

## 🚀 快速开始

### 标准滚动预测（推荐）

```bash
python3 ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

### 快速测试（小样本 <100 天）

```bash
python3 ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2019-10-31 \
  --val-end 2019-11-30
```

### 购买量优化预测

```bash
# 购买量预测需要更长的训练窗口（150+ 天）
python3 optimize_purchase_v3.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

---

## 📦 安装指南

### 前置要求

- Python 3.6+
- MySQL/MariaDB 或 DuckDB
- pip 包管理器

### 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost pymysql python-dotenv matplotlib
```

### 克隆或下载

```bash
# 如果使用 git
git clone <repository-url>
cd ecommerce-predictor

# 或复制脚本到工作区
cp *.py /your/workspace/
```

---

## 🏆 模型性能

### 当前最佳配置

| 预测目标 | 推荐版本 | 训练窗口 | 模型 | MAPE | 使用场景 |
|--------|---------|---------|-------|------|----------|
| **PV 预测** | v3.0 / v4.0 | 92-123 天 | GradientBoosting | **2.73%** | 正常时期 |
| **购买量预测** | v6.0 | 183 天 | GradientBoosting | **12.65%** | 长期训练 |
| **节假日预测** | v1.0 | 31 天 | Exponential Smoothing | 11.39% (Black Friday) | 小样本 |
| **快速基线** | v1.0 | 31 天 | Exponential Smoothing | 26.92% | 数据不足 |

### 版本性能历史

| 版本 | 训练天数 | 最佳模型 | PV 误差 | 购买量误差 | 发生了什么 |
|------|---------|---------|---------|-----------|-----------|
| v1.0 | 31 天 | 指数平滑 | 26.92% | 11.39% | 基线模型，Black Friday 误差 45% |
| v2.0 | 61 天 | RandomForest | 8.15% | 23.17% | ✅ 训练数据翻倍 |
| v3.0 | 92 天 | GradientBoosting | 2.73% | 42.28% | ✅ PV 预测历史最优 |
| v4.0 | 123 天 | GradientBoosting | 2.73% | 42.28% | ✅ 正式 v1 |
| v5.0 | 152 天 | Ridge | 29.99% | 36.99% | ⚠️ 撞上美国疫情爆发 |
| v6.0 | 183 天 | GradientBoosting | 10.03% | 12.65% | ✅ 购买量预测最优 |

---

## 📊 数据集

### 数据源信息

- **数据集名称:** eCommerce behavior data from multi category store
- **平台:** [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- **提供方:** REES46 Marketing Platform（美国纽约）
- **公司:** 成立于 2013 年，创始人：Michael Kechinov
- **网站:** [rees46.com](https://rees46.com/)

### 数据统计

| 指标 | 数值 |
|--------|-------|
| **总事件数** | ~1.1 亿 |
| **独立用户数** | ~530 万 |
| **独立商品数** | ~20 万 |
| **品类数量** | ~700 |
| **地区** | 🇺🇸 美国 |
| **时间范围** | 2019-10-01 ~ 2020-04-30（7 个月） |

### 事件类型

| 类型 | 描述 | 占比 |
|------|-------------|-------|
| `view` | 商品浏览 | ~90% |
| `cart` | 加入购物车 | ~7% |
| `purchase` | 购买 | ~3% |

### 美国节假日因素

| 日期 | 节假日 | 影响 |
|------|---------|--------|
| 11 月 11 日 | Veterans Day | 联邦假日促销 |
| 11 月 28 日 | Thanksgiving | 购物季开始 |
| 11 月 29 日 | Black Friday | 最大购物活动 |
| 12 月 2 日 | Cyber Monday | 电商专属优惠 |
| 12 月 25 日 | Christmas | 圣诞购物季 |
| 1 月 1 日 | New Year's Day | 新年促销 |

> **重要提示:** 这是 🇺🇸 **美国电商数据**。请使用 Black Friday/Thanksgiving，**不要使用中国的双 11/618**。

### 特殊时期

| 时期 | 事件 | 影响 |
|--------|-------|--------|
| 2020-03（中下旬） | 🦠 美国疫情爆发 | 居家令、电商激增、模式转变 |
| 2020-04 | 全面封锁 | 在线购物高峰 |

**建议:** 2020 年 3 月之后的数据代表疫情时期 — 请单独标记或从正常时期模型中排除。

---

## ⚙️ 配置指南

### 数据库连接

#### 方法一：环境变量（推荐）

```bash
export RDS_HOST=<read-replica-public-endpoint>
export RDS_USER=<db-username>
export RDS_PASSWORD=<password>
export RDS_DATABASE=<db-name>
export RDS_PORT=3306
```

#### 方法二：`.env` 文件

创建 `{baseDir}/.env` 文件（脚本会自动加载）：

```bash
RDS_HOST=<read-replica-public-endpoint>
RDS_PORT=3306
RDS_USER=<db-username>
RDS_PASSWORD=<password>
RDS_DATABASE=<db-name>
```

**依赖:**
```bash
pip install python-dotenv
```

### 验证配置

```bash
# 检查环境变量
echo $RDS_HOST

# 测试数据库连接
python3 ecommerce_predictor_auto.py --help
```

### 数据库表结构

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

### 每日聚合查询

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

## 🔧 特征工程

### 20 个核心特征

#### 时间特征（3 个）
- `day_of_week` — 星期几（0-6）
- `day_of_month` — 月份中的日期（1-31）
- `is_weekend` — 周末标志（周六/周日）

#### 美国节假日（4 个）
- `is_veterans_day` — Veterans Day（11 月 11 日）
- `is_thanksgiving` — Thanksgiving（11 月第 4 个周四）
- `is_black_friday` — Black Friday（Thanksgiving 次日）
- `is_cyber_monday` — Cyber Monday（Thanksgiving 后的周一）

#### 滞后特征（4 个）
- `lag_1_pv` — PV 滞后 1 天
- `lag_7_pv` — PV 滞后 7 天
- `lag_1_uv` — UV 滞后 1 天
- `lag_7_uv` — UV 滞后 7 天

#### 滚动统计（4 个）
- `ma_7_pv` — 7 日移动平均 PV
- `ma_7_uv` — 7 日移动平均 UV
- `median_7_pv` — 7 日中位数 PV
- `std_7_pv` — 7 日标准差 PV

#### 增长率（2 个）
- `mom_growth_pv` — 环比 PV 增长率
- `wow_growth_pv` — 同比 PV 增长率

#### 转化率（2 个）
- `uv_purchase` — 购买/UV 转化率
- `cart_to_purchase_rate` — 购买/购物车转化率

#### 购物季（1 个）
- `is_christmas_season` — 圣诞季标志（12 月 1-25 日）

### 特征选择指南

- **特征/样本比 < 1.0** — 例如，31 个样本使用 20 个特征（0.65 比率）
- **避免过拟合** — v3.0 有 51 个特征/31 个样本（1.65 比率）→ R²=-0.30
- **从简单开始** — 从核心特征开始，仅在需要时增加复杂度

---

## 📚 最佳实践

### 推荐模型（购买量用 v6.0，PV 用 v3.0/v4.0）

| 训练天数 | 推荐模型 | 原因 |
|---------------|------------------|--------|
| < 100 | Lasso (alpha=0.01) | 防止小样本过拟合 |
| ≥ 100 | GradientBoosting (n_estimators=50, max_depth=3) | 捕捉非线性模式 |

### 验证工作流

1. **配置数据库连接**（环境变量或 `.env`）
2. **提取数据**，使用正确的日期范围
3. **构建模型**（根据训练天数自动选择）
4. **验证**，使用 MAPE 和 R² 指标
5. **交叉验证**，小样本使用 LOO CV
6. **生成报告**（Markdown + 图表 + 历史记录）

### 性能指标

| 指标 | 描述 | 良好阈值 | 优秀 |
|--------|-------------|----------------|-----------|
| **MAPE** | 平均绝对百分比误差 | <20% | <10% |
| **R²** | 决定系数 | >0.5 | >0.7 |
| **LOO CV MAPE** | 交叉验证稳定性 | 越低越好 | <5% |

### 经验总结

#### ✅ 成功模式

1. **特征多≠效果好** — v3.0 的 51 个特征导致严重过拟合
2. **小样本用简单模型** — v4.0 的 Lasso 在 31 个样本上表现最佳
3. **交叉验证必不可少** — LOO CV 确认模型稳定性
4. **节假日匹配地区** — 美国使用 Black Friday/Thanksgiving，不是双 11
5. **数据越多越稳定** — 购买量 MAPE 从 42% 降至 13%（更多数据）
6. **滚动验证捕捉异常** — v8.0 的疫情数据导致性能下降
7. **模型选择很重要** — <100 天：Lasso，≥100 天：GradientBoosting

#### ❌ 常见陷阱

1. **假设偏差** — 假设 11 月=双 11（这是美国数据！）
2. **特征过多** — 51 个特征 / 31 个样本 = 1.65 比率 → 过拟合
3. **过早使用复杂模型** — XGBoost 在小样本上不如 Lasso
4. **地理混淆** — 美国疫情（2020-03）vs 中国疫情（2020-02）

---

## 🔍 故障排查

| 问题 | 解决方案 |
|---------|----------|
| **预测误差高** | 检查特征/样本比（<1.0）；增加训练数据；使用 LOO CV 验证稳定性 |
| **模型不收敛** | 增加迭代次数：`Lasso(max_iter=10000)` |
| **节假日因子无效** | 验证节假日日期；确保训练集包含节假日数据 |
| **数据库连接超时** | 检查白名单；添加 `connect_timeout=30` |
| **性能突然下降** | 检查验证集是否包含特殊时期（疫情/促销） |
| **环境变量不生效** | 运行 `echo $RDS_HOST` 验证；检查 `.env` 文件路径 |
| **脚本未找到** | 使用 `{baseDir}/script_name.py` 指定正确路径 |
| **检测到过拟合** | 减少特征；使用 Lasso；应用 LOO CV；收集更多数据 |

---

## 🤝 贡献指南

欢迎贡献！请遵循以下指南：

### 如何贡献

1. **Fork** 仓库
2. **创建功能分支** (`git checkout -b feature/amazing-feature`)
3. **提交更改** (`git commit -m 'Add amazing feature'`)
4. **推送到分支** (`git push origin feature/amazing-feature`)
5. **创建 Pull Request**

### 开发指南

- 遵循现有代码风格
- 为新功能添加测试
- 更新文档
- 保持提交原子化和描述性
- 发布使用语义化版本控制

### 报告问题

- 使用 GitHub Issues 报告 bug
- 包含 Python 版本、操作系统和错误日志
- 提供最小可复现示例
- 适当标记问题（bug、enhancement、question）

---

## 📄 许可证

本项目采用 **MIT 许可证** — 详见 [LICENSE](LICENSE) 文件。

### MIT 许可证摘要

- ✅ 可用于商业和非商业目的
- ✅ 可自由修改和分发
- ✅ 不提供任何保证
- ⚠️ 副本中需包含版权声明

---

## 📞 支持

- **文档:** 参见 `SKILL.md`、`DATASET.md`、`HISTORY.md`、`CHANGELOG.md`
- **问题:** 在 GitHub 提交 issue 报告 bug 或功能请求
- **讨论:** 使用 GitHub Discussions 提问和分享想法

---

## 🙏 致谢

- **数据集提供方:** REES46 Marketing Platform（美国纽约）
- **数据源:** [Kaggle Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- **框架:** OpenClaw Skill 系统

---

**版本:** v6.0  
**最后更新:** 2026-03-09  
**数据周期:** 2019-10-01 ~ 2020-04-30（7 个月）  
**维护者:** E-Commerce Predictor Team

---

<div align="center">

**Made with ❤️ for the OpenClaw Community**

[⬆ 返回顶部](#ecommerce-predictor)

</div>
