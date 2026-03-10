# E-Commerce Predictor - Version History

> Complete evolution record v1.0 → v6.0

---

## 📊 Version Comparison Table

| 版本 | 训练天数 | 最佳模型 | PV 误差 | 购买量误差 | 发生了什么 |
|------|---------|---------|---------|-----------|-----------|
| **v1.0** | 31 天 | 指数平滑 | 26.92% | 11.39% | 基线模型，黑五误差 45% |
| **v2.0** | 61 天 | RandomForest | 8.15% | 23.17% | ✅ 训练数据翻倍 |
| **v3.0** | 92 天 | GradientBoosting | 2.73% | 42.28% | ✅ PV 预测历史最优 |
| **v4.0** | 123 天 | GradientBoosting | 2.73% | 42.28% | ✅ 正式 v1 |
| **v5.0** | 152 天 | Ridge | 29.99% | 36.99% | ⚠️ 撞上美国疫情爆发 |
| **v6.0** | 183 天 | GradientBoosting | 10.03% | 12.65% | ✅ 购买量预测最优 |

---

## 📝 Version Details

### v1.0 - Baseline Model
- **Training Days:** 31
- **Model:** Exponential Smoothing
- **Features:** 5 base features
- **PV MAPE:** 26.92%
- **Purchase MAPE:** 11.39%
- **Black Friday Error:** 45%
- **Notes:** Initial baseline, no holiday factors

### v2.0 - Data Expansion
- **Training Days:** 61 (doubled from v1.0)
- **Model:** RandomForest
- **PV MAPE:** 8.15% (significant improvement)
- **Purchase MAPE:** 23.17%
- **Key Change:** Training data doubled, better PV prediction

### v3.0 - PV Prediction Optimal
- **Training Days:** 92
- **Model:** GradientBoosting
- **PV MAPE:** 2.73% (historical best for PV)
- **Purchase MAPE:** 42.28%
- **Key Achievement:** Best PV prediction accuracy

### v4.0 - Production Ready v1
- **Training Days:** 123
- **Model:** GradientBoosting
- **PV MAPE:** 2.73% (maintained)
- **Purchase MAPE:** 42.28%
- **Status:** Confirmed as production-ready v1

### v5.0 - Pandemic Impact
- **Training Days:** 152
- **Model:** Ridge
- **PV MAPE:** 29.99% (anomaly)
- **Purchase MAPE:** 36.99%
- **⚠️ Warning:** Training data includes US COVID outbreak period (March 2020), causing data pattern shift

### v6.0 - Purchase Prediction Optimal
- **Training Days:** 183 (7 months)
- **Model:** GradientBoosting
- **PV MAPE:** 10.03%
- **Purchase MAPE:** 12.65% (historical best for purchase)
- **Key Achievement:** Best purchase volume prediction accuracy

---

## 🏆 Recommended Configurations

| Prediction Target | Recommended Version | Training Window | Model | Expected MAPE | Use Case |
|------------------|---------------------|-----------------|-------|---------------|----------|
| **PV Prediction** | v3.0 / v4.0 | 92-123 days | GradientBoosting | **2.73%** | Traffic forecasting |
| **Purchase Volume** | v6.0 | 183 days | GradientBoosting | **12.65%** | Inventory planning |
| **Small Sample** | v1.0 | 31 days | Exponential Smoothing | 26.92% | Quick testing |

---

## 📈 Key Learnings

### Success Patterns
1. **More data helps** - v2.0 (61 days) significantly outperformed v1.0 (31 days)
2. **GradientBoosting excels** - Best overall performance for both PV and purchase
3. **Stable periods matter** - v5.0 suffered from pandemic data contamination

### Common Pitfalls
1. **Data contamination** - v5.0 included COVID outbreak period, causing 29.99% error
2. **Overfitting risk** - Earlier versions with too many features showed poor generalization
3. **Holiday mismatch** - Initial versions incorrectly used Chinese holidays for US data

### Model Selection Strategy
- **< 100 days:** Use simpler models (Lasso, Ridge, Exponential Smoothing)
- **≥ 100 days:** GradientBoosting recommended
- **Special periods:** Label and handle separately (e.g., pandemic, major promotions)

---

## 📚 Related Documentation

- **[README.md](README.md)** - English documentation
- **[README_CN.md](README_CN.md)** - Chinese documentation
- **[SKILL.md](SKILL.md)** - OpenClaw Skill usage guide
- **[DATASET.md](DATASET.md)** - Dataset details and holiday modeling
- **[CHANGELOG.md](CHANGELOG.md)** - Release notes

---

**Last Updated:** 2026-03-09  
**Current Version:** v6.0  
**Maintained by:** E-Commerce Predictor Team
