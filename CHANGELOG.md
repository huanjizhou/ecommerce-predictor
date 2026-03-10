# Changelog

## v1.0.0 (2026-03-08)

### 🎉 首次发布

**核心功能：**
- 电商用户行为时间序列预测
- 自动进化模型（滚动训练 + 自动验证）
- 支持 PV、UV、购买量预测
- 美国节假日因子集成（感恩节、黑五、网络星期一）

**模型特性：**
- 20 个核心特征（时间、节假日、滞后、滚动统计、增长率、转化率）
- 模型自动选择：<100 天用 Lasso，≥100 天用 GradientBoosting
- LOO 交叉验证，确保稳定性

**性能表现：**
- PV 预测 MAPE: 26.92% → 10.03%（↓62.7%）
- 购买量预测 MAPE: 12.65%（优秀水平 <15%）
- 训练数据：31 天 → 183 天

**技术栈：**
- Python 3.6+
- pandas, scikit-learn, xgboost
- MySQL/DuckDB
- OpenClaw Skill 系统

**使用方法：**
```bash
python3 ecommerce_predictor_auto.py \
  --train-start 2019-10-01 \
  --train-end 2020-03-01 \
  --val-end 2020-04-01
```

**注意事项：**
- 这是🇺🇸美国电商数据，节假日用黑五/感恩节
- 需要配置 RDS 数据库连接（环境变量或 .env 文件）
- 建议使用 DuckDB 加速查询（1000+ 倍提升）
