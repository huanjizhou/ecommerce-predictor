# 电商预测模型 - 数据集说明

---

## 📊 数据来源

**数据集名称：** eCommerce behavior data from multi category store  
**平台：** [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)  
**提供方：** REES46 Marketing Platform（美国纽约）

**公司信息：**
- 🏢 总部：New York, USA
- 📅 成立：2013 年
- 👨‍💼 创始人：Michael Kechinov
- 🌐 官网：rees46.com

---

## 📅 数据时间范围

| 时间段 | 天数 | 状态 |
|--------|------|------|
| 2019-10-01 ~ 2019-11-30 | 61 天 | ✅ Kaggle 原始数据 |
| 2019-12-01 ~ 2020-04-30 | 151 天 | ✅ 已导入数据库 |
| **总计** | **212 天** | **7 个月** |

---

## 📈 数据规模

| 指标 | 数值 |
|------|------|
| **总事件数** | 约 1.1 亿条 |
| **独立用户** | 约 530 万 |
| **独立商品** | 约 20 万 |
| **品类数** | 约 700 个 |
| **地区** | 🇺🇸 美国电商 |

---

## 🎯 事件类型

| 类型 | 说明 | 占比 |
|------|------|------|
| `view` | 商品浏览 | ~90% |
| `cart` | 加入购物车 | ~7% |
| `purchase` | 购买 | ~3% |

---

## 🗄️ 数据库表结构

**表名：** `ecommerce_events`

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

**每日聚合查询：**
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

## 🎉 美国节假日因子

| 日期 | 节日 | 影响 |
|------|------|------|
| 11 月 11 日 | 退伍军人节 (Veterans Day) | 联邦假日促销 |
| 11 月 28 日 | 感恩节 (Thanksgiving) | 购物季开始 |
| 11 月 29 日 | 黑色星期五 (Black Friday) | 全年最大购物节 |
| 12 月 2 日 | 网络星期一 (Cyber Monday) | 电商专属大促 |
| 12 月 25 日 | 圣诞节 (Christmas) | 圣诞购物季 |
| 1 月 1 日 | 元旦 (New Year's Day) | 新年促销 |

**重要说明：**
- ❌ **美国没有双 11 购物节**
- ✅ 11 月 11 日流量高峰是**退伍军人节促销 + 黑五预热**

---

## ⚠️ 特殊时期标记

| 时间 | 事件 | 影响 |
|------|------|------|
| 2020-03 中下旬起 | 🦠 美国疫情爆发 | 居家令、电商激增、模式突变 |
| 2020-04 | 全面封锁 | 线上购物高峰 |

**建议：** 2020 年 3 月起的数据为疫情特殊时期，建议单独标注或排除在正常时期模型之外。

---

## 🔗 参考资料

- [Kaggle 数据集](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- [REES46 公司官网](https://rees46.com/)
- [美国节假日列表](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/)

---

**最后更新：** 2026-03-08
