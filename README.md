# 🧹 E-Commerce Data Cleaning & Feature Engineering

Transforming a raw, chaotic e-commerce order dataset into a mathematically clean, ML-ready dataset using statistical imputation, outlier treatment, and feature engineering.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Overview

This project takes a raw **e-commerce order dataset** (1,176 transactions, 11 original fields) and prepares it for machine learning by:

1. **Handling missing data** with statistical imputation
2. **Detecting and removing outliers** using IQR & Z-Score methods
3. **Engineering 4 new predictive features** from existing columns
4. **Validating** every feature with correlation analysis and visualizations
5. **Encoding & scaling** the dataset for direct use in ML models

**Result:** a fully populated, statistically clean, feature-enriched dataset ready for classification or regression modeling.

---

## 📂 Dataset

| Field | Description |
|---|---|
| `OrderID`, `TrackingNumber` | Unique identifiers |
| `Date` | Order date |
| `CustomerID` | Customer identifier |
| `Product` | Item purchased (7 categories) |
| `Quantity`, `UnitPrice`, `TotalPrice` | Core transaction values |
| `PaymentMethod` | Online, Cash, Credit Card, Debit Card, Gift Card |
| `OrderStatus` | Shipped, Delivered, Cancelled, Returned, Pending |
| `ItemsInCart` | Items added to cart |
| `CouponCode` | Discount code applied (if any) |
| `ReferralSource` | Marketing channel (Instagram, Email, Google, Facebook, Referral) |

**Final shape:** `1,176 rows × 23 columns` (11 original + 3 identifiers + 9 engineered/derived fields)

---

## 🧼 1. Missing Data Handling

An initial `df.isnull().sum()` audit found **zero missing values** in all numeric fields. Only `CouponCode` had missing data — **309 out of 1,176 rows (26.3%)**.

Since a missing coupon simply means *"no coupon was used"*, this was treated as informative rather than random noise:

```python
df['UsedCoupon'] = df['CouponCode'].notna().astype(int)
df['CouponCode'] = df['CouponCode'].fillna('No Coupon')
```

✅ Preserves all 309 rows (no data loss)
✅ Converts the missingness pattern into a usable binary feature
❌ Avoided `df.dropna()` — would have deleted real transactions and biased the dataset

> A `SimpleImputer(strategy='median')` / `KNNImputer` fallback was prepared for numeric fields, but wasn't needed since none had missing values.

---

## 📊 2. Outlier Detection (IQR + Z-Score)

Outliers were tested on all core numeric fields using the **IQR method** (robust to skewed data), cross-validated with **Z-scores**:

```python
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]
```

| Field | Outliers Found |
|---|---|
| UnitPrice | 0 |
| Quantity | 0 |
| ItemsInCart | 0 |
| **TotalPrice** | **1** (verified, then removed) |

<p align="center">
  <img src="assets/boxplots_grid.png" width="600" alt="Boxplots of core numeric fields"/>
</p>
<p align="center"><em>Figure 1 — Boxplot distributions for UnitPrice, Quantity, ItemsInCart, and TotalPrice</em></p>

<p align="center">
  <img src="assets/totalprice_boxplot.png" width="450" alt="TotalPrice with IQR bounds"/>
</p>
<p align="center"><em>Figure 2 — TotalPrice distribution with upper/lower IQR bounds overlaid</em></p>

<p align="center">
  <img src="assets/distributions.png" width="600" alt="Distribution histograms"/>
</p>
<p align="center"><em>Figure 3 — Right-skewed distributions of TotalPrice and UnitPrice, justifying the use of IQR over a normal-distribution assumption</em></p>

---

## 🛠️ 3. Feature Engineering

Four new features were engineered — exceeding the 3-feature minimum:

| Feature | Formula | Purpose |
|---|---|---|
| `IsWeekend` | `Date.dayofweek in [5,6]` | Captures weekend vs. weekday shopping behavior |
| `CartConversionRatio` | `Quantity / ItemsInCart` | Proxy for purchase intent / cart abandonment |
| `AvgItemPrice` | `TotalPrice / Quantity` | Distinguishes premium vs. bulk/discount orders |
| `CustomerOrderCount` | `groupby('CustomerID').count()` | Flags repeat customers vs. one-time buyers |

```python
df['Date'] = pd.to_datetime(df['Date'])
df['IsWeekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

df['CartConversionRatio'] = df['Quantity'] / df['ItemsInCart']
df['AvgItemPrice'] = df['TotalPrice'] / df['Quantity']
df['CustomerOrderCount'] = df.groupby('CustomerID')['OrderID'].transform('count')
```

### Feature Validation

<p align="center">
  <img src="assets/correlation_heatmap.png" width="550" alt="Correlation heatmap"/>
</p>
<p align="center"><em>Figure 4 — Correlation matrix of engineered and core features against TotalPrice</em></p>

| Feature | Correlation with `TotalPrice` | Takeaway |
|---|---|---|
| `CartConversionRatio` | **+0.22** | Strongest engineered predictor of order value |
| `CustomerOrderCount` | −0.05 | Weak — dataset is dominated by first-time buyers |
| `IsWeekend` | −0.01 | Negligible linear effect |
| `UsedCoupon` | constant | No variance in this sample |

<p align="center">
  <img src="assets/conversion_by_status.png" width="500" alt="Cart conversion by order status"/>
</p>
<p align="center"><em>Figure 5 — CartConversionRatio varies meaningfully across OrderStatus, supporting its use in predicting order outcome</em></p>

---

## 🗂️ 4. Categorical Distribution Check

Verified class balance before encoding, to avoid a model that just learns the majority class:

<p align="center">
  <img src="assets/orderstatus_bar.png" width="500" alt="OrderStatus distribution"/>
</p>

<p align="center">
  <img src="assets/categorical_bars.png" width="600" alt="PaymentMethod and ReferralSource distribution"/>
</p>
<p align="center"><em>Figure 6/7 — OrderStatus, PaymentMethod, and ReferralSource are all well-balanced across categories</em></p>

---

## 🤖 5. ML-Ready Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=['PaymentMethod','OrderStatus','ReferralSource'], drop_first=True)

# Scale numeric fields
scaler = StandardScaler()
numeric_cols = ['UnitPrice','Quantity','TotalPrice','ItemsInCart']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Drop non-predictive identifiers
df_model = df_encoded.drop(columns=['OrderID','CustomerID','TrackingNumber'])
```

---

## 📁 Repo Structure

```
├── data/
│   ├── Dataset_for_Data_Analytics.xlsx        # raw input
│   └── Cleaned_Dataset_for_Data_Analytics.csv # cleaned + feature-engineered output
├── notebooks/
│   └── data_cleaning_pipeline.ipynb
├── assets/                                     # charts used in this README
└── README.md
```

---

## 🧰 Tech Stack

`Python` · `Pandas` · `NumPy` · `SciPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## 🚀 Getting Started

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl
```

```python
import pandas as pd
df = pd.read_csv("data/Cleaned_Dataset_for_Data_Analytics.csv")
```

---

## 📈 Summary

| Metric | Value |
|---|---|
| Rows before cleaning | 1,177 |
| Rows after cleaning | 1,176 |
| Missing values remaining | 0 |
| Outliers removed | 1 |
| New features engineered | 4 |
| Final columns | 23 |

</br>

