# Dairy Sales Forecasting — MSc Data Science Dissertation

**University of Hertfordshire · MSc Data Science & Analytics · 2023**

A full machine learning pipeline built on real Indian dairy industry data (2019–2022) to predict sales volume and forecast demand — with the goal of reducing overstock waste and improving revenue planning for dairy producers.

---

## The business problem

Dairy products are perishable. Overstock means waste; understock means lost sales. Indian dairy farms and brands had no reliable way to predict how much product would sell on a given day, across different regions, brands, and sales channels.

This project asked: **can machine learning predict dairy sales well enough to drive real inventory decisions?**

---

## What I built

A complete data science pipeline across three stages:

### 1. Exploratory analysis & correlation
- Identified key drivers of revenue: **total inventory value** and **quantity in stock** were the strongest predictors of sales outcomes
- Found pronounced seasonal and brand-based patterns (Amul dominated volume; Warana led by total inventory value)
- Showed refrigerated storage correlated with highest stock availability; products with shorter shelf lives required more aggressive turnover strategies

### 2. Machine learning models for sales prediction
Trained and compared 7 models using **k-fold cross-validation** and **MSE** as the evaluation metric:

| Model | Mean MSE (after tuning) |
|---|---|
| Boosting (AdaBoost) | 0.0143 |
| Bagging | 0.0132 ✅ best |
| Stacking | 0.0274 |
| Decision Tree | 0.0372 |
| Voting | 0.0394 |
| Linear Regression | 0.0945 |
| KNN | 0.2175 |

Used **SHAP values** to explain model decisions — total farm value and quantity in stock drove ~95% of prediction importance.

### 3. Time series forecasting by day of week
- **Neural Network (Keras/TensorFlow):** RMSE 0.240, MAE 0.188
- **XGBoost:** RMSE 0.218, MAE 0.176 — outperformed the neural network
- Both models captured average sales well but struggled with day-to-day variance — a known limitation when true time series data is unavailable

---

## Tech stack

```
Python · Scikit-learn · XGBoost · TensorFlow/Keras · Pandas · NumPy
Matplotlib · Seaborn · SHAP · GridSearchCV
```

---

## Key findings

- **Bagging and Boosting** were the strongest models for sales prediction — ensemble methods consistently outperformed single models
- **Premium brands** (Amul, Warana) sustained higher pricing without significant volume impact — price elasticity is low at the top end
- **Refrigerated storage** products had the highest stock availability; frozen storage represented an under-served market opportunity
- XGBoost outperformed neural networks on this tabular dataset — consistent with broader ML research on structured data

---

## Dataset

Dairy Goods Sales Dataset — 2019 to 2022, covering farms across 15 Indian states. Features include: farm location, land area, cow count, product type, brand, sales channel (retail/wholesale/online), storage conditions, shelf life, and revenue.

---

## Structure

```
├── notebooks/
│   ├── 01_EDA_and_correlation.ipynb
│   ├── 02_ML_models_and_tuning.ipynb
│   └── 03_time_series_forecasting.ipynb
├── data/
│   └── dairy_dataset.csv
├── outputs/
│   ├── shap_summary.png
│   ├── mse_comparison.png
│   └── actual_vs_predicted.png
└── README.md
```

---

## What I'd do differently

- Collect **daily time series data** — the dataset had dates but not a true sequential time series, which limited ARIMA and RNN approaches
- Explore **LSTM networks** for sequential forecasting once proper time-indexed data is available
- Deploy the best model as a **FastAPI endpoint** for real-time inventory decisions

---

## Author

**Rohith Jayadevan**
MSc Data Science & Analytics, University of Hertfordshire
[LinkedIn](https://linkedin.com/in/rohithjayadevan13) · rohithjayadevan40@gmail.com
