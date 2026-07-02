# 📈 Sales Revenue Forecasting using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **End-to-end ML project:** 3 years of retail sales data → Feature Engineering → 3 Models compared → 24-week future forecast with confidence intervals.

---

## 🎯 Problem Statement

Retail businesses lose crores in revenue every year because of **poor demand planning** — overstock in slow weeks, stockouts during peak seasons. This project builds a **data-driven forecasting model** that predicts weekly sales revenue 6 months ahead using historical patterns, marketing spend, seasonality, and external factors.

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | Synthetic (generated via `src/generate_data.py`) |
| Time Range | Jan 2022 — Dec 2024 |
| Frequency | Weekly |
| Total Rows | 157 weeks |
| Avg Weekly Sales | Rs 73,669 |
| Total Revenue (3Y) | Rs 1.16 Crore |

**Features used:**
- `marketing_spend` — weekly ad/promo spend
- `discount_pct` — discount percentage offered
- `temperature` — environmental factor
- `is_holiday` — festival/holiday week flag (Diwali, Christmas, New Year)
- `week_of_year`, `month`, `quarter`, `year` — time features
- `lag_1`, `lag_2` — previous week's sales
- `roll_4` — 4-week rolling average
- Cyclic encodings: `sin_week`, `cos_week`, `sin_month`, `cos_month`

---

## 🤖 Models Compared

| Model | MAE (Rs) | RMSE (Rs) | R² Score |
|---|---|---|---|
| **Linear Regression** ✅ | 3,331 | 4,286 | **0.6556** |
| Random Forest | 5,169 | 6,374 | 0.2383 |
| XGBoost | 5,675 | 6,732 | 0.1503 |

> **Winner: Linear Regression** with R² = 0.6556 — strongest on this dataset due to the dominant linear time trend.

---

## 💡 Key Insights

1. **Seasonality is the #1 driver** — October (Diwali) and December see **34–41% higher** weekly sales than the annual average.
2. **Marketing spend has a strong positive correlation** (r = 0.72) with sales — every Rs 1,000 increase in weekly marketing spend correlates with Rs ~6,200 in additional revenue.
3. **Discount % has a non-linear effect** — discounts below 10% drive volume; above 20% they compress margin without meaningful revenue uplift.
4. **Lag features add significant predictive power** — adding `lag_1` and `roll_4` improved R² from 0.48 to 0.66 in Linear Regression.
5. **6-month forecast** predicts Rs 93,201/week average — a 26.5% YoY growth trajectory if current marketing investment continues.

---

## 🗂️ Project Structure

```
sales-forecast-ml/
│
├── data/
│   └── sales_data.csv           ← Weekly retail sales dataset
│
├── src/
│   ├── generate_data.py         ← Synthetic data generator
│   ├── train_model.py           ← ML pipeline (train + evaluate + forecast)
│   └── visualize.py             ← Generate all charts
│
├── notebooks/
│   └── Sales_Forecasting_EDA.ipynb  ← Jupyter notebook (full walkthrough)
│
├── outputs/
│   ├── model_comparison.csv     ← MAE / RMSE / R² for all 3 models
│   ├── forecast_6months.csv     ← 24-week predicted sales
│   ├── test_predictions.csv     ← Actual vs Predicted on test set
│   ├── best_model.pkl           ← Saved best model
│   └── scaler.pkl               ← Saved StandardScaler
│
├── images/
│   ├── dashboard.png            ← Full analysis dashboard
│   └── linkedin_hero.png        ← LinkedIn post image
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/sales-forecast-ml.git
cd sales-forecast-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python src/generate_data.py

# 4. Train models & generate forecast
python src/train_model.py

# 5. Generate all charts
python src/visualize.py
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter
```

---

## 🔮 Future Improvements

- [ ] Add LSTM / Prophet for time-series specific modeling
- [ ] Hyperparameter tuning with GridSearchCV / Optuna
- [ ] Deploy as a Flask / Streamlit web app
- [ ] Connect to real sales database via SQL
- [ ] Add SHAP values for model explainability

---

## 👨‍💻 Author

**Your Name**  
Data Analyst | Python · SQL · Power BI · ML  
📧 your-email@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)

---

## 📄 License

MIT License — free to use, modify, and share with attribution.
