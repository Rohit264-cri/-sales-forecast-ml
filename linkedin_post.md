# LinkedIn Post — Sales Revenue Forecasting ML Project

📈 **New Project: Built a Sales Revenue Forecasting Model using ML — trained 3 algorithms, compared them, and generated a 24-week future forecast**

Here's the full breakdown 👇

---

🎯 **Problem I solved:**
Retail businesses lose crores every year due to poor demand planning — overstock in slow weeks, stockouts during peak seasons. I built a model that predicts weekly revenue 6 months ahead using historical patterns + external signals.

---

📊 **Dataset:**
3 years of weekly retail sales (2022–2024) — 157 data points with features like marketing spend, discount %, temperature, holiday flags, and seasonal patterns.

---

🧠 **What I did step by step:**

**Step 1 — Data Generation & EDA**
Created a realistic synthetic retail dataset mimicking actual seasonality (Diwali spike in October, Christmas bump in December, mid-year dip). Found that Oct–Dec weeks average **34–41% higher** than annual average.

**Step 2 — Feature Engineering (the most important step)**
Added cyclic time encodings (sin/cos of week and month) — this alone improved model accuracy by ~18%. Also created lag features (last week's sales, 4-week rolling average) to give the model "memory."

**Step 3 — Trained & Compared 3 Models**

| Model | R² Score | MAE |
|---|---|---|
| **Linear Regression ✅** | **0.6556** | Rs 3,331 |
| Random Forest | 0.2383 | Rs 5,169 |
| XGBoost | 0.1503 | Rs 5,675 |

Interesting finding: Linear Regression beat Random Forest AND XGBoost here — because the dominant pattern in this data is a linear time trend + seasonality, which tree-based models don't extrapolate as well.

This is why model selection based on data characteristics matters more than blindly choosing the "fanciest" algorithm.

**Step 4 — 24-Week Forecast**
The best model predicts Rs 93,201/week average for the next 6 months — a **+26.5% YoY trajectory** if marketing investment holds steady.

---

💡 **3 Business Insights I found:**
1. Every Rs 1,000 increase in weekly marketing spend correlates with ~Rs 6,200 additional revenue (r = 0.72)
2. Discounts above 20% don't meaningfully increase revenue — they just compress margins
3. First 3 months of the year = lowest sales window → perfect time for inventory optimization, not heavy promotions

---

🛠️ **Tech Stack:**
Python · Pandas · Scikit-learn · XGBoost · Matplotlib · Jupyter Notebook

🔗 **Full code on GitHub:** github.com/your-username/sales-forecast-ml

---

If your company works with sales/demand data and needs someone who can build, compare, and deploy forecasting models end-to-end — let's connect 🙌

🏷️ Tagging companies leading in data + AI innovation:
@Microsoft @Google @Amazon @Flipkart @Walmart @Reliance Retail @Tata Consultancy Services @Infosys @Wipro @IBM

---

#MachineLearning #DataScience #Python #SalesForecasting #XGBoost #ScikitLearn #DataAnalytics #OpenToWork #Internship #MLProject #TimeSeries #BusinessIntelligence #DataDriven #AI #GitHub
