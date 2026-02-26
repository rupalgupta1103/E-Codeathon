# 🍽️ SmartServe AI — Smart Mess Optimization & Dynamic Meal Planning System

> **Hackathon Project** | Team: Rupal Gupta, Priya Singh Rana, Rishika Cherukuri

Reduces hostel mess food waste by **83%** using AI-powered demand forecasting.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

---

## 🏗️ Architecture

```
Data Layer          →  data_generator.py   (historical + synthetic data)
ML Layer            →  model.py            (XGBoost + Random Forest)
Dashboard Layer     →  app.py              (Streamlit UI)
```

### ML Pipeline

| Step | Details |
|------|---------|
| **Input Features** | Attendance, Day of Week, Weather, Holiday flag, Exam flag, Lag-1, Lag-7, Rolling-7 avg |
| **Target** | `actual_consumption_kg` |
| **Primary Model** | XGBoost Regressor |
| **Baseline Model** | Random Forest Regressor |
| **Optimization** | Predicted Demand × 1.025 (2.5% safety buffer) |

### Why XGBoost?
- Handles structured tabular data very well
- Gradient boosting minimizes prediction errors iteratively
- Built-in regularization reduces overfitting
- Fast training, scalable to larger datasets

---

## 📊 Key Results

| Metric | Manual | SmartServe AI |
|--------|--------|---------------|
| Prepared | 900 kg | 820 kg |
| Waste | 120 kg | 20 kg |
| Waste % | 13.3% | 2.4% |
| Waste Reduction | — | **83%** |
| Daily Cost Savings | — | ₹8,000–10,000 |

---

## 📁 Project Structure

```
smartserve/
├── app.py              # Streamlit dashboard (4 pages)
├── model.py            # XGBoost ML model + feature engineering
├── data_generator.py   # Synthetic data generation
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🌱 SDG Impact

- **SDG 12** – Responsible Consumption and Production
- **SDG 13** – Climate Action (CO₂ reduction tracking)

---

## 🔁 Continuous Learning

The system supports retraining with real consumption data:
1. Upload actual daily consumption CSV
2. Model retrains with combined historical + new data
3. Predictions improve over time (adaptive intelligence)
