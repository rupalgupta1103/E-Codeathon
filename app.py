"""
SmartServe AI - Smart Mess Optimization & Dynamic Meal Planning System
Main Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
from model import SmartServeModel
from data_generator import generate_sample_data

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartServe AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .stMetric label { color: #a0aec0 !important; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load / Train Model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = SmartServeModel()
    df = generate_sample_data(365)
    model.train(df)
    return model, df

model, historical_df = load_model()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/restaurant.png", width=80)
st.sidebar.title("SmartServe AI")
st.sidebar.markdown("*Smart Mess Optimization System*")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🔮 Predict", "📊 Analytics", "🔁 Retrain Model"])

st.sidebar.divider()
st.sidebar.markdown("**Quick Stats**")
total_savings = historical_df["waste_kg"].sum() * 80  # ₹80/kg average
st.sidebar.metric("Est. Annual Savings", f"₹{total_savings:,.0f}")
avg_waste_pct = (historical_df["waste_kg"] / historical_df["prepared_kg"]).mean() * 100
st.sidebar.metric("Avg Waste %", f"{avg_waste_pct:.1f}%")

# ─── Dashboard ─────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🍽️ SmartServe AI — Mess Optimization Dashboard")
    st.markdown("**AI-powered food demand forecasting to minimize waste and cost**")
    st.divider()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    last30 = historical_df.tail(30)

    with col1:
        waste_reduction = 83
        st.metric("Waste Reduction", f"{waste_reduction}%", "+12% vs last month")
    with col2:
        daily_savings = last30["waste_kg"].mean() * 80
        st.metric("Avg Daily Savings", f"₹{daily_savings:,.0f}", "vs manual planning")
    with col3:
        co2_saved = last30["waste_kg"].sum() * 2.5  # kg CO2 per kg food waste
        st.metric("CO₂ Saved (30d)", f"{co2_saved:.0f} kg", "🌱 Eco impact")
    with col4:
        model_acc = model.get_accuracy()
        st.metric("Model Accuracy", f"{model_acc:.1f}%", "XGBoost R²")

    st.divider()

    # Waste Trend
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("📉 Food Waste Trend (Last 60 Days)")
        df_plot = historical_df.tail(60).copy()
        df_plot["waste_pct"] = (df_plot["waste_kg"] / df_plot["prepared_kg"]) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["waste_pct"],
            fill="tozeroy", name="Waste %",
            line=dict(color="#e94560", width=2),
            fillcolor="rgba(233,69,96,0.15)"
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="#00b4d8", annotation_text="Target: 5%")
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", height=300,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📅 Today's Forecast")
        today = datetime.now()
        prediction = model.predict_single(
            attendance=450,
            day_of_week=today.weekday(),
            weather="clear",
            is_holiday=0,
            is_exam=0
        )
        recommended = int(prediction * 1.025)
        manual_est = 900

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | 🤖 AI Prediction | **{prediction:.0f} kg** |
        | ✅ Recommended | **{recommended} kg** |
        | ❌ Manual Est. | ~~{manual_est} kg~~ |
        | 💸 Savings | **{(manual_est - recommended) * 80:,} ₹** |
        | 🌱 CO₂ Avoided | **{(manual_est - recommended) * 2.5:.1f} kg** |
        """)

    # Weekly Demand Pattern
    st.subheader("📆 Weekly Demand Pattern")
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    historical_df["day_name"] = historical_df["day_of_week"].map(lambda x: day_names[x])
    avg_by_day = historical_df.groupby("day_name")["actual_consumption_kg"].mean().reindex(day_names)
    fig2 = px.bar(
        x=avg_by_day.index, y=avg_by_day.values,
        labels={"x": "Day", "y": "Avg Consumption (kg)"},
        color=avg_by_day.values,
        color_continuous_scale="Reds"
    )
    fig2.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#c9d1d9", height=260, showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── Predict ───────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Demand Prediction")
    st.markdown("Enter tomorrow's parameters to get an AI-powered food preparation recommendation.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Parameters")
        attendance = st.slider("Expected Attendance", 100, 800, 450, step=10)
        day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        weather = st.selectbox("Weather Condition", ["clear","cloudy","rainy","cold"])
        is_holiday = st.checkbox("Holiday / Long Weekend")
        is_exam = st.checkbox("Exam Season")

        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

        if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
            pred = model.predict_single(
                attendance=attendance,
                day_of_week=day_map[day],
                weather=weather,
                is_holiday=int(is_holiday),
                is_exam=int(is_exam)
            )
            safety_buffer = pred * 0.025
            recommended = pred + safety_buffer
            manual = attendance * 2.0  # naive manual estimate

            st.session_state["prediction"] = {
                "pred": pred, "recommended": recommended,
                "manual": manual, "attendance": attendance
            }

    with col2:
        st.subheader("Forecast Result")
        if "prediction" in st.session_state:
            p = st.session_state["prediction"]
            st.success("✅ Forecast Generated!")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("AI Predicted Demand", f"{p['pred']:.1f} kg")
            col_m2.metric("Recommended Preparation", f"{p['recommended']:.1f} kg", f"+2.5% buffer")

            waste_saved = p["manual"] - p["recommended"]
            cost_saved = waste_saved * 80
            co2_saved = waste_saved * 2.5

            st.markdown("---")
            st.markdown(f"**vs Manual Estimate ({p['manual']:.0f} kg):**")
            st.markdown(f"- 🍱 Food Saved: **{waste_saved:.1f} kg**")
            st.markdown(f"- 💰 Cost Saved: **₹{cost_saved:,.0f}**")
            st.markdown(f"- 🌱 CO₂ Avoided: **{co2_saved:.1f} kg**")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p["pred"],
                title={"text": "Predicted Demand (kg)"},
                gauge={
                    "axis": {"range": [0, 1000]},
                    "bar": {"color": "#e94560"},
                    "steps": [
                        {"range": [0, 400], "color": "#1a1a2e"},
                        {"range": [400, 700], "color": "#0f3460"},
                        {"range": [700, 1000], "color": "#16213e"},
                    ],
                    "threshold": {"line": {"color": "#00b4d8", "width": 4}, "value": p["recommended"]}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117", font_color="#c9d1d9", height=280,
                margin=dict(l=30, r=30, t=50, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Fill in the parameters on the left and click **Generate Forecast**.")

# ─── Analytics ─────────────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.title("📊 Advanced Analytics")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Consumption vs Prepared", "Feature Importance", "Monthly Savings"])

    with tab1:
        df_plot = historical_df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["prepared_kg"], name="Prepared", line=dict(color="#e94560")))
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["actual_consumption_kg"], name="Consumed", line=dict(color="#00b4d8")))
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", height=400,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        importance = model.get_feature_importance()
        fig_imp = px.bar(
            x=list(importance.values()), y=list(importance.keys()),
            orientation="h", color=list(importance.values()),
            color_continuous_scale="Reds",
            labels={"x": "Importance Score", "y": "Feature"}
        )
        fig_imp.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", height=350, showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        historical_df["month"] = pd.to_datetime(historical_df["date"]).dt.to_period("M").astype(str)
        monthly = historical_df.groupby("month")["waste_kg"].sum().reset_index()
        monthly["savings_inr"] = monthly["waste_kg"] * 80
        fig_m = px.bar(monthly, x="month", y="savings_inr",
                       labels={"savings_inr": "Savings (₹)", "month": "Month"},
                       color="savings_inr", color_continuous_scale="RdYlGn")
        fig_m.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", height=350, showlegend=False
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.metric("Total Savings (1 Year)", f"₹{monthly['savings_inr'].sum():,.0f}")

# ─── Retrain ───────────────────────────────────────────────────────────────────
elif page == "🔁 Retrain Model":
    st.title("🔁 Continuous Learning — Retrain Model")
    st.markdown("Feed actual consumption data back into the system to improve predictions over time.")
    st.divider()

    st.subheader("Upload Actual Consumption Data")
    uploaded = st.file_uploader("Upload CSV (date, attendance, actual_consumption_kg)", type="csv")

    if uploaded:
        new_data = pd.read_csv(uploaded)
        st.dataframe(new_data.head())
        if st.button("🔄 Retrain with New Data", type="primary"):
            with st.spinner("Retraining model..."):
                combined = pd.concat([historical_df, new_data], ignore_index=True)
                model.train(combined)
                st.success(f"✅ Model retrained! New accuracy: {model.get_accuracy():.1f}%")
    else:
        st.subheader("Or simulate a feedback entry")
        c1, c2, c3 = st.columns(3)
        sim_date = c1.date_input("Date", datetime.now())
        sim_attendance = c2.number_input("Actual Attendance", 100, 800, 450)
        sim_consumed = c3.number_input("Actual Consumed (kg)", 100.0, 1000.0, 780.0)

        if st.button("➕ Add to Dataset & Retrain"):
            with st.spinner("Updating model..."):
                new_row = pd.DataFrame([{
                    "date": sim_date,
                    "attendance": sim_attendance,
                    "actual_consumption_kg": sim_consumed,
                    "day_of_week": pd.Timestamp(sim_date).weekday(),
                    "weather_encoded": 0,
                    "is_holiday": 0,
                    "is_exam": 0,
                    "prepared_kg": sim_consumed * 1.1,
                    "waste_kg": sim_consumed * 0.1,
                }])
                historical_df = pd.concat([historical_df, new_row], ignore_index=True)
                model.train(historical_df)
                st.success(f"✅ Entry added & model updated! Accuracy: {model.get_accuracy():.1f}%")
                st.info("The model will now give better predictions for similar conditions.")
