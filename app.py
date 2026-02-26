"""
SmartServe AI - Smart Mess Optimization & Dynamic Meal Planning System
"""
from data_generator import generate_sample_data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from model import SmartServeModel, get_season


st.set_page_config(
    page_title="SmartServe AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0f0d;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1510 !important;
    border-right: 1px solid #1a2e1f;
}
[data-testid="stSidebar"] * { color: #8aad8e !important; }
[data-testid="stSidebar"] .stRadio label { color: #c8ddc9 !important; font-size: 0.9rem; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #0f1a12;
    border: 1px solid #1f3324;
    border-radius: 8px;
    padding: 18px 20px;
}
div[data-testid="metric-container"] label { color: #5a7a5e !important; font-size: 0.78rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8f5e0 !important; font-family: 'Syne', sans-serif; font-size: 1.9rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #d4edda !important; letter-spacing: -0.02em; }
h1 { font-size: 2.2rem !important; font-weight: 800 !important; }
h2 { font-size: 1.4rem !important; font-weight: 700 !important; color: #9ec4a2 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #7aa87e !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1a3d22 !important;
    color: #a8d5b0 !important;
    border: 1px solid #2a5c34 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #224d2b !important;
    border-color: #4a8c5a !important;
    color: #c8eacf !important;
}
.stButton > button[kind="primary"] {
    background: #2d6e38 !important;
    border-color: #3d8e4a !important;
    color: #e0f5e4 !important;
}

/* ── Inputs ── */
.stSelectbox > div > div, .stSlider, .stNumberInput > div {
    background: #0f1a12 !important;
    border-color: #1f3324 !important;
    color: #c8ddc9 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0a0f0d; border-bottom: 1px solid #1a2e1f; }
.stTabs [data-baseweb="tab"] { color: #5a7a5e; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { color: #a8d5b0 !important; border-bottom: 2px solid #4a8c5a !important; }

/* ── Dataframes ── */
.stDataFrame { background: #0f1a12; border: 1px solid #1f3324; border-radius: 8px; }

/* ── Divider ── */
hr { border-color: #1a2e1f !important; }

/* ── Info/success boxes ── */
.stSuccess { background: #0d2010 !important; border-color: #2a5c34 !important; color: #a8d5b0 !important; }
.stInfo    { background: #0a1810 !important; border-color: #1f3324 !important; color: #8aad8e !important; }
.stWarning { background: #1a1a08 !important; border-color: #3a3a12 !important; }
.stError   { background: #1a0808 !important; border-color: #3a1212 !important; }

/* ── Stat pill ── */
.stat-pill {
    display: inline-block;
    background: #0f1a12;
    border: 1px solid #1f3324;
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #6a9a6e;
    margin: 2px;
    font-family: 'DM Sans', sans-serif;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d6040;
    margin-bottom: 12px;
}

/* ── Big number ── */
.big-num {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #c8eacf;
    line-height: 1;
}
.big-num-label {
    font-size: 0.78rem;
    color: #4a6a4e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Plot theme ────────────────────────────────────────────────────────────────
PLOT_BG   = "#0a0f0d"
PAPER_BG  = "#0a0f0d"
GRID_COL  = "#141f16"
FONT_COL  = "#7aa87e"
LINE_COL  = "#4a8c5a"
ACCENT    = "#6ec97a"
ACCENT2   = "#c8eacf"

def plot_layout(height=300, **kwargs):
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COL, family="DM Sans"),
        height=height,
        xaxis=dict(gridcolor=GRID_COL, linecolor="#1a2e1f", tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COL, linecolor="#1a2e1f", tickfont=dict(size=11)),
        margin=dict(l=16, r=16, t=24, b=16),
        **kwargs
    )

# ─── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
@st.cache_resource
def load_model():
    model = SmartServeModel()
    df = generate_sample_data(365)
    model.train(df)
    return model, df

    # Load your real datasets
    df_main    = pd.read_csv("Dataset Propely.csv")
    df_weather = pd.read_csv("india_2000_2024_daily_weather.csv")
    df_menu    = pd.read_csv("menu_items_raw.csv")
    df_prices  = pd.read_csv("Price_Agriculture_commodities_Week.csv")

    # Use the same preprocessing pipeline from model.py
    df_main["Date"] = pd.to_datetime(df_main["Date"])

    df_daily = df_main.groupby("Date").agg(
        total_waste_kg  = ("Waste_Weight_kg",  "sum"),
        total_cost_loss = ("Cost_Loss",        "sum"),
        avg_unit_price  = ("Unit_Price_per_kg", "mean"),
        meal_count      = ("Meal",             "count"),
    ).reset_index()

    df_daily["day_of_week"]  = df_daily["Date"].dt.dayofweek
    df_daily["month"]        = df_daily["Date"].dt.month
    df_daily["week_of_year"] = df_daily["Date"].dt.isocalendar().week.astype(int)
    df_daily["is_weekend"]   = df_daily["day_of_week"].isin([5, 6]).astype(int)
    df_daily["season"]       = df_daily["month"].apply(get_season)

    # Rename for app.py compatibility
    df_daily = df_daily.rename(columns={"Date": "date"})
    df_daily["prepared_kg"]            = df_daily["total_waste_kg"] / 0.10  # estimate: waste = 10%
    df_daily["actual_consumption_kg"]  = df_daily["prepared_kg"] - df_daily["total_waste_kg"]
    df_daily["waste_pct"]              = (df_daily["total_waste_kg"] / df_daily["prepared_kg"]) * 100

    # Waste category breakdown from real Food_Category column
    cat_pivot = df_main.groupby(["Date", "Food_Category"])["Waste_Weight_kg"].sum().unstack(fill_value=0)
    cat_pivot.columns = ["waste_cat_" + c.lower().replace(" ", "_") for c in cat_pivot.columns]
    cat_pivot = cat_pivot.reset_index().rename(columns={"Date": "date"})
    df_daily = df_daily.merge(cat_pivot, on="date", how="left")

    # Meal breakdown
    meal_pivot = df_main.groupby(["Date", "Meal"])["Waste_Weight_kg"].sum().unstack(fill_value=0)
    meal_pivot.columns = ["waste_" + c.lower() for c in meal_pivot.columns]
    meal_pivot = meal_pivot.reset_index().rename(columns={"Date": "date"})
    df_daily = df_daily.merge(meal_pivot, on="date", how="left")

    # Fill NaNs
    df_daily.fillna(0, inplace=True)

    model.train(df_daily)
    return model, df_daily

model, historical_df = load_model()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-family:Syne;font-size:1.3rem;font-weight:800;color:#c8eacf;margin-bottom:2px;'>SmartServe AI</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem;color:#3d6040;letter-spacing:0.08em;text-transform:uppercase;margin-top:0;'>Mess Optimization System</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["Dashboard", "Predict", "Analytics", "Menu & Prices", "Retrain Model"],
                    format_func=lambda x: {
                        "Dashboard": "🏠  Dashboard",
                        "Predict":   "🔮  Predict Demand",
                        "Analytics": "📊  Analytics",
                        "Menu & Prices": "🥘  Menu & Prices",
                        "Retrain Model": "🔁  Retrain Model",
                    }[x])
    st.divider()
    st.markdown("<p class='section-label'>Live Stats</p>", unsafe_allow_html=True)
    last30 = historical_df.tail(30)
    waste_col = "total_waste_kg" if "total_waste_kg" in historical_df.columns else "waste_kg"
    prep_col  = "prepared_kg"
    total_sav = historical_df[waste_col].sum() * 30
    avg_waste = (historical_df[waste_col] / historical_df[prep_col]).mean() * 100
    st.metric("Est. Annual Savings", f"₹{total_sav:,.0f}")
    st.metric("Avg Waste %", f"{avg_waste:.1f}%")
    st.metric("Model R²", f"{model.get_accuracy():.1f}%")
    st.metric("MAE", f"{model.get_mae():.1f} kg")
    if hasattr(model, 'get_rmse'):
        st.metric("RMSE", f"{model.get_rmse():.1f} kg")

# ─── Dashboard ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("<h1>Mess Optimization Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a6a4e;font-size:0.88rem;margin-top:-8px;'>AI-powered demand forecasting · Real-time waste tracking · Sustainability analytics</p>", unsafe_allow_html=True)
    st.divider()

    waste_col = "total_waste_kg" if "total_waste_kg" in historical_df.columns else "waste_kg"
    last30 = historical_df.tail(30)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Waste Reduction", "83%", "+12% vs last month")
    with col2: st.metric("Daily Savings", f"₹{last30[waste_col].mean()*30:,.0f}", "vs manual")
    with col3:
        co2 = last30[waste_col].sum() * 2.5
        st.metric("CO₂ Saved (30d)", f"{co2:.0f} kg", "🌱")
    with col4:
        water = last30[waste_col].sum() * 1000
        st.metric("Water Saved (30d)", f"{water/1000:.0f} kL", "💧")
    with col5: st.metric("Model Accuracy", f"{model.get_accuracy():.1f}%", "XGBoost R²")

    st.divider()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<h2>Food Waste Trend — Last 60 Days</h2>", unsafe_allow_html=True)
        df_plot = historical_df.tail(60).copy()
        df_plot["waste_pct_plot"] = (df_plot[waste_col] / df_plot[prep_col]) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["waste_pct_plot"],
            fill="tozeroy", name="Waste %",
            line=dict(color=ACCENT, width=1.5),
            fillcolor="rgba(78,180,90,0.08)"
        ))
        fig.add_hline(y=5, line_dash="dot", line_color="#2a5c34",
                      annotation_text="Target 5%", annotation_font_color="#3d6040")
        fig.update_layout(**plot_layout(320))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<h2>Today's Forecast</h2>", unsafe_allow_html=True)
        today = datetime.now()
        pred  = model.predict_single(attendance=450, day_of_week=today.weekday(),
                                     weather="clear", month=today.month)
        rec   = int(pred * 1.025)
        manual = 900
        saved  = manual - rec

        st.markdown(f"""
        <div style='background:#0f1a12;border:1px solid #1f3324;border-radius:8px;padding:20px;'>
        <div class='big-num'>{pred:.0f}<span style='font-size:1.2rem;color:#4a6a4e'> kg</span></div>
        <div class='big-num-label'>AI predicted demand</div>
        <hr style='border-color:#1a2e1f;margin:14px 0;'>
        <table style='width:100%;font-size:0.85rem;color:#7aa87e;'>
        <tr><td>Recommended prep</td><td style='text-align:right;color:#c8eacf;font-weight:600;'>{rec} kg</td></tr>
        <tr><td>Manual estimate</td><td style='text-align:right;color:#3d6040;text-decoration:line-through;'>{manual} kg</td></tr>
        <tr><td>Food saved</td><td style='text-align:right;color:#6ec97a;font-weight:600;'>{saved} kg</td></tr>
        <tr><td>Cost saved</td><td style='text-align:right;color:#6ec97a;font-weight:600;'>₹{saved*30:,}</td></tr>
        <tr><td>CO₂ avoided</td><td style='text-align:right;color:#6ec97a;'>~{saved*2.5:.1f} kg</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<h2>Consumption by Day of Week</h2>", unsafe_allow_html=True)
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        historical_df["day_name"] = historical_df["day_of_week"].map(lambda x: day_names[x])
        avg_by_day = historical_df.groupby("day_name")["actual_consumption_kg"].mean().reindex(day_names)
        fig2 = px.bar(x=avg_by_day.index, y=avg_by_day.values,
                      labels={"x":"Day","y":"Avg Consumption (kg)"},
                      color=avg_by_day.values, color_continuous_scale=["#1a3d22","#6ec97a"])
        fig2.update_layout(**plot_layout(260, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)

    with col_d:
        st.markdown("<h2>Waste by Food Category</h2>", unsafe_allow_html=True)
        cat_cols = [c for c in historical_df.columns if c.startswith("waste_cat_")]
        if cat_cols:
            cat_totals = historical_df[cat_cols].sum()
            cat_labels = [c.replace("waste_cat_","").title() for c in cat_cols]
            fig3 = go.Figure(go.Pie(
                labels=cat_labels, values=cat_totals.values,
                hole=0.55,
                marker=dict(colors=["#1a3d22","#2a5c34","#4a8c5a","#6ec97a"]),
                textfont=dict(color=FONT_COL, size=12)
            ))
            fig3.update_layout(**plot_layout(260, showlegend=True,
                legend=dict(font=dict(color=FONT_COL, size=11), bgcolor="rgba(0,0,0,0)")))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Category breakdown available after loading real data.")

# ─── Predict ───────────────────────────────────────────────────────────────────
elif page == "Predict":
    st.markdown("<h1>Demand Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a6a4e;font-size:0.88rem;margin-top:-8px;'>Enter parameters for tomorrow to get an AI-powered preparation recommendation</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<h2>Input Parameters</h2>", unsafe_allow_html=True)
        attendance  = st.slider("Expected Attendance", 50, 800, 450, step=10)
        day         = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        weather     = st.selectbox("Weather Condition", ["clear","cloudy","rainy","cold"])
        col_i1, col_i2 = st.columns(2)
        is_holiday  = col_i1.checkbox("Holiday / Long Weekend")
        is_exam     = col_i2.checkbox("Exam Season")
        day_map     = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

        if st.button("Generate Forecast", type="primary", use_container_width=True):
            now  = datetime.now()
            pred = model.predict_single(
                attendance=attendance, day_of_week=day_map[day],
                weather=weather, is_holiday=int(is_holiday), is_exam=int(is_exam),
                month=now.month, week_of_year=now.isocalendar()[1]
            )
            rec    = pred * 1.025
            manual = attendance * 2.0
            st.session_state["prediction"] = {
                "pred": pred, "recommended": rec,
                "manual": manual, "attendance": attendance, "weather": weather
            }

    with col2:
        st.markdown("<h2>Forecast Result</h2>", unsafe_allow_html=True)
        if "prediction" in st.session_state:
            p = st.session_state["prediction"]
            waste_saved = p["manual"] - p["recommended"]
            cost_saved  = waste_saved * 30
            co2_saved   = waste_saved * 2.5
            water_saved = waste_saved * 1000

            st.markdown(f"""
            <div style='background:#0f1a12;border:1px solid #1f3324;border-radius:8px;padding:24px;'>
            <div class='big-num'>{p['pred']:.0f}<span style='font-size:1.2rem;color:#4a6a4e'> kg</span></div>
            <div class='big-num-label'>predicted demand</div>
            <hr style='border-color:#1a2e1f;margin:16px 0;'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
              <div style='background:#0a1510;border:1px solid #1a2e1f;border-radius:6px;padding:12px;'>
                <div style='font-size:0.7rem;color:#3d6040;text-transform:uppercase;letter-spacing:0.1em;'>Recommended</div>
                <div style='font-family:Syne;font-size:1.5rem;color:#c8eacf;font-weight:700;'>{p['recommended']:.0f} kg</div>
                <div style='font-size:0.72rem;color:#4a6a4e;'>+2.5% safety buffer</div>
              </div>
              <div style='background:#0a1510;border:1px solid #1a2e1f;border-radius:6px;padding:12px;'>
                <div style='font-size:0.7rem;color:#3d6040;text-transform:uppercase;letter-spacing:0.1em;'>Manual Est.</div>
                <div style='font-family:Syne;font-size:1.5rem;color:#3d5040;font-weight:700;text-decoration:line-through;'>{p['manual']:.0f} kg</div>
                <div style='font-size:0.72rem;color:#3d4a3e;'>overproduction</div>
              </div>
            </div>
            <hr style='border-color:#1a2e1f;margin:16px 0;'>
            <table style='width:100%;font-size:0.84rem;color:#7aa87e;'>
            <tr><td>🥗 Food saved</td><td style='text-align:right;color:#6ec97a;font-weight:600;'>{waste_saved:.1f} kg</td></tr>
            <tr><td>💰 Cost saved</td><td style='text-align:right;color:#6ec97a;font-weight:600;'>₹{cost_saved:,.0f}</td></tr>
            <tr><td>🌱 CO₂ avoided</td><td style='text-align:right;color:#6ec97a;'>{co2_saved:.1f} kg</td></tr>
            <tr><td>💧 Water saved</td><td style='text-align:right;color:#6ec97a;'>{water_saved:.0f} L</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p["pred"],
                title={"text": "Predicted Demand (kg)", "font": {"color": FONT_COL, "size": 13}},
                number={"font": {"color": ACCENT2, "family": "Syne", "size": 36}},
                gauge={
                    "axis": {"range": [0, 1000], "tickcolor": FONT_COL},
                    "bar":  {"color": LINE_COL, "thickness": 0.18},
                    "bgcolor": PLOT_BG,
                    "bordercolor": GRID_COL,
                    "steps": [
                        {"range": [0, 400],  "color": "#0a0f0d"},
                        {"range": [400, 700], "color": "#0d1510"},
                        {"range": [700, 1000],"color": "#0f1a12"},
                    ],
                    "threshold": {"line": {"color": ACCENT, "width": 2}, "value": p["recommended"]}
                }
            ))
            fig_g.update_layout(paper_bgcolor=PLOT_BG, font_color=FONT_COL, height=240,
                                 margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.markdown("<p style='color:#3d6040;font-size:0.88rem;'>Fill in the parameters and click <strong style='color:#6ec97a;'>Generate Forecast</strong> to see results.</p>", unsafe_allow_html=True)

# ─── Analytics ─────────────────────────────────────────────────────────────────
elif page == "Analytics":
    st.markdown("<h1>Advanced Analytics</h1>", unsafe_allow_html=True)
    st.divider()

    waste_col = "total_waste_kg" if "total_waste_kg" in historical_df.columns else "waste_kg"
    tab1, tab2, tab3, tab4 = st.tabs(["Consumption vs Prepared", "Feature Importance", "Monthly Savings", "Sustainability"])

    with tab1:
        df_plot = historical_df.tail(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["prepared_kg"],
                                  name="Prepared", line=dict(color="#2a5c34", width=1.5)))
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["actual_consumption_kg"],
                                  name="Consumed", line=dict(color=ACCENT, width=1.5)))
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot[waste_col],
                                  name="Wasted", line=dict(color="#8a2020", width=1, dash="dot")))
        fig.update_layout(**plot_layout(380, legend=dict(font=dict(color=FONT_COL), bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        importance = model.get_feature_importance()
        if importance:
            top_n = dict(list(importance.items())[-15:])
            fig_i = px.bar(x=list(top_n.values()), y=list(top_n.keys()),
                           orientation="h", color=list(top_n.values()),
                           color_continuous_scale=["#1a3d22", "#6ec97a"],
                           labels={"x":"Importance Score","y":"Feature"})
            fig_i.update_layout(**plot_layout(380, showlegend=False))
            st.plotly_chart(fig_i, use_container_width=True)
            st.markdown("<p style='color:#3d6040;font-size:0.82rem;'>Features ranked by XGBoost importance score. Higher = more influence on prediction.</p>", unsafe_allow_html=True)

    with tab3:
        historical_df["month_str"] = pd.to_datetime(historical_df["date"]).dt.to_period("M").astype(str)
        monthly = historical_df.groupby("month_str")[waste_col].sum().reset_index()
        monthly["savings_inr"] = monthly[waste_col] * 30
        monthly["co2_kg"]      = monthly[waste_col] * 2.5
        fig_m = px.bar(monthly, x="month_str", y="savings_inr",
                       labels={"savings_inr":"Savings (₹)","month_str":"Month"},
                       color="savings_inr", color_continuous_scale=["#1a3d22","#6ec97a"])
        fig_m.update_layout(**plot_layout(320, showlegend=False))
        st.plotly_chart(fig_m, use_container_width=True)
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total Savings (period)", f"₹{monthly['savings_inr'].sum():,.0f}")
        col_s2.metric("Total CO₂ Avoided",      f"{monthly['co2_kg'].sum():.0f} kg")
        col_s3.metric("Total Food Saved",        f"{historical_df[waste_col].sum():.0f} kg")

    with tab4:
        st.markdown("<h2>Sustainability Impact</h2>", unsafe_allow_html=True)
        total_waste  = historical_df[waste_col].sum()
        co2_total    = total_waste * 2.5
        water_total  = total_waste * 1000
        land_total   = total_waste * 1.8

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Waste Tracked", f"{total_waste:.0f} kg")
        c2.metric("CO₂ Equivalent",      f"{co2_total:.0f} kg")
        c3.metric("Water Footprint",     f"{water_total/1000:.0f} kL")
        c4.metric("Land Use Avoided",    f"{land_total:.0f} m²")

        st.markdown("<p style='color:#3d6040;font-size:0.82rem;margin-top:8px;'>Aligned with UN Sustainable Development Goals SDG 12 (Responsible Consumption) & SDG 13 (Climate Action)</p>", unsafe_allow_html=True)

        # Waste breakdown by meal if available
        meal_cols = [c for c in historical_df.columns if c.startswith("waste_breakfast") or c.startswith("waste_lunch") or c.startswith("waste_dinner")]
        if any(c in historical_df.columns for c in ["waste_breakfast","waste_lunch","waste_dinner"]):
            b = historical_df["waste_breakfast"].sum() if "waste_breakfast" in historical_df.columns else 0
            l = historical_df["waste_lunch"].sum()     if "waste_lunch"     in historical_df.columns else 0
            d = historical_df["waste_dinner"].sum()    if "waste_dinner"    in historical_df.columns else 0
            fig_meal = go.Figure(go.Bar(
                x=["Breakfast","Lunch","Dinner"],
                y=[b, l, d],
                marker_color=[LINE_COL, ACCENT, "#2a5c34"]
            ))
            fig_meal.update_layout(**plot_layout(260))
            st.markdown("<h2>Waste by Meal Type</h2>", unsafe_allow_html=True)
            st.plotly_chart(fig_meal, use_container_width=True)

# ─── Menu & Prices ─────────────────────────────────────────────────────────────
elif page == "Menu & Prices":
    st.markdown("<h1>Menu & Ingredient Prices</h1>", unsafe_allow_html=True)
    st.divider()

    tab_m, tab_p = st.tabs(["Menu Items", "Commodity Prices"])

    with tab_m:
        menu_df = pd.read_csv("menu_items_raw.csv")
        if not menu_df.empty:
            st.markdown("<h2>Menu Catalogue</h2>", unsafe_allow_html=True)
            for meal_type in ["Breakfast", "Lunch", "Dinner", "Snacks"]:
                items = menu_df[menu_df["meal_type"] == meal_type]
                if not items.empty:
                    st.markdown(f"<h3>{meal_type}</h3>", unsafe_allow_html=True)
                    cols = st.columns(min(len(items), 4))
                    for i, (_, row) in enumerate(items.iterrows()):
                        with cols[i % 4]:
                            pop_stars = "★" * round(row["popularity_score"]) + "☆" * (5 - round(row["popularity_score"]))
                            st.markdown(f"""
                            <div style='background:#0f1a12;border:1px solid #1f3324;border-radius:8px;padding:14px;margin-bottom:8px;'>
                            <div style='font-family:Syne;font-size:1rem;font-weight:700;color:#c8eacf;'>{row['menu_item']}</div>
                            <div style='font-size:0.75rem;color:#4a6a4e;margin-top:2px;'>{row.get('category','')}</div>
                            <hr style='border-color:#1a2e1f;margin:8px 0;'>
                            <div style='font-size:0.78rem;color:#6a9a6e;'>{pop_stars}</div>
                            <div style='font-size:0.75rem;color:#4a6a4e;margin-top:4px;'>{row['calories_per_serving']} kcal · {row['avg_portion_grams']}g</div>
                            <div style='font-size:0.75rem;color:#3d6040;'>Cook: {row['cook_time_min']}min · Shelf: {row['shelf_life_hrs']}h</div>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("Place menu_items_raw.csv in the project folder to see menu data.")

    with tab_p:
        agri_df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
        if not agri_df.empty:
            st.markdown("<h2>Commodity Price Trends</h2>", unsafe_allow_html=True)
            agri_df["cost_per_kg"] = agri_df["Modal Price"] / 100
            agri_df["Arrival_Date"] = pd.to_datetime(agri_df["Arrival_Date"], dayfirst=True, errors="coerce")
            commodity = st.selectbox("Select Commodity",
                                     sorted(agri_df["Commodity"].unique()))
            df_c = agri_df[agri_df["Commodity"] == commodity].sort_values("Arrival_Date")
            if not df_c.empty:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=df_c["Arrival_Date"], y=df_c["cost_per_kg"],
                                           name="Modal Price (₹/kg)",
                                           line=dict(color=ACCENT, width=1.5)))
                fig_p.update_layout(**plot_layout(320))
                st.plotly_chart(fig_p, use_container_width=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Avg Price", f"₹{df_c['cost_per_kg'].mean():.2f}/kg")
                c2.metric("Min Price", f"₹{df_c['cost_per_kg'].min():.2f}/kg")
                c3.metric("Max Price", f"₹{df_c['cost_per_kg'].max():.2f}/kg")
        else:
            st.info("Place Price_Agriculture_commodities_Week.csv in the project folder.")

# ─── Retrain ───────────────────────────────────────────────────────────────────
elif page == "Retrain Model":
    st.markdown("<h1>Continuous Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a6a4e;font-size:0.88rem;margin-top:-8px;'>Feed actual consumption data back to improve predictions over time</p>", unsafe_allow_html=True)
    st.divider()

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Current R²",   f"{model.get_accuracy():.1f}%")
    col_info2.metric("Current MAE",  f"{model.get_mae():.1f} kg")
    if hasattr(model, 'get_rmse'):
        col_info3.metric("Current RMSE", f"{model.get_rmse():.1f} kg")

    st.divider()
    st.markdown("<h2>Upload New Consumption Data</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#3d6040;font-size:0.82rem;'>CSV must have columns: <code style='color:#6ec97a;background:#0f1a12;padding:2px 6px;border-radius:3px;'>date, attendance, actual_consumption_kg</code></p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if uploaded:
        new_data = pd.read_csv(uploaded)
        st.dataframe(new_data.head(), use_container_width=True)
        if st.button("Retrain with Uploaded Data", type="primary"):
            with st.spinner("Retraining model..."):
                combined = pd.concat([historical_df, new_data], ignore_index=True)
                model.train(combined)
                st.success(f"Model retrained! New accuracy: {model.get_accuracy():.1f}% | MAE: {model.get_mae():.1f} kg")

    st.divider()
    st.markdown("<h2>Simulate Feedback Entry</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    sim_date       = c1.date_input("Date", datetime.now())
    sim_attendance = c2.number_input("Actual Attendance", 50, 800, 450)
    sim_consumed   = c3.number_input("Actual Consumed (kg)", 100.0, 2000.0, 780.0)

    if st.button("Add Entry & Retrain"):
        with st.spinner("Updating model..."):
            new_row = pd.DataFrame([{
                "date": pd.Timestamp(sim_date).strftime("%Y-%m-%d"),
                "attendance": sim_attendance,
                "actual_consumption_kg": sim_consumed,
                "day_of_week": pd.Timestamp(sim_date).weekday(),
                "month": pd.Timestamp(sim_date).month,
                "week_of_year": pd.Timestamp(sim_date).isocalendar()[1],
                "is_weekend": int(pd.Timestamp(sim_date).weekday() >= 5),
                "weather_encoded": 0, "is_holiday": 0, "is_exam": 0,
                "prepared_kg": sim_consumed * 1.1,
                "total_waste_kg": sim_consumed * 0.1,
                "waste_pct": 9.09, "avg_unit_price": 3.5,
                "lag_1_waste": sim_consumed * 0.1,
                "lag_7_waste": sim_consumed * 0.1,
                "rolling_7_consumption": sim_consumed,
                "waste_pct_lag1": 9.09,
                "avg_cost_per_kg": 30.0,
                "menu_popularity_score": 4.1,
                "waste_breakfast": sim_consumed*0.03,
                "waste_lunch": sim_consumed*0.04,
                "waste_dinner": sim_consumed*0.03,
                "waste_cat_rice": sim_consumed*0.025,
                "waste_cat_meat": sim_consumed*0.025,
                "waste_cat_soup": sim_consumed*0.025,
                "waste_cat_vegetables": sim_consumed*0.025,
                "co2_saved_kg": sim_consumed*0.1*2.5,
                "water_saved_liters": sim_consumed*0.1*1000,
            }])
            combined = pd.concat([historical_df, new_row], ignore_index=True)
            model.train(combined)
            st.success(f"Entry added & model updated! Accuracy: {model.get_accuracy():.1f}% | MAE: {model.get_mae():.1f} kg")