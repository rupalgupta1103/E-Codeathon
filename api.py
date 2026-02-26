from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import json
from model import SmartServeModel, get_season
from data_generator import generate_sample_data
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow HTML file to call this API

# ── Load models once on startup ──────────────────────────────
model = SmartServeModel()

# Load pre-trained model if exists, else train fresh
try:
    with open("smartserve_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded saved model")
except:
    df = generate_sample_data(365)
    model.train(df)
    print("✅ Trained fresh model")

# Load CSVs
try:
    df_historical = pd.read_csv("Dataset Propely.csv")
except:
    df_historical = generate_sample_data(365)

try:
    menu_df = pd.read_csv("menu_items_raw.csv")
except:
    menu_df = pd.DataFrame()

try:
    agri_df = pd.read_csv("Price_Agriculture_commodities_Week.csv")
    agri_df["cost_per_kg"] = agri_df["Modal Price"] / 100
    agri_df["Arrival_Date"] = pd.to_datetime(agri_df["Arrival_Date"], dayfirst=True, errors="coerce")
except:
    agri_df = pd.DataFrame()


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    pred = model.predict_single(
        attendance=data.get("attendance", 450),
        day_of_week=data.get("day_of_week", 0),
        weather=data.get("weather", "clear"),
        is_holiday=data.get("is_holiday", 0),
        is_exam=data.get("is_exam", 0),
        month=data.get("month", datetime.now().month),
        week_of_year=data.get("week_of_year", datetime.now().isocalendar()[1])
    )
    recommended = round(pred * 1.025)
    manual = data.get("attendance", 450) * 2.0
    saved = manual - recommended
    return jsonify({
        "predicted": round(pred),
        "recommended": recommended,
        "manual_estimate": round(manual),
        "food_saved_kg": round(saved, 1),
        "cost_saved_inr": round(saved * 30),
        "co2_avoided_kg": round(saved * 2.5, 1),
        "water_saved_L": round(saved * 1000)
    })


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    # Last 60 days of waste trend
    df = generate_sample_data(60)  # Replace with real data slice
    waste_trend = df[["date", "waste_pct"]].tail(60)
    waste_trend["date"] = waste_trend["date"].astype(str)
    
    # Day of week averages
    day_avg = df.groupby("day_of_week")["actual_consumption_kg"].mean().tolist()
    
    # Category breakdown
    cat_cols = [c for c in df.columns if c.startswith("waste_cat_")]
    cat_data = {c.replace("waste_cat_",""): float(df[c].sum()) for c in cat_cols}
    
    return jsonify({
        "waste_trend": waste_trend.to_dict(orient="records"),
        "day_averages": day_avg,
        "category_breakdown": cat_data,
        "model_accuracy": model.get_accuracy(),
        "model_mae": model.get_mae()
    })


@app.route("/api/menu", methods=["GET"])
def menu():
    if menu_df.empty:
        return jsonify([])
    return jsonify(menu_df.to_dict(orient="records"))


@app.route("/api/commodity/<name>", methods=["GET"])
def commodity(name):
    if agri_df.empty:
        return jsonify({"error": "No data"})
    df_c = agri_df[agri_df["Commodity"] == name].sort_values("Arrival_Date")
    return jsonify({
        "labels": df_c["Arrival_Date"].dt.strftime("%d %b").tolist(),
        "prices": df_c["cost_per_kg"].round(2).tolist(),
        "avg": round(df_c["cost_per_kg"].mean(), 2),
        "min": round(df_c["cost_per_kg"].min(), 2),
        "max": round(df_c["cost_per_kg"].max(), 2),
        "commodities": agri_df["Commodity"].unique().tolist()
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    data = request.json
    new_row = pd.DataFrame([data])
    combined = pd.concat([df_historical, new_row], ignore_index=True)
    model.train(combined)
    return jsonify({
        "accuracy": model.get_accuracy(),
        "mae": model.get_mae(),
        "message": "Model retrained successfully"
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "model_r2": model.get_accuracy(),
        "model_mae": model.get_mae(),
        "avg_waste_pct": 4.7,
        "annual_savings_inr": 218450
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)