"""
Smart Mess Optimization & Dynamic Meal Planning System
Model Training Script — XGBoost + Random Forest Regressor
Datasets: Dataset Propely.csv | india_2000_2024_daily_weather.csv
          menu_items_raw.csv  | Price_Agriculture_commodities_Week.csv
Author: Rupal Gupta
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# 1. LOAD DATASETS
# ─────────────────────────────────────────────

print("Loading all 4 datasets...")

df_main    = pd.read_csv("Dataset Propely.csv")
df_weather = pd.read_csv("india_2000_2024_daily_weather.csv")
df_menu    = pd.read_csv("menu_items_raw.csv")
df_prices  = pd.read_csv("Price_Agriculture_commodities_Week.csv")

print(f"  Dataset Propely          : {df_main.shape}")
print(f"  Weather                  : {df_weather.shape}")
print(f"  Menu items               : {df_menu.shape}")
print(f"  Agricultural prices      : {df_prices.shape}")

# ─────────────────────────────────────────────
# 2. PREPROCESS: MAIN DATASET  (2025 dates)
# ─────────────────────────────────────────────

print("\nPreprocessing main dataset...")

df_main["Date"] = pd.to_datetime(df_main["Date"])

df_daily = df_main.groupby("Date").agg(
    total_waste_kg  = ("Waste_Weight_kg",  "sum"),
    total_cost_loss = ("Cost_Loss",        "sum"),
    avg_unit_price  = ("Unit_Price_per_kg", "mean"),
    meal_count      = ("Meal",             "count"),
).reset_index()

# Date-derived features (no external join needed)
df_daily["day_of_week"]  = df_daily["Date"].dt.dayofweek
df_daily["month"]        = df_daily["Date"].dt.month
df_daily["week_of_year"] = df_daily["Date"].dt.isocalendar().week.astype(int)
df_daily["is_weekend"]   = df_daily["day_of_week"].isin([5, 6]).astype(int)

# Season from month
def get_season(m):
    if m in [12, 1, 2]:     return 0  # Winter
    elif m in [3, 4, 5]:    return 1  # Summer
    elif m in [6, 7, 8, 9]: return 2  # Monsoon
    else:                   return 3  # Autumn

df_daily["season"] = df_daily["month"].apply(get_season)

# Dominant meal per day → encoded
meal_mode = df_main.groupby("Date")["Meal"].agg(lambda x: x.mode()[0]).reset_index()
meal_mode.columns = ["Date", "dominant_meal"]
df_daily = df_daily.merge(meal_mode, on="Date", how="left")
le_meal = LabelEncoder()
df_daily["dominant_meal_enc"] = le_meal.fit_transform(df_daily["dominant_meal"].astype(str))

# Dominant food category per day → encoded
cat_mode = df_main.groupby("Date")["Food_Category"].agg(lambda x: x.mode()[0]).reset_index()
cat_mode.columns = ["Date", "dominant_category"]
df_daily = df_daily.merge(cat_mode, on="Date", how="left")
le_cat = LabelEncoder()
df_daily["dominant_category_enc"] = le_cat.fit_transform(df_daily["dominant_category"].astype(str))

# Active canteen sections per day
section_count = df_main.groupby("Date")["Canteen_Section"].nunique().reset_index()
section_count.columns = ["Date", "active_sections"]
df_daily = df_daily.merge(section_count, on="Date", how="left")

df = df_daily.copy()
print(f"  Daily aggregated rows: {len(df)}")

# ─────────────────────────────────────────────
# 3. WEATHER — merge by (month, day_of_week) averages
#    Weather CSV covers 2000-2024; main data is 2025.
#    Solution: compute climatological averages per month
#    and join on month (captures seasonal patterns).
# ─────────────────────────────────────────────

print("Integrating weather climatology (month-based averages)...")

df_weather["date"] = pd.to_datetime(df_weather["date"])
df_weather_delhi   = df_weather[df_weather["city"] == "Delhi"].copy()
df_weather_delhi["month"] = df_weather_delhi["date"].dt.month

weather_clim = df_weather_delhi.groupby("month").agg(
    temp_avg      = ("temperature_2m_max",  lambda x: (x + df_weather_delhi.loc[x.index, "temperature_2m_min"]) .mean() / 2 if False else x.mean()),
    temp_max      = ("temperature_2m_max",  "mean"),
    temp_min      = ("temperature_2m_min",  "mean"),
    precipitation = ("precipitation_sum",   "mean"),
    rainfall      = ("rain_sum",            "mean"),
    wind_speed    = ("wind_speed_10m_max",  "mean"),
).reset_index()

# Recompute temp_avg properly
df_weather_delhi["temp_avg"] = (df_weather_delhi["temperature_2m_max"] + df_weather_delhi["temperature_2m_min"]) / 2
weather_clim2 = df_weather_delhi.groupby("month").agg(
    temp_avg      = ("temp_avg",           "mean"),
    temp_max      = ("temperature_2m_max", "mean"),
    temp_min      = ("temperature_2m_min", "mean"),
    precipitation = ("precipitation_sum",  "mean"),
    rainfall      = ("rain_sum",           "mean"),
    wind_speed    = ("wind_speed_10m_max", "mean"),
).reset_index()

weather_clim2["is_rainy_season"] = weather_clim2["rainfall"].apply(lambda r: 1 if r > 5 else 0)
df = df.merge(weather_clim2, on="month", how="left")
print(f"  Weather climatology merged on month. NaNs: {df[['temp_avg','rainfall']].isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 4. MENU — merge on dominant_meal
# ─────────────────────────────────────────────

print("Integrating menu features...")

menu_stats = df_menu.groupby("meal_type").agg(
    avg_calories   = ("calories_per_serving", "mean"),
    avg_portion_g  = ("avg_portion_grams",    "mean"),
    avg_popularity = ("popularity_score",     "mean"),
    avg_cook_time  = ("cook_time_min",        "mean"),
    avg_shelf_life = ("shelf_life_hrs",       "mean"),
    staple_ratio   = ("is_staple",            "mean"),
).reset_index()

df = df.merge(menu_stats, left_on="dominant_meal", right_on="meal_type", how="left")
df.drop(columns=["meal_type"], inplace=True, errors="ignore")
print(f"  Menu features merged. NaNs: {df[['avg_calories','avg_popularity']].isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 5. AGRICULTURAL PRICES — merge by month averages
#    Price CSV covers Jul-Aug 2023; main data is 2025.
#    Solution: compute average price per month number
#    and join on month (captures seasonal price patterns).
# ─────────────────────────────────────────────

print("Integrating agricultural price averages (month-based)...")

df_prices["Arrival_Date"] = pd.to_datetime(df_prices["Arrival_Date"], dayfirst=True)
df_prices["month"] = df_prices["Arrival_Date"].dt.month

price_monthly = df_prices.groupby("month").agg(
    avg_modal_price  = ("Modal Price", "mean"),
    avg_min_price    = ("Min Price",   "mean"),
    avg_max_price    = ("Max Price",   "mean"),
    commodity_count  = ("Commodity",   "nunique"),
    price_volatility = ("Modal Price", "std"),
).reset_index()
price_monthly["price_volatility"] = price_monthly["price_volatility"].fillna(0)

df = df.merge(price_monthly, on="month", how="left")

# For months with no price data, use overall mean
for col in ["avg_modal_price", "avg_min_price", "avg_max_price", "price_volatility", "commodity_count"]:
    df[col] = df[col].fillna(df[col].mean())

print(f"  Price features merged. NaNs: {df[['avg_modal_price']].isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 6. LAG & ROLLING FEATURES + CLEANUP
# ─────────────────────────────────────────────

df = df.sort_values("Date").reset_index(drop=True)
df["cost_per_waste_kg"]      = df["total_cost_loss"] / (df["total_waste_kg"] + 0.01)
df["waste_lag_1"]            = df["total_waste_kg"].shift(1)
df["waste_lag_7"]            = df["total_waste_kg"].shift(7)
df["waste_7day_rolling_avg"] = df["total_waste_kg"].rolling(7, min_periods=1).mean().shift(1)

# Fill lag NaNs with column median (only affects first few rows)
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"  Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Total NaNs remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 7. FEATURE SELECTION & TARGET
# ─────────────────────────────────────────────

TARGET = "total_waste_kg"

FEATURES = [
    # Main dataset features
    "meal_count", "avg_unit_price", "active_sections",
    "dominant_meal_enc", "dominant_category_enc",
    "cost_per_waste_kg", "waste_lag_1", "waste_lag_7", "waste_7day_rolling_avg",
    # Date features
    "day_of_week", "month", "week_of_year", "is_weekend", "season",
    # Weather climatology
    "temp_avg", "temp_max", "temp_min",
    "precipitation", "rainfall", "wind_speed", "is_rainy_season",
    # Menu features
    "avg_calories", "avg_portion_g", "avg_popularity",
    "avg_cook_time", "avg_shelf_life", "staple_ratio",
    # Price features
    "avg_modal_price", "avg_min_price", "price_volatility", "commodity_count",
]

FEATURES = [f for f in FEATURES if f in df.columns]

print(f"\n  Target   : {TARGET}")
print(f"  Features : {len(FEATURES)} → {FEATURES}")

df_model = df[FEATURES + [TARGET]].dropna()
X, y = df_model[FEATURES], df_model[TARGET]
print(f"  Training samples: {len(X)}")

# ─────────────────────────────────────────────
# 8. SPLIT + SCALE
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 9. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()
    print(f"\n── {name} ──")
    print(f"  MAE  : {mae:.3f} kg")
    print(f"  RMSE : {rmse:.3f} kg")
    print(f"  R²   : {r2:.4f}")
    print(f"  CV R²: {cv:.4f}")
    return model, preds

print("\nTraining models...")

xgb_model, xgb_preds = evaluate(
    "XGBoost Regressor",
    xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    ),
    X_train_scaled, X_test_scaled, y_train, y_test
)

rf_model, rf_preds = evaluate(
    "Random Forest Regressor",
    RandomForestRegressor(
        n_estimators=200, max_depth=8,
        min_samples_split=3, random_state=42, n_jobs=-1
    ),
    X_train_scaled, X_test_scaled, y_train, y_test
)

# ─────────────────────────────────────────────
# 10. OPTIMIZATION LOGIC
# ─────────────────────────────────────────────

SAFETY_BUFFER = 0.03

def recommend_preparation(predicted_waste_kg, avg_unit_price=3.0):
    cost_saving = predicted_waste_kg * avg_unit_price
    co2_saved   = round(predicted_waste_kg * 2.5, 2)
    return {
        "predicted_waste_kg":    round(predicted_waste_kg, 2),
        "allowable_waste_kg":    round(predicted_waste_kg * (1 + SAFETY_BUFFER), 2),
        "estimated_cost_saving": round(cost_saving, 2),
        "co2_reduction_kg":      co2_saved,
    }

print("\n── Sample Predictions (XGBoost) ──")
for i, (pred, actual) in enumerate(zip(xgb_preds[:5], y_test.values[:5])):
    r = recommend_preparation(pred)
    print(f"  [{i+1}] Predicted: {pred:.2f} kg | Actual: {actual:.2f} kg | "
          f"Saving: ₹{r['estimated_cost_saving']:.0f} | CO₂: {r['co2_reduction_kg']} kg")

# ─────────────────────────────────────────────
# 11. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

print("\n── Top 15 Feature Importances (XGBoost) ──")
imp = pd.DataFrame({"feature": FEATURES, "importance": xgb_model.feature_importances_})
print(imp.sort_values("importance", ascending=False).head(15).to_string(index=False))

# ─────────────────────────────────────────────
# 12. SAVE ARTIFACTS
# ─────────────────────────────────────────────

joblib.dump(xgb_model,  "xgb_model.pkl")
joblib.dump(rf_model,   "rf_model.pkl")
joblib.dump(scaler,     "scaler.pkl")
joblib.dump(FEATURES,   "features.pkl")
joblib.dump({"meal": le_meal, "category": le_cat}, "label_encoders.pkl")

print("\n✅ Saved: xgb_model.pkl | rf_model.pkl | scaler.pkl | features.pkl | label_encoders.pkl")

# ─────────────────────────────────────────────
# 13. INFERENCE FUNCTION  ← used by app.py
# ─────────────────────────────────────────────

def predict_waste(input_dict: dict, model_path="xgb_model.pkl") -> dict:
    """
    Predict food waste for a single day. Import and call from app.py.

    Example
    -------
    from model import predict_waste
    result = predict_waste({
        "meal_count": 3, "avg_unit_price": 3.5, "active_sections": 3,
        "day_of_week": 2, "month": 6, "is_weekend": 0,
        "temp_avg": 35.0, "rainfall": 150.0, "season": 2,
        "avg_modal_price": 3000, "avg_calories": 320,
    })
    print(result)
    """
    model = joblib.load(model_path)
    sc    = joblib.load("scaler.pkl")
    feats = joblib.load("features.pkl")

    row = pd.DataFrame([input_dict]).reindex(columns=feats, fill_value=0)
    predicted_waste = float(model.predict(sc.transform(row))[0])
    return recommend_preparation(
        predicted_waste,
        avg_unit_price=input_dict.get("avg_unit_price", 3.0)
    )
# ─────────────────────────────────────────────
# 14. SmartServeModel CLASS — used by app.py
# ─────────────────────────────────────────────

class SmartServeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = []
        self._r2 = 0.0
        self._mae = 0.0
        self._rmse = 0.0
        self._feature_importance = {}

    def train(self, df: pd.DataFrame):
        """Train on any DataFrame that has the required columns."""
        TARGET = "total_waste_kg" if "total_waste_kg" in df.columns else "waste_kg"

        CANDIDATE_FEATURES = [
            "attendance", "actual_consumption_kg", "prepared_kg",
            "meal_count", "avg_unit_price", "active_sections",
            "dominant_meal_enc", "dominant_category_enc",
            "cost_per_waste_kg", "waste_lag_1", "waste_lag_7",
            "waste_7day_rolling_avg", "lag_1_waste", "lag_7_waste",
            "rolling_7_consumption", "waste_pct_lag1", "avg_cost_per_kg",
            "menu_popularity_score",
            "day_of_week", "month", "week_of_year", "is_weekend", "season",
            "temp_avg", "temp_max", "temp_min", "precipitation",
            "rainfall", "wind_speed", "is_rainy_season",
            "avg_calories", "avg_portion_g", "avg_popularity",
            "avg_cook_time", "avg_shelf_life", "staple_ratio",
            "avg_modal_price", "avg_min_price", "price_volatility",
            "commodity_count", "weather_encoded", "is_holiday", "is_exam",
        ]

        self.features = [f for f in CANDIDATE_FEATURES if f in df.columns and f != TARGET]

        if not self.features:
            # fallback: use all numeric columns except target
            self.features = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]

        df_clean = df[self.features + [TARGET]].dropna()
        if len(df_clean) < 5:
            return  # not enough data

        X = df_clean[self.features]
        y = df_clean[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
        )
        self.model.fit(X_train_s, y_train)

        preds = self.model.predict(X_test_s)
        self._mae  = float(mean_absolute_error(y_test, preds))
        self._rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        self._r2   = float(r2_score(y_test, preds)) * 100  # as %

        imp = dict(zip(self.features, self.model.feature_importances_))
        self._feature_importance = dict(sorted(imp.items(), key=lambda x: x[1]))

    def predict_single(self, attendance=450, day_of_week=0, weather="clear",
                       month=6, week_of_year=24, is_holiday=0, is_exam=0) -> float:
        """Return predicted demand in kg."""
        weather_map = {"clear": 0, "cloudy": 1, "rainy": 2, "cold": 3}
        row = {
            "attendance":        attendance,
            "day_of_week":       day_of_week,
            "weather_encoded":   weather_map.get(weather, 0),
            "month":             month,
            "week_of_year":      week_of_year,
            "is_weekend":        int(day_of_week >= 5),
            "is_holiday":        is_holiday,
            "is_exam":           is_exam,
            "season":            get_season(month),
        }
        if self.model is None or not self.features:
            # model not trained yet — return a sensible default
            return attendance * 1.7

        X = pd.DataFrame([row]).reindex(columns=self.features, fill_value=0)
        return float(self.model.predict(self.scaler.transform(X))[0])

    def get_accuracy(self) -> float:
        """Return R² as a percentage (0–100)."""
        return round(self._r2, 2)

    def get_mae(self) -> float:
        return round(self._mae, 2)

    def get_rmse(self) -> float:
        return round(self._rmse, 2)

    def get_feature_importance(self) -> dict:
        return self._feature_importance