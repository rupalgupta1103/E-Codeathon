"""
SmartServe AI - Enriched Data Generator
Uses all 4 real datasets with full feature engineering.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
MESS_CSV    = os.path.join(DATA_DIR, "Dataset Propely.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "india_2000_2024_daily_weather.csv")
AGRI_CSV    = os.path.join(DATA_DIR, "Price_Agriculture_commodities_Week.csv")
MENU_CSV    = os.path.join(DATA_DIR, "menu_items_raw.csv")

WEATHER_MAP = {"clear": 0, "cloudy": 1, "rainy": 2, "cold": 3}

# Monthly average temperatures for Delhi (from 25yr historical data)
DELHI_MONTHLY_TEMP = {1:19.7,2:23.6,3:29.8,4:36.6,5:39.4,6:38.4,7:33.8,8:32.5,9:32.5,10:32.0,11:27.3,12:22.0}
DELHI_MONTHLY_PRECIP = {1:0.83,2:1.0,3:0.75,4:0.43,5:0.85,6:2.16,7:6.4,8:5.39,9:3.57,10:0.37,11:0.16,12:0.20}


def _encode_weather(temp, precip):
    if precip > 5:  return 2
    if temp  < 15:  return 3
    if precip > 1:  return 1
    return 0


def _load_weather_monthly_avg():
    """Load actual weather or return historical monthly averages."""
    if not os.path.exists(WEATHER_CSV):
        return None
    w = pd.read_csv(WEATHER_CSV)
    w['date'] = pd.to_datetime(w['date'])
    delhi = w[w['city'] == 'Delhi'].copy()
    delhi['month'] = delhi['date'].dt.month
    return delhi.groupby('month')[['temperature_2m_max', 'precipitation_sum']].mean().to_dict()


def _load_agri_cost():
    if not os.path.exists(AGRI_CSV):
        return 30.0
    agri = pd.read_csv(AGRI_CSV)
    agri['Arrival_Date'] = pd.to_datetime(agri['Arrival_Date'], dayfirst=True, errors='coerce')
    agri = agri.dropna(subset=['Arrival_Date', 'Modal Price'])
    mess_items = ['Rice','Wheat','Tomato','Potato','Onion','Cabbage','Cauliflower','Brinjal']
    filtered = agri[agri['Commodity'].isin(mess_items)].copy()
    if filtered.empty:
        filtered = agri.copy()
    filtered['cost_per_kg'] = filtered['Modal Price'] / 100
    filtered['week'] = filtered['Arrival_Date'].dt.to_period('W').apply(lambda r: r.start_time)
    return filtered.groupby('week')['cost_per_kg'].mean().reset_index().rename(columns={'cost_per_kg':'avg_cost_per_kg'})


def _load_menu_popularity():
    if not os.path.exists(MENU_CSV):
        return {'Breakfast': 4.0, 'Lunch': 4.2, 'Dinner': 4.1, 'Snacks': 3.8}
    menu = pd.read_csv(MENU_CSV)
    return menu.groupby('meal_type')['popularity_score'].mean().to_dict()


def _load_real_data():
    if not os.path.exists(MESS_CSV):
        return None

    mess = pd.read_csv(MESS_CSV)
    mess['Date'] = pd.to_datetime(mess['Date'])

    # Daily aggregation
    daily = mess.groupby('Date').agg(
        total_waste_kg   =('Waste_Weight_kg', 'sum'),
        total_cost_loss  =('Cost_Loss',       'sum'),
        meals_served     =('Meal',            'count'),
        avg_unit_price   =('Unit_Price_per_kg','mean'),
    ).reset_index()

    # Meal-type waste breakdown
    meal_waste = mess.groupby(['Date','Meal'])['Waste_Weight_kg'].sum().unstack(fill_value=0).reset_index()
    meal_waste.columns = ['Date'] + [f'waste_{c.lower()}' for c in meal_waste.columns[1:]]
    daily = daily.merge(meal_waste, on='Date', how='left')

    # Food category waste breakdown
    cat_waste = mess.groupby(['Date','Food_Category'])['Waste_Weight_kg'].sum().unstack(fill_value=0).reset_index()
    cat_waste.columns = ['Date'] + [f'waste_cat_{c.lower().replace(" ","")}' for c in cat_waste.columns[1:]]
    daily = daily.merge(cat_waste, on='Date', how='left')

    # Core features
    daily['attendance']            = (daily['meals_served'] / 3).clip(lower=50).astype(int)
    daily['actual_consumption_kg'] = (daily['total_waste_kg'] / 0.15).round(2)
    daily['prepared_kg']           = (daily['actual_consumption_kg'] + daily['total_waste_kg']).round(2)
    daily['day_of_week']           = daily['Date'].dt.dayofweek
    daily['month']                 = daily['Date'].dt.month
    daily['week_of_year']          = daily['Date'].dt.isocalendar().week.astype(int)
    daily['is_weekend']            = (daily['day_of_week'] >= 5).astype(int)
    daily['waste_pct']             = (daily['total_waste_kg'] / daily['prepared_kg'] * 100).round(2)
    daily['is_holiday']            = 0
    daily['is_exam']               = 0

    # Weather using historical monthly averages (2025 data beyond weather CSV range)
    daily['temp_avg']   = daily['month'].map(DELHI_MONTHLY_TEMP)
    daily['precip_avg'] = daily['month'].map(DELHI_MONTHLY_PRECIP)
    daily['weather_encoded'] = daily.apply(
        lambda r: _encode_weather(r['temp_avg'], r['precip_avg']), axis=1
    )

    # Lag & rolling features
    daily = daily.sort_values('Date').reset_index(drop=True)
    mean_waste = daily['total_waste_kg'].mean()
    mean_cons  = daily['actual_consumption_kg'].mean()
    daily['lag_1_waste']          = daily['total_waste_kg'].shift(1).fillna(mean_waste)
    daily['lag_7_waste']          = daily['total_waste_kg'].shift(7).fillna(mean_waste)
    daily['rolling_7_consumption']= daily['actual_consumption_kg'].rolling(7, min_periods=1).mean()
    daily['waste_pct_lag1']       = daily['waste_pct'].shift(1).fillna(daily['waste_pct'].mean())

    # Agriculture cost
    agri_weekly = _load_agri_cost()
    if isinstance(agri_weekly, pd.DataFrame):
        daily['week'] = daily['Date'].dt.to_period('W').apply(lambda r: r.start_time)
        daily = daily.merge(agri_weekly, on='week', how='left')
        daily['avg_cost_per_kg'] = daily['avg_cost_per_kg'].fillna(agri_weekly['avg_cost_per_kg'].mean()).round(2)
        daily.drop(columns=['week'], inplace=True)
    else:
        daily['avg_cost_per_kg'] = agri_weekly

    # Menu popularity
    menu_pop = _load_menu_popularity()
    daily['menu_popularity_score'] = daily['month'].apply(
        lambda m: round(np.mean(list(menu_pop.values())), 2)
    )

    # Sustainability metrics
    daily['co2_saved_kg']     = (daily['total_waste_kg'] * 2.5).round(2)
    daily['water_saved_liters']= (daily['total_waste_kg'] * 1000).round(0)

    daily['date'] = daily['Date'].dt.strftime('%Y-%m-%d')
    daily.drop(columns=['Date', 'meals_served', 'total_cost_loss', 'temp_avg', 'precip_avg'], errors='ignore', inplace=True)

    return daily


def _generate_synthetic_data(days=365, seed=42):
    rng   = np.random.default_rng(seed)
    start = datetime.now() - timedelta(days=days)
    dates = [start + timedelta(days=i) for i in range(days)]
    records = []
    for i, date in enumerate(dates):
        dow, month = date.weekday(), date.month
        base = 450
        if dow in (5,6): base -= rng.integers(30,80)
        if dow == 0:     base += rng.integers(10,30)
        wp = [0.50,0.25,0.15,0.10]
        if month in (12,1,2):  wp = [0.25,0.30,0.10,0.35]
        elif month in (6,7,8): wp = [0.30,0.30,0.30,0.10]
        weather     = rng.choice(["clear","cloudy","rainy","cold"], p=wp)
        weather_enc = WEATHER_MAP[weather]
        is_holiday  = int(rng.random() < 0.055)
        is_exam     = int((i % 120) < 14)
        attendance  = int(base*(0.85 if is_holiday else 1.0)*(1.05 if is_exam else 1.0)
                        *(0.95 if weather=="rainy" else 1.0)+rng.integers(-20,20))
        attendance  = max(50, min(800, attendance))
        wm = {"clear":1.0,"cloudy":1.02,"rainy":1.08,"cold":1.12}
        actual   = (attendance*1.73*wm[weather]*(0.82 if is_holiday else 1.0)
                   *(1.08 if is_exam else 1.0)*rng.uniform(0.95,1.05))
        prepared = actual * rng.uniform(1.10,1.20)
        waste    = prepared - actual
        waste_pct= (waste/prepared*100)
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "attendance": attendance, "day_of_week": dow, "month": month,
            "week_of_year": date.isocalendar()[1], "is_weekend": int(dow>=5),
            "weather_encoded": weather_enc, "is_holiday": is_holiday, "is_exam": is_exam,
            "actual_consumption_kg": round(actual,2), "prepared_kg": round(prepared,2),
            "total_waste_kg": round(waste,2), "waste_pct": round(waste_pct,2),
            "avg_unit_price": round(rng.uniform(2,8),2),
            "waste_breakfast": round(waste*0.3,2), "waste_lunch": round(waste*0.4,2),
            "waste_dinner": round(waste*0.3,2),
            "waste_cat_rice": round(waste*0.25,2), "waste_cat_meat": round(waste*0.25,2),
            "waste_cat_soup": round(waste*0.25,2), "waste_cat_vegetables": round(waste*0.25,2),
            "lag_1_waste": round(waste*rng.uniform(0.8,1.2),2),
            "lag_7_waste": round(waste*rng.uniform(0.8,1.2),2),
            "rolling_7_consumption": round(actual*rng.uniform(0.9,1.1),2),
            "waste_pct_lag1": round(waste_pct*rng.uniform(0.9,1.1),2),
            "avg_cost_per_kg": round(rng.uniform(20,45),2),
            "menu_popularity_score": round(rng.uniform(3.5,4.8),2),
            "co2_saved_kg": round(waste*2.5,2),
            "water_saved_liters": round(waste*1000,0),
        })
    return pd.DataFrame(records)


def generate_sample_data(days=365, seed=42):
    real = _load_real_data()
    if real is not None and len(real) > 0:
        print(f"[SmartServe] Loaded {len(real)} rows of REAL enriched data ({real.shape[1]} features).")
        if len(real) < days:
            needed    = days - len(real)
            synthetic = _generate_synthetic_data(needed, seed)
            for col in real.columns:
                if col not in synthetic.columns:
                    synthetic[col] = 0
            for col in synthetic.columns:
                if col not in real.columns:
                    real[col] = 0
            df = pd.concat([synthetic, real], ignore_index=True)
            print(f"[SmartServe] Padded with {needed} synthetic rows → total {len(df)}")
        else:
            df = real
    else:
        print("[SmartServe] Real data not found — using synthetic data.")
        df = _generate_synthetic_data(days, seed)
    return df


def get_menu_items():
    if os.path.exists(MENU_CSV):
        return pd.read_csv(MENU_CSV)
    return pd.DataFrame()


def get_agri_prices():
    if not os.path.exists(AGRI_CSV):
        return pd.DataFrame()
    agri = pd.read_csv(AGRI_CSV)
    agri['Arrival_Date'] = pd.to_datetime(agri['Arrival_Date'], dayfirst=True, errors='coerce')
    mess_items = ['Rice','Wheat','Tomato','Potato','Onion','Cabbage','Cauliflower']
    return agri[agri['Commodity'].isin(mess_items)].copy()


def get_sample_prediction_input():
    return {"attendance":460,"day_of_week":2,"weather":"clear","is_holiday":0,"is_exam":0}


if __name__ == "__main__":
    df = generate_sample_data(365)
    print(df.head(2).to_string())
    print(f"\nShape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Avg waste: {df['total_waste_kg'].mean():.1f} kg/day")
    df.to_csv("sample_data.csv", index=False)
    print("Saved!")