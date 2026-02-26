"""
SmartServe AI - Synthetic Data Generator
Generates realistic hostel mess consumption data for training/demo.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_data(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Simulate daily mess consumption data for a hostel.
    
    Factors modelled:
      - Base demand ~780 kg/day
      - Day-of-week effect (weekends slightly lower)
      - Weather effect (rain/cold → more indoor eating)
      - Holiday dip  (-15%)
      - Exam season spike (+10%) — students eat more
      - Random noise (±5%)
    """
    rng = np.random.default_rng(seed)
    start = datetime.now() - timedelta(days=days)
    dates = [start + timedelta(days=i) for i in range(days)]

    records = []
    for i, date in enumerate(dates):
        dow = date.weekday()  # 0=Mon, 6=Sun
        month = date.month

        # Base attendance
        base_attendance = 450
        if dow in (5, 6):   # weekend
            base_attendance -= rng.integers(30, 80)
        if dow == 0:        # Monday surge
            base_attendance += rng.integers(10, 30)

        # Weather
        weather_choices = ["clear", "cloudy", "rainy", "cold"]
        weather_probs = [0.50, 0.25, 0.15, 0.10]
        if month in (12, 1, 2):
            weather_probs = [0.25, 0.30, 0.10, 0.35]
        elif month in (6, 7, 8):
            weather_probs = [0.30, 0.30, 0.30, 0.10]
        weather = rng.choice(weather_choices, p=weather_probs)
        weather_map = {"clear": 0, "cloudy": 1, "rainy": 2, "cold": 3}
        weather_enc = weather_map[weather]

        # Holidays: roughly ~20 days/year
        is_holiday = int(rng.random() < 0.055)
        # Exams: roughly 3 exam seasons / year, each ~2 weeks
        is_exam = int((i % 120) < 14)

        # Attendance adjustments
        attendance = int(base_attendance
                         * (0.85 if is_holiday else 1.0)
                         * (1.05 if is_exam else 1.0)
                         * (0.95 if weather == "rainy" else 1.0)
                         + rng.integers(-20, 20))
        attendance = max(100, min(800, attendance))

        # Food consumption (kg)
        base_consumption = attendance * 1.73  # ~1.73 kg per student
        weather_multiplier = {"clear": 1.0, "cloudy": 1.02, "rainy": 1.08, "cold": 1.12}
        actual = (base_consumption
                  * weather_multiplier[weather]
                  * (0.82 if is_holiday else 1.0)
                  * (1.08 if is_exam else 1.0)
                  * rng.uniform(0.95, 1.05))

        # Manual over-preparation (simulate old system: +15%)
        prepared = actual * rng.uniform(1.10, 1.20)
        waste = prepared - actual

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "attendance": attendance,
            "day_of_week": dow,
            "weather": weather,
            "weather_encoded": weather_enc,
            "is_holiday": is_holiday,
            "is_exam": is_exam,
            "actual_consumption_kg": round(actual, 2),
            "prepared_kg": round(prepared, 2),
            "waste_kg": round(waste, 2),
        })

    df = pd.DataFrame(records)
    return df


def get_sample_prediction_input() -> dict:
    """Return a sample input dict for prediction."""
    return {
        "attendance": 460,
        "day_of_week": 2,       # Wednesday
        "weather": "clear",
        "is_holiday": 0,
        "is_exam": 0,
    }


if __name__ == "__main__":
    df = generate_sample_data(365)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Avg waste: {df['waste_kg'].mean():.1f} kg/day")
    print(f"Avg waste %: {(df['waste_kg']/df['prepared_kg']).mean()*100:.1f}%")
    df.to_csv("sample_data.csv", index=False)
    print("Saved to sample_data.csv")
