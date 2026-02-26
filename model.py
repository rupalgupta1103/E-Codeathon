"""
SmartServe AI - Machine Learning Model
XGBoost Regressor for food demand forecasting
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


WEATHER_MAP = {"clear": 0, "cloudy": 1, "rainy": 2, "cold": 3}


class SmartServeModel:
    """
    XGBoost-based demand forecasting model for mess food optimization.
    
    Features:
      - attendance_count
      - day_of_week (0=Mon … 6=Sun)
      - weather_encoded
      - is_holiday
      - is_exam
      - lag_1 (yesterday's consumption)
      - lag_7 (same day last week)
      - rolling_7 (7-day average)
    
    Target: actual_consumption_kg
    """

    def __init__(self):
        self.xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.rf = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        self.feature_names = [
            "attendance", "day_of_week", "weather_encoded",
            "is_holiday", "is_exam",
            "lag_1", "lag_7", "rolling_7"
        ]
        self._r2 = 0.0
        self._mae = 0.0
        self._trained = False

    # ── Feature Engineering ────────────────────────────────────────────────────
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("date").reset_index(drop=True)
        df["lag_1"] = df["actual_consumption_kg"].shift(1).fillna(df["actual_consumption_kg"].mean())
        df["lag_7"] = df["actual_consumption_kg"].shift(7).fillna(df["actual_consumption_kg"].mean())
        df["rolling_7"] = df["actual_consumption_kg"].rolling(7, min_periods=1).mean()
        return df

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        df = self._engineer_features(df)
        X = df[self.feature_names]
        y = df["actual_consumption_kg"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.xgb.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        y_pred = self.xgb.predict(X_test)
        self._r2 = r2_score(y_test, y_pred) * 100
        self._mae = mean_absolute_error(y_test, y_pred)
        self._trained = True

        # Store rolling stats for inference
        self._mean_lag1 = df["actual_consumption_kg"].iloc[-1]
        self._mean_lag7 = df["actual_consumption_kg"].iloc[-7] if len(df) >= 7 else df["actual_consumption_kg"].mean()
        self._mean_rolling7 = df["actual_consumption_kg"].tail(7).mean()

        print(f"[SmartServe] XGBoost trained | R²={self._r2:.1f}% | MAE={self._mae:.2f} kg")

    # ── Single-day Prediction ──────────────────────────────────────────────────
    def predict_single(
        self,
        attendance: int,
        day_of_week: int,
        weather: str = "clear",
        is_holiday: int = 0,
        is_exam: int = 0,
    ) -> float:
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        weather_enc = WEATHER_MAP.get(weather, 0)
        X = np.array([[
            attendance, day_of_week, weather_enc,
            is_holiday, is_exam,
            self._mean_lag1, self._mean_lag7, self._mean_rolling7
        ]])
        return float(self.xgb.predict(X)[0])

    # ── Batch Prediction ───────────────────────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["lag_1"] = self._mean_lag1
        df["lag_7"] = self._mean_lag7
        df["rolling_7"] = self._mean_rolling7
        return self.xgb.predict(df[self.feature_names])

    # ── Metrics & Explainability ───────────────────────────────────────────────
    def get_accuracy(self) -> float:
        return self._r2

    def get_mae(self) -> float:
        return self._mae

    def get_feature_importance(self) -> dict:
        scores = self.xgb.feature_importances_
        return dict(sorted(zip(self.feature_names, scores), key=lambda x: x[1]))

    # ── Optimization Logic ─────────────────────────────────────────────────────
    @staticmethod
    def recommended_quantity(predicted_kg: float, buffer_pct: float = 2.5) -> float:
        """Add safety buffer to avoid food shortage."""
        return predicted_kg * (1 + buffer_pct / 100)

    @staticmethod
    def waste_metrics(prepared_kg: float, consumed_kg: float, cost_per_kg: float = 80.0):
        waste = max(prepared_kg - consumed_kg, 0)
        return {
            "waste_kg": waste,
            "waste_pct": (waste / prepared_kg) * 100 if prepared_kg > 0 else 0,
            "cost_saved_inr": waste * cost_per_kg,
            "co2_saved_kg": waste * 2.5,  # 2.5 kg CO2e per kg food waste
        }
