"""
predict_service.py
-------------------
Loads the most recently trained (timestamped) Sales + Customers model
pair from models/, builds the same feature set used at training time,
and returns predictions with an approximate 95% prediction interval
(same tree-variance approach as Task_2.ipynb).

If no trained model is found (e.g. the real Kaggle train.csv/store.csv
were never supplied to train_model.py), falls back to a small demo
model trained on synthetic data on first use, purely so the dashboard
stays usable end-to-end. This is clearly flagged in the API response
as "demo_mode": true so it's never mistaken for a real forecast.
"""

import os
import glob
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from features import NUM_FEATURES, CAT_FEATURES, FEATURES, build_feature_frame, STORE_METADATA_DEFAULTS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
logger = logging.getLogger("PREDICT")


def _latest_meta():
    metas = sorted(glob.glob(os.path.join(MODEL_DIR, "meta-*.json")))
    if not metas:
        return None
    with open(metas[-1]) as f:
        return json.load(f)


def _make_demo_pipeline(target_scale=1.0):
    """Tiny RF trained on synthetic data, used only when no real model
    is available yet, so the app remains testable out of the box."""
    rng = np.random.RandomState(42)
    n = 3000
    synth = pd.DataFrame({
        "Store": rng.randint(1, 20, n),
        "Date": pd.date_range("2015-01-01", periods=n, freq="D")[rng.randint(0, 365, n)],
        "Promo": rng.randint(0, 2, n),
        "SchoolHoliday": rng.randint(0, 2, n),
        "StateHoliday": rng.choice(["0", "0", "0", "a"], n),
        "Open": 1,
        "StoreType": rng.choice(["a", "b", "c", "d"], n),
        "Assortment": rng.choice(["a", "b", "c"], n),
        "CompetitionDistance": rng.uniform(50, 15000, n),
        "CompetitionOpenSinceMonth": rng.randint(1, 12, n),
        "CompetitionOpenSinceYear": rng.randint(2000, 2015, n),
        "Promo2": rng.randint(0, 2, n),
        "Promo2SinceWeek": rng.randint(1, 52, n),
        "Promo2SinceYear": rng.randint(2010, 2015, n),
        "PromoInterval": "None",
    })
    synth = build_feature_frame(synth)
    base = 4000 + synth["Promo"] * 1500 - synth["IsWeekend"] * 500
    noise = rng.normal(0, 400, n)
    target = np.clip(base + noise, 0, None) * target_scale

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, NUM_FEATURES),
                                       ("cat", categorical_pipeline, CAT_FEATURES)])
    pipe = Pipeline([("preprocess", preprocessor),
                      ("model", RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42))])
    pipe.fit(synth[FEATURES], target)
    return pipe


class PredictionService:
    def __init__(self):
        self.demo_mode = False
        self.comp_dist_median = None
        meta = _latest_meta()

        if meta is not None:
            sales_path = os.path.join(MODEL_DIR, meta["sales_model"])
            customers_path = os.path.join(MODEL_DIR, meta["customers_model"])
            if os.path.exists(sales_path) and os.path.exists(customers_path):
                logger.info(f"Loading trained models from {meta['timestamp']}")
                self.sales_model = joblib.load(sales_path)
                self.customers_model = joblib.load(customers_path)
                self.comp_dist_median = meta.get("competition_distance_median")
                self.metrics = meta.get("metrics", {})
                return

        logger.warning("No trained model found in models/. Falling back to DEMO MODE "
                        "(synthetic data) so the app stays usable. Run train_model.py "
                        "with the real Rossmann data/train.csv, data/store.csv to replace this.")
        self.demo_mode = True
        self.sales_model = _make_demo_pipeline(target_scale=1.0)
        self.customers_model = _make_demo_pipeline(target_scale=0.15)
        self.metrics = {}

    def _predict_with_ci(self, pipeline, X):
        preds = pipeline.predict(X)
        rf = pipeline.named_steps["model"]
        X_t = pipeline.named_steps["preprocess"].transform(X)
        tree_preds = np.array([tree.predict(X_t) for tree in rf.estimators_])
        std = tree_preds.std(axis=0)
        lower = np.clip(preds - 1.96 * std, 0, None)
        upper = preds + 1.96 * std
        return preds, lower, upper

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """df must contain at least: Store, Date, and ideally Promo,
        SchoolHoliday, StateHoliday, StoreType, Assortment,
        CompetitionDistance, Promo2, etc. Missing columns are filled
        with sane defaults via features.STORE_METADATA_DEFAULTS."""
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        feat = build_feature_frame(df, competition_distance_fill=self.comp_dist_median)

        X = feat[FEATURES]
        sales_pred, sales_lo, sales_hi = self._predict_with_ci(self.sales_model, X)
        cust_pred, cust_lo, cust_hi = self._predict_with_ci(self.customers_model, X)

        result = pd.DataFrame({
            "Store": feat["Store"].values,
            "Date": feat["Date"].dt.strftime("%Y-%m-%d").values,
            "Predicted_Sales": np.round(sales_pred, 2),
            "Sales_CI_Lower_95": np.round(sales_lo, 2),
            "Sales_CI_Upper_95": np.round(sales_hi, 2),
            "Predicted_Customers": np.round(cust_pred).astype(int),
            "Customers_CI_Lower_95": np.round(np.clip(cust_lo, 0, None)).astype(int),
            "Customers_CI_Upper_95": np.round(cust_hi).astype(int),
        })
        return result


# Module-level singleton, loaded once per process.
_service = None


def get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
