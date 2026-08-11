"""
train_model.py
---------------
Trains the Random Forest pipelines used by the Task 3 web app and
serializes them with the timestamp naming convention from Task 2
(e.g. models/sales-10-08-2026-16-32-31-00.pkl).

Run this once (or on a schedule) after you have train.csv / test.csv /
store.csv from the Kaggle Rossmann dataset placed in data/.

    python train_model.py

Trains TWO pipelines, both sharing the same feature set (features.py):
  - sales_model      -> predicts Sales      (as in Task_2.ipynb)
  - customers_model  -> predicts Customers  (added for the Task 3
                         dashboard requirement: "output predicted sales
                         amount AND customer numbers", which Task 2 did
                         not originally cover)

Also saves the CompetitionDistance median used at train time, so the
Flask app can reproduce identical missing-value imputation for single-row
/ partial user input at inference time.
"""

import os
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import NUM_FEATURES, CAT_FEATURES, FEATURES, build_feature_frame

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

logger = logging.getLogger("TRAIN")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler("logs/train.log")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)


def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100)


def make_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=150, max_depth=25,
            min_samples_split=5, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        )),
    ])


def fit_and_report(pipeline, X_train, y_train, X_valid, y_valid, label):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    metrics = {
        "rmspe": rmspe(y_valid, preds),
        "mae": float(mean_absolute_error(y_valid, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, preds))),
        "r2": float(r2_score(y_valid, preds)),
    }
    logger.info(f"[{label}] RMSPE={metrics['rmspe']:.2f}%  MAE={metrics['mae']:.2f}  "
                f"RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.4f}")
    return metrics


def main(data_dir="data", model_dir="models"):
    logger.info("Loading train/test/store data...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(os.path.join(data_dir, "store.csv"))

    train["Date"] = pd.to_datetime(train["Date"])
    train = train.merge(store, on="Store", how="left")

    comp_dist_median = float(train["CompetitionDistance"].median())
    train = build_feature_frame(train, competition_distance_fill=comp_dist_median)

    # Sales forecasting is performed for open stores with valid sales.
    train = train[train["Open"] == 1].copy()
    train = train[train["Sales"] > 0].copy()
    train = train.sort_values("Date").reset_index(drop=True)
    logger.info(f"Training rows after filtering: {len(train):,}")

    split_date = train["Date"].quantile(0.80)
    train_df = train[train["Date"] <= split_date]
    valid_df = train[train["Date"] > split_date]

    X_train, X_valid = train_df[FEATURES], valid_df[FEATURES]

    results = {}

    # ---- Sales model ----
    sales_pipeline = make_pipeline()
    results["sales"] = fit_and_report(
        sales_pipeline, X_train, train_df["Sales"], X_valid, valid_df["Sales"], "Sales"
    )

    # ---- Customers model (added for Task 3 dashboard requirement) ----
    customers_pipeline = make_pipeline()
    results["customers"] = fit_and_report(
        customers_pipeline, X_train, train_df["Customers"], X_valid, valid_df["Customers"], "Customers"
    )

    # Refit both on full data before serialization (as Task_2.ipynb does for the final model)
    logger.info("Refitting final models on full training data...")
    sales_pipeline_final = make_pipeline()
    sales_pipeline_final.fit(train[FEATURES], train["Sales"])

    customers_pipeline_final = make_pipeline()
    customers_pipeline_final.fit(train[FEATURES], train["Customers"])

    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-00")
    sales_path = os.path.join(model_dir, f"sales-{timestamp}.pkl")
    customers_path = os.path.join(model_dir, f"customers-{timestamp}.pkl")
    meta_path = os.path.join(model_dir, f"meta-{timestamp}.json")

    joblib.dump(sales_pipeline_final, sales_path)
    joblib.dump(customers_pipeline_final, customers_path)

    with open(meta_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "sales_model": os.path.basename(sales_path),
            "customers_model": os.path.basename(customers_path),
            "competition_distance_median": comp_dist_median,
            "metrics": results,
        }, f, indent=2)

    logger.info(f"Saved: {sales_path}")
    logger.info(f"Saved: {customers_path}")
    logger.info(f"Saved: {meta_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
