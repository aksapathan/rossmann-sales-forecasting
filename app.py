"""
rossmann_app_merged.py
-----------------------
Merged single-file version of the Rossmann Sales Forecasting project.
Combines (in order):
  1. features.py         -> feature engineering (shared by train + serve)
  2. train_model.py       -> offline model training script (run via train_model_main())
  3. predict_service.py   -> loads trained models, serves predictions + CI
  4. app.py                -> Streamlit dashboard (runs on `streamlit run rossmann_app_merged.py`)

Original files are unchanged in logic; only the cross-file imports
(`from features import ...`, `from predict_service import get_service`)
were removed since everything now lives in one module.

To train:    python rossmann_app_merged.py --train
To serve:    streamlit run rossmann_app_merged.py
"""

import os
import sys
import glob
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. features.py — Single source of truth for feature engineering
#
# This is a direct port of the cleaning / feature-engineering logic from
# Task_2.ipynb (handle_missing_values + add_date_features + FEATURES list).
# Both training and live inference use these same transformations so the
# exact same features are built at train time and at serve time.

NUM_FEATURES = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpenMonths",
    "Promo2",
    "Promo2OpenWeeks",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "Quarter",
    "DayOfYear",
    "IsWeekend",
    "IsBeginningOfMonth",
    "IsMidMonth",
    "IsEndOfMonth",
    "IsMonthStart",
    "IsMonthEnd",
    "DaysToHoliday",
    "DaysAfterHoliday",
]

CAT_FEATURES = ["StoreType", "Assortment", "StateHoliday"]

FEATURES = NUM_FEATURES + CAT_FEATURES
SALES_TARGET = "Sales"
CUSTOMERS_TARGET = "Customers"

# Columns the store metadata (store.csv) must contain for a full feature
# build. Used to fill in sane defaults when a user only supplies a subset
# via the web form instead of the full store.csv.
STORE_METADATA_DEFAULTS = {
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": np.nan,  # filled with median at handle_missing_values time
    "CompetitionOpenSinceMonth": 0,
    "CompetitionOpenSinceYear": 0,
    "Promo2": 0,
    "Promo2SinceWeek": 0,
    "Promo2SinceYear": 0,
    "PromoInterval": "None",
}


def handle_missing_values(data: pd.DataFrame, competition_distance_fill=None) -> pd.DataFrame:
    """Same imputation rules as Task_2.ipynb cell 5.

    competition_distance_fill: if provided (e.g. the median saved at train
    time), used instead of recomputing the median from `data` itself -
    important at inference time when `data` may be a single row.
    """
    data = data.copy()

    if "CompetitionDistance" in data.columns:
        fill_value = competition_distance_fill
        if fill_value is None:
            fill_value = data["CompetitionDistance"].median()
        data["CompetitionDistance"] = data["CompetitionDistance"].fillna(fill_value)

    for col in [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    if "PromoInterval" in data.columns:
        data["PromoInterval"] = data["PromoInterval"].fillna("None")

    if "Open" in data.columns:
        data["Open"] = data["Open"].fillna(1)

    return data


def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
    """Same date / holiday / competition / promo feature engineering as
    Task_2.ipynb cell 7."""
    data = data.copy()

    if "StateHoliday" in data.columns:
        data["StateHoliday"] = data["StateHoliday"].astype("string").fillna("0").astype(str)
    else:
        data["StateHoliday"] = "0"

    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter
    data["DayOfYear"] = data["Date"].dt.dayofyear

    data["IsWeekend"] = (data["DayOfWeek"].isin([5, 6])).astype(int)
    data["IsBeginningOfMonth"] = (data["Day"] <= 10).astype(int)
    data["IsMidMonth"] = (data["Day"].between(11, 20)).astype(int)
    data["IsEndOfMonth"] = (data["Day"] > 20).astype(int)
    data["IsMonthStart"] = (data["Date"].dt.is_month_start).astype(int)
    data["IsMonthEnd"] = (data["Date"].dt.is_month_end).astype(int)

    state_holiday = data["StateHoliday"].astype(str)
    data["IsStateHoliday"] = (state_holiday != "0").astype(int)

    holiday_dates = (
        data.loc[data["IsStateHoliday"] == 1, "Date"].drop_duplicates().sort_values()
    )

    if len(holiday_dates) > 0:
        hol_arr = holiday_dates.values.astype("datetime64[D]")
        d_arr = data["Date"].values.astype("datetime64[D]")
        idx = np.searchsorted(hol_arr, d_arr, side="left")

        days_to = np.full(len(d_arr), 9999, dtype=int)
        days_after = np.full(len(d_arr), 9999, dtype=int)

        for i, dd in enumerate(d_arr):
            pos = idx[i]
            if pos < len(hol_arr):
                days_to[i] = int((hol_arr[pos] - dd).astype("timedelta64[D]").astype(int))
            if pos > 0:
                days_after[i] = int((dd - hol_arr[pos - 1]).astype("timedelta64[D]").astype(int))

        data["DaysToHoliday"] = days_to
        data["DaysAfterHoliday"] = days_after
    else:
        data["DaysToHoliday"] = 9999
        data["DaysAfterHoliday"] = 9999

    for col in ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
                "Promo2SinceYear", "Promo2SinceWeek"]:
        if col not in data.columns:
            data[col] = 0

    data["CompetitionOpenMonths"] = (
        (data["Year"] - data["CompetitionOpenSinceYear"]) * 12
        + (data["Month"] - data["CompetitionOpenSinceMonth"])
    ).clip(lower=0).fillna(0)

    data["Promo2OpenWeeks"] = (
        (data["Year"] - data["Promo2SinceYear"]) * 52
        + (data["WeekOfYear"] - data["Promo2SinceWeek"])
    ).clip(lower=0).fillna(0)

    return data


def build_feature_frame(data: pd.DataFrame, competition_distance_fill=None) -> pd.DataFrame:
    """Full pipeline: missing-value handling + date/business features.
    Returns a frame that has (at least) all columns in FEATURES."""
    data = handle_missing_values(data, competition_distance_fill=competition_distance_fill)
    data = add_date_features(data)

    for col in FEATURES:
        if col not in data.columns:
            data[col] = STORE_METADATA_DEFAULTS.get(col, 0)

    return data


# 2. train_model.py — Offline training script
#
# Trains the Random Forest pipelines used by the dashboard and serializes
# them with a timestamp naming convention (e.g. models/sales-10-08-2026-16-32-31-00.pkl).
#
# Run this once (or on a schedule) after you have train.csv / test.csv /
# store.csv from the Kaggle Rossmann dataset placed in data/.
#
#     python rossmann_app_merged.py --train
#
# Trains TWO pipelines, both sharing the same feature set above:
#   - sales_model      -> predicts Sales
#   - customers_model  -> predicts Customers
#
# Also saves the CompetitionDistance median used at train time, so the
# serving code can reproduce identical missing-value imputation for
# single-row / partial user input at inference time.

train_logger = logging.getLogger("TRAIN")
train_logger.setLevel(logging.INFO)


def _init_train_logging():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not train_logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler("logs/train.log")
        ch = logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        train_logger.addHandler(fh)
        train_logger.addHandler(ch)


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
    train_logger.info(f"[{label}] RMSPE={metrics['rmspe']:.2f}%  MAE={metrics['mae']:.2f}  "
                       f"RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.4f}")
    return metrics


def train_model_main(data_dir="data", model_dir="models"):
    _init_train_logging()
    train_logger.info("Loading train/test/store data...")
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
    train_logger.info(f"Training rows after filtering: {len(train):,}")

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

    # ---- Customers model (added for the dashboard requirement) ----
    customers_pipeline = make_pipeline()
    results["customers"] = fit_and_report(
        customers_pipeline, X_train, train_df["Customers"], X_valid, valid_df["Customers"], "Customers"
    )

    # Refit both on full data before serialization (as Task_2.ipynb does for the final model)
    train_logger.info("Refitting final models on full training data...")
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

    train_logger.info(f"Saved: {sales_path}")
    train_logger.info(f"Saved: {customers_path}")
    train_logger.info(f"Saved: {meta_path}")
    train_logger.info("Training complete.")


# 3. predict_service.py — Loads trained models, serves predictions + CI
#
# Loads the most recently trained (timestamped) Sales + Customers model
# pair from models/, builds the same feature set used at training time,
# and returns predictions with an approximate 95% prediction interval
# (same tree-variance approach as Task_2.ipynb).
#
# If no trained model is found (e.g. the real Kaggle train.csv/store.csv
# were never supplied), falls back to a small demo model trained on
# synthetic data on first use, purely so the dashboard stays usable
# end-to-end. This is clearly flagged as "demo_mode": True so it's never
# mistaken for a real forecast.

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
predict_logger = logging.getLogger("PREDICT")


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
                predict_logger.info(f"Loading trained models from {meta['timestamp']}")
                self.sales_model = joblib.load(sales_path)
                self.customers_model = joblib.load(customers_path)
                self.comp_dist_median = meta.get("competition_distance_median")
                self.metrics = meta.get("metrics", {})
                return

        predict_logger.warning("No trained model found in models/. Falling back to DEMO MODE "
                                "(synthetic data) so the app stays usable. Run training with the "
                                "real Rossmann data/train.csv, data/store.csv to replace this.")
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
        with sane defaults via STORE_METADATA_DEFAULTS."""
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


# 4. app.py — Streamlit dashboard
#
# Only runs when this file is executed via `streamlit run rossmann_app_merged.py`
# (guarded so `python rossmann_app_merged.py --train` doesn't try to import
# streamlit / build the UI).

def run_streamlit_app():
    import streamlit as st

    # Page configuration
    st.set_page_config(
        page_title="Rossmann Sales Forecasting",
        page_icon="📊",
        layout="wide"
    )

    # Load prediction service
    @st.cache_resource
    def load_service():
        return get_service()

    service = load_service()

    # Title
    st.title("Rossmann Sales Forecasting Dashboard")
    st.write(
        "Predict future Sales and Customers using the trained Rossmann ML model."
    )

    # Model status
    if service.demo_mode:
        st.warning(" Demo model is being used.")
    else:
        st.success(" Real trained model loaded successfully.")

    # Sidebar
    st.sidebar.header("Store Information")

    store_id = st.sidebar.number_input(
        "Store ID",
        min_value=1,
        value=1,
        step=1
    )

    store_type = st.sidebar.selectbox(
        "Store Type",
        ["a", "b", "c", "d"]
    )

    assortment = st.sidebar.selectbox(
        "Assortment",
        ["a", "b", "c"]
    )

    competition_distance = st.sidebar.number_input(
        "Competition Distance",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    promo2 = st.sidebar.selectbox(
        "Promo2",
        [0, 1]
    )

    promo = st.sidebar.selectbox(
        "Promo",
        [0, 1]
    )

    school_holiday = st.sidebar.selectbox(
        "School Holiday",
        [0, 1]
    )

    state_holiday = st.sidebar.selectbox(
        "State Holiday",
        ["0", "a", "b", "c"]
    )

    # Date
    st.subheader(" Prediction Date")

    prediction_date = st.date_input(
        "Select Date"
    )

    # Manual prediction
    st.subheader("🔮 Single Prediction")

    if st.button("Predict Sales", type="primary"):

        input_df = pd.DataFrame([{
            "Store": int(store_id),
            "Date": pd.to_datetime(prediction_date),
            "Promo": int(promo),
            "SchoolHoliday": int(school_holiday),
            "StateHoliday": state_holiday,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": float(competition_distance),
            "Promo2": int(promo2)
        }])

        try:

            result = service.predict(input_df)

            st.success("Prediction generated successfully!")

            # KPI cards
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Sales",
                    f"{result['Predicted_Sales'].iloc[0]:,.2f}"
                )

            with col2:
                st.metric(
                    "Predicted Customers",
                    f"{result['Predicted_Customers'].iloc[0]:,.0f}"
                )

            # Confidence interval
            st.subheader("📈 Prediction Interval")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Sales 95% Confidence Interval")
                st.write(
                    f"{result['Sales_CI_Lower_95'].iloc[0]:,.2f}"
                    f" – "
                    f"{result['Sales_CI_Upper_95'].iloc[0]:,.2f}"
                )

            with col2:
                st.write("Customers 95% Confidence Interval")
                st.write(
                    f"{result['Customers_CI_Lower_95'].iloc[0]:,.0f}"
                    f" – "
                    f"{result['Customers_CI_Upper_95'].iloc[0]:,.0f}"
                )

            # Result table
            st.subheader("📋 Prediction Result")

            st.dataframe(
                result,
                use_container_width=True
            )

            # Download
            csv = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction CSV",
                data=csv,
                file_name="rossmann_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")

    # CSV Upload
    st.divider()

    st.subheader("📂 Bulk Prediction using CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV containing future dates",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.write("Uploaded Data:")
            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            if "Date" not in uploaded_df.columns:
                st.error("CSV must contain a 'Date' column.")

            else:

                uploaded_df["Store"] = int(store_id)

                if "IsPromo" in uploaded_df.columns:
                    uploaded_df["Promo"] = uploaded_df["IsPromo"].astype(int)

                if "SchoolHoliday" in uploaded_df.columns:
                    uploaded_df["SchoolHoliday"] = (
                        uploaded_df["SchoolHoliday"].astype(int)
                    )

                if "IsHoliday" in uploaded_df.columns:
                    uploaded_df["StateHoliday"] = uploaded_df[
                        "IsHoliday"
                    ].apply(
                        lambda x: "a" if int(x) == 1 else "0"
                    )

                if "Promo" not in uploaded_df.columns:
                    uploaded_df["Promo"] = int(promo)

                if "SchoolHoliday" not in uploaded_df.columns:
                    uploaded_df["SchoolHoliday"] = int(school_holiday)

                if "StateHoliday" not in uploaded_df.columns:
                    uploaded_df["StateHoliday"] = state_holiday

                uploaded_df["StoreType"] = store_type
                uploaded_df["Assortment"] = assortment
                uploaded_df["CompetitionDistance"] = competition_distance
                uploaded_df["Promo2"] = promo2

                if st.button("🚀 Predict Uploaded Data"):

                    result = service.predict(uploaded_df)

                    st.success(
                        f"Predictions generated for {len(result)} rows."
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    csv = result.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Bulk Predictions",
                        data=csv,
                        file_name="rossmann_bulk_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"Error processing CSV: {e}")

    # Model Performance
    st.divider()

    st.subheader("📊 Model Performance")

    metrics = service.metrics

    if metrics:

        sales_metrics = metrics.get("sales", {})
        customer_metrics = metrics.get("customers", {})

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Sales Model")

            st.write(
                f"**RMSPE:** {sales_metrics.get('rmspe', 0):.2f}%"
            )

            st.write(
                f"**MAE:** {sales_metrics.get('mae', 0):,.2f}"
            )

            st.write(
                f"**RMSE:** {sales_metrics.get('rmse', 0):,.2f}"
            )

            st.write(
                f"**R²:** {sales_metrics.get('r2', 0):.4f}"
            )

        with col2:

            st.markdown("### Customers Model")

            st.write(
                f"**RMSPE:** {customer_metrics.get('rmspe', 0):.2f}%"
            )

            st.write(
                f"**MAE:** {customer_metrics.get('mae', 0):,.2f}"
            )

            st.write(
                f"**RMSE:** {customer_metrics.get('rmse', 0):,.2f}"
            )

            st.write(
                f"**R²:** {customer_metrics.get('r2', 0):.4f}"
            )

    # Footer
    st.divider()

    st.caption(
        "Rossmann Sales Forecasting | Machine Learning Project"
    )


# ============================================================================
# Entry point
# ============================================================================
#
# - `python rossmann_app_merged.py --train`  -> runs training only
# - `streamlit run rossmann_app_merged.py`   -> runs the dashboard
#   (Streamlit executes the whole module top-to-bottom, so the app code
#   below runs automatically in that case; the --train branch is skipped
#   because sys.argv won't contain "--train" when Streamlit launches it.)

if __name__ == "__main__":
    if "--train" in sys.argv:
        train_model_main()
    else:
        run_streamlit_app()
