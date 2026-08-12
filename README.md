# Rossmann Sales Forecasting

A Streamlit dashboard that predicts daily **Sales** and **Customers** for Rossmann drug stores using Random Forest models, with a 95% prediction interval for each forecast.

## Overview

The project trains two separate Random Forest regressors — one for `Sales`, one for `Customers` — on the [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset, then serves them through a Streamlit UI for both single-store and bulk (CSV) predictions.

The core design principle is a **single shared feature pipeline** (`features.py`) used identically at both training time and inference time, so the model never sees a distribution shift between how it was trained and how it's queried.

## Project Structure

| File | Responsibility |
|---|---|
| `features.py` | Single source of truth for feature engineering — missing-value imputation and date/holiday/competition/promo feature construction. Imported by both training and serving code. |
| `train_model.py` | Offline training script. Loads `data/train.csv` + `data/store.csv`, builds features, trains/evaluates both models, and serializes them with metrics to `models/`. |
| `predict_service.py` | Loads the most recently trained model pair, rebuilds features identically to training, and returns predictions + 95% CI. Falls back to a synthetic "demo mode" model if no trained model exists. |
| `app.py` | Streamlit front-end: sidebar inputs for a single prediction, CSV upload for bulk predictions, and a model performance panel. |

## How It Works

### 1. Feature Engineering (`features.py`)

- **`handle_missing_values`** — fills `CompetitionDistance` with a median (recomputed from data, or a value passed in from training so inference on a single row stays consistent), fills competition/promo2 "since" columns and `PromoInterval`/`Open` with sane defaults.
- **`add_date_features`** — derives `DayOfWeek`, `Year`, `Month`, `Day`, `WeekOfYear`, `Quarter`, `DayOfYear`, weekend/month-position flags, and holiday proximity (`DaysToHoliday`, `DaysAfterHoliday` via nearest-holiday search). Also computes `CompetitionOpenMonths` and `Promo2OpenWeeks` as months/weeks since competition/promo2 started.
- **`build_feature_frame`** — runs both steps above and back-fills any feature the model expects but the input lacks (`STORE_METADATA_DEFAULTS`), guaranteeing the model always receives all 25 columns in `FEATURES` (22 numeric + 3 categorical: `StoreType`, `Assortment`, `StateHoliday`).

### 2. Training (`train_model.py`)

- Merges `train.csv` with `store.csv` on `Store`.
- Filters to open stores with `Sales > 0` (standard Rossmann practice, since closed-store days are zero-sales by definition and not informative).
- **Chronological 80/20 split** (`Date.quantile(0.80)`) rather than a random split, appropriate for a forecasting task.
- Each pipeline is `ColumnTransformer` (median-impute + scale numeric, most-frequent-impute + one-hot encode categorical) → `RandomForestRegressor(n_estimators=150, max_depth=25)`.
- Reports **RMSPE, MAE, RMSE, R²** on the validation split for both targets.
- Refits both models on the **full** dataset before saving (validation split was only for reporting).
- Saves artifacts with a timestamp naming convention:
  - `models/sales-<timestamp>.pkl`
  - `models/customers-<timestamp>.pkl`
  - `models/meta-<timestamp>.json` (which files to load, the training-time `CompetitionDistance` median, and validation metrics)

### 3. Serving (`predict_service.py`)

- On startup, finds the newest `meta-*.json` and loads the corresponding model pair. If none exists, silently trains a small demo pipeline on synthetic data (`demo_mode = True`) so the app is runnable without any real data.
- **`predict(df)`**: rebuilds features via `build_feature_frame` (using the *training-time* median for `CompetitionDistance`, not a re-derived one — important for single-row inference), then predicts.
- **Prediction interval**: instead of a parametric interval, it uses the spread across individual trees in the forest (`std` of all `tree.predict(X)`) to build an approximate 95% CI: `pred ± 1.96 * std`.

### 4. Dashboard (`app.py`)

- **Single prediction**: sidebar form (store ID, store type, assortment, competition distance, promo/promo2, holidays, date) → predicts Sales + Customers with CI, shown as metric cards, a CI range, a result table, and a CSV download.
- **Bulk prediction**: upload a CSV of future dates; the app maps common Kaggle test-set columns (`IsPromo` → `Promo`, `IsHoliday` → `StateHoliday`) and fills any other required fields from the sidebar values, then predicts and offers a CSV download.
- **Model performance panel**: shows RMSPE/MAE/RMSE/R² for both models, pulled from `meta-*.json` via `service.metrics`.
- Displays a warning banner if running in demo mode vs. a success banner for a real trained model.

## Setup & Usage

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

**1. Train the models** (optional — the app works in demo mode without this):

Place the Kaggle Rossmann `train.csv` and `store.csv` in `data/`, then:

```bash
python train_model.py
```

This writes trained models + metrics to `models/`.

**2. Run the dashboard:**

```bash
streamlit run app.py
```

## Notes & Potential Improvements

- **Demo mode fallback**: convenient for testing without the real dataset, but worth double-checking `app.py`'s warning banner is prominent enough that demo predictions are never mistaken for real forecasts.
- **`Open` flag**: training filters to `Open == 1`, but the dashboard's single/bulk prediction forms don't collect or default an `Open` field — this is fine since the model isn't trained on it as a feature (it's a filter, not in `FEATURES`), just worth knowing it isn't a safety check to prevent nonsensical predictions for closed-store dates.
- **CI method**: tree-variance intervals are a reasonable heuristic but aren't calibrated prediction intervals in the statistical sense — good enough for a dashboard, worth flagging if this were used for business decisions.
- **`predict_service.py`** loads only the *latest* meta file; there's no model versioning/rollback UI, which would be a natural next step if this were deployed longer-term.
