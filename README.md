# Rossmann Sales Forecasting (Merged Single-File Version)

A Streamlit dashboard that predicts daily **Sales** and **Customers** for Rossmann drug stores using Random Forest models, with a 95% prediction interval for each forecast — all packed into a single file, `rossmann_app_merged.py`.

## What's in this file

This is a merged version of the original four-file project. Everything lives in `rossmann_app_merged.py`, organized into four clearly-labeled sections in this order:

| Section | Original file | Responsibility |
|---|---|---|
| 1 | `features.py` | Feature engineering — missing-value imputation and date/holiday/competition/promo feature construction. Used identically by both training and serving. |
| 2 | `train_model.py` | Offline training logic, wrapped in `train_model_main()`. Loads `data/train.csv` + `data/store.csv`, builds features, trains/evaluates both models, and serializes them with metrics to `models/`. |
| 3 | `predict_service.py` | `PredictionService` class — loads the most recently trained model pair, rebuilds features identically to training, and returns predictions + 95% CI. Falls back to a synthetic "demo mode" model if no trained model exists. |
| 4 | `app.py` | The Streamlit UI, wrapped in `run_streamlit_app()`: sidebar inputs for a single prediction, CSV upload for bulk predictions, and a model performance panel. |

The original per-file imports (`from features import ...`, `from predict_service import get_service`) were removed since everything is now in one module — no other logic was changed.

## How It Works

### 1. Feature Engineering

- **`handle_missing_values`** — fills `CompetitionDistance` with a median (recomputed from data, or a value passed in from training so inference on a single row stays consistent), fills competition/promo2 "since" columns and `PromoInterval`/`Open` with sane defaults.
- **`add_date_features`** — derives `DayOfWeek`, `Year`, `Month`, `Day`, `WeekOfYear`, `Quarter`, `DayOfYear`, weekend/month-position flags, and holiday proximity (`DaysToHoliday`, `DaysAfterHoliday` via nearest-holiday search). Also computes `CompetitionOpenMonths` and `Promo2OpenWeeks`.
- **`build_feature_frame`** — runs both steps above and back-fills any feature the model expects but the input lacks (`STORE_METADATA_DEFAULTS`), guaranteeing the model always receives all 25 columns in `FEATURES` (22 numeric + 3 categorical: `StoreType`, `Assortment`, `StateHoliday`).

### 2. Training (`train_model_main()`)

- Merges `train.csv` with `store.csv` on `Store`.
- Filters to open stores with `Sales > 0`.
- **Chronological 80/20 split** (`Date.quantile(0.80)`), appropriate for a forecasting task.
- Each pipeline: `ColumnTransformer` (median-impute + scale numeric, most-frequent-impute + one-hot encode categorical) → `RandomForestRegressor(n_estimators=150, max_depth=25)`.
- Reports **RMSPE, MAE, RMSE, R²** on the validation split for both targets, then refits both models on the **full** dataset before saving.
- Saves artifacts with a timestamp naming convention:
  - `models/sales-<timestamp>.pkl`
  - `models/customers-<timestamp>.pkl`
  - `models/meta-<timestamp>.json` (which files to load, the training-time `CompetitionDistance` median, and validation metrics)

### 3. Serving (`PredictionService` / `get_service()`)

- On first use, finds the newest `meta-*.json` and loads the corresponding model pair. If none exists, silently trains a small demo pipeline on synthetic data (`demo_mode = True`) so the app is runnable without any real data.
- **`predict(df)`**: rebuilds features via `build_feature_frame` (using the *training-time* median for `CompetitionDistance`), then predicts.
- **Prediction interval**: uses the spread across individual trees in the forest (`std` of all `tree.predict(X)`) to build an approximate 95% CI: `pred ± 1.96 * std`.

### 4. Dashboard (`run_streamlit_app()`)

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
python rossmann_app_merged.py --train
```

This writes trained models + metrics to `models/`.

**2. Run the dashboard:**

```bash
streamlit run rossmann_app_merged.py
```

> Note: Streamlit executes the whole module top-to-bottom, so `run_streamlit_app()` runs automatically when launched with `streamlit run`. The `--train` branch only triggers when running the file directly with `python ... --train`.

## Why merged into one file

Useful when you want a single portable script — e.g. for sharing, deploying to a platform that expects one entry-point file, or pasting into a single Streamlit Cloud app — without managing separate `features.py` / `train_model.py` / `predict_service.py` / `app.py` imports.

## Notes & Potential Improvements

- **Demo mode fallback**: convenient for testing without the real dataset, but worth double-checking the warning banner is prominent enough that demo predictions are never mistaken for real forecasts.
- **`Open` flag**: training filters to `Open == 1`, but the dashboard's prediction forms don't collect an `Open` field — this is fine since the model isn't trained on it as a feature, just worth knowing it isn't a safety check against closed-store dates.
- **CI method**: tree-variance intervals are a reasonable heuristic but aren't calibrated prediction intervals in the statistical sense — good enough for a dashboard, worth flagging if used for business decisions.
- **Single-file trade-off**: easier to share/deploy, but harder to unit-test each stage in isolation than the original four-file layout — if the project grows, splitting back out may be worth it.
- **Model versioning**: only the *latest* `meta-*.json` is loaded; there's no rollback UI if a new training run performs worse.
