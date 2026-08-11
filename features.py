"""
features.py
------------
Single source of truth for feature engineering.

This is a direct port of the cleaning / feature-engineering logic from
Task_2.ipynb (handle_missing_values + add_date_features + FEATURES list).
Both train_model.py (offline training) and predict_service.py (live
inference in the Flask app) import from here so that the exact same
transformations are applied at train time and at serve time.
"""

import numpy as np
import pandas as pd

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
