import streamlit as st
import pandas as pd
import numpy as np
from predict_service import get_service

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load prediction service
# --------------------------------------------------
@st.cache_resource
def load_service():
    return get_service()


service = load_service()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📊 Rossmann Sales Forecasting Dashboard")
st.write(
    "Predict future Sales and Customers using the trained Rossmann ML model."
)

# --------------------------------------------------
# Model status
# --------------------------------------------------
if service.demo_mode:
    st.warning("⚠️ Demo model is being used.")
else:
    st.success("✅ Real trained model loaded successfully.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
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

# --------------------------------------------------
# Date
# --------------------------------------------------
st.subheader("📅 Prediction Date")

prediction_date = st.date_input(
    "Select Date"
)

# --------------------------------------------------
# Manual prediction
# --------------------------------------------------
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

        # --------------------------------------------------
        # KPI cards
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Confidence interval
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Result table
        # --------------------------------------------------
        st.subheader("📋 Prediction Result")

        st.dataframe(
            result,
            use_container_width=True
        )

        # --------------------------------------------------
        # Download
        # --------------------------------------------------
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction CSV",
            data=csv,
            file_name="rossmann_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# CSV Upload
# --------------------------------------------------
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


# --------------------------------------------------
# Model Performance
# --------------------------------------------------
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

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Rossmann Sales Forecasting | Machine Learning Project"
)