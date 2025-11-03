import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import gdown
import traceback

# ==========================================================
# 💰 EMIPredict AI – Intelligent Financial Risk Assessment
# ==========================================================
st.set_page_config(page_title="💰 EMIPredict AI", layout="centered")

st.title("💰 EMIPredict AI – Intelligent Financial Risk Assessment")
st.markdown("""
Predict EMI eligibility and maximum EMI limit using your financial data.  
_Powered by AI models trained on real credit datasets._
""")

# ==========================================================
# 🔽 Google Drive Model Downloader (gdown)
# ==========================================================
@st.cache_resource
def load_model_from_drive(file_id, local_path):
    """Download and cache large models from Google Drive using gdown."""
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 50_000_000:
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"📥 Downloading {os.path.basename(local_path)}...")
        gdown.download(url, local_path, quiet=False, fuzzy=True)
        st.success(f"✅ Downloaded {os.path.basename(local_path)} "
                   f"({os.path.getsize(local_path)/1e6:.2f} MB)")
    else:
        st.info(f"✅ Using cached {os.path.basename(local_path)} "
                f"({os.path.getsize(local_path)/1e6:.2f} MB)")
    return joblib.load(local_path)


# === 🔗 Google Drive File IDs ===
clf_id = "1HDVTRFddt98iwgPrQJbjYL1ZYxBlhQPI"
reg_id = "1HWkwyGHbvEydh36Q4Gwzm5H8KYGl92Gj"

# === Load Models and Preprocessors ===
clf = load_model_from_drive(clf_id, "models/best_classifier.joblib")
reg = load_model_from_drive(reg_id, "models/best_regressor.joblib")
encoder = joblib.load("models/encoder.joblib")
scaler = joblib.load("models/scaler.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")

st.success("✅ All models and preprocessors loaded successfully!")

# ==========================================================
# 🧮 User Input Section
# ==========================================================
st.header("🧮 EMI Eligibility Prediction")

col1, col2 = st.columns(2)
with col1:
    salary = st.number_input("Monthly Salary (₹)", min_value=1000, value=50000, step=1000)
    current_emi = st.number_input("Current EMI (₹)", min_value=0, value=0, step=500)
    expenses = st.number_input("Other Expenses (₹)", min_value=0, value=2000, step=500)
    years = st.number_input("Years of Employment", min_value=0, value=5, step=1)
    travel = st.number_input("Travel Expenses (₹)", min_value=0, value=1000, step=500)
    groceries = st.number_input("Groceries & Utilities (₹)", min_value=0, value=5000, step=500)

with col2:
    rent = st.number_input("Monthly Rent (₹)", min_value=0, value=8000, step=500)
    loan = st.number_input("Existing Loans (₹)", min_value=0, value=0, step=500)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750, step=10)
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=50000, value=500000, step=50000)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, value=60, step=6)
    age = st.number_input("Age", min_value=18, value=30, step=1)

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col4:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
with col5:
    education = st.selectbox("Education", ["Graduate", "Postgraduate", "HighSchool"])

col6, col7, col8 = st.columns(3)
with col6:
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
with col7:
    company_type = st.selectbox("Company Type", ["Private", "Public", "Startup", "Other"])
with col8:
    house_type = st.selectbox("House Type", ["Owned", "Rented", "Company Provided"])

emi_scenario = st.selectbox("EMI Scenario", ["Standard", "Flexible", "High Risk"])

# ==========================================================
# 🔮 Prediction Section (no manual preprocessing)
# ==========================================================
if st.button("🔍 Predict EMI Eligibility & Limit"):
    try:
        # ✅ Prepare raw input DataFrame directly for pipeline
        df_raw = pd.DataFrame([{
            "age": age,
            "monthly_salary": salary,
            "years_of_employment": years,
            "monthly_rent": rent,
            "family_size": 3,
            "dependents": 1,
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": travel,
            "groceries_utilities": groceries,
            "other_monthly_expenses": expenses,
            "existing_loans": loan,
            "current_emi_amount": current_emi,
            "credit_score": credit_score,
            "bank_balance": 50000,
            "emergency_fund": 10000,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "employment_type": employment_type,
            "company_type": company_type,
            "house_type": house_type,
            "emi_scenario": emi_scenario
        }])

        st.write("### 🧾 Input Data Sent to Model")
        st.dataframe(df_raw)

        # === Predict EMI Eligibility ===
        class_pred_raw = clf.predict(df_raw)
        class_pred = int(class_pred_raw[0])
        label = label_encoder.inverse_transform([class_pred])[0]

        st.markdown("### 🏦 **Predicted EMI Eligibility**")
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(df_raw)[0]
            confidence = float(proba[class_pred])
            st.success(f"{label} ({confidence*100:.1f}% confidence)")
            st.progress(confidence)
        else:
            st.success(label)

        # === Predict Maximum EMI ===
        emi_pred_raw = reg.predict(df_raw)
        emi_val = float(emi_pred_raw[0])
        st.markdown("### 💸 **Predicted Maximum Affordable EMI**")
        st.info(f"₹{emi_val:,.0f}")

        # === Summary ===
        st.divider()
        colA, colB = st.columns(2)
        colA.metric("Eligibility", label)
        colB.metric("Predicted EMI", f"₹{emi_val:,.0f}")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)
        st.text(traceback.format_exc())

# ==========================================================
# 🧠 Debug Info (Expandable)
# ==========================================================
with st.expander("🧩 Model & Preprocessing Info"):
    try:
        if hasattr(clf, "feature_names_in_"):
            st.write("Classifier features:", list(clf.feature_names_in_))
        if hasattr(reg, "feature_names_in_"):
            st.write("Regressor features:", list(reg.feature_names_in_))
    except Exception as e:
        st.exception(e)

