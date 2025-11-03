# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

DATA_PATH = "data/emi_prediction_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Loaded dataset with shape: {df.shape}")

# === Targets ===
target_class = "emi_eligibility"
target_reg = "max_monthly_emi"

# Handle string labels → binary 0/1
label_encoder = LabelEncoder()
df[target_class] = label_encoder.fit_transform(df[target_class])

# === Column definitions ===
categorical_cols = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "emi_scenario"
]
numeric_cols = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "requested_amount", "requested_tenure"
]

# === Clean up data ===
df[categorical_cols] = df[categorical_cols].fillna("missing")

# Convert numeric columns to float safely
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(0.0)

X = df[categorical_cols + numeric_cols]
y_class = df[target_class]
y_reg = df[target_reg]

# === Preprocessing ===
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ("cat", encoder, categorical_cols),
    ("num", scaler, numeric_cols)
])

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# === Train models ===
print("🚀 Training classifier...")
clf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42, n_jobs=-1))
])
clf_pipeline.fit(X_train, y_train)
print("✅ Classifier trained.")

print("🚀 Training regressor...")
reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=250, max_depth=15, random_state=42, n_jobs=-1))
])
reg_pipeline.fit(Xr_train, yr_train)
print("✅ Regressor trained.")

# === Save artifacts ===
fitted_encoder = clf_pipeline.named_steps["preprocess"].named_transformers_["cat"]
fitted_scaler = clf_pipeline.named_steps["preprocess"].named_transformers_["num"]

joblib.dump(clf_pipeline, os.path.join(MODEL_DIR, "best_classifier.joblib"))
joblib.dump(reg_pipeline, os.path.join(MODEL_DIR, "best_regressor.joblib"))
joblib.dump(fitted_encoder, os.path.join(MODEL_DIR, "encoder.joblib"))
joblib.dump(fitted_scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

print("\n✅ Training complete. Files saved in /models:")
for f in os.listdir(MODEL_DIR):
    print(f"  - {f} ({os.path.getsize(os.path.join(MODEL_DIR, f))/1e6:.2f} MB)")

