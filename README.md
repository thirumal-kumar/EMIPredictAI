# 💰 EMIPredict AI
AI-powered EMI Eligibility and Limit Prediction Web App built with **Streamlit** and **Scikit-learn**.

---

## 🚀 Overview
**EMIPredict AI** uses machine learning to evaluate EMI eligibility and predict the maximum EMI a user can afford based on demographic and financial details.

The models are trained on a real-world dataset of 400K+ credit profiles, saved as high-performance `.joblib` files, and dynamically loaded from Google Drive for efficient deployment.

---

## 🧩 Features
✅ Predict EMI eligibility (Eligible / Not Eligible)  
✅ Estimate maximum affordable EMI (₹/month)  
✅ User-friendly Streamlit interface  
✅ Google Drive model hosting using `gdown`  
✅ Caching for fast repeated predictions  
✅ Deployment-ready for **Streamlit Cloud**

---

## 🧠 Model Architecture
- **Classifier:** RandomForestClassifier pipeline  
- **Regressor:** RandomForestRegressor pipeline  
- Both pipelines include preprocessing (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`)

---

## 📦 Project Structure
EMIPredictAI/
│
├── data/
│ └── emi_prediction_dataset.csv
│
├── models/
│ ├── best_classifier.joblib ← (hosted on Google Drive)
│ ├── best_regressor.joblib ← (hosted on Google Drive)
│ ├── encoder.joblib
│ ├── scaler.joblib
│ └── label_encoder.joblib
│
├── streamlit_app/
│ └── app.py
│
├── train_model.py
├── requirements.txt
└── README.md
