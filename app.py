import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Set page configuration for a premium look
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Caching heavy resource loading functions to prevent lagging on every widget interaction
@st.cache_resource
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_preprocessors(le_path, ohe_path, scaler_path):
    with open(le_path, "rb") as file:
        le = pickle.load(file)
    with open(ohe_path, "rb") as file:
        ohe = pickle.load(file)
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    return le, ohe, scaler

# Load assets
try:
    model = load_keras_model("churn_model.h5")
    label_encoder, onehot_encoder, scaler = load_preprocessors("label_encoder.pkl", "onehot_encoder.pkl", "scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()

# Header Section
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict their probability of churning (leaving the bank).")
st.markdown("---")

# Layout: Split into logical columns for a premium dashboard look
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    geography = st.selectbox("Geography", onehot_encoder.categories_[0])
    gender = st.selectbox("Gender", label_encoder.classes_)
    age = st.slider("Age", 18, 92, value=40)
    tenure = st.slider("Tenure (Years)", 0, 10, value=5)

with col2:
    st.subheader("💳 Financial Status")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

with col3:
    st.subheader("🏦 Bank Relationship")
    num_of_products = st.slider("Number of Products used", 1, 4, value=2)
    has_cr_card = st.selectbox("Has Credit Card?", ["No (0)", "Yes (1)"])
    is_active_member = st.selectbox("Is Active Member?", ["No (0)", "Yes (1)"])

# Convert selected binary categories to numeric integers (0 or 1)
has_cr_card_val = 1 if "Yes" in has_cr_card else 0
is_active_member_val = 1 if "Yes" in is_active_member else 0

# 2. Build input DataFrame matching the exact training column order
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_val],
    'IsActiveMember': [is_active_member_val],
    'EstimatedSalary': [estimated_salary]
})

# 3. One-hot encode Geography using a DataFrame to prevent scikit-learn feature name warnings
geo_df = pd.DataFrame({'Geography': [geography]})
onehot_encoded = onehot_encoder.transform(geo_df)
onehot_encoded_df = pd.DataFrame(
    onehot_encoded.toarray(), 
    columns=onehot_encoder.get_feature_names_out(['Geography'])
)

# Concatenate all features
input_data = pd.concat([input_data, onehot_encoded_df], axis=1)

# 4. Scaling the input data using the loaded StandardScaler
scaled_input = scaler.transform(input_data)

# 5. Prediction logic
prediction = model.predict(scaled_input)
prediction_probability = float(prediction[0][0])

st.markdown("---")
st.subheader("🎯 Prediction Analysis")

# Visual Display of Results
if prediction_probability > 0.5:
    st.error(
        f"🚨 **High Risk of Churn:** This customer is **likely to churn** with a probability of **{prediction_probability:.2%}**."
    )
    # Colored progress bar for churn
    st.progress(prediction_probability)
else:
    st.success(
        f"✅ **Safe Customer:** This customer is **not likely to churn** (churn probability is **{prediction_probability:.2%}**)."
    )
    # Colored progress bar for safe (retaining)
    st.progress(prediction_probability)
