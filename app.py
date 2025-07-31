# streamlit_app.py
import streamlit as st
import pandas as pd
import mlflow.sklearn
import numpy as np

# Title and description
st.title("Insurance Premium Estimator")
st.markdown("""
This app uses a trained machine learning model to predict **Insurance Premium Amount** based on user inputs.
""")

# Load the model from MLflow
model_uri = "file:///Users/pavankalyankarri/Desktop/Guvi/Project%203/mlruns/831797079015474252/models/m-1cb7dc6b88e841fcb98880f9644cebd5/artifacts/model"
model = mlflow.sklearn.load_model(model_uri)

# Input fields
st.sidebar.header("Enter Customer Details")
age = st.sidebar.slider("Age", 18, 65, 30)
annual_income = st.sidebar.number_input("Annual Income", min_value=1000, max_value=200000, value=30000)
dependents = st.sidebar.slider("Number of Dependents", 0, 5, 2)
health_score = st.sidebar.slider("Health Score", 0.0, 60.0, 25.0)
previous_claims = st.sidebar.slider("Previous Claims", 0, 9, 1)
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 20, 10)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
insurance_duration = st.sidebar.slider("Insurance Duration (Years)", 1, 10, 5)

# One-hot encoded categorical fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.sidebar.selectbox("Occupation", ["Employed", "Unemployed", "Self-Employed"])
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
feedback = st.sidebar.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
smoking = st.sidebar.radio("Smoking Status", ["Yes", "No"])
exercise = st.sidebar.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])
property_type = st.sidebar.selectbox("Property Type", ["House", "Condo", "Apartment"])

# Convert to model input format
input_data = pd.DataFrame({
    "Age": [age],
    "Annual Income": [annual_income],
    "Number of Dependents": [dependents],
    "Health Score": [health_score],
    "Previous Claims": [previous_claims],
    "Vehicle Age": [vehicle_age],
    "Credit Score": [credit_score],
    "Insurance Duration": [insurance_duration],
    # One-hot encodings (example)
    "Gender_Male": [gender == "Male"],
    "Marital Status_Married": [marital_status == "Married"],
    "Marital Status_Single": [marital_status == "Single"],
    "Education Level_High School": [education == "High School"],
    "Education Level_Master's": [education == "Master's"],
    "Education Level_PhD": [education == "PhD"],
    "Occupation_Self-Employed": [occupation == "Self-Employed"],
    "Occupation_Unemployed": [occupation == "Unemployed"],
    "Location_Suburban": [location == "Suburban"],
    "Location_Urban": [location == "Urban"],
    "Policy Type_Comprehensive": [policy_type == "Comprehensive"],
    "Policy Type_Premium": [policy_type == "Premium"],
    "Customer Feedback_Good": [feedback == "Good"],
    "Customer Feedback_Poor": [feedback == "Poor"],
    "Smoking Status_Yes": [smoking == "Yes"],
    "Exercise Frequency_Monthly": [exercise == "Monthly"],
    "Exercise Frequency_Rarely": [exercise == "Rarely"],
    "Exercise Frequency_Weekly": [exercise == "Weekly"],
    "Property Type_Condo": [property_type == "Condo"],
    "Property Type_House": [property_type == "House"]
})

# Prediction
if st.button("Predict Premium"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Premium: ₹{prediction:,.2f}")
