#IMPORTANT: Please save this content as 'app.py' and ensure you have 'randomforest_churn_model.pkl' in the same directory.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Configuration ---

MODEL_FILE = 'randomforest_churn_model.pkl'
FEATURE_NAMES = ["Age", "Gender", "Tenure", "MonthlyCharges", "TotalCharges"]

#--- Utility Functions for Model Loading ---

@st.cache_resource
def create_mock_model(feature_names):
    """Creates a mock pipeline if the model file is missing to prevent app crash."""
    st.warning(f"⚠️ MOCK MODEL ACTIVE: Using a dummy model because '{MODEL_FILE}' was not found.")

    # Create and fit a simple mock Logistic Regression pipeline
    df_mock = pd.DataFrame({
        "Age": np.random.randint(18, 80, 20),
        "Gender": np.random.randint(0, 2, 20),
        "Tenure": np.random.randint(1, 72, 20),
        "MonthlyCharges": np.random.uniform(20, 120, 20),
        "TotalCharges": np.random.uniform(100, 8000, 20),
        "Churn": np.random.randint(0, 2, 20)
    })
    df_mock['TotalCharges'] = np.log1p(df_mock['TotalCharges'])

    X_mock = df_mock[feature_names]
    y_mock = df_mock['Churn']

    mock_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
        
    mock_pipeline.fit(X_mock, y_mock)

    joblib.dump(mock_pipeline, MODEL_FILE)
    return mock_pipeline


@st.cache_resource
def load_model(file_path):
    """Loads the trained model pipeline or creates a mock one if the file is missing."""
    if not os.path.exists(file_path):
        return create_mock_model(FEATURE_NAMES)
    
    try:
        model = joblib.load(file_path)
        st.sidebar.success(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_mock_model(FEATURE_NAMES)


#--- Streamlit App Configuration ---

st.set_page_config(
page_title="Customer Churn Prediction",
layout="wide",
initial_sidebar_state="collapsed"
)

#CRITICAL LINE: Load the model once at the start

model_pipeline = load_model(MODEL_FILE)

# --- SIDEBAR ---

with st.sidebar:
    st.header("Model Pipeline Details")

    final_classifier_name = type(model_pipeline.named_steps[model_pipeline.steps[-1][0]]).__name__
    st.success(f"Primary Classifier: **{final_classifier_name}**")

    st.markdown("---")
    st.caption("Developed for Customer Churn Analysis.")


#--- MAIN PAGE LAYOUT ---

st.title(" Customer Churn Risk Predictor")
st.subheader("Input Customer Features to Check Likelihood of Leaving")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Customer Demographics & Duration")
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, step=1)
    gender_map = {'Male': 0, 'Female': 1}
    gender_display = st.selectbox("Gender", options=list(gender_map.keys()))
    gender_encoded = gender_map[gender_display]
    tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=24, step=1)

with col2:
    st.markdown("### Service Charges")
    monthly_charges = st.number_input(
    "Monthly Charges ($)", 
    min_value=10.0, max_value=150.0, value=75.0, step=0.5,
    help="The amount charged to the customer monthly."
    )
total_charges = st.number_input(
"Total Charges ($)",
min_value=0.0, value=1800.0, step=10.0,
help="The total amount charged. This will be log-transformed before prediction."
)

st.markdown("---")

#--- Prediction Button and Logic ---

if st.button("Calculate Churn Risk", type="primary"):

# 1. Prepare input data
    input_data = {
        "Age": age,
        "Gender": gender_encoded,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        # CRITICAL STEP: Apply Log Transformation
        "TotalCharges": np.log1p(total_charges) 
    }

# 2. Convert to DataFrame and ensure correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_NAMES]

    try:
        # 3. Predict the probability of Churn (Class 1)
        # Note: The pipeline handles the internal StandardScaler transform here.
        churn_probability = model_pipeline.predict_proba(input_df)[:, 1][0]
        
        # 4. Define risk categories and display result
        if churn_probability >= 0.70:
            risk_level = "HIGH Risk 🔴"
            st.error(f"The customer is at **{risk_level}** of churning.")
        elif churn_probability >= 0.50:
            risk_level = "MODERATE Risk 🟡"
            st.warning(f"The customer is at **{risk_level}** of churning.")
        else:
            risk_level = "LOW Risk 🟢"
            st.success(f"The customer is at **{risk_level}** of churning.")


    # --- Display Results ---
        st.markdown("### Risk Score")
        st.metric(
            label="Probability of Churn",
            value=f"{churn_probability * 100:.2f}%"
        )
        st.progress(churn_probability)
        st.info("The model estimates the probability of the customer leaving the service.")
        
    except Exception as e:
        st.error("An error occurred during prediction. Please verify your model file.")
        print(f"Prediction Error: {e}")
