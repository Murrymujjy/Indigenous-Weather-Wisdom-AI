import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import joblib

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    """Loads the trained models and data for the prediction page."""
    try:
        cat_model = CatBoostClassifier()
        cat_model.load_model("best_catboost_model.cbm")
        
        # We need to recreate the LightGBM model structure to load the ensemble
        lgbm_model = LGBMClassifier(objective='multiclass', random_state=42)
        
        # Dummy fit to create the booster, then reload from joblib
        X_dummy = pd.DataFrame(np.zeros((1, 10)), columns=[f'col_{i}' for i in range(10)])
        y_dummy = np.zeros(1)
        lgbm_model.fit(X_dummy, y_dummy)
        lgbm_model = joblib.load('lgbm_model.joblib') # Assuming you have this saved
        
        ensemble_model = VotingClassifier(
            estimators=[('cat', cat_model), ('lgbm', lgbm_model)],
            voting='soft'
        )

        train_df = pd.read_csv('train.csv')
        
        return ensemble_model, train_df
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'best_catboost_model.cbm' and 'lgbm_model.joblib' are in your project directory.")
        st.stop()

# --- Page Content ---
def render():
    st.title("🌦️ Make a New Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")

    ensemble_model, train_df = load_models()

    # Get unique values for dropdowns from training data
    communities = sorted(train_df['community'].unique())
    districts = sorted(train_df['district'].unique())
    indicators = sorted(train_df['indicator'].unique())
    indicator_descriptions = sorted(train_df['indicator_description'].unique())
    forecast_lengths = sorted(train_df['forecast_length'].unique())

    # Create input widgets for user data
    with st.form("prediction_form"):
        st.subheader("Input Features")
        community = st.selectbox("Community", communities)
        district = st.selectbox("District", districts)
        confidence = st.slider("Confidence Level", 0.0, 1.0, 0.5)
        predicted_intensity = st.selectbox("Predicted Intensity", sorted(train_df['predicted_intensity'].unique()))
        indicator = st.selectbox("Indicator", indicators)
        indicator_description = st.selectbox("Indicator Description", indicator_descriptions)
        forecast_length = st.selectbox("Forecast Length", forecast_lengths)

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Create a DataFrame for the user's input
        input_data = pd.DataFrame({
            'confidence': [confidence],
            'predicted_intensity': [predicted_intensity],
            'community': [community],
            'district': [district],
            'indicator': [indicator],
            'indicator_description': [indicator_description],
            'forecast_length': [forecast_length]
        })

        # Make prediction
        prediction = ensemble_model.predict(input_data)
        
        st.subheader("Prediction Result")
        st.success(f"The predicted rainfall category is: **{prediction[0]}**")
        st.balloons()
