import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    """Loads the trained LightGBM model and data for the prediction page."""
    try:
        # Load the LightGBM model directly from the file
        lgbm_model = joblib.load('lgbm_model.joblib')
        
        # Load the training data to be used for getting dropdown options
        train_df = pd.read_csv('train_data.csv')
        
        return lgbm_model, train_df
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'lgbm_model.joblib' and 'train_data.csv' are in your project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()


# --- Page Content ---
def render():
    st.title("🌦️ Make a New Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")

    lgbm_model, train_df = load_models()

    # Get unique values for dropdowns from training data, handling potential NaN values
    communities = sorted(train_df['community'].dropna().unique())
    districts = sorted(train_df['district'].dropna().unique())
    indicators = sorted(train_df['indicator'].dropna().unique())
    indicator_descriptions = sorted(train_df['indicator_description'].dropna().unique())
    forecast_lengths = sorted(train_df['forecast_length'].dropna().unique())
    predicted_intensities = sorted(train_df['predicted_intensity'].dropna().unique())

    # Create input widgets for user data
    with st.form("prediction_form"):
        st.subheader("Input Features")
        community = st.selectbox("Community", communities)
        district = st.selectbox("District", districts)
        confidence = st.slider("Confidence Level", 0.0, 1.0, 0.5)
        predicted_intensity = st.selectbox("Predicted Intensity", predicted_intensities)
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
        prediction = lgbm_model.predict(input_data)
        
        st.subheader("Prediction Result")
        st.success(f"The predicted rainfall category is: **{prediction[0]}**")
        st.balloons()
