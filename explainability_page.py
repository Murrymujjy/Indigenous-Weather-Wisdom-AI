import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Helper Functions ---
@st.cache_resource
def load_models():
    """
    Loads the trained LightGBM model and the training data.
    """
    try:
        # Load the LightGBM model
        lgbm_model = joblib.load('lgbm_model.joblib')
        
        # Load the training data to get feature names and categories
        train_df = pd.read_csv('train.csv')
        
        return lgbm_model, train_df
    except FileNotFoundError:
        st.error("One or more files not found. Please ensure `lgbm_model.joblib` and `train.csv` are in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models or data: {e}")
        st.stop()

# --- Page Content ---
def render():
    st.title("🌦️ Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")
    
    lgbm_model, train_df = load_models()
    
    # Drop the rainfall column to get features for one-hot encoding
    X_train = train_df.drop(columns=['rainfall']).copy()
    
    # Drop rows with any missing values to ensure unique() works correctly
    train_df = train_df.dropna(subset=['community', 'district', 'indicator', 'time_observed'])

    # Get unique values for dropdown menus
    communities = sorted(train_df['community'].unique())
    districts = sorted(train_df['district'].unique())
    indicators = sorted(train_df['indicator'].unique())
    times = sorted(train_df['time_observed'].unique())
    
    with st.form("prediction_form"):
        st.subheader("Ecological Indicators")
        community = st.selectbox("Community", communities)
        district = st.selectbox("District", districts)
        indicator = st.selectbox("Indicator", indicators)
        indicator_description = st.text_input("Indicator Description", "e.g., Small white mushrooms grow on trees")
        time_observed = st.selectbox("Time Observed", times)
        
        st.subheader("Meteorological Data")
        forecast_length = st.number_input("Forecast Length (days)", min_value=1, max_value=30, value=5)
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame([{
                'community': community,
                'district': district,
                'indicator': indicator,
                'indicator_description': indicator_description,
                'time_observed': time_observed,
                'forecast_length': forecast_length
            }])
            
            # Use `get_dummies` for one-hot encoding on both user input and a template of training data
            input_encoded = pd.get_dummies(input_data)
            X_train_encoded = pd.get_dummies(X_train)
            
            # Align columns to ensure the input data matches the model's training data
            # This is crucial to prevent `ValueError`
            input_aligned, _ = input_encoded.align(X_train_encoded, join='right', axis=1, fill_value=0)
            
            # Make the prediction
            try:
                prediction = lgbm_model.predict(input_aligned)
                st.success(f"Predicted Rainfall: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
