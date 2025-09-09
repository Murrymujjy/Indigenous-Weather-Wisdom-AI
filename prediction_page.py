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
    
    # Drop rows with any missing values in the key categorical columns
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
            
            # --- CRITICAL FIX: Ensure columns are aligned ---
            
            # 1. Separate features from the target in the training data
            X_train = train_df.drop(columns=['rainfall', 'ID', 'user_id', 'confidence', 'predicted_intensity', 'Target'], errors='ignore')
            
            # 2. Get a list of all columns after one-hot encoding the training data
            # This is our master list of features the model expects
            encoded_train_columns = pd.get_dummies(X_train).columns
            
            # 3. One-hot encode the user input
            input_encoded = pd.get_dummies(input_data)
            
            # 4. Align the user's input with the master list of features from training data
            input_aligned = input_encoded.reindex(columns=encoded_train_columns, fill_value=0)
            
            # Make the prediction
            try:
                prediction = lgbm_model.predict(input_aligned)
                st.success(f"Predicted Rainfall: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
