import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# --- Helper Functions ---
@st.cache_resource
def load_models():
    """
    Loads the trained LightGBM model and prepares the data for encoding.
    """
    try:
        # Load the LightGBM model
        lgbm_model = joblib.load('lgbm_model.joblib')
        
        # Load the training and test data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        # Drop the problematic columns as done in the notebook
        columns_to_drop_from_raw = ['ID', 'user_id', 'prediction_time', 'time_observed']
        for col in columns_to_drop_from_raw:
            if col in train_df.columns:
                train_df = train_df.drop(col, axis=1)
            # The test set might not have the 'Target' column.
            if col in test_df.columns:
                test_df = test_df.drop(col, axis=1)

        # Separate features from the target
        y_train = train_df['Target']
        X_train = train_df.drop(['Target'], axis=1)
        X_test = test_df.drop(['Target'], axis=1)

        # Get the list of all categorical features
        categorical_features = ['community', 'district', 'predicted_intensity', 'indicator', 'indicator_description', 'forecast_length']

        # Fit LabelEncoders for each categorical feature on the combined dataset
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            all_data = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            le.fit(all_data)
            label_encoders[col] = le
            
        return lgbm_model, train_df, label_encoders
    except FileNotFoundError:
        st.error("One or more files not found. Please ensure `lgbm_model.joblib`, `train.csv`, and `test.csv` are in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models or data: {e}")
        st.stop()

# --- Page Content ---
def render():
    st.title("🌦️ Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")
    
    lgbm_model, train_df, label_encoders = load_models()
    
    # Drop rows with any missing values in the key categorical columns for the dropdowns
    train_df = train_df.dropna(subset=['community', 'district', 'indicator', 'predicted_intensity', 'time_observed'])

    # Get unique values for dropdown menus
    communities = sorted(train_df['community'].unique())
    districts = sorted(train_df['district'].unique())
    indicators = sorted(train_df['indicator'].unique())
    
    with st.form("prediction_form"):
        st.subheader("Ecological Indicators")
        community = st.selectbox("Community", communities)
        district = st.selectbox("District", districts)
        indicator = st.selectbox("Indicator", indicators)
        
        st.subheader("Meteorological Data")
        predicted_intensity = st.selectbox("Predicted Intensity", sorted(train_df['predicted_intensity'].unique()))
        forecast_length = st.number_input("Forecast Length (days)", min_value=1, value=5)
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame([{
                'community': community,
                'district': district,
                'predicted_intensity': predicted_intensity,
                'indicator': indicator,
                'forecast_length': forecast_length,
                # Add dummy columns for other features to match the model's training data
                'ID': None, 'user_id': None, 'prediction_time': None, 'time_observed': None, 'Target': None
            }])
            
            # --- CRITICAL FIX: Replicate notebook's preprocessing on user input ---
            # Drop the same columns as the notebook
            input_data = input_data.drop(columns=['ID', 'user_id', 'prediction_time', 'time_observed'], errors='ignore')

            # Label encode the user input
            encoded_input_data = input_data.copy()
            for col, le in label_encoders.items():
                encoded_input_data[col] = le.transform(encoded_input_data[col].astype(str))
            
            # Align user input with training data columns
            X_train = train_df.drop(['Target'], axis=1)
            input_df = encoded_input_data.reindex(columns=X_train.columns, fill_value=0)

            # Make the prediction
            try:
                prediction = lgbm_model.predict(input_df)
                st.success(f"Predicted Rainfall: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
