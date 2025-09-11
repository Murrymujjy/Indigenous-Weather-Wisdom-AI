import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# --- Helper Functions ---
@st.cache_resource
def load_models_and_data():
    """
    Loads the trained model and prepares data and encoders.
    """
    try:
        # Load the LightGBM model
        lgbm_model = joblib.load('lgbm_model.joblib')
        
        # Load the raw training and test data
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        # Drop the problematic columns as done in the notebook
        columns_to_drop_from_raw = ['ID', 'user_id', 'prediction_time', 'time_observed']
        
        # Create a single dataframe for fitting encoders
        combined_df = pd.concat([train_df.drop('Target', axis=1, errors='ignore'), test_df], ignore_index=True)
        combined_df = combined_df.drop(columns_to_drop_from_raw, axis=1, errors='ignore')

        # Get the list of all categorical features
        categorical_features = ['community', 'district', 'predicted_intensity', 'indicator', 'indicator_description', 'forecast_length']
        
        # Fit LabelEncoders for each categorical feature on the combined dataset
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            # Handle potential NaNs by converting to string first
            le.fit(combined_df[col].astype(str))
            label_encoders[col] = le
            
        return lgbm_model, train_df, combined_df, label_encoders
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all required files are in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models or data: {e}")
        st.stop()

# Custom function to transform a label gracefully, handling unseen values
def safe_transform(le, value):
    try:
        # The LabelEncoder is fitted on all values (train + test), so this should work
        return le.transform([str(value)])[0]
    except ValueError:
        # If the value is completely new (not in train or test), assign a unique code
        # A good practice is to return -1 or a specific high number.
        # This will be treated as an out-of-vocabulary category by LightGBM.
        return -1

# --- Page Content ---
def render():
    st.title("🌦️ Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")
    
    lgbm_model, train_df, combined_df, label_encoders = load_models_and_data()
    
    # Drop rows with any missing values in the key categorical columns for the dropdowns
    # Use the combined_df for UI unique values to make sure all possibilities are covered
    ui_df = combined_df.dropna(subset=['community', 'district', 'indicator', 'predicted_intensity', 'forecast_length'])

    # Get unique values for dropdown menus
    communities = sorted(ui_df['community'].unique())
    districts = sorted(ui_df['district'].unique())
    indicators = sorted(ui_df['indicator'].unique())
    
    with st.form("prediction_form"):
        st.subheader("Ecological Indicators")
        community = st.selectbox("Community", communities)
        district = st.selectbox("District", districts)
        indicator = st.selectbox("Indicator", indicators)
        
        st.subheader("Meteorological Data")
        predicted_intensity = st.selectbox("Predicted Intensity", sorted(ui_df['predicted_intensity'].unique()))
        forecast_length = st.number_input("Forecast Length (days)", min_value=1, value=5)
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame([{
                'community': community,
                'district': district,
                'predicted_intensity': predicted_intensity,
                'indicator': indicator,
                'forecast_length': forecast_length
            }])
            
            # Find the corresponding 'indicator_description' from the combined data
            # This is CRITICAL to avoid the 'unseen labels' error for a non-user-input column.
            # We assume a 1:1 mapping between indicator and its description
            indicator_description_map = combined_df.set_index('indicator')['indicator_description'].astype(str).to_dict()
            input_data['indicator_description'] = input_data['indicator'].map(indicator_description_map)
            
            # --- CRITICAL FIX: Replicate notebook's preprocessing on user input ---
            encoded_input_data = input_data.copy()
            for col, le in label_encoders.items():
                encoded_input_data[col] = encoded_input_data[col].apply(lambda x: safe_transform(le, x))

            # Align user input with training data columns
            # This is important to ensure the order and presence of all features
            encoded_input_data = encoded_input_data.reindex(columns=combined_df.columns, fill_value=-1)

            try:
                # Make the prediction
                prediction = lgbm_model.predict(encoded_input_data)
                
                # Assuming the target variable is 'Rainfall'
                if prediction[0] == 0:
                    st.success("Prediction: No Rainfall Expected")
                else:
                    st.success(f"Prediction: Rainfall Expected (Category {prediction[0]})")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    render()
