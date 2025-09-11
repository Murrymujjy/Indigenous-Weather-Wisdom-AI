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

        # Drop the 'Target' from the training data for combining
        train_features = train_df.drop('Target', axis=1)

        # Combine for consistent preprocessing
        combined_df = pd.concat([train_features, test_df], ignore_index=True)
        
        # --- Define Categorical Features (Must Match Training) ---
        categorical_features = ['community', 'district', 'predicted_intensity', 'indicator', 'indicator_description', 'forecast_length']
        
        # --- Replicate Feature Engineering from Notebook ---
        for col in categorical_features:
            combined_df[col] = combined_df[col].astype('category')
        
        # Create new numerical features from categorical data using .cat.codes
        combined_df['community_code'] = combined_df['community'].cat.codes
        combined_df['district_code'] = combined_df['district'].cat.codes
        combined_df['predicted_intensity_code'] = combined_df['predicted_intensity'].cat.codes
        combined_df['indicator_code'] = combined_df['indicator'].cat.codes
        combined_df['indicator_description_code'] = combined_df['indicator_description'].cat.codes
        combined_df['forecast_length_code'] = combined_df['forecast_length'].cat.codes

        # Handle and engineer time-based features
        combined_df['time_observed'] = combined_df['time_observed'].replace(['MORNING', 'AFTERNOON', 'EVENING'], np.nan)
        combined_df['prediction_time'] = pd.to_datetime(combined_df['prediction_time'], errors='coerce')
        combined_df['time_observed'] = pd.to_datetime(combined_df['time_observed'], errors='coerce')

        combined_df['prediction_month'] = combined_df['prediction_time'].dt.month
        combined_df['prediction_dayofweek'] = combined_df['prediction_time'].dt.dayofweek
        combined_df['prediction_hour'] = combined_df['prediction_time'].dt.hour
        combined_df['time_diff_hours'] = (combined_df['prediction_time'] - combined_df['time_observed']).dt.total_seconds() / 3600
        combined_df['time_diff_hours'] = combined_df['time_diff_hours'].fillna(0)

        # Create interaction features
        combined_df['prediction_distance'] = np.abs(combined_df['community_code'] - combined_df['district_code'])
        combined_df['intensity_x_distance'] = combined_df['predicted_intensity_code'] * combined_df['prediction_distance']
        combined_df['forecast_length_x_distance'] = combined_df['forecast_length_code'] * combined_df['prediction_distance']

        # Drop original non-predictive columns
        columns_to_drop = ['ID', 'user_id', 'prediction_time', 'time_observed']
        final_feature_columns = combined_df.drop(columns_to_drop, axis=1, errors='ignore').columns.tolist()

        return lgbm_model, combined_df, categorical_features, final_feature_columns
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all required files are in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models or data: {e}")
        st.stop()

# --- Page Content ---
def render():
    st.title("🌦️ Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")
    
    lgbm_model, combined_df, categorical_features, final_feature_columns = load_models_and_data()
    
    # Get unique values for dropdown menus from the combined dataset
    ui_df = combined_df.dropna(subset=['community', 'district', 'indicator', 'predicted_intensity', 'forecast_length'])
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
                'forecast_length': float(forecast_length), # Ensure this is float to match training
                'time_observed': '2025-09-11 12:00', # A dummy value for feature engineering
                'prediction_time': '2025-09-11 13:00'
            }])

            # Find the corresponding 'indicator_description' from the combined data
            indicator_description_map = combined_df.set_index('indicator')['indicator_description'].astype(str).to_dict()
            input_data['indicator_description'] = input_data['indicator'].map(indicator_description_map)

            # --- CRITICAL: REPLICATE FEATURE ENGINEERING ---
            for col in categorical_features:
                input_data[col] = input_data[col].astype('category')
                
            # Create numerical features from categorical data
            input_data['community_code'] = input_data['community'].cat.codes
            input_data['district_code'] = input_data['district'].cat.codes
            input_data['predicted_intensity_code'] = input_data['predicted_intensity'].cat.codes
            input_data['indicator_code'] = input_data['indicator'].cat.codes
            input_data['indicator_description_code'] = input_data['indicator_description'].cat.codes
            input_data['forecast_length_code'] = input_data['forecast_length'].cat.codes

            # Engineer time features
            input_data['time_observed'] = pd.to_datetime(input_data['time_observed'], errors='coerce')
            input_data['prediction_time'] = pd.to_datetime(input_data['prediction_time'], errors='coerce')
            input_data['prediction_month'] = input_data['prediction_time'].dt.month
            input_data['prediction_dayofweek'] = input_data['prediction_time'].dt.dayofweek
            input_data['prediction_hour'] = input_data['prediction_time'].dt.hour
            input_data['time_diff_hours'] = (input_data['prediction_time'] - input_data['time_observed']).dt.total_seconds() / 3600
            input_data['time_diff_hours'] = input_data['time_diff_hours'].fillna(0)

            # Create interaction features
            input_data['prediction_distance'] = np.abs(input_data['community_code'] - input_data['district_code'])
            input_data['intensity_x_distance'] = input_data['predicted_intensity_code'] * input_data['prediction_distance']
            input_data['forecast_length_x_distance'] = input_data['forecast_length_code'] * input_data['prediction_distance']

            # Drop original non-predictive columns
            input_data = input_data.drop(['time_observed', 'prediction_time'], axis=1, errors='ignore')

            # Reindex and align the user input DataFrame with the model's expected features
            final_input_df = input_data.reindex(columns=final_feature_columns, fill_value=0)
            
            # Make the prediction
            try:
                prediction = lgbm_model.predict(final_input_df)
                
                if prediction[0] == 0:
                    st.success("Prediction: No Rainfall Expected")
                else:
                    st.success(f"Prediction: Rainfall Expected (Category {prediction[0]})")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    render()
