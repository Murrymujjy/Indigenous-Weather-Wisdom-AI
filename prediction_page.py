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
        
        # Get the list of all categorical features
        categorical_features = ['community', 'district', 'predicted_intensity', 'indicator', 'indicator_description', 'forecast_length']
        
        # Fit LabelEncoders for each categorical feature on the combined dataset
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            # Handle potential NaNs by converting to string first
            le.fit(combined_df[col].astype(str))
            label_encoders[col] = le
        
        # Get the final list of columns the model was trained on
        # This is a critical step to ensure we match the feature set.
        # We need to replicate the full feature engineering from the training script.
        
        # Create a new dataframe with engineered features to get the final column list
        engineered_df = combined_df.copy()
        
        # Re-introduce and engineer time features from the original columns
        engineered_df['time_observed'] = engineered_df['time_observed'].replace(['MORNING', 'AFTERNOON', 'EVENING'], np.nan)
        engineered_df['prediction_time'] = pd.to_datetime(engineered_df['prediction_time'], errors='coerce')
        engineered_df['time_observed'] = pd.to_datetime(engineered_df['time_observed'], errors='coerce')
        
        # Extract useful time-based features
        engineered_df['prediction_month'] = engineered_df['prediction_time'].dt.month
        engineered_df['prediction_dayofweek'] = engineered_df['prediction_time'].dt.dayofweek
        engineered_df['prediction_hour'] = engineered_df['prediction_time'].dt.hour
        engineered_df['time_diff_hours'] = (engineered_df['prediction_time'] - engineered_df['time_observed']).dt.total_seconds() / 3600
        engineered_df['time_diff_hours'].fillna(0, inplace=True)
        
        # Create 'prediction_distance' feature using categorical codes
        engineered_df['community_code'] = engineered_df['community'].astype('category').cat.codes
        engineered_df['district_code'] = engineered_df['district'].astype('category').cat.codes
        engineered_df['prediction_distance'] = np.abs(engineered_df['community_code'] - engineered_df['district_code'])

        # Feature Interaction
        engineered_df['intensity_x_distance'] = engineered_df['predicted_intensity'].astype('category').cat.codes * engineered_df['prediction_distance']
        engineered_df['forecast_length_x_distance'] = engineered_df['forecast_length'].astype('category').cat.codes * engineered_df['prediction_distance']

        # Drop original time and other unnecessary columns
        engineered_df = engineered_df.drop(columns_to_drop_from_raw, axis=1, errors='ignore')

        # Get the final list of columns to ensure the user input dataframe matches
        final_feature_columns = engineered_df.columns.tolist()
            
        return lgbm_model, combined_df, label_encoders, final_feature_columns
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all required files are in the project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the models or data: {e}")
        st.stop()

# Custom function to transform a label gracefully, handling unseen values
def safe_transform(le, value):
    try:
        return le.transform([str(value)])[0]
    except ValueError:
        return -1 # Assign a unique code for unseen labels

# --- Page Content ---
def render():
    st.title("🌦️ Prediction")
    st.markdown("Enter the details below to get a prediction for rainfall in the Pra River Basin.")
    
    lgbm_model, combined_df, label_encoders, final_feature_columns = load_models_and_data()
    
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
        
        # Optional: Add dummy inputs for other features needed by the model
        time_observed_str = st.text_input("Time Observed (e.g., '2025-09-10 12:00')", value='2025-09-10 12:00')
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            # Create a DataFrame from user inputs
            input_data = pd.DataFrame([{
                'community': community,
                'district': district,
                'predicted_intensity': predicted_intensity,
                'indicator': indicator,
                'forecast_length': forecast_length,
                'time_observed': time_observed_str,
                'prediction_time': '2025-09-11 13:00' # Placeholder for a live timestamp
            }])
            
            # Find the corresponding 'indicator_description' from the combined data
            indicator_description_map = combined_df.set_index('indicator')['indicator_description'].astype(str).to_dict()
            input_data['indicator_description'] = input_data['indicator'].map(indicator_description_map)

            # --- CRITICAL STEP: REPLICATE FEATURE ENGINEERING ---
            # 1. Engineer time features
            input_data['time_observed'] = input_data['time_observed'].replace(['MORNING', 'AFTERNOON', 'EVENING'], np.nan)
            input_data['prediction_time'] = pd.to_datetime(input_data['prediction_time'], errors='coerce')
            input_data['time_observed'] = pd.to_datetime(input_data['time_observed'], errors='coerce')
            input_data['prediction_month'] = input_data['prediction_time'].dt.month
            input_data['prediction_dayofweek'] = input_data['prediction_time'].dt.dayofweek
            input_data['prediction_hour'] = input_data['prediction_time'].dt.hour
            input_data['time_diff_hours'] = (input_data['prediction_time'] - input_data['time_observed']).dt.total_seconds() / 3600
            input_data['time_diff_hours'].fillna(0, inplace=True)
            
            # 2. Create 'prediction_distance' feature
            input_data['community_code'] = input_data['community'].astype('category').cat.codes
            input_data['district_code'] = input_data['district'].astype('category').cat.codes
            input_data['prediction_distance'] = np.abs(input_data['community_code'] - input_data['district_code'])

            # 3. Feature Interaction
            input_data['intensity_x_distance'] = input_data['predicted_intensity'].astype('category').cat.codes * input_data['prediction_distance']
            input_data['forecast_length_x_distance'] = input_data['forecast_length'].astype('category').cat.codes * input_data['prediction_distance']

            # 4. Drop original columns that are no longer needed for prediction
            input_data = input_data.drop(['time_observed', 'prediction_time'], axis=1)

            # 5. Transform categorical features using the pre-fitted encoders
            for col, le in label_encoders.items():
                input_data[col] = input_data[col].apply(lambda x: safe_transform(le, x))

            # 6. Align the user input DataFrame with the model's expected features
            # This is the most important step to prevent the feature count mismatch.
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
