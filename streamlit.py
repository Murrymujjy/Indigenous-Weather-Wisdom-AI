import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# --- 1. Load the Trained Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained CatBoost model from the .cbm file."""
    model = CatBoostClassifier()
    model.load_model('best_catboost_model.cbm')
    return model

# --- IMPORTANT: PLACEHOLDER DATA FOR FEATURE ENGINEERING ---
# You must replace these with the actual mappings from your training data.
# This ensures that the app performs the same feature engineering as your training script.
# You can get these mappings by running a small script on your training data.
community_map = {
    '1': 1, '2': 2, '3': 3, # Add all your communities and their integer codes here
}
district_map = {
    '1': 1, '2': 2, '3': 3, # Add all your districts and their integer codes here
}
mean_map_intensity = {
    'small': 0.5, 'moderate': 1.2, 'heavy': 2.5 # Replace with the means from your training data
}
mean_map_community = {
    '1': 0.8, '2': 1.5, '3': 2.0 # Replace with the means from your training data
}


# --- 2. Streamlit App Interface ---
st.title('☔️ Ghanaian Indigenous Rain Prediction App')
st.markdown("""
Welcome to the Indigenous Weather Wisdom AI Model!
This application predicts the type of rainfall—heavy, moderate, or small—expected in the next 12 to 24 hours based on traditional ecological indicators.
""")

st.header('Make a Prediction')

# --- 3. User Input Fields ---
# Get all unique values from your original dataframes to create selectboxes
# Assuming your original 'community' and 'district' columns are strings.
communities = sorted(list(set(community_map.keys())))
districts = sorted(list(set(district_map.keys())))
intensities = ['small', 'moderate', 'heavy'] # Or get from your data

user_community = st.selectbox('Community:', communities)
user_district = st.selectbox('District:', districts)
user_intensity = st.selectbox('Predicted Intensity:', intensities)


# --- 4. Prediction Logic ---
if st.button('Predict Rainfall'):
    # a. Create a DataFrame from user input
    user_input = pd.DataFrame([{
        'community': user_community,
        'district': user_district,
        'predicted_intensity': user_intensity
    }])

    # b. Re-create the same feature engineering steps on user input
    user_input['community_code'] = user_input['community'].map(community_map)
    user_input['district_code'] = user_input['district'].map(district_map)

    # Convert to numeric for the calculation
    user_input['community_code'] = pd.to_numeric(user_input['community_code'], errors='coerce')
    user_input['district_code'] = pd.to_numeric(user_input['district_code'], errors='coerce')

    # Recreate engineered features
    user_input['prediction_distance'] = np.abs(user_input['community_code'] - user_input['district_code'])
    user_input['avg_distance_by_intensity'] = user_input['predicted_intensity'].map(mean_map_intensity)
    user_input['avg_distance_by_community'] = user_input['community'].map(mean_map_community)

    # Drop the temporary code columns
    user_input = user_input.drop(columns=['community_code', 'district_code'])

    # Ensure all categorical columns are strings for CatBoost
    user_input['community'] = user_input['community'].astype(str)
    user_input['district'] = user_input['district'].astype(str)
    user_input['predicted_intensity'] = user_input['predicted_intensity'].astype(str)

    # c. Make the prediction
    model = load_model()
    prediction_proba = model.predict_proba(user_input)
    prediction = np.argmax(prediction_proba, axis=1)[0] # Get the class with the highest probability
    prediction_class = ['heavy', 'moderate', 'small'][prediction]

    # d. Display the results
    st.subheader('Prediction Results')
    st.markdown(f"**Predicted Rainfall Type:** <h3 style='color:green;'>{prediction_class.upper()}</h3>", unsafe_allow_html=True)
    st.write(f"Confidence (Probability): {np.max(prediction_proba):.2f}")
    
    st.info('**Disclaimer:** This prediction is for informational purposes and should not be used as the sole source for agricultural decisions.')
