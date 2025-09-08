import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

st.set_page_config(page_title="Prediction", page_icon="🌤️")

# --- 1. Load the Trained Model ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('best_catboost_model.cbm')
    return model

# --- 2. IMPORTANT: PLACEHOLDER DATA FOR FEATURE ENGINEERING ---
# You must replace these with the actual mappings from your training data.
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


# --- 3. Streamlit App Interface ---
st.title('Predict Rainfall')
st.markdown("Use the model to make a new prediction.")
st.header('Make a Prediction')

# --- 4. User Input Fields ---
communities = sorted(list(set(community_map.keys())))
districts = sorted(list(set(district_map.keys())))
intensities = ['small', 'moderate', 'heavy'] 

with st.form("prediction_form"):
    user_community = st.selectbox('Community:', communities)
    user_district = st.selectbox('District:', districts)
    user_intensity = st.selectbox('Predicted Intensity:', intensities)
    submitted = st.form_submit_button("Predict Rainfall")

# --- 5. Prediction Logic ---
if submitted:
    user_input = pd.DataFrame([{
        'community': user_community,
        'district': user_district,
        'predicted_intensity': user_intensity
    }])
    user_input['community_code'] = user_input['community'].map(community_map)
    user_input['district_code'] = user_input['district'].map(district_map)
    user_input['community_code'] = pd.to_numeric(user_input['community_code'], errors='coerce')
    user_input['district_code'] = pd.to_numeric(user_input['district_code'], errors='coerce')

    user_input['prediction_distance'] = np.abs(user_input['community_code'] - user_input['district_code'])
    user_input['avg_distance_by_intensity'] = user_input['predicted_intensity'].map(mean_map_intensity)
    user_input['avg_distance_by_community'] = user_input['community'].map(mean_map_community)

    user_input = user_input.drop(columns=['community_code', 'district_code'])

    for col in ['community', 'district', 'predicted_intensity']:
        user_input[col] = user_input[col].astype(str)

    model = load_model()
    prediction_proba = model.predict_proba(user_input)
    prediction = np.argmax(prediction_proba, axis=1)[0]
    prediction_class = ['heavy', 'moderate', 'small'][prediction]

    # Display the results
    st.subheader('Prediction Results')
    st.markdown(f"**Predicted Rainfall Type:** <h3 style='color:green;'>{prediction_class.upper()}</h3>", unsafe_allow_html=True)
    st.write(f"Confidence (Probability): {np.max(prediction_proba):.2f}")
    st.info('**Disclaimer:** This prediction is for informational purposes only.')
