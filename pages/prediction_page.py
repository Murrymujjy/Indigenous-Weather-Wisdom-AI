import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

def render():
    st.title('Predict Rainfall')
    st.markdown("Use the model to make a new prediction.")
    
    # Use st.cache_resource for the model
    @st.cache_resource
    def load_model():
        model = CatBoostClassifier()
        # Ensure the model file path is correct relative to the app.py location
        model.load_model('best_catboost_model.cbm')
        return model
    
    # --- IMPORTANT: Replace these with your actual mappings from the training data ---
    community_map = {'community_a': 1, 'community_b': 2, 'community_c': 3}
    district_map = {'district_1': 1, 'district_2': 2, 'district_3': 3}
    mean_map_intensity = {'small': 0.5, 'moderate': 1.2, 'heavy': 2.5}
    mean_map_community = {'community_a': 0.8, 'community_b': 1.5, 'community_c': 2.0}

    # Streamlit UI elements for user input
    communities = sorted(list(set(community_map.keys())))
    districts = sorted(list(set(district_map.keys())))
    intensities = ['small', 'moderate', 'heavy']

    with st.form("prediction_form"):
        user_community = st.selectbox('Community:', communities)
        user_district = st.selectbox('District:', districts)
        user_intensity = st.selectbox('Predicted Intensity:', intensities)
        submitted = st.form_submit_button("Predict Rainfall")

    if submitted:
        user_input = pd.DataFrame([{
            'community': user_community,
            'district': user_district,
            'predicted_intensity': user_intensity
        }])
        user_input['community_code'] = user_input['community'].map(community_map)
        user_input['district_code'] = user_input['district'].map(district_map)
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

        st.subheader('Prediction Results')
        st.markdown(f"**Predicted Rainfall Type:** <h3 style='color:green;'>{prediction_class.upper()}</h3>", unsafe_allow_html=True)
        st.write(f"Confidence (Probability): {np.max(prediction_proba):.2f}")
        st.info('**Disclaimer:** This prediction is for informational purposes only.')
