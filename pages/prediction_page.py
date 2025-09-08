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
    community_map = {'1': 1, '2': 2, '3': 3,} # Replace with your real data
    district_map = {'1': 1, '2': 2, '3': 3,} # Replace with your real data
    mean_map_intensity = {'small': 0.5, 'moderate': 1.2, 'heavy': 2.5}
    mean_map_community = {'1': 0.8, '2': 1.5, '3': 2.0}

    # Streamlit UI elements for user input
    communities = sorted(list(set(community_map.keys())))
    districts = sorted(list(set(district_map.keys())))
    intensities = ['small', 'moderate', 'heavy']

    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        
        # Community selection with description
        user_community = st.selectbox('Community:', communities)
        st.markdown(
            "<sub>The specific community where the traditional weather observation was made.</sub>",
            unsafe_allow_html=True
        )

        # District selection with description
        user_district = st.selectbox('District:', districts)
        st.markdown(
            "<sub>The larger administrative district the observation belongs to.</sub>",
            unsafe_allow_html=True
        )

        # Predicted intensity selection with description
        user_intensity = st.selectbox('Predicted Intensity:', intensities)
        st.markdown(
            "<sub>The initial rainfall prediction (small, moderate, or heavy) made using traditional knowledge.</sub>",
            unsafe_allow_html=True
        )
        
        submitted = st.form_submit_button("Predict Rainfall")

    if submitted:
        # Create a DataFrame from user input
        user_input = pd.DataFrame([{
            'community': user_community,
            'district': user_district,
            'predicted_intensity': user_intensity
        }])
        
        # Re-create the same feature engineering steps as in your training script
        user_input['community_code'] = user_input['community'].map(community_map)
        user_input['district_code'] = user_input['district'].map(district_map)
        
        # The fix: Fill NaN values with a default value to prevent the CatBoostError
        # A good default is the mean of the original training feature
        user_input['community_code'] = user_input['community_code'].fillna(0)
        user_input['district_code'] = user_input['district_code'].fillna(0)

        user_input['prediction_distance'] = np.abs(user_input['community_code'] - user_input['district_code'])
        
        user_input['avg_distance_by_intensity'] = user_input['predicted_intensity'].map(mean_map_intensity)
        user_input['avg_distance_by_community'] = user_input['community'].map(mean_map_community)
        
        # The fix: Fill NaN values for the new features as well
        user_input['avg_distance_by_intensity'] = user_input['avg_distance_by_intensity'].fillna(0.0)
        user_input['avg_distance_by_community'] = user_input['avg_distance_by_community'].fillna(0.0)

        user_input = user_input.drop(columns=['community_code', 'district_code'])
        
        for col in ['community', 'district', 'predicted_intensity']:
            user_input[col] = user_input[col].astype(str)

        # Make the prediction
        model = load_model()
        prediction_proba = model.predict_proba(user_input)
        prediction = np.argmax(prediction_proba, axis=1)[0]
        prediction_class = ['heavy', 'moderate', 'small'][prediction]

        # Display the results
        st.subheader('Prediction Results')
        st.markdown(f"**Predicted Rainfall Type:** <h3 style='color:green;'>{prediction_class.upper()}</h3>", unsafe_allow_html=True)
        st.write(f"Confidence (Probability): {np.max(prediction_proba):.2f}")
        st.info('**Disclaimer:** This prediction is for informational purposes only.')
