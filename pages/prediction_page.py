import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

def render():
    st.title('Predict Rainfall')
    st.markdown("Use the model to make a new prediction.")
    
    @st.cache_resource
    def load_model():
        model = CatBoostClassifier()
        model.load_model('best_catboost_model.cbm')
        return model
    
    # --- IMPORTANT: These dictionaries have been corrected based on your data ---
    community_map = {'Tumfa': 15, 'Kwabeng': 1, 'Akropong': 2, 'Asamama': 3, 'Akwaduuso': 21, 'Banso': 6, 'Awenare': 8, 'mouso': 9, 'Abomosu': 43, 'Foso Odumasi': 11, 'Amonom': 12, 'Asunafo': 30, 'Apampatia': 24, 'FOSO ODUMASI': 16, 'Assin ATONSU': 18, 'odumasi': 34, 'Assin Wurakese': 20, 'Assin Atonsu': 22, 'odumasi Adansi': 23, 'Assin Aponsie': 36, 'Assin nyankomasi': 26, 'Assin Foso Odumasi': 27, 'akwaduuso': 28, 'assin mesre nyame': 29, 'ASSIN BROFOYEDUR': 31, 'Mampamhwe': 32, 'Atonsu': 33, 'ODUMASI': 38, 'jimiso': 39, 'Asonkore': 40, 'Dompim': 41, 'Domeabra': 42}
    district_map = {'atiwa_west': 0, 'assin_fosu': 1, 'obuasi_east': 2}
    # Corrected keys from floats to strings
    mean_map_intensity = {'small': 15.54, 'moderate': 11.64, 'heavy': 8.51}
    # Corrected keys by removing spaces
    mean_map_community = {'Assin Wurakese': 19.0, 'ASSIN BROFOYEDUR': 30.0, 'Abomosu': 43.0, 'Akropong': 2.0, 'Akwaduuso': 20.05, 'Amonom': 12.0, 'Apampatia': 24.0, 'Asamama': 3.0, 'Assin Aponsie': 36.0, 'Assin Atonsu': 21.0, 'Assin Foso Odumasi': 26.0, 'Assin Nyankomasi': 34.0, 'Asunafo': 30.0, 'Atonsu': 32.0, 'Awenare': 8.0, 'Banso': 6.0, 'FOSO ODUMASI': 15.0, 'Foso Odumasi': 10.0, 'Kwabeng': 1.0, 'Mampamhwe': 30.0, 'Tumfa': 15.0, 'akwaduuso': 27.0, 'assin mesre nyame': 28.0, 'mouso': 9.0, 'odumasi': 32.0, 'odumasi Adansi': 21.0, 'jimiso': 39.0, 'Asonkore': 40.0, 'Dompim': 41.0, 'Domeabra': 42.0}
    
    # Streamlit UI elements for user input
    communities = sorted(list(set(community_map.keys())))
    districts = sorted(list(set(district_map.keys())))
    intensities = ['small', 'moderate', 'heavy']

    with st.form("prediction_form"):
        st.subheader("Input Parameters")
        
        user_community = st.selectbox('Community:', communities)
        st.markdown("<sub>The specific community where the traditional weather observation was made.</sub>", unsafe_allow_html=True)

        user_district = st.selectbox('District:', districts)
        st.markdown("<sub>The larger administrative district the observation belongs to.</sub>", unsafe_allow_html=True)

        user_intensity = st.selectbox('Predicted Intensity:', intensities)
        st.markdown("<sub>The initial rainfall prediction (small, moderate, or heavy) made using traditional knowledge.</sub>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Predict Rainfall")

    if submitted:
        user_input = pd.DataFrame([{
            'community': user_community,
            'district': user_district,
            'predicted_intensity': user_intensity
        }])
        
        # Add .str.strip() to remove any invisible leading/trailing spaces
        user_input['community'] = user_input['community'].astype(str).str.strip()
        user_input['district'] = user_input['district'].astype(str).str.strip()
        user_input['predicted_intensity'] = user_input['predicted_intensity'].astype(str).str.strip()

        user_input['community_code'] = user_input['community'].map(community_map)
        user_input['district_code'] = user_input['district'].map(district_map)
        
        # The fix: Fill NaN values with a default value to prevent the CatBoostError
        user_input['community_code'] = user_input['community_code'].fillna(0)
        user_input['district_code'] = user_input['district_code'].fillna(0)

        user_input['prediction_distance'] = np.abs(user_input['community_code'] - user_input['district_code'])
        
        user_input['avg_distance_by_intensity'] = user_input['predicted_intensity'].map(mean_map_intensity)
        user_input['avg_distance_by_community'] = user_input['community'].map(mean_map_community)
        
        user_input['avg_distance_by_intensity'] = user_input['avg_distance_by_intensity'].fillna(0.0)
        user_input['avg_distance_by_community'] = user_input['avg_distance_by_community'].fillna(0.0)

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
