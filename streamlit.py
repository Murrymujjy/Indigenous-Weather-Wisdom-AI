import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# --- Set a broad page configuration for the app ---
st.set_page_config(
    page_title="Indigenous Weather App",
    page_icon="🏠",
    layout="wide"
)

# --- Initialize session state for navigation ---
# Set the default page to 'Home'
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# --- Functions for each page content ---
def home_page():
    st.title("Welcome to the Indigenous Weather Wisdom AI Model")
    st.markdown("""
    This web application is a professional showcase of a machine learning model built to forecast rainfall in Ghana's Pra River Basin.
    It combines a unique dataset of Indigenous Ecological Indicators with modern AI.
    """)
    st.subheader("Explore the App's Sections")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🌦️ Go to Prediction"):
            st.session_state.page = 'Prediction'
            st.experimental_rerun()
    with col2:
        if st.button("📊 Go to Insights"):
            st.session_state.page = 'Insights'
            st.experimental_rerun()
    with col3:
        if st.button("🧠 Go to Explainability"):
            st.session_state.page = 'Explainability'
            st.experimental_rerun()

    st.markdown("---")
    st.info("Click on any of the buttons above to navigate to a specific section.")


def prediction_page():
    # Code for the Prediction Page
    st.title('Predict Rainfall')
    st.markdown("Use the model to make a new prediction.")
    
    @st.cache_resource
    def load_model():
        model = CatBoostClassifier()
        model.load_model('best_catboost_model.cbm')
        return model

    # You must replace these with the actual mappings from your training data.
    community_map = {'1': 1, '2': 2, '3': 3,}
    district_map = {'1': 1, '2': 2, '3': 3,}
    mean_map_intensity = {'small': 0.5, 'moderate': 1.2, 'heavy': 2.5}
    mean_map_community = {'1': 0.8, '2': 1.5, '3': 2.0}

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


def insights_page():
    # Code for the Insights Page
    st.title('Data Insights')
    st.markdown("Explore key distributions and relationships in the training data.")

    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('path/to/your/train_data.csv')
            df['community'] = df['community'].astype(str).str.strip()
            df['predicted_intensity'] = df['predicted_intensity'].astype(str).str.strip()
            df['Target'] = df['Target'].astype(str)
            return df
        except FileNotFoundError:
            st.error("Training data file not found. Please place 'train.csv' in the same folder.")
            return pd.DataFrame()

    df = load_data()
    if not df.empty:
        st.subheader('Target Distribution')
        target_counts = df['Target'].value_counts().reset_index()
        target_counts.columns = ['Target', 'Count']
        fig = px.pie(target_counts, names='Target', values='Count', title='Distribution of Rainfall Targets')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Rainfall Intensity Distribution')
        intensity_counts = df['predicted_intensity'].value_counts().reset_index()
        intensity_counts.columns = ['Predicted_Intensity', 'Count']
        fig_intensity = px.bar(intensity_counts, x='Predicted_Intensity', y='Count',
                               title='Distribution of Predicted Rainfall Intensity')
        st.plotly_chart(fig_intensity, use_container_width=True)

        st.subheader('Community and Intensity')
        community_intensity = df.groupby(['community', 'predicted_intensity']).size().reset_index(name='count')
        fig_community_intensity = px.bar(community_intensity, x='community', y='count', color='predicted_intensity',
                                         title='Predicted Intensity by Community', barmode='group')
        st.plotly_chart(fig_community_intensity, use_container_width=True)


def explainability_page():
    # Code for the Explainability Page
    st.title('Model Explainability (SHAP)')
    st.markdown("SHAP (SHapley Additive exPlanations) helps us understand how each feature contributed to the model's predictions.")

    @st.cache_resource
    def load_model_and_data_for_shap():
        try:
            model = CatBoostClassifier()
            model.load_model('best_catboost_model.cbm')
            data = pd.read_csv('path/to/your/train_data.csv')
            sample_data = data.sample(n=100, random_state=42).copy()
            
            # --- PLACEHOLDER DATA FOR FEATURE ENGINEERING (MUST BE THE SAME) ---
            community_map = {'1': 1, '2': 2, '3': 3,}
            district_map = {'1': 1, '2': 2, '3': 3,}
            mean_map_intensity = {'small': 0.5, 'moderate': 1.2, 'heavy': 2.5}
            mean_map_community = {'1': 0.8, '2': 1.5, '3': 2.0}

            # Replicate feature engineering on the sample
            sample_data['community'] = sample_data['community'].astype(str).str.strip()
            sample_data['district'] = sample_data['district'].astype(str).str.strip()
            sample_data['predicted_intensity'] = sample_data['predicted_intensity'].astype(str).str.strip()
            sample_data['community_code'] = sample_data['community'].map(community_map)
            sample_data['district_code'] = sample_data['district'].map(district_map)
            sample_data['prediction_distance'] = np.abs(sample_data['community_code'] - sample_data['district_code'])
            sample_data['avg_distance_by_intensity'] = sample_data['predicted_intensity'].map(mean_map_intensity)
            sample_data['avg_distance_by_community'] = sample_data['community'].map(mean_map_community)
            sample_data = sample_data.drop(columns=['community_code', 'district_code', 'Target', 'ID', 'user_id'])
            for col in ['community', 'district', 'predicted_intensity']:
                sample_data[col] = sample_data[col].astype(str)
            
            return model, sample_data
        except FileNotFoundError:
            st.error("Model or training data not found. Please ensure all files are in the directory.")
            return None, None

    model, sample_data = load_model_and_data_for_shap()
    if model and not sample_data.empty:
        with st.spinner("Generating SHAP plots... this may take a moment."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_data)

        st.subheader("1. SHAP Summary Plot")
        st.info("This plot shows the feature importance for all predictions.")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        shap.summary_plot(shap_values[0], sample_data, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("2. SHAP Dependency Plot")
        st.info("This plot shows how a single feature's value affects the model's prediction.")
        feature_to_plot = st.selectbox("Select a feature to see its dependency plot:", sample_data.columns)
        fig_dep, ax_dep = plt.subplots(1, 1, figsize=(10, 6))
        shap.dependence_plot(feature_to_plot, shap_values[0], sample_data, show=False)
        plt.tight_layout()
        st.pyplot(fig_dep)


# --- Page Navigation Logic ---
# Add a home button to the sidebar
with st.sidebar:
    st.image("https://github.com/streamlit/docs/raw/main/src/pages/images/home.png", width=150)
    if st.button("Return to Home"):
        st.session_state.page = 'Home'
        st.experimental_rerun()

# Use an if/elif block to display the correct page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == 'Prediction':
    prediction_page()
elif st.session_state.page == 'Insights':
    insights_page()
elif st.session_state.page == 'Explainability':
    explainability_page()
