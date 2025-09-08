import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

def render():
    st.title('Model Explainability (SHAP)')
    st.markdown("SHAP (SHapley Additive exPlanations) helps us understand how each feature contributed to the model's predictions.")

    @st.cache_resource
    def load_model_and_data_for_shap():
        try:
            model = CatBoostClassifier()
            # Path is relative to the app.py file
            model.load_model('best_catboost_model.cbm')
            data = pd.read_csv('train_data.csv')
            sample_data = data.sample(n=100, random_state=42).copy()
            
            # --- IMPORTANT: Replicate your exact feature engineering here ---
            community_map = {'community_a': 1, 'community_b': 2, 'community_c': 3}
            district_map = {'district_1': 1, 'district_2': 2, 'district_3': 3}
            mean_map_intensity = {'small': 0.5, 'moderate': 1.2, 'heavy': 2.5}
            mean_map_community = {'community_a': 0.8, 'community_b': 1.5, 'community_c': 2.0}

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
            st.error("Model or training data not found. Please ensure all files are in the main directory.")
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
