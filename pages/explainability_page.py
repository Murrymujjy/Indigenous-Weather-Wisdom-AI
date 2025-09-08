import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier

@st.cache_resource
def load_resources_for_shap():
    try:
        model = CatBoostClassifier()
        model.load_model('../best_catboost_model.cbm')

        train_df = pd.read_csv('../train.csv')
        test_df = pd.read_csv('../test.csv')
        
        train_df['community'] = train_df['community'].astype(str).str.strip()
        train_df['district'] = train_df['district'].astype(str).str.strip()
        train_df['predicted_intensity'] = train_df['predicted_intensity'].astype(str).str.strip()
        test_df['community'] = test_df['community'].astype(str).str.strip()
        test_df['district'] = test_df['district'].astype(str).str.strip()
        test_df['predicted_intensity'] = test_df['predicted_intensity'].astype(str).str.strip()

        all_communities = pd.concat([train_df['community'], test_df['community']]).unique()
        all_districts = pd.concat([train_df['district'], test_df['district']]).unique()
        community_map = {community: i for i, community in enumerate(all_communities)}
        district_map = {district: i for i, district in enumerate(all_districts)}

        train_df['community_code'] = train_df['community'].map(community_map)
        train_df['district_code'] = train_df['district'].map(district_map)
        train_df['prediction_distance'] = np.abs(train_df['community_code'] - train_df['district_code'])

        mean_map_intensity = train_df.groupby('predicted_intensity')['prediction_distance'].mean().to_dict()
        mean_map_community = train_df.groupby('community')['prediction_distance'].mean().to_dict()

        sample_data = train_df.sample(n=100, random_state=42).copy()
        
        sample_data['community_code'] = sample_data['community'].map(community_map)
        sample_data['district_code'] = sample_data['district'].map(district_map)
        sample_data['prediction_distance'] = np.abs(sample_data['community_code'] - sample_data['district_code'])
        sample_data['avg_distance_by_intensity'] = sample_data['predicted_intensity'].map(mean_map_intensity)
        sample_data['avg_distance_by_community'] = sample_data['community'].map(mean_map_community)
        
        # --- FIX: Drop columns only if they exist in the DataFrame ---
        columns_to_drop = ['community_code', 'district_code']
        if 'Target' in sample_data.columns:
            columns_to_drop.append('Target')
        if 'ID' in sample_data.columns:
            columns_to_drop.append('ID')
        if 'user_id' in sample_data.columns:
            columns_to_drop.append('user_id')

        sample_data = sample_data.drop(columns=columns_to_drop)

        for col in ['community', 'district', 'predicted_intensity']:
            sample_data[col] = sample_data[col].astype(str)

        return model, sample_data
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'best_catboost_model.cbm', 'train_data.csv', and 'test_data.csv' are in the main directory.")
        return None, None

def render():
    st.title('Model Explainability (SHAP)')
    st.markdown("SHAP helps us understand how each feature contributed to the model's predictions.")

    model, sample_data = load_resources_for_shap()
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
