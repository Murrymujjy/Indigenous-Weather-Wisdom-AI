import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# --- Load Data and Model ---
@st.cache_resource
def load_data():
    """Loads the training and test data."""
    try:
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')
        return df_train, df_test
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'train.csv' and 'test.csv' are in your project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()


@st.cache_resource
def load_model():
    """Loads the trained LightGBM model."""
    try:
        lgbm_model = joblib.load('lgbm_model.joblib')
        return lgbm_model
    except FileNotFoundError:
        st.error("Model file 'lgbm_model.joblib' not found. Please ensure it's in your project directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()


# --- Page Content ---
def render():
    st.title("🧠 Model Explainability")
    st.markdown("Understand how the model makes its decisions using **SHAP (SHapley Additive exPlanations)**.")
    st.warning("This page may take a few seconds to load as it computes complex feature importance values.")
    
    df_train, _ = load_data()
    lgbm_model = load_model()

    if df_train is not None and lgbm_model is not None:
        # Prepare data for SHAP
        X = df_train.drop('rainfall', axis=1)
        y = df_train['rainfall']

        # Get categorical features for the explainer
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        
        # Instantiate and fit the LightGBM model to get a Booster object for SHAP
        lgbm_model.fit(X, y)
        
        # SHAP Explainer
        explainer = shap.TreeExplainer(lgbm_model)
        
        st.subheader("Feature Importance Summary")
        st.write("This summary plot shows which features are most important for the model's predictions.")
        
        # Calculate SHAP values for a sample of the training data
        sample_size = min(500, len(X))
        X_sample = X.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig)
        
        st.subheader("Dependency Plots")
        st.write("These plots show the effect of a single feature on the model's prediction. They can reveal complex relationships.")
        
        # Create an interactive SHAP dependency plot for a selected feature
        feature_to_plot = st.selectbox("Select a feature to visualize:", X.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(feature_to_plot, shap_values[0], X_sample, show=False)
        st.pyplot(fig)
