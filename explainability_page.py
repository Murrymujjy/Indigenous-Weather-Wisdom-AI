import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
    
    df_train, df_test = load_data()
    lgbm_model = load_model()

    if df_train is not None and lgbm_model is not None:
        # --- REPLICATE NOTEBOOK PREPROCESSING ---
        # 1. Drop the problematic columns as done in the notebook
        columns_to_drop_from_raw = ['ID', 'user_id', 'prediction_time', 'time_observed']
        for col in columns_to_drop_from_raw:
            if col in df_train.columns:
                df_train = df_train.drop(col, axis=1)
            if col in df_test.columns:
                df_test = df_test.drop(col, axis=1)

        # 2. Define the target and features based on the notebook
        y_train = df_train['Target']
        X_train = df_train.drop(['Target'], axis=1)

        # FIX: The line below causes the error. The 'Target' column does not exist in df_test.
        # Use errors='ignore' to prevent the app from crashing.
        X_test = df_test.drop(['Target'], axis=1, errors='ignore')
        
        # 3. Label encode categorical features exactly as in the notebook
        categorical_features = ['community', 'district', 'predicted_intensity', 'indicator', 'indicator_description', 'forecast_length']
        for col in categorical_features:
            le = LabelEncoder()
            all_data = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
            le.fit(all_data)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str)) # FIX: Must transform X_test too

        # 4. Fit the LightGBM model to ensure the SHAP explainer has a booster object
        lgbm_model.fit(X_train, y_train)

        # SHAP Explainer
        explainer = shap.TreeExplainer(lgbm_model)
        
        st.subheader("Feature Importance Summary")
        st.write("This summary plot shows which features are most important for the model's predictions.")
        
        # Calculate SHAP values for a sample of the training data
        sample_size = min(500, len(X_train))
        X_sample = X_train.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig)
        
        st.subheader("Dependency Plots")
        st.write("These plots show the effect of a single feature on the model's prediction. They can reveal complex relationships.")
        
        # Create an interactive SHAP dependency plot for a selected feature
        feature_to_plot = st.selectbox("Select a feature to visualize:", X_train.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(feature_to_plot, shap_values, X_sample, show=False)
        st.pyplot(fig)
