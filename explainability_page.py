import streamlit as st
import pandas as pd
import shap
from catboost import CatBoostClassifier
import joblib

# --- Load Models and Data ---
@st.cache_resource
def load_assets():
    """Loads the trained CatBoost model and data for explainability."""
    try:
        # Load the training data to be used as a background dataset for SHAP
        df_train = pd.read_csv('train.csv')
        df_train = df_train.drop(['ID', 'user_id', 'prediction_time', 'time_observed', 'Target'], axis=1)
        
        # Load the trained CatBoost model
        model = CatBoostClassifier()
        model.load_model("best_catboost_model.cbm")
        
        return model, df_train
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'best_catboost_model.cbm' and 'train_data.csv' are in the project directory.")
        st.stop()

# --- Page Content ---
def render():
    st.title("🧠 Model Explainability")
    st.markdown("This page uses SHAP (SHapley Additive exPlanations) to help you understand how the model arrives at its predictions.")
    
    st.warning("Note: Generating SHAP plots can be computationally intensive and may take a moment to load.")

    model, df_train = load_assets()

    st.subheader("Feature Importance Summary")
    st.write("This summary plot shows which features are most important for the model's predictions.")
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_train)
    
    # Create the SHAP summary plot
    fig = shap.summary_plot(shap_values, df_train, show=False)
    st.pyplot(fig, bbox_inches='tight')

    st.subheader("Individual Prediction Explanation")
    st.write("Select a row from the data to see a detailed explanation for that specific prediction.")
    
    # Display data table for selection
    df_sample = df_train.sample(n=100, random_state=42)
    selected_row = st.selectbox("Select a sample prediction to explain:", df_sample.index)

    if selected_row:
        instance = df_train.loc[[selected_row]]
        instance_shap_values = explainer.shap_values(instance)
        
        st.write("Feature values for the selected instance:")
        st.dataframe(instance)
        
        # Create a force plot for the individual instance
        st.write("This plot shows how each feature value contributes to the final prediction.")
        force_plot = shap.force_plot(
            explainer.expected_value[0],
            instance_shap_values[0],
            instance
        )
        st.pyplot(force_plot, bbox_inches='tight')
