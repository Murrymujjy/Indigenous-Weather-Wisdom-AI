import streamlit as st

st.set_page_config(
    page_title="Indigenous Weather App",
    page_icon="🏠",
    layout="wide"
)

st.title("Welcome to the Indigenous Weather Wisdom AI Model")
st.markdown("""
This web application is a professional showcase of a machine learning model built to forecast rainfall in Ghana's Pra River Basin.
It is based on a unique dataset of Indigenous Ecological Indicators (IEIs) and demonstrates the power of combining traditional knowledge with modern AI.

👈 Select a page from the navigation bar on the left to get started.

- **Prediction**: Use the model to predict rainfall based on new data.
- **Insights**: Explore key data distributions and relationships.
- **Explainability**: Understand how the model makes its predictions using SHAP.
""")
