import streamlit as st
from streamlit_option_menu import option_menu
import pages.prediction_page as prediction_page
import pages.insights_page as insights_page
import pages.explainability_page as explainability_page
import pages.forecasting_page as forecasting_page

# ---- Session Setup ----
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Home"

# ---- Sidebar Navigation ----
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["🏠 Home", "🌦️ Prediction", "📊 Insights", "🧠 Explainability", "📈 Forecasting"],
        icons=["house", "cloud-drizzle", "bar-chart-line", "brain", "graph-up-arrow"],
        menu_icon="cast",
        default_index=["🏠 Home", "🌦️ Prediction", "📊 Insights", "🧠 Explainability", "📈 Forecasting"].index(st.session_state.selected_nav)
    )
    st.session_state.selected_nav = selected

# ---- Pages ----
if st.session_state.selected_nav == "🏠 Home":
    st.title("🌾 Indigenous Weather Wisdom AI Model")
    st.markdown("""
    This web application is a professional showcase of a machine learning model built to forecast rainfall in Ghana's Pra River Basin.
    It combines a unique dataset of Indigenous Ecological Indicators with modern AI.
    
    Use the navigation menu on the left to explore the app's features.
    """)
    st.subheader("Explore the App's Sections")
    st.markdown("---")
    
    st.markdown("### **🌦️ Prediction**")
    st.write("Make new predictions based on your own data.")
    
    st.markdown("### **📊 Insights**")
    st.write("Explore data visualizations and key insights from the dataset.")

    st.markdown("### **📈 Forecasting**")
    st.write("View the time series forecast for future weather patterns.")

    st.markdown("### **🧠 Explainability**")
    st.write("Understand how the model makes its decisions.")

    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Made with ❤️ for the Indigenous Weather Competition</div>", unsafe_allow_html=True)

elif st.session_state.selected_nav == "🌦️ Prediction":
    prediction_page.render()

elif st.session_state.selected_nav == "📊 Insights":
    insights_page.render()

elif st.session_state.selected_nav == "🧠 Explainability":
    explainability_page.render()

elif st.session_state.selected_nav == "📈 Forecasting":
    forecasting_page.render()
