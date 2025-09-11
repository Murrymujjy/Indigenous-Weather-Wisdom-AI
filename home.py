import streamlit as st
from streamlit_option_menu import option_menu
import prediction_page as prediction_page
import insights_page as insights_page
import explainability_page as explainability_page
import forecasting_page as forecasting_page
import live_forecast_page as live_forecast_page  # New import

# ---- Session Setup ----
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Home"

# ---- Sidebar Navigation ----
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["🏠 Home", "🌦️ Prediction", "📊 Insights", "📈 Forecasting", "🛰️ Live Forecast", "🧠 Explainability"],
        icons=["house", "cloud-drizzle", "bar-chart-line", "graph-up-arrow", "wifi", "brain"],
        menu_icon="cast",
        default_index=["🏠 Home", "🌦️ Prediction", "📊 Insights", "📈 Forecasting", "🛰️ Live Forecast", "🧠 Explainability"].index(st.session_state.selected_nav)
    )
    st.session_state.selected_nav = selected

# ---- Page Content ----
def set_nav_to(page_name):
    st.session_state.selected_nav = page_name

if st.session_state.selected_nav == "🏠 Home":
    # --- Custom Home Page Content ---
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    body, .stApp { font-family: 'Poppins', sans-serif; }
    .main-header { font-size: 3rem; font-weight: bold; color: #4CAF50; text-align: center; margin-bottom: 2rem; }
    .subheader { font-size: 1.5rem; color: #333; margin-top: 2rem; }
    .feature-card-button { 
        background-color: #f0f2f6; 
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 15px; 
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
        transition: transform 0.3s ease-in-out;
        width: 100%;
        text-align: left;
        border: none;
        cursor: pointer;
    }
    .feature-card-button:hover {
        transform: translateY(-5px);
        background-color: #e6e8eb;
    }
    .feature-card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🌾 Indigenous Weather Wisdom AI Model</h1>", unsafe_allow_html=True)
    st.markdown("### A professional showcase of an AI model to forecast rainfall in Ghana's Pra River Basin, combining Indigenous Ecological Indicators with modern machine learning.")
    
    # Using the new parameter to avoid deprecation warning
    st.image("https://images.unsplash.com/photo-1598444738743-f66184547900?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", use_container_width=True, caption="Credit: Unsplash")

    st.markdown("---")
    st.markdown("<h2 class='subheader'>Explore the App's Powerful Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🌦️ Prediction", key="pred_button", use_container_width=True):
            set_nav_to("🌦️ Prediction")
        st.markdown("<p style='text-align:left;'>Get a rainfall forecast by inputting new data. Our model combines CatBoost and LightGBM for high-accuracy predictions.</p>", unsafe_allow_html=True)
    
    with col2:
        if st.button("📊 Insights", key="insights_button", use_container_width=True):
            set_nav_to("📊 Insights")
        st.markdown("<p style='text-align:left;'>Dive into the data with interactive visualizations. Understand trends, patterns, and the distribution of rainfall categories.</p>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("📈 Forecasting", key="forecast_button", use_container_width=True):
            set_nav_to("📈 Forecasting")
        st.markdown("<p style='text-align:left;'>See future rainfall predictions using a dedicated time-series model (Prophet). View forecasts with a clear confidence interval.</p>", unsafe_allow_html=True)
    
    with col4:
        if st.button("🧠 Explainability", key="explain_button", use_container_width=True):
            set_nav_to("🧠 Explainability")
        st.markdown("<p style='text-align:left;'>Understand the 'why' behind the predictions. Our SHAP-powered explainability features reveal which factors most influence the model's decisions.</p>", unsafe_allow_html=True)

    # --- New section for the Live Forecast button ---
    col5, col6 = st.columns(2)
    with col5:
        if st.button("🛰️ Live Forecast", key="live_forecast_button", use_container_width=True):
            set_nav_to("🛰️ Live Forecast")
        st.markdown("<p style='text-align:left;'>Get real-time weather information from a live API. See current conditions and weather parameters for selected communities.</p>", unsafe_allow_html=True)

elif st.session_state.selected_nav == "🌦️ Prediction":
    prediction_page.render()

elif st.session_state.selected_nav == "📊 Insights":
    insights_page.render()

elif st.session_state.selected_nav == "🧠 Explainability":
    explainability_page.render()

elif st.session_state.selected_nav == "📈 Forecasting":
    forecasting_page.render()

elif st.session_state.selected_nav == "🛰️ Live Forecast":
    live_forecast_page.render()
