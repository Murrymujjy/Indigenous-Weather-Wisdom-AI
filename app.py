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
        ["🏠 Home", "🌦️ Prediction", "📊 Insights", "📈 Forecasting", "🧠 Explainability"],
        icons=["house", "cloud-drizzle", "bar-chart-line", "graph-up-arrow", "brain"],
        menu_icon="cast",
        default_index=["🏠 Home", "🌦️ Prediction", "📊 Insights", "📈 Forecasting", "🧠 Explainability"].index(st.session_state.selected_nav)
    )
    st.session_state.selected_nav = selected

# ---- Page Content ----
if st.session_state.selected_nav == "🏠 Home":
    # --- Custom Home Page Content ---
    st.markdown("""<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #4CAF50; text-align: center; margin-bottom: 2rem; }
    .subheader { font-size: 1.5rem; color: #333; margin-top: 2rem; }
    .stButton button { background-color: #4CAF50; color: white; border-radius: 12px; padding: 10px 24px; font-size: 16px; border: none; cursor: pointer; }
    .stButton button:hover { background-color: #45a049; }
    .feature-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>""", unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>🌾 Indigenous Weather Wisdom AI Model</h1>", unsafe_allow_html=True)
    st.markdown("### A professional showcase of an AI model to forecast rainfall in Ghana's Pra River Basin, combining Indigenous Ecological Indicators with modern machine learning.")

    st.image("https://images.unsplash.com/photo-1549487779-1d440d9b4334?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80", use_column_width=True, caption="Credit: Unsplash")

    st.markdown("---")
    st.markdown("<h2 class='subheader'>Explore the App's Powerful Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### 🌦️ Prediction")
        st.write("Get a rainfall forecast by inputting new data. Our model combines CatBoost and LightGBM for high-accuracy predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Insights")
        st.write("Dive into the data with interactive visualizations. Understand trends, patterns, and the distribution of rainfall categories.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Forecasting")
        st.write("See future rainfall predictions using a dedicated time-series model (Prophet). View forecasts with a clear confidence interval.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Explainability")
        st.write("Understand the 'why' behind the predictions. Our SHAP-powered explainability features reveal which factors most influence the model's decisions.")
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.selected_nav == "🌦️ Prediction":
    pages.prediction_page.render()

elif st.session_state.selected_nav == "📊 Insights":
    pages.insights_page.render()

elif st.session_state.selected_nav == "🧠 Explainability":
    pages.explainability_page.render()

elif st.session_state.selected_nav == "📈 Forecasting":
    pages.forecasting_page.render()
