import streamlit as st
from streamlit_option_menu import option_menu
import pages.prediction_page as prediction_page
import pages.insights_page as insights_page
import pages.explainability_page as explainability_page

# ---- Session Setup ----
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Home"

# ---- Sidebar Navigation ----
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["🏠 Home", "🌦️ Prediction", "📊 Insights", "🧠 Explainability"],
        icons=["house", "cloud-drizzle", "bar-chart-line", "brain"],
        menu_icon="cast",
        default_index=["🏠 Home", "🌦️ Prediction", "📊 Insights", "🧠 Explainability"].index(st.session_state.selected_nav)
    )
    st.session_state.selected_nav = selected

# ---- Home Page ----
if st.session_state.selected_nav == "🏠 Home":
    st.title("🌾 Indigenous Weather Wisdom AI Model")
    st.markdown("""
    This web application is a professional showcase of a machine learning model built to forecast rainfall in Ghana's Pra River Basin.
    It combines a unique dataset of Indigenous Ecological Indicators with modern AI.
    """)
    st.subheader("Explore the App's Sections")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌦️ Prediction")
        if st.button("Make a New Prediction"):
            st.session_state.selected_nav = "🌦️ Prediction"
            st.rerun()

    with col2:
        st.markdown("### 📊 Insights")
        if st.button("View Data Visualizations"):
            st.session_state.selected_nav = "📊 Insights"
            st.rerun()
            
    st.markdown("---")
    
    st.markdown("### 🧠 Explainability")
    if st.button("Understand the Model"):
        st.session_state.selected_nav = "🧠 Explainability"
        st.rerun()
    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>Made with ❤️ for the Indigenous Weather Competition</div>", unsafe_allow_html=True)

# ---- Pages ----
elif st.session_state.selected_nav == "🌦️ Prediction":
    prediction_page.render()

elif st.session_state.selected_nav == "📊 Insights":
    insights_page.render()

elif st.session_state.selected_nav == "🧠 Explainability":
    explainability_page.render()
