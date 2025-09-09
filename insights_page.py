import streamlit as st
import pandas as pd
import plotly.express as px

# --- Load Data ---
@st.cache_data
def load_data():
    """Loads the training data for the insights page."""
    try:
        df = pd.read_csv('train.csv')
        return df
    except FileNotFoundError:
        st.error("Training data file ('train.csv') not found. Please ensure it's in the project directory.")
        st.stop()

# --- Page Content ---
def render():
    st.title("📊 Data Insights")
    st.markdown("Explore key visualizations and patterns in the Indigenous Weather Wisdom dataset.")

    df = load_data()

    # --- Target Distribution Chart ---
    st.subheader("Distribution of Target Variable")
    target_counts = df['Target'].value_counts().reset_index()
    target_counts.columns = ['Target', 'Count']
    fig1 = px.bar(target_counts, x='Target', y='Count',
                  title='Count of Each Rainfall Category',
                  labels={'Target': 'Rainfall Category', 'Count': 'Number of Occurrences'})
    st.plotly_chart(fig1)

    # --- Community and Rainfall Chart ---
    st.subheader("Community-level Insights")
    community_counts = df.groupby(['community', 'Target']).size().reset_index(name='Count')
    fig2 = px.bar(community_counts, x='community', y='Count', color='Target',
                  title='Rainfall Categories per Community',
                  labels={'community': 'Community', 'Count': 'Number of Occurrences'})
    st.plotly_chart(fig2)

    # --- Indicator Description and Rainfall Chart ---
    st.subheader("Indicator Insights")
    indicator_counts = df.groupby(['indicator_description', 'Target']).size().reset_index(name='Count')
    fig3 = px.bar(indicator_counts, x='indicator_description', y='Count', color='Target',
                  title='Rainfall Categories per Indicator Description',
                  labels={'indicator_description': 'Indicator Description', 'Count': 'Number of Occurrences'})
    st.plotly_chart(fig3)
