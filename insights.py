import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Insights", page_icon="📊")

@st.cache_data
def load_data():
    try:
        # REPLACE WITH THE PATH TO YOUR TRAINING DATA
        df = pd.read_csv('path/to/your/train_data.csv')
        df['community'] = df['community'].astype(str).str.strip()
        df['predicted_intensity'] = df['predicted_intensity'].astype(str).str.strip()
        df['Target'] = df['Target'].astype(str) # Convert target to string for plotting
        return df
    except FileNotFoundError:
        st.error("Training data file not found. Please place 'train.csv' in the same folder.")
        return pd.DataFrame()

st.title('Data Insights')
st.markdown("Explore key distributions and relationships in the training data.")

df = load_data()
if not df.empty:
    st.subheader('Target Distribution')
    target_counts = df['Target'].value_counts().reset_index()
    target_counts.columns = ['Target', 'Count']
    fig = px.pie(target_counts, names='Target', values='Count', title='Distribution of Rainfall Targets')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Rainfall Intensity Distribution')
    intensity_counts = df['predicted_intensity'].value_counts().reset_index()
    intensity_counts.columns = ['Predicted_Intensity', 'Count']
    fig_intensity = px.bar(intensity_counts, x='Predicted_Intensity', y='Count',
                           title='Distribution of Predicted Rainfall Intensity')
    st.plotly_chart(fig_intensity, use_container_width=True)

    st.subheader('Community and Intensity')
    community_intensity = df.groupby(['community', 'predicted_intensity']).size().reset_index(name='count')
    fig_community_intensity = px.bar(community_intensity, x='community', y='count', color='predicted_intensity',
                                     title='Predicted Intensity by Community', barmode='group')
    st.plotly_chart(fig_community_intensity, use_container_width=True)
