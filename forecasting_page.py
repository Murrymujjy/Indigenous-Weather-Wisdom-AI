import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# --- Load Model and Encoder ---
@st.cache_resource
def load_prophet_assets():
    """Loads the trained Prophet model and LabelEncoder."""
    try:
        prophet_model = load('prophet_model.joblib')
        target_encoder = load('target_encoder.joblib')
        return prophet_model, target_encoder
    except FileNotFoundError:
        st.error("Error: Prophet model or LabelEncoder files not found. Please ensure they are in the project directory.")
        st.stop()

# --- Page Content ---
def render():
    st.title("📈 Time Series Forecasting")
    st.markdown("This section provides a forecast of future rainfall categories based on historical trends using the Prophet model.")
    
    prophet_model, target_encoder = load_prophet_assets()
    
    # Create a future dataframe for forecasting (e.g., for the next 30 days)
    future_days = st.slider("Select the number of days to forecast:", 1, 365, 30)
    future_df = prophet_model.make_future_dataframe(periods=future_days, include_history=False)
    
    # Make predictions
    forecast = prophet_model.predict(future_df)
    
    # Function to find the closest known label (from our previous fix)
    def get_closest_label(value, encoder):
        """Maps a predicted continuous value to the closest integer label."""
        all_labels = np.array(encoder.transform(encoder.classes_))
        closest_label_idx = np.abs(all_labels - value).argmin()
        return encoder.classes_[closest_label_idx]
    
    # Apply the function to the 'yhat' predictions
    forecast['Target_Forecast'] = forecast['yhat'].apply(lambda x: get_closest_label(x, target_encoder))
    
    # --- Explanations for the user ---
    st.info("💡 **Understanding the Forecast:**\n\n"
            "- **`yhat`**: This is the model's primary numerical prediction for the rainfall category. The `Target_Forecast` column is derived from this value.\n"
            "- **`yhat_lower` and `yhat_upper`**: These columns represent the **prediction's uncertainty interval**. They show the range within which the true value is expected to fall with a high degree of confidence. The wider the interval, the less certain the model is about its prediction.")

    st.subheader("Forecasted Rainfall")
    st.write("Below is a table showing the forecasted rainfall for each day.")
    st.warning("👉 **Tip:** Use your mouse to scroll down the table to see the full forecast for your selected date range.")
    st.dataframe(forecast[['ds', 'Target_Forecast', 'yhat_lower', 'yhat_upper']])
    
    st.subheader("Interactive Forecast Plot")
    
    # Create a Plotly figure for the forecast
    fig = go.Figure()
    
    # Add the forecasted line
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines', name='Forecasted Value'))
    
    # Add the uncertainty interval
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                             mode='lines', line=dict(width=0),
                             marker=dict(color="#444"),
                             name='Upper Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                             marker=dict(color="#444"),
                             line=dict(width=0),
                             fillcolor='rgba(68, 68, 68, 0.3)',
                             fill='tonexty',
                             name='Lower Bound'))
    
    fig.update_layout(
        title='Time Series Forecast with Uncertainty Interval',
        xaxis_title='Date',
        yaxis_title='Predicted Value'
    )
    
    st.plotly_chart(fig)
