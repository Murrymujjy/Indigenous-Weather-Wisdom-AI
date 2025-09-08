import streamlit as st
import pandas as pd
from joblib import load
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np

def render():
    """Renders the time series forecasting page of the app."""
    
    st.header("Time Series Forecast")
    
    # Load the saved Prophet model and LabelEncoder
    try:
        prophet_model = load('prophet_model.joblib')
        target_encoder = load('target_encoder.joblib')
    except FileNotFoundError:
        st.error("Error: Prophet model or LabelEncoder files not found. Please ensure they have been saved correctly.")
        st.stop()
    
    st.write("This section provides a forecast of the 'Target' variable using the Prophet model.")
    
    # Create a future dataframe for forecasting
    future_days = st.slider("Select the number of days to forecast:", 1, 365, 30)
    
    # Make a dataframe with future dates only
    future_df = prophet_model.make_future_dataframe(periods=future_days, include_history=False)
    
    # Make predictions
    forecast = prophet_model.predict(future_df)
    
    # Function to find the closest known label
    def get_closest_label(value, encoder):
        """
        Maps a predicted continuous value to the closest integer label
        known by the LabelEncoder.
        """
        all_labels = np.array(encoder.transform(encoder.classes_))
        # Find the index of the closest integer label
        closest_label_idx = np.abs(all_labels - value).argmin()
        # Return the corresponding original label
        return encoder.classes_[closest_label_idx]
    
    # Apply the function to the 'yhat' predictions and get confidence intervals
    forecast['Target_Forecast'] = forecast['yhat'].apply(lambda x: get_closest_label(x, target_encoder))
    
    st.subheader("Forecasted Labels")
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
