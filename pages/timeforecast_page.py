import streamlit as st
import pandas as pd
from joblib import load
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# Load the saved Prophet model and LabelEncoder
try:
    prophet_model = load('prophet_model.joblib')
    target_encoder = load('target_encoder.joblib')
except FileNotFoundError:
    st.error("Error: Prophet model or LabelEncoder files not found. Please run the training script first.")
    st.stop()

st.title("Time Series Forecast")
st.write("This section provides a forecast of the 'Target' variable using the Prophet model.")

# Create a future dataframe for forecasting (e.g., for the next 30 days)
future_days = st.slider("Select the number of days to forecast:", 1, 365, 30)
future = prophet_model.make_future_dataframe(periods=future_days, include_history=True)

# Make predictions
forecast = prophet_model.predict(future)

# Function to find the closest known label (from our previous fix)
def get_closest_label(value, encoder):
    all_labels = np.array(encoder.transform(encoder.classes_))
    closest_label_idx = np.abs(all_labels - value).argmin()
    return encoder.classes_[closest_label_idx]

# Apply the function to the 'yhat' predictions and get confidence intervals
forecast['Target_Forecast'] = forecast['yhat'].apply(lambda x: get_closest_label(x, target_encoder))
forecast['Target_Lower'] = forecast['yhat_lower'].apply(lambda x: get_closest_label(x, target_encoder))
forecast['Target_Upper'] = forecast['yhat_upper'].apply(lambda x: get_closest_label(x, target_encoder))

st.subheader("Forecasted Labels")
st.dataframe(forecast[['ds', 'Target_Forecast', 'yhat_lower', 'yhat_upper']].tail(future_days))

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
