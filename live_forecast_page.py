import streamlit as st
import requests

# --- Live Forecast Page Content ---
def render():
    st.title("🛰️ Live Forecast")
    st.markdown("Get a live weather forecast for a location in Ghana. The data is fetched in real-time from a weather API.")

    st.info("The live forecast feature requires an API key. For demonstration purposes, we will use a publicly available service.")
    st.markdown("---")

    # List of communities in the Pra River Basin for demonstration
    communities = [
        "Ahafo Ano North", "Asunafo North", "Asunafo South", "Atwima Kwanwoma", 
        "Bia West", "Bia East", "Bodi", "Bosomtwe", "Prestea Huni Valley", 
        "Wassa Amenfi Central", "Wassa Amenfi East", "Wassa Amenfi West"
    ]
    
    # User selects a community
    selected_community = st.selectbox("Select a community:", communities)

    # Simple API call to a placeholder service
    # NOTE: In a real-world scenario, you would need to use a valid API key and endpoint.
    # The URL below is a placeholder and will not return live data.
    api_url = f"https://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={selected_community},Ghana"

    # Use a button to trigger the API call
    if st.button("Get Live Weather", use_container_width=True):
        try:
            # For this example, we'll use mock data as the API key is not provided
            # In your actual code, uncomment the line below and use your API key
            # response = requests.get(api_url)
            # response.raise_for_status() # Raise an exception for bad status codes
            # weather_data = response.json()
            
            # --- Mock data for demonstration ---
            weather_data = {
                "location": {
                    "name": selected_community,
                    "region": "Ghana",
                    "country": "Ghana",
                    "lat": 6.67,
                    "lon": -1.61,
                    "tz_id": "Africa/Accra",
                    "localtime_epoch": 1694380800,
                    "localtime": "2025-09-10 16:30"
                },
                "current": {
                    "temp_c": 28.5,
                    "condition": {
                        "text": "Partly cloudy",
                        "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
                        "code": 1003
                    },
                    "humidity": 75,
                    "cloud": 50,
                    "wind_kph": 15,
                    "precip_mm": 0.5,
                    "feelslike_c": 32.1
                }
            }
            
            st.success("Weather data fetched successfully!")

            # Display weather information
            st.markdown(f"### Weather in {selected_community}")
            st.markdown(f"**Date & Time:** {weather_data['location']['localtime']}")
            st.write(f"**Temperature:** {weather_data['current']['temp_c']}°C")
            st.write(f"**Condition:** {weather_data['current']['condition']['text']}")
            st.write(f"**Humidity:** {weather_data['current']['humidity']}%")
            st.write(f"**Precipitation (last 24h):** {weather_data['current']['precip_mm']} mm")
            st.write(f"**Wind Speed:** {weather_data['current']['wind_kph']} kph")

            # Display the weather icon
            st.image(f"https:{weather_data['current']['condition']['icon']}", width=64)

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching weather data: {e}. Please check your API key and network connection.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
