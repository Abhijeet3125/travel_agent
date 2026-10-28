from pydantic import BaseModel
from langchain_core.tools import tool
import requests


@tool
def get_coordinates(city_name: str) -> str:
    """Get the exact latitude and longitude for a given city or place name."""

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    
    response = requests.get(url).json()
    
    if "results" in response and len(response["results"]) > 0:
        lat = response["results"][0]["latitude"]
        lon = response["results"][0]["longitude"]
        country = response["results"][0].get("country", "Unknown")
        return f"Coordinates for {city_name}, {country}: Latitude {lat}, Longitude {lon}"
    return "Location not found."


@tool
def get_weather(latitude: float , longitude: float, start_date: str , end_date: str) -> str:
    """
    Get the historical or forecasted daily weather for a specific location and date range.
    Args:
        latitude: Latitude of the destination.
        longitude: Longitude of the destination.
        start_date: The start date in 'YYYY-MM-DD' format.
        end_date: The end date in 'YYYY-MM-DD' format.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto"
    }

    try:
        response = requests.get(url , params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])

        forcast_lines = []

        for i in range(len(dates)):
            forcast_lines.append(
                f"{dates[i]}: High {max_temps[i]}°C, Low  {min_temps[i]}°C, Rain: {precip[i]}mm"
            )

        return "\n".join(forcast_lines)

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"




    

