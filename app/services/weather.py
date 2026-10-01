import requests


class WeatherServiceError(Exception):
    pass


def current_weather(city, timeout=8):
    try:
        location_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        location_response.raise_for_status()
        locations = location_response.json().get("results", [])
        if not locations:
            raise WeatherServiceError("We could not find that city. Check the spelling and try again.")

        location = locations[0]
        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=timeout,
        )
        forecast_response.raise_for_status()
        data = forecast_response.json()
        current = data["current"]
        daily = data["daily"]
        return {
            "city": location["name"],
            "country": location.get("country", ""),
            "temperature_c": round(current["temperature_2m"], 1),
            "humidity_percent": int(current["relative_humidity_2m"]),
            "rainfall_mm": round(current.get("precipitation", 0), 1),
            "weather_code": current.get("weather_code", 0),
            "forecast": [
                {
                    "date": daily["time"][index],
                    "high_c": round(daily["temperature_2m_max"][index], 1),
                    "low_c": round(daily["temperature_2m_min"][index], 1),
                    "rain_chance": daily["precipitation_probability_max"][index],
                }
                for index in range(len(daily["time"]))
            ],
        }
    except WeatherServiceError:
        raise
    except (requests.RequestException, KeyError, TypeError, ValueError) as error:
        raise WeatherServiceError("Weather data is temporarily unavailable. Please try again later.") from error