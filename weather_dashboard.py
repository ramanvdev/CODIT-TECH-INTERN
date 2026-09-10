"""
Task 1 - API Integration and Data Visualization
--------------------------------------------------
Fetches live current-weather data for a set of cities from the OpenWeatherMap
public REST API and builds a multi-panel visualization dashboard using
Matplotlib / Seaborn.

Usage:
    export OPENWEATHER_API_KEY="your_api_key_here"
    python weather_dashboard.py

If no API key / internet connection is available (e.g. while running in an
offline/sandboxed environment), the script automatically falls back to a
cached sample response with the exact same JSON schema returned by
OpenWeatherMap, so the rest of the pipeline (parsing -> DataFrame ->
visualization) can still be demonstrated end to end.
"""

import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

CITIES = ["Chennai", "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Kolkata"]

# ---------------------------------------------------------------------------
# Cached sample responses (same schema as the real API) - used only as an
# offline fallback so the script always produces output.
# ---------------------------------------------------------------------------
SAMPLE_RESPONSES = {
    "Chennai":   {"main": {"temp": 31.2, "humidity": 74, "feels_like": 36.1}, "wind": {"speed": 4.1}, "weather": [{"description": "haze"}]},
    "Mumbai":    {"main": {"temp": 29.4, "humidity": 81, "feels_like": 33.9}, "wind": {"speed": 5.6}, "weather": [{"description": "scattered clouds"}]},
    "Delhi":     {"main": {"temp": 33.8, "humidity": 46, "feels_like": 35.0}, "wind": {"speed": 3.2}, "weather": [{"description": "clear sky"}]},
    "Bengaluru": {"main": {"temp": 24.6, "humidity": 68, "feels_like": 25.1}, "wind": {"speed": 2.8}, "weather": [{"description": "light rain"}]},
    "Hyderabad": {"main": {"temp": 28.9, "humidity": 58, "feels_like": 30.2}, "wind": {"speed": 3.9}, "weather": [{"description": "few clouds"}]},
    "Kolkata":   {"main": {"temp": 30.1, "humidity": 79, "feels_like": 34.5}, "wind": {"speed": 4.4}, "weather": [{"description": "overcast clouds"}]},
}


def fetch_weather(city: str) -> dict:
    """Fetch current weather for a city. Falls back to cached sample data
    if the API key is missing or the request fails (e.g. no network)."""
    if API_KEY:
        try:
            params = {"q": city, "appid": API_KEY, "units": "metric"}
            resp = requests.get(BASE_URL, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"[warn] live API call failed for {city} ({exc}); using cached sample data")
    return SAMPLE_RESPONSES[city]


def build_dataframe(cities) -> pd.DataFrame:
    rows = []
    for city in cities:
        data = fetch_weather(city)
        rows.append({
            "city": city,
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "humidity_pct": data["main"]["humidity"],
            "wind_speed_mps": data["wind"]["speed"],
            "condition": data["weather"][0]["description"].title(),
        })
    return pd.DataFrame(rows)


def build_dashboard(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Live Weather Dashboard - Major Indian Cities", fontsize=16, fontweight="bold")

    # Panel 1: Temperature by city (bar)
    sns.barplot(data=df, x="city", y="temperature_c", ax=axes[0, 0], palette="flare")
    axes[0, 0].set_title("Current Temperature (°C)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # Panel 2: Humidity by city (bar)
    sns.barplot(data=df, x="city", y="humidity_pct", ax=axes[0, 1], palette="crest")
    axes[0, 1].set_title("Humidity (%)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Humidity (%)")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Panel 3: Temperature vs Feels-Like (grouped bar)
    melted = df.melt(id_vars="city", value_vars=["temperature_c", "feels_like_c"],
                      var_name="metric", value_name="value")
    sns.barplot(data=melted, x="city", y="value", hue="metric", ax=axes[1, 0], palette="mako")
    axes[1, 0].set_title("Actual vs Feels-Like Temperature")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("°C")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].legend(labels=["Actual", "Feels Like"], title="")

    # Panel 4: Wind speed (horizontal bar)
    sns.barplot(data=df.sort_values("wind_speed_mps"), y="city", x="wind_speed_mps",
                ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("Wind Speed (m/s)")
    axes[1, 1].set_xlabel("Wind Speed (m/s)")
    axes[1, 1].set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    print(f"Dashboard saved to {out_path}")


if __name__ == "__main__":
    df = build_dataframe(CITIES)
    print(df.to_string(index=False))
    df.to_csv("weather_data.csv", index=False)
    build_dashboard(df, "weather_dashboard.png")
