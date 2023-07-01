# TODO:
# - Incorporate elevation data into query 

import datetime
import urllib.parse
from abc import ABC, abstractmethod

import pandas as pd
import requests

from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def retry_get(url: str) -> requests.Response:
    return requests.get(url)


class WindProvider(ABC):
    @abstractmethod
    def get_wind_data(self, lat: str, lon: str, day: datetime.datetime) -> pd.DataFrame:
        """
        Returns a dataframe with columns:
        wind_speed (m/s)
        wind_direction (Degrees clockwise from true north)
        timestamp (GMT+0 datetime.datetime)

        Each row in the dataframe corresponds to an hour in the given day
        """


class OpenMeteoProvider(WindProvider):
    def __init__(self):
        self.url = "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&timezone={timezone}&start_date={day_str}&end_date={day_str}&hourly=windspeed_10m,winddirection_10m&windspeed_unit=ms&min={day_str}&max={day_str}"

    def get_wind_data(self, lat: float, lon: float, day: datetime.datetime):
        timezone = urllib.parse.quote_plus("GMT")

        day_str = day.strftime("%Y-%m-%d")
        url = self.url.format(lat=lat, lon=lon, day_str=day_str, timezone=timezone)
        response = retry_get(url).json()
        try:
            dataframe = pd.DataFrame(response["hourly"])
        except Exception as e:
            raise Exception(
                f"""
            Error parsing response from OpenMeteo: {response}
            with parameters: lat={lat}, lon={lon}, date={day_str}, timezone={timezone}
            """
            ) from e
        dataframe = dataframe.rename(
            columns={
                "windspeed_10m": "wind_speed",
                "winddirection_10m": "wind_direction",
            }
        )
        dataframe["timestamp"] = pd.to_datetime(dataframe["time"])
        return dataframe


class OWMWindProvider(WindProvider):
    def __init__(self, appid: str):
        self.base_url = "https://history.openweathermap.org/data/2.5/history/city"
        self.appid = appid

    def get_wind_data(self, lat: str, lon: str, day: datetime.datetime) -> pd.DataFrame:
        unix_timestamp = int(day.timestamp())

        params = {
            "lat": lat,
            "lon": lon,
            "type": "hour",
            "start": unix_timestamp,
            "end": unix_timestamp + 3600,  # End time is one hour later
            "appid": self.appid,
        }

        response = requests.get(self.base_url, params=params)

        response.raise_for_status()

        data = response.json()

        try:
            wind_data = data["list"][0]["wind"]
        except Exception as e:
            raise Exception(
                f"""
            Error parsing response from OpenWeatherMap: {response}
            with parameters: lat={lat}, lon={lon}, date={day} 
            """
            ) from e

        return pd.DataFrame(wind_data)
