import datetime
import urllib.parse
from abc import ABC, abstractmethod

import pandas as pd
import requests

    
class WindProvider(ABC):
    @abstractmethod
    def get_wind_data(self, lat: str, lon: str, day: datetime.datetime):
        """
        Returns a dataframe with columns:
        wind_speed (m/s)
        wind_direction (Degrees)
        timestamp (datetime.datetime)

        Each row in the dataframe corresponds to an hour in the given day
        """

class OpenMeteoProvider(WindProvider):

    def __init__(self):
        self.url = "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&timezone={timezone}&start_date={day_str}&end_date={day_str}&hourly=windspeed_10m,winddirection_10m&windspeed_unit=ms&min={day_str}&max={day_str}"

    def get_wind_data(self, lat: float, lon: float, day: datetime.datetime):
        # Check that the day is timezone aware
        # if day.tzinfo is None:
        #     raise Exception('Must provide a timezone aware datetime object') 
        
        # TODO: variable timezone
        #timezone = day.tzname()
        timezone = 'America/New_York'
        
        timezone = urllib.parse.quote_plus(timezone)

        day_str = day.strftime('%Y-%m-%d')
        url = self.url.format(lat = lat, lon = lon, day_str = day_str, timezone = timezone) 
        response = requests.get(url).json()
        dataframe = pd.DataFrame(response['hourly'])
        dataframe = dataframe.rename(columns={'windspeed_10m': 'wind_speed', 'winddirection_10m': 'wind_direction'})
        dataframe['timestamp'] = pd.to_datetime(dataframe['time'])
        return dataframe
        
class OWMWindProvider(WindProvider):
    def __init__(self, appid: str):
        self.base_url = "https://history.openweathermap.org/data/2.5/history/city"
        self.appid = appid
   
    def get_wind_data(self,
                      lat: str,
                      lon: str,
                      timestamp: datetime.datetime):
        unix_timestamp = int(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp())

        params = {
            "lat": lat,
            "lon": lon,
            "type": "hour",
            "start": unix_timestamp,
            "end": unix_timestamp + 3600,  # End time is one hour later
            "appid": self.appid
        }

        response = requests.get(self.base_url, params=params)

        response.raise_for_status()

        data = response.json()

        if not data["list"]:
            raise ValueError("No data available for the given coordinates and timestamp.")
        
        wind_data = data["list"][0]["wind"]

        return wind_data["speed"], wind_data["deg"]

