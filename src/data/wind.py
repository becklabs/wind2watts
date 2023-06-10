import requests
import datetime

class WindAPI:
    def __init__(self, appid):
        self.base_url = "https://history.openweathermap.org/data/2.5/history/city"
        self.appid = appid
    
    def get_wind_data(self, lat, lon, timestamp):
        # Convert datetime to unix timestamp
        unix_timestamp = int(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp())

        # Define API call parameters
        params = {
            "lat": lat,
            "lon": lon,
            "type": "hour",
            "start": unix_timestamp,
            "end": unix_timestamp + 3600,  # End time is one hour later
            "appid": self.appid
        }

        # Make API call
        response = requests.get(self.base_url, params=params)

        # Raise exception if request was not successful
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Check if list is not empty
        if not data["list"]:
            raise ValueError("No data available for the given coordinates and timestamp.")
        
        # Extract wind data from the first (and only) list element
        wind_data = data["list"][0]["wind"]

        # Return wind speed and direction
        return wind_data["speed"], wind_data["deg"]
