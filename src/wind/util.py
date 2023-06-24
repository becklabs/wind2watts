
import datetime

import pandas as pd

from .providers import WindProvider


def add_wind_data(df: pd.DataFrame,
                    timestamp_column: str,
                    latitude_column: str,
                    longitude_column: str,
                  wind: WindProvider):
    """
    Returns a clone of the dataframe with the following columns added:
    wind_speed (m/s)
    wind_direction (Degrees) 

    Data is added hourly, with the mean lat/lon for the hour being used for geolocation.
    # TODO: Evaluate model performace when a mean of lat/lon is used for a given hour, vs the actual lat/lon
    """
    df = df.copy()
    df['year'] = df[timestamp_column].dt.year
    df['month'] = df[timestamp_column].dt.month
    df['day'] = df[timestamp_column].dt.day
    df['hour'] =  df[timestamp_column].dt.hour

    for (year, month, day, hour), group in df.groupby(['year',
                                                        'month',
                                                        'day',
                                                        'hour']):

        dt = datetime.datetime(int(year), int(month), int(day))

        lat = group[latitude_column].mean()
        lon = group[longitude_column].mean()
        wind_data = wind.get_wind_data(lat, lon, dt)

        wind_data['hour'] = wind_data['timestamp'].dt.hour
        wind_data_hour = wind_data[wind_data['hour'] == hour]

        speed = wind_data_hour['wind_speed'].values[0]
        direction = wind_data_hour['wind_direction'].values[0]

        df.loc[group.index, 'wind_speed'] = speed
        df.loc[group.index, 'wind_direction'] = direction
    
    return df
