# TODO:
# - Vectorize window creation, pop out into helper

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List
from sklearn.preprocessing import MinMaxScaler

from .util import velocity_vector, airspeed


class LatLonPowerDataset(Dataset):
    """
    Input features:
    Concatenation of 5 time-series datapoints, each having:
    - Time delta from previous point (seconds)
    - Latitude delta from previous point (decimal degrees)
    - Longitude delta from previous point (decimal degrees)
    - Elevation delta from previous point (meters)
    - Wind speed (m/s)
    - Wind direction (degrees)

    Justifications
    - Time delta: Need to represent time, but want the model to be robust to time of day, date in time.
    - Latitude delta: Need to represent distance, but want the model to be robust to location
    - Longitude delta: Need to represent distance, but want the model to be robust to location
    - Elevation delta: Need to represent elevation, but want the model to be robust to location

    Output labels:
    - 5 second power in watts (target)
    """

    def __init__(
        self,
        activities: List[pd.DataFrame],
        window_size: int,
        n_features: int = 6,
        time_column="timestamp",
        latitude_column="position_lat",
        longitude_column="position_long",
        elevation_column="enhanced_altitude",
        speed_column="enhanced_speed",
        wind_speed_column="wind_speed",
        wind_direction_column="wind_direction",
        power_column="power",
        window_scaler=None,
        power_scaler=None,
        device="cpu",
    ):
        self.window_size = window_size
        self.n_features = n_features
        self.device = device
        self.time_column = time_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.elevation_column = elevation_column
        self.wind_speed_column = wind_speed_column
        self.wind_direction_column = wind_direction_column
        self.power_column = power_column

        windows = []
        window_powers = []

        for fit_dataframe in activities:

            fit_dataframe = fit_dataframe[
                (fit_dataframe[power_column].notnull())
                & (fit_dataframe[latitude_column].notnull())
                & (fit_dataframe[longitude_column].notnull())
                & (fit_dataframe[time_column].notnull())
                & (fit_dataframe[elevation_column].notnull())
                & (fit_dataframe[wind_direction_column].notnull())
                & (fit_dataframe[wind_speed_column].notnull())
            ].copy()

            fit_dataframe["power_window"] = (
                fit_dataframe[self.power_column].rolling(window_size).mean()
            )

            # Convert the time column to seconds
            fit_dataframe["time_seconds"] = (
                pd.to_datetime(fit_dataframe[self.time_column]).astype("int64")
                // 10**9
            )

            if len(fit_dataframe) < window_size:
                continue

            # TODO: Vectorize window creation
            # Convert DataFrame columns to NumPy arrays
            latitude = fit_dataframe[self.latitude_column].values
            longitude = fit_dataframe[self.longitude_column].values
            elevation = fit_dataframe[self.elevation_column].values
            time_seconds = fit_dataframe["time_seconds"].values
            wind_speed = fit_dataframe["wind_speed"].values
            wind_direction = fit_dataframe["wind_direction"].values
            power_window = fit_dataframe["power_window"].values

            activity_windows = []
            activity_powers = []

            for i in range(window_size - 1, len(fit_dataframe)):
                lat_delta = (
                    latitude[i - window_size + 1 : i + 1]
                    - latitude[i - window_size + 1]
                )
                lon_delta = (
                    longitude[i - window_size + 1 : i + 1]
                    - longitude[i - window_size + 1]
                )
                elevation_delta = (
                    elevation[i - window_size + 1 : i + 1]
                    - elevation[i - window_size + 1]
                )
                time_delta = (
                    time_seconds[i - window_size + 1 : i + 1]
                    - time_seconds[i - window_size + 1]
                )

                wind_speed_window = wind_speed[i - window_size + 1 : i + 1]
                wind_direction_window = wind_direction[i - window_size + 1 : i + 1]

                window = np.column_stack(
                    (
                        lat_delta,
                        lon_delta,
                        elevation_delta,
                        time_delta,
                        wind_speed_window,
                        wind_direction_window,
                    )
                )

                activity_windows.append(window)
                activity_powers.append(power_window[i])

            windows.extend(activity_windows)
            window_powers.extend(activity_powers)

        self.windows = np.array(windows)
        self.window_powers = np.array(window_powers).reshape(-1, 1)

        if window_scaler is None:
            window_scaler = MinMaxScaler()
            window_scaler.fit(self.windows.reshape(-1, self.n_features))

        if power_scaler is None:
            power_scaler = MinMaxScaler()
            power_scaler.fit(self.window_powers)

        self.window_scaler = window_scaler
        self.power_scaler = power_scaler

        self.len_windows = len(self.windows)

        # Check if the windows are empty
        if self.len_windows == 0:
            return

        self.windows_scaled = self.window_scaler.transform(
            self.windows.reshape(-1, self.n_features)
        ).reshape(-1, self.window_size, self.n_features)

        self.windows_scaled = torch.FloatTensor(self.windows_scaled).to(self.device)

        self.window_powers_scaled = self.power_scaler.transform(
            self.window_powers
        ).reshape(-1, 1)

        self.window_powers_scaled = torch.FloatTensor(
            self.window_powers_scaled
        ).to(self.device)

    def __len__(self):
        return self.len_windows 

    def __getitem__(self, idx):
        return self.windows_scaled[idx], self.window_powers_scaled[idx]


class AirspeedPowerDataset(LatLonPowerDataset):
    """
    Features:
    window_length x 3 matrix of:
    - Airspeed (m/s)
    - Elevation delta (meters)
    - Time delta (seconds)

    Airspeed of rider will reduce overfitting on wind direction.

    Labels:
    - Moving Average Power (Watts)
    """

    def __init__(
        self,
        activities: List[pd.DataFrame],
        window_size: int,
        n_features: int = 3,
        time_column="timestamp",
        latitude_column="position_lat",
        longitude_column="position_long",
        elevation_column="enhanced_altitude",
        wind_speed_column="wind_speed",
        wind_direction_column="wind_direction",
        power_column="power",
        window_scaler=None,
        power_scaler=None,
        device="cpu",
    ):
        self.window_size = window_size
        self.n_features = n_features
        self.device = device
        self.time_column = time_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.elevation_column = elevation_column
        self.wind_speed_column = wind_speed_column
        self.wind_direction_column = wind_direction_column
        self.power_column = power_column

        windows = []
        window_powers = []

        for fit_dataframe in activities:
            fit_dataframe = fit_dataframe[
                (fit_dataframe[power_column].notnull())
                & (fit_dataframe[latitude_column].notnull())
                & (fit_dataframe[longitude_column].notnull())
                & (fit_dataframe[time_column].notnull())
                & (fit_dataframe[elevation_column].notnull())
                & (fit_dataframe[wind_direction_column].notnull())
                & (fit_dataframe[wind_speed_column].notnull())
            ].copy()

            if len(fit_dataframe) < window_size:
                continue

            fit_dataframe["power_window"] = (
                fit_dataframe[self.power_column].rolling(window_size).mean()
            )

            fit_dataframe[self.time_column] = pd.to_datetime(
                fit_dataframe[self.time_column]
            )

            # Convert the time column to seconds
            fit_dataframe["time_seconds"] = (
                fit_dataframe[self.time_column].astype("int64") // 10**9
            )

            for i in range(1, len(fit_dataframe)):
                lat1 = fit_dataframe.iloc[i - 1][self.latitude_column]
                lon1 = fit_dataframe.iloc[i - 1][self.longitude_column]
                lat2 = fit_dataframe.iloc[i][self.latitude_column]
                lon2 = fit_dataframe.iloc[i][self.longitude_column]
                time1 = fit_dataframe.iloc[i - 1][self.time_column]
                time2 = fit_dataframe.iloc[i][self.time_column]

                try:
                    direction, speed = velocity_vector(
                        lat1, lon1, time1, lat2, lon2, time2
                    )
                except ZeroDivisionError:
                    direction, speed = np.nan, np.nan

                fit_dataframe.at[i, "direction"] = direction
                fit_dataframe.at[i, "speed_calc"] = speed

            fit_dataframe = fit_dataframe.iloc[1:].copy()

            if fit_dataframe["speed_calc"].isnull().any():
                continue

            fit_dataframe["airspeed"] = airspeed(
                fit_dataframe["speed_calc"].values,
                fit_dataframe["direction"].values,
                fit_dataframe["wind_speed"].values,
                fit_dataframe["wind_direction"].values,
            )

            fit_dataframe = fit_dataframe.query("airspeed.notnull()")
            fit_dataframe = fit_dataframe.copy()

            # TODO: Vectorize window creation
            # Convert DataFrame columns to NumPy arrays
            elevation = fit_dataframe[self.elevation_column].values
            time_seconds = fit_dataframe["time_seconds"].values
            airspeed_arr = fit_dataframe["airspeed"].values
            power_window = fit_dataframe["power_window"].values

            activity_windows = []
            activity_powers = []

            for i in range(window_size - 1, len(fit_dataframe)):
                elevation_delta = (
                    elevation[i - window_size + 1 : i + 1]
                    - elevation[i - window_size + 1]
                )
                time_delta = (
                    time_seconds[i - window_size + 1 : i + 1]
                    - time_seconds[i - window_size + 1]
                )
                airspeed_window = airspeed_arr[i - window_size + 1 : i + 1]

                window = np.column_stack(
                    (
                        airspeed_window,
                        elevation_delta,
                        time_delta,
                    )
                )

                activity_windows.append(window)
                activity_powers.append(power_window[i])

            windows.extend(activity_windows)
            window_powers.extend(activity_powers)

        self.windows = np.array(windows)
        self.window_powers = np.array(window_powers).reshape(-1, 1)

        if window_scaler is None:
            window_scaler = MinMaxScaler()
            window_scaler.fit(self.windows.reshape(-1, 3))

        if power_scaler is None:
            power_scaler = MinMaxScaler()
            power_scaler.fit(self.window_powers)

        self.window_scaler = window_scaler
        self.power_scaler = power_scaler

        if len(self.windows) == 0:
            self.windows_scaled = self.windows
            self.window_powers_scaled = self.window_powers
            return

        self.windows_scaled = self.window_scaler.transform(
            self.windows.reshape(-1, self.n_features)
        ).reshape(-1, self.window_size, self.n_features)

        self.window_powers_scaled = self.power_scaler.transform(self.window_powers)
