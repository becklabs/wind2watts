# TODO:
# - Vectorize window creation, pop out into helper

from datetime import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from .util import airspeed, velocity, smooth_gradient


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
        """
        Initializes the Dataset object with input features and parameters.
        Args:
        - activities: List of dataframes, each representing an activity. Each dataframe contains
                      columns corresponding to features and power output.
        - window_size: Size of the sliding window for input data.
        - n_features: Number of features.
        - time_column: Column name for the time stamp in each dataframe.
        - latitude_column: Column name for latitude in each dataframe.
        - longitude_column: Column name for longitude in each dataframe.
        - elevation_column: Column name for elevation in each dataframe.
        - speed_column: Column name for speed in each dataframe.
        - wind_speed_column: Column name for wind speed in each dataframe.
        - wind_direction_column: Column name for wind direction in each dataframe.
        - power_column: Column name for power in each dataframe.
        - window_scaler: Scaler object for normalizing input data (optional).
        - power_scaler: Scaler object for normalizing power data (optional).
        - device: Device type for PyTorch computations.
        """
        self.window_size = window_size
        self.n_features = n_features
        self.device = device
        self.time_column = time_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.elevation_column = elevation_column
        self.speed_column = speed_column
        self.wind_speed_column = wind_speed_column
        self.wind_direction_column = wind_direction_column
        self.power_column = power_column

        windows = []
        window_powers = []

        for fit_dataframe in activities:
            fit_dataframe = self.preprocess_activity(fit_dataframe)

            if len(fit_dataframe) < window_size:
                continue

            latitude = fit_dataframe[self.latitude_column].values
            longitude = fit_dataframe[self.longitude_column].values
            elevation = fit_dataframe[self.elevation_column].values
            time_seconds = fit_dataframe["time_seconds"].values  # generated column
            wind_speed = fit_dataframe[self.wind_speed_column].values
            wind_direction = fit_dataframe[self.wind_direction_column].values

            power_window = fit_dataframe["power_window"].values  # generated column

            activity_windows, activity_powers = self.create_windows(
                fit_dataframe,
                rel_features=[latitude, longitude, elevation, time_seconds],
                abs_features=[wind_speed, wind_direction],
                target=power_window,
            )

            windows.extend(activity_windows)
            window_powers.extend(activity_powers)

        self.prep_windows(windows, window_powers, window_scaler, power_scaler)

    def prep_windows(self, windows, window_powers, window_scaler, power_scaler):
        """
        Prepares the input features and power output data for the model.
        Args:
        - windows: List of arrays, each representing a window of features.
        - window_powers: List of power output values corresponding to each window.
        - window_scaler: Scaler object for normalizing input data.
        - power_scaler: Scaler object for normalizing power data.
        """

        self.windows = np.array(windows)
        self.window_powers = np.array(window_powers).reshape(-1, 1)

        self.windows_scaled, self.window_powers_scaled = self.scale_data(
            self.windows, self.window_powers, window_scaler, power_scaler
        )

        self.len_windows = len(self.windows_scaled)

        self.windows_scaled = torch.FloatTensor(self.windows_scaled).to(self.device)
        self.window_powers_scaled = torch.FloatTensor(self.window_powers_scaled).to(
            self.device
        )

    def preprocess_activity(self, fit_dataframe: pd.DataFrame):
        """
        Preprocesses a dataframe representing an activity by dropping NaN values and adding
        new calculated columns:
        - power_window: rolling mean of power over window size
        - time_seconds: time in seconds since first datapoint

        Args:
        - fit_dataframe: DataFrame representing an activity.
        Returns:
        - fit_dataframe: Preprocessed DataFrame.
        """
        fit_dataframe = fit_dataframe[
            (fit_dataframe[self.power_column].notnull())
            & (fit_dataframe[self.latitude_column].notnull())
            & (fit_dataframe[self.longitude_column].notnull())
            & (fit_dataframe[self.time_column].notnull())
            & (fit_dataframe[self.elevation_column].notnull())
            & (fit_dataframe[self.wind_direction_column].notnull())
            & (fit_dataframe[self.wind_speed_column].notnull())
        ].copy()

        if len(fit_dataframe) < self.window_size:
            return fit_dataframe

        fit_dataframe["power_window"] = (
            fit_dataframe[self.power_column].rolling(self.window_size).mean()
        )

        fit_dataframe[self.time_column] = pd.to_datetime(
            fit_dataframe[self.time_column]
        )

        fit_dataframe["time_seconds"] = (
            fit_dataframe[self.time_column] - fit_dataframe[self.time_column].iloc[0]
        ).dt.total_seconds()

        return fit_dataframe

    def create_windows(
        self,
        fit_dataframe: pd.DataFrame,
        rel_features: List[np.ndarray],
        abs_features: List[np.ndarray],
        target: np.ndarray,
    ):
        """
        Creates sliding windows from a given DataFrame for both features and power output.
        Args:
        - fit_dataframe: DataFrame representing an activity.
        - rel_features: List of arrays, each representing relative feature values.
        - abs_features: List of arrays, each representing absolute feature values.
        - target: Array representing power output.
        Returns:
        - activity_windows: List of arrays, each array representing a window of features.
        - activity_powers: List of power output values corresponding to each window.
        """
        activity_windows = []
        activity_powers = []

        for i in range(self.window_size - 1, len(fit_dataframe)):
            rel_window = np.column_stack(
                (
                    [
                        feature[i - self.window_size + 1 : i + 1]
                        - feature[i - self.window_size + 1]
                        for feature in rel_features
                    ]
                )
            )

            abs_window = np.column_stack(
                (
                    [
                        feature[i - self.window_size + 1 : i + 1]
                        for feature in abs_features
                    ]
                )
            )

            window = np.column_stack(
                (
                    rel_window,
                    abs_window,
                )
            )

            activity_windows.append(window)
            activity_powers.append(target[i])

        return activity_windows, activity_powers

    def scale_data(
        self, windows: np.ndarray, powers: np.ndarray, window_scaler, power_scaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales the input features and power output data.
        Args:
        - windows: Numpy array of windows, each window containing input features.
        - powers: Numpy array of power output values.
        - window_scaler: Scaler object for normalizing input data.
        - power_scaler: Scaler object for normalizing power data.
        Returns:
        - windows_scaled: Numpy array of scaled windows.
        - powers_scaled: Numpy array of scaled power output values.
        """
        if window_scaler is None:
            window_scaler = MinMaxScaler()
            window_scaler.fit(windows.reshape(-1, self.n_features))

        if power_scaler is None:
            power_scaler = MinMaxScaler()
            power_scaler.fit(powers)

        self.window_scaler = window_scaler
        self.power_scaler = power_scaler

        if len(self.windows) == 0:
            return windows, powers

        windows_scaled = self.window_scaler.transform(
            windows.reshape(-1, self.n_features)
        ).reshape(-1, self.window_size, self.n_features)

        powers_scaled = self.power_scaler.transform(powers)

        return windows_scaled, powers_scaled

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
            fit_dataframe = self.preprocess_activity(fit_dataframe)

            if len(fit_dataframe) < window_size:
                continue

            rider_speed, rider_direction = velocity(
                fit_dataframe[self.latitude_column].values,
                fit_dataframe[self.longitude_column].values,
                fit_dataframe["time_seconds"].values,
            )

            fit_dataframe["direction"] = rider_direction
            fit_dataframe["speed_calc"] = rider_speed

            fit_dataframe = fit_dataframe.iloc[1:].copy()

            if fit_dataframe["speed_calc"].isnull().any():
                continue

            fit_dataframe["airspeed"] = airspeed(
                fit_dataframe["speed_calc"].values,
                fit_dataframe["direction"].values,
                fit_dataframe[self.wind_speed_column].values,
                fit_dataframe[self.wind_direction_column].values,
            )

            fit_dataframe = fit_dataframe[fit_dataframe["airspeed"].notnull()]
            fit_dataframe = fit_dataframe.copy()

            elevation = fit_dataframe[self.elevation_column].values
            time_seconds = fit_dataframe["time_seconds"].values
            airspeed_arr = fit_dataframe["airspeed"].values
            power_window = fit_dataframe["power_window"].values

            activity_windows, activity_powers = self.create_windows(
                fit_dataframe,
                rel_features=[time_seconds, elevation],
                abs_features=[airspeed_arr],
                target=power_window,
            )

            windows.extend(activity_windows)
            window_powers.extend(activity_powers)

        self.windows = np.array(windows)
        self.window_powers = np.array(window_powers).reshape(-1, 1)

        self.windows_scaled, self.window_powers_scaled = self.scale_data(
            self.windows, self.window_powers, window_scaler, power_scaler
        )

        self.len_windows = len(self.windows_scaled)

        self.windows_scaled = torch.FloatTensor(self.windows_scaled).to(self.device)
        self.window_powers_scaled = torch.FloatTensor(self.window_powers_scaled).to(
            self.device
        )


class WindAnglePowerDataset(LatLonPowerDataset):
    """
    Features:
    window_length x 3 matrix of:
    - Speed (m/s)
    - Relative Elevation (meters)
    - Wind angle wrt rider (degrees)
    - Wind speed (m/s)

    Labels:
    - Moving Average Power (Watts)
    """

    def __init__(
        self,
        activities: List[pd.DataFrame],
        window_size: int,
        n_features: int = 4,
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
            fit_dataframe = self.preprocess_activity(fit_dataframe)

            if len(fit_dataframe) < window_size:
                continue

            rider_speed, rider_direction = velocity(
                fit_dataframe[self.latitude_column].values,
                fit_dataframe[self.longitude_column].values,
                fit_dataframe["time_seconds"].values,
            )

            fit_dataframe["direction"] = rider_direction
            fit_dataframe["speed_calc"] = rider_speed

            fit_dataframe = fit_dataframe.iloc[1:].copy()

            if fit_dataframe["speed_calc"].isnull().any():
                continue

            fit_dataframe["rel_wind_angle"] = np.abs(
                fit_dataframe[self.wind_direction_column].values
                - fit_dataframe["direction"].values
            )

            fit_dataframe = fit_dataframe[fit_dataframe["rel_wind_angle"].notnull()]
            fit_dataframe = fit_dataframe.copy()

            # TODO: Vectorize window creation
            # Convert DataFrame columns to NumPy arrays
            elevation = fit_dataframe[self.elevation_column].values
            rel_wind_angle = fit_dataframe["rel_wind_angle"].values
            wind_speed = fit_dataframe[self.wind_speed_column].values
            speed = fit_dataframe["speed_calc"].values
            power_window = fit_dataframe["power_window"].values

            activity_windows, activity_powers = self.create_windows(
                fit_dataframe,
                rel_features=[elevation],
                abs_features=[speed, rel_wind_angle, wind_speed],
                target=power_window,
            )

            windows.extend(activity_windows)
            window_powers.extend(activity_powers)

        self.prep_windows(windows, window_powers, window_scaler, power_scaler)


class WindowedAvgPowerDataset(LatLonPowerDataset):
    """
    Features:
    - Avg gradient over window (degrees)
    - Avg speed over window (m/s)
    - Wind speed (m/s)
    - Wind direction wrt rider (degrees)

    Labels:
    - Avg Power over window (Watts)
    """

    ALTIUDE_WINDOW = 5

    def __init__(
        self,
        activities: List[pd.DataFrame],
        window_size: int,
        n_features: int = 4,
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
            fit_dataframe = self.preprocess_activity(fit_dataframe)

            if len(fit_dataframe) < window_size:
                continue

            rider_speed, rider_direction = velocity(
                fit_dataframe[self.latitude_column].values,
                fit_dataframe[self.longitude_column].values,
                fit_dataframe["time_seconds"].values,
            )

            gradient = smooth_gradient(
                fit_dataframe[self.latitude_column].values,
                fit_dataframe[self.longitude_column].values,
                fit_dataframe[self.elevation_column].values,
                altitude_window=self.ALTIUDE_WINDOW,
            )

            fit_dataframe["gradient"] = gradient
            fit_dataframe["direction"] = rider_speed
            fit_dataframe["speed_calc"] = rider_direction

            fit_dataframe = fit_dataframe[
                fit_dataframe["gradient"].notnull()
                & fit_dataframe["speed_calc"].notnull()
                & fit_dataframe["direction"].notnull()
            ]

            if len(fit_dataframe) < window_size:
                continue

            fit_dataframe["rel_wind_angle"] = np.abs(
                fit_dataframe[self.wind_direction_column].values
                - fit_dataframe["direction"].values
            )

            fit_dataframe = fit_dataframe[fit_dataframe["rel_wind_angle"].notnull()]
            fit_dataframe = fit_dataframe.copy()

            window_gradient = (
                fit_dataframe["gradient"].rolling(window_size).mean().values
            )
            window_speed = (
                fit_dataframe["speed_calc"].rolling(window_size).mean().values
            )
            window_power = (
                fit_dataframe["power_window"].rolling(window_size).mean().values
            )
            rel_wind_angle = fit_dataframe["rel_wind_angle"].values
            wind_speed = fit_dataframe[self.wind_speed_column].values

            windows.append(
                np.column_stack(
                    (window_gradient, window_speed, rel_wind_angle, wind_speed)
                )[window_size - 1 :]
            )

            window_powers.extend(window_power[window_size - 1 :])

        self.windows = np.concatenate(windows, axis=0)
        self.prep_windows(self.windows, window_powers, window_scaler, power_scaler)

    def scale_data(
        self, windows: np.ndarray, powers: np.ndarray, window_scaler, power_scaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        if window_scaler is None:
            window_scaler = MinMaxScaler()
            window_scaler.fit(windows)

        if power_scaler is None:
            power_scaler = MinMaxScaler()
            power_scaler.fit(powers)

        self.window_scaler = window_scaler
        self.power_scaler = power_scaler

        if len(self.windows) == 0:
            return windows, powers

        windows_scaled = self.window_scaler.transform(windows)

        powers_scaled = self.power_scaler.transform(powers)

        return windows_scaled, powers_scaled
