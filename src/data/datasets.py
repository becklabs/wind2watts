import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class STPowerDataset:
    def __init__(self,
                 fit_dataframe: pd.DataFrame,
                 window_size: int,
                 time_column = 'timestamp',
                 latitude_column = 'position_lat',
                 longitude_column = 'position_long',
                 elevation_column = 'enhanced_altitude',
                 wind_speed_column = 'wind_speed',
                 wind_direction_column = 'wind_direction',
                 power_column = 'power',
                 input_scaler = None,
                 output_scaler = None,
                 device = 'cpu'):

        self.device = device
        self.time_column = time_column
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.elevation_column = elevation_column
        self.wind_speed_column = wind_speed_column
        self.wind_direction_column = wind_direction_column
        self.power_column = power_column

        fit_dataframe['power_window'] = fit_dataframe[self.power_column].rolling(window_size).mean()

        # Convert the time column to seconds
        fit_dataframe['time_seconds'] = pd.to_datetime(fit_dataframe[self.time_column]).astype(int)

        # TODO: Vectorize window creation 
        windows = []
        window_powers = []
        for i in range(window_size - 1, len(fit_dataframe)):
            window = fit_dataframe.iloc[i - window_size + 1 : i + 1]
            window = window.copy()
            # Ensure that the deltas start with 0
            window['lat_delta'] = window[self.latitude_column] - window[self.latitude_column].iloc[0]
            window['lon_delta'] = window[self.longitude_column] - window[self.longitude_column].iloc[0]
            window['elevation_delta'] = window[self.elevation_column] - window[self.elevation_column].iloc[0]
            window['time_delta'] = window['time_seconds'] - window['time_seconds'].iloc[0]

            windows.append(window[['lat_delta',
                                   'lon_delta',
                                   'elevation_delta',
                                   'time_delta',
                                   'wind_speed',
                                   'wind_direction']].values)

            window_powers.append(fit_dataframe.iloc[i]['power_window'])
        
        self.windows = np.array(windows)
        self.window_powers = np.array(window_powers).reshape(-1, 1)

        if input_scaler is None:
            input_scaler = MinMaxScaler()
            input_scaler.fit(self.windows.reshape(-1, 6))

        if output_scaler is None:
            output_scaler = MinMaxScaler()
            output_scaler.fit(self.window_powers)
        
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window_scaled = self.input_scaler.transform(self.windows[idx])
        power_scaled = self.output_scaler.transform(self.window_powers[idx].reshape(-1, 1)).reshape(-1)

        return torch.FloatTensor(window_scaled).to(self.device), torch.FloatTensor(power_scaled).to(self.device)