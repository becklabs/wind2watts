from typing import Tuple, Any
import datetime
import math
import numpy as np
import pandas as pd

EARTH_RADIUS = 6371e3  # meters


@np.vectorize
def airspeed(
    rider_speed: float, rider_direction: float, wind_speed: float, wind_direction: float
) -> float:
    """
    Calculates the airspeed of the rider in (m/s)
    Args:
        rider_speed: the rider's speed (m/s)
        rider_direction: the rider's direction (degrees clockwise from true north)
        wind_direction: the wind direction (degrees clockwise from true north)
        wind_speed: the wind speed (m/s)
    Returns:
        airspeed: the airspeed of the rider (m/s)
    """

    # Shift the wind direction by 90 degrees to make it clockwise from the x-axis
    rider_direction = np.deg2rad(90 - rider_direction)
    wind_direction = np.deg2rad(90 - wind_direction)

    wind_vector = wind_speed * np.array(
        (np.cos(wind_direction), np.sin(wind_direction))
    )
    rider_vector = rider_speed * np.array(
        (np.cos(rider_direction), np.sin(rider_direction))
    )
    resultant = rider_vector - wind_vector

    magnitude = np.linalg.norm(resultant)

    return float(magnitude)


@np.vectorize
def haversine(
    lat1: float, lon1: float, lat2: float, lon2: float, r=EARTH_RADIUS
) -> float:
    """
    Implements the Haversine formula to calculate the distance between two points
    on a sphere with the given radius in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c  # output distance in meters

def velocity(
    lat: np.ndarray,
    long: np.ndarray,
    time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        lat: latitude (dd)
        long: longitude (dd)
        time: time (seconds)
    Returns:
        speed:  (m/s)
        direction: (degrees clockwise from north)
    """
    lat1 = lat[1:]
    lon1 = long[1:]
    time1 = time[1:]
    lat2 = lat[:-1]
    lon2 = long[:-1]
    time2 = time[:-1]

    distance = haversine(lat1, lon1, lat2, lon2)

    time_diff = time1 - time2

    # handle ZeroDivisionError
    eps = 1e-6
    mask = time_diff < eps
    time_diff[mask] = np.nan
    speed = distance / time_diff

    # prepend np.nan to maintain same size as input
    speed = np.insert(speed, 0, np.nan)

    y = np.sin(lon1 - lon2) * np.cos(lat1)
    x = np.cos(lat2) * np.sin(lat1) - np.sin(lat2) * np.cos(lat1) * np.cos(lon1 - lon2)  # type: ignore

    direction = (np.degrees(np.arctan2(y, x)) + 360) % 360  # normalize to 0-360
    direction = (direction + 180) % 360  # reverse direction

    # prepend np.nan to maintain same size as input
    direction = np.insert(direction, 0, np.nan)

    return speed, direction



def smooth_gradient(
    lat: np.ndarray,
    long: np.ndarray,
    altitude: np.ndarray,
    altitude_window: int = 5,
):
    """
    Args:
        lat: 1D Numpy array representing latitudes of the points.
        long: 1D Numpy array representing longitudes of the points.
        altitude: 1D Numpy array representing altitudes of the points.
        gradient_window: The size of the moving window for smoothing the calculated gradient. Default is 3.
        altitude_window: The size of the moving window for smoothing the altitude values before gradient calculation. Default is 5.

    Returns:
        gradient: A 1D numpy array representing the smoothed gradient (meters y / meters x).
    """
    lat1 = lat[:-1]
    lon1 = long[:-1]

    lat2 = lat[1:]
    lon2 = long[1:]

    dx = haversine(lat1, lon1, lat2, lon2)
    dx[dx == 0] = np.nan

    smoothed_y = pd.Series(altitude).rolling(altitude_window).mean().values  # type: ignore
    dy = smoothed_y[1:] - smoothed_y[:-1]  # type: ignore

    return pd.Series(dy / dx).interpolate().values  # type: ignore


def validate_lat_long(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    valid_latitudes = np.logical_and(-90 <= latitudes, latitudes <= 90)
    valid_longitudes = np.logical_and(-180 <= longitudes, longitudes <= 180)
    return np.logical_and(valid_latitudes, valid_longitudes)

# def velocity(
#     lat: np.ndarray,
#     long: np.ndarray,
#     time: np.ndarray,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Args:
#         lat: latitude (dd)
#         long: longitude (dd)
#         time: time (seconds)
#     Returns:
#         speed:  (m/s)
#         direction: (degrees clockwise from north)
#     """
#     lat1 = lat[:-1]
#     lon1 = long[:-1]
#     time1 = time[:-1]
#     lat2 = lat[1:]
#     lon2 = long[1:]
#     time2 = time[1:]

#     distance = haversine(lat1, lon1, lat2, lon2)

#     time_diff = time2 - time1

#     # handle ZeroDivisionError
#     eps = 1e-6
#     mask = time_diff < eps
#     time_diff[mask] = np.nan
#     speed = distance / time_diff

#     # but we need to handle the case where time_diff is close to zero

#     y = np.sin(lon2 - lon1) * np.cos(lat2)

#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)  # type: ignore

#     direction = (np.degrees(np.arctan2(y, x)) + 360) % 360  # normalize to 0-360

#     return speed, direction