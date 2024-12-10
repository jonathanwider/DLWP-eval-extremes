import numpy as np
import xarray as xr


def saturation_vapor_pressure_from_temperature(
    T: xr.DataArray, b: float = 17.625, l: float = 243.04
) -> xr.DataArray:
    """Given an array of temperatures (in °C), compute the saturation water pressure in hPa, using the August-Roche-Magnus formula

    Args:
    ----
        T (xr.DataArray): Temperatures in °C
        b (float, optional): Coefficient 1 of August-Roche-Magnus formula. Defaults to 17.625.
        l (float, optional): Coefficient 2 of August-Roche-Magnus formula. Defaults to 243.04.

    Returns:
    -------
        xr.DataArray: Saturation water pressure in hPa.

    """

    def f(x):
        return 6.1094 * np.exp((b * x) / (l + x))

    return xr.apply_ufunc(f, T, vectorize=True, dask="allowed")


def celsius_to_fahrenheit(T: xr.DataArray) -> xr.DataArray:
    return T * 1.8 + 32


def fahrenheit_to_celsius(T: xr.DataArray) -> xr.DataArray:
    return (T - 32) / 1.8


def kelvin_to_fahrenheit(T: xr.DataArray) -> xr.DataArray:
    return T * 1.8 - 459.67


def fahrenheit_to_kelvin(T: xr.DataArray) -> xr.DataArray:
    return (T + 459.67) / 1.8


def kelvin_to_celsius(T: xr.DataArray) -> xr.DataArray:
    return T - 273.15


def celsius_to_kelvin(T: xr.DataArray) -> xr.DataArray:
    return T + 273.15


def ms_to_kmh(speed: xr.DataArray) -> xr.DataArray:
    return speed * 3.6


def kmh_to_ms(speed: xr.DataArray) -> xr.DataArray:
    return speed / 3.6


def mph_to_kmh(speed: xr.DataArray) -> xr.DataArray:
    return speed * 1.609


def mph_to_ms(speed: xr.DataArray) -> xr.DataArray:
    return kmh_to_ms(mph_to_kmh(speed))


def relative_humidity_from_temperature_and_dew_point_temperature(
    T: xr.DataArray, T_d: xr.DataArray
) -> xr.DataArray:
    """Given an array of temperatures (in °C) and dew point temperatures (in °C), compute the relative humidity

    Args:
    ----
        T (xr.DataArray): Temperatures in °C
        T_d (xr.DataArray): Dewpoint temperatures in °C

    Returns:
    -------
        xr.DataArray: Relative humidity.

    """
    return (
        saturation_vapor_pressure_from_temperature(T_d)
        / saturation_vapor_pressure_from_temperature(T)
        * 100
    )


def HI_polynomial(T: xr.DataArray, R: xr.DataArray) -> xr.DataArray:
    return (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * T * T
        - 0.05481717 * R * R
        + 0.00122874 * T * T * R
        + 0.00085282 * T * R * R
        - 0.00000199 * T * T * R * R
    )


def HI_a1(T: xr.DataArray, R: xr.DataArray) -> xr.DataArray:
    return ((13 - R) / 4) * np.sqrt((17 - np.abs(T - 95)) / 17)


def HI_a2(T: xr.DataArray, R: xr.DataArray) -> xr.DataArray:
    return ((R - 85) / 10) * ((87 - T) / 5)


def heat_index_fahrenheit(T: xr.DataArray, R: xr.DataArray) -> xr.DataArray:
    """Compute the heat index (in Fahrenheit) from temperature (in Fahrenheit) and relative humidity.

    Args:
    ----
        T (xr.DataArray): Temperatures in Fahrenheit
        R (xr.DataArray): Relative humidity (surface layer) in dimensionless units.

    Returns:
    -------
        xr.DataArray: The heat index in F, computed according NOAA standards (https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml).

    """
    hi = 0.5 * (T + 61 + ((T - 68) * 1.2) + (R * 0.094))

    # if this condition is fulfilled use a more elaborate calculation
    is_above_80 = (hi + T) / 2 > 80

    hi_above_80 = HI_polynomial(T, R)

    # in some regions we need to adapt the polynomial formula.
    use_adjustment_a1 = np.logical_and.reduce((R < 13, T >= 80, T <= 112))
    use_adjustment_a2 = np.logical_and.reduce((R > 85, T >= 80, T <= 87))

    # apply corrections
    hi_above_80 = xr.where(use_adjustment_a1, -HI_a1(T, R) + hi_above_80, hi_above_80)
    hi_above_80 = xr.where(use_adjustment_a2, HI_a2(T, R) + hi_above_80, hi_above_80)

    return xr.where(is_above_80, hi_above_80, hi)


def heat_index_celsius(T: xr.DataArray, R: xr.DataArray) -> xr.DataArray:
    """Compute the heat index (in °C) from temperature (in °C) and relative humidity.

    Args:
    ----
        T (xr.DataArray): Temperatures in °C
        R (xr.DataArray): Relative humidity (surface layer) in dimensionless units.

    Returns:
    -------
        xr.DataArray: The heat index in °C, computed according NOAA standards (https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml).

    """
    return fahrenheit_to_celsius(heat_index_fahrenheit(T=celsius_to_fahrenheit(T), R=R))


def mixing_ratio_at_saturation_from_pressure_and_saturation_vapor_pressure(
    p: xr.DataArray, e_s: xr.DataArray
) -> xr.DataArray:
    return 0.622 * (e_s) / (p - e_s)


def relative_humidity_from_specific_humidity_and_mixing_ratio_at_saturation(
    q: xr.DataArray, w_s: xr.DataArray
) -> xr.DataArray:
    return 100 * q / (1 - q) / w_s


def relative_humidity_from_specific_humidity_and_pressure_and_temperature(
    q: xr.DataArray, p: xr.DataArray, T: xr.DataArray
) -> xr.DataArray:
    return relative_humidity_from_specific_humidity_and_mixing_ratio_at_saturation(
        q=q,
        w_s=mixing_ratio_at_saturation_from_pressure_and_saturation_vapor_pressure(
            p=p, e_s=saturation_vapor_pressure_from_temperature(T=T)
        ),
    )


def get_indices(array, u_array):
    res = np.zeros(len(array), dtype=int)
    for i in range(len(u_array)):
        res[array == u_array[i]] = i
    return res


def time_to_doy_and_hour(da: xr.DataArray) -> xr.DataArray:
    """Generate an xarray dataarray with dimensions (dayofyear, hour) from the dataarray with dimension time. Should only be applied to small subsets of large datasets! The tuples (dayofyear, hour) have to be unique! Assumes time to be in first dimension.

    Args:
    ----
        da (xr.DataArray): _description_

    Returns:
    -------
        xr.DataArray: _description_

    """
    dayofyear = da["time"].dt.dayofyear
    hour = da["time"].dt.hour

    u_dayofyear = np.unique(dayofyear)
    u_hour = np.unique(hour)

    i_doy = get_indices(dayofyear, u_dayofyear)
    i_hour = get_indices(hour, u_hour)

    old_dims = tuple(x for x in da.dims if x != "time")
    old_coords = dict(da.coords)
    del old_coords["time"]
    old_coords_lengths = [len(old_coords[dim]) for dim in old_dims]

    new_da = xr.DataArray(
        name=da.name,
        dims=("dayofyear", "hour", *old_dims),
        coords={"dayofyear": u_dayofyear, "hour": u_hour, **old_coords},
        data=np.full((len(u_dayofyear), len(u_hour), *old_coords_lengths), np.nan),
    )
    values = da.values
    for i in range(len(da.values)):
        new_da.data[i_doy[i], i_hour[i], ...] = values[i]
    return new_da


def get_windspeed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    return np.sqrt(u**2 + v**2)


def get_wind_chill(T_2m: xr.DataArray, speed_10m: xr.DataArray) -> xr.DataArray:
    """Given the 2m temperature (in °C), and the wind speed at a height of 10m (in km/h), calculate the wind chill tempearture (in °C).

    Using the formula of "Environment Canada" given in https://en.wikipedia.org/wiki/Wind_chill.
    Wind chill is defined only for temperatures at or below 10 °C (50 °F) and wind speeds above 4.8 km/h (3.0 mph).

    Args:
    ----
        T_2m (xr.DataArray): Array of 2m temperatures, in °C
        speed_10m (xr.DataArray): Array of 10m wind speed in km/h

    Returns:
    -------
        xr.DataArray: Wind chill in °C

    """
    return (
        13.12
        + 0.6215 * T_2m
        - 11.37 * np.power(speed_10m, 0.16)
        + 0.3965 * T_2m * np.power(speed_10m, 0.16)
    )


def get_wind_chill_mask(T_2m: xr.DataArray, speed_10m: xr.DataArray) -> xr.DataArray:
    """Given the 2m temperature (in °C), and the wind speed at a height of 10m (in km/h), calculate the wind chill tempearture (in °C). Using the formula of "Environment Canada" given in https://en.wikipedia.org/wiki/Wind_chill.
    Wind chill is defined only for temperatures at or below 10 °C (50 °F) and wind speeds above 4.8 km/h (3.0 mph).

    Args:
    ----
        T_2m (xr.DataArray): Array of 2m temperatures, in °C
        speed_10m (xr.DataArray): Array of 10m wind speed in km/h

    Returns:
    -------
        xr.DataArray: Wind chill in °C

    """
    return xr.where(np.logical_and(T_2m < 10, speed_10m > 4.8), 1, 0)
