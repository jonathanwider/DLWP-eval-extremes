import numpy as np
import xarray as xr

from meteo_utils import time_to_doy_and_hour


def get_data_for_barrier_plot(gt: xr.DataArray, fc: xr.DataArray) -> xr.DataArray:
    """Extract data from forecast and ground truth data that can be used in predictability barrier plots.

    Args:
        gt (xr.DataArray): Ground truth data.
        fc (xr.DataArray): Forecast data.

    Returns:
        xr.DataArray: Extracted data.

    """
    res = xr.DataArray(
        coords={
            **gt.rename(time="valid_time").coords,
            "prediction_timedelta": fc.prediction_timedelta,
        },
        dims=(*gt.rename(time="valid_time").dims, "prediction_timedelta"),
    )
    for i in range(len(res.valid_time)):
        vt = res.valid_time[i]
        r = xr.where(fc.valid_time == vt, 1, 0)
        itemindex = dict(zip(r.dims, np.where(r == 1), strict=True))

        extracted = fc.to_numpy()[tuple(itemindex.values())]
        perm = np.roll(np.arange(len(extracted.shape)), -1)

        res.loc[
            {
                "valid_time": vt,
                "prediction_timedelta": fc.prediction_timedelta[
                    itemindex["prediction_timedelta"]
                ],
            }
        ] = extracted.transpose(perm)
    return res - gt.rename(time="valid_time")


def time_of_maximum_error(arr):
    # only works for 2d arrays when taking argmax over second axis!
    # makes everything nan to start with
    res = np.zeros(arr.shape[0]) + np.nan
    # finds the indices where the entire column would be nan, so the nanargmin would raise an error
    d0 = np.nanmin(arr, axis=1)
    # on the indices where we do not have a nan-column, get the right index with nanargmin, and than put the right value in those points
    res[~np.isnan(d0)] = np.nanargmax(abs(arr[~np.isnan(d0), :]), axis=1)
    return res


def get_anomaly(da, ts, climatology, lead_time=None, start_date=None):
    assert not (lead_time is not None and start_date is not None)
    if lead_time is not None:
        ts_mins_lead_time = slice(
            ts.start - np.timedelta64(lead_time, "D"),
            ts.stop - np.timedelta64(lead_time, "D"),
        )
        res = da.sel(
            {
                "time": ts_mins_lead_time,
                "prediction_timedelta": np.timedelta64(lead_time, "D"),
            }
        )
    elif start_date is not None:
        res = da.sel(
            {
                "time": start_date,
                "prediction_timedelta": slice(
                    ts.start - start_date, ts.stop - start_date
                ),
            }
        )
        res = (
            res.assign_coords(
                {"prediction_timedelta": start_date + res.prediction_timedelta}
            )
            .drop("time")
            .rename({"prediction_timedelta": "time"})
        )
    else:
        res = da.sel({"time": ts})

    return time_to_doy_and_hour(res) - climatology


def get_absolute(da, ts, lead_time=None, start_date=None):
    assert not (lead_time is not None and start_date is not None)
    if lead_time is not None:
        ts_mins_lead_time = slice(
            ts.start - np.timedelta64(lead_time, "D"),
            ts.stop - np.timedelta64(lead_time, "D"),
        )
        res = da.sel(
            {
                "time": ts_mins_lead_time,
                "prediction_timedelta": np.timedelta64(lead_time, "D"),
            }
        )
    elif start_date is not None:
        res = da.sel(
            {
                "time": start_date,
                "prediction_timedelta": slice(
                    ts.start - start_date, ts.stop - start_date
                ),
            }
        )
        res = (
            res.assign_coords(
                {"prediction_timedelta": start_date + res.prediction_timedelta}
            )
            .drop("time")
            .rename({"prediction_timedelta": "time"})
        )
    else:
        res = da.sel({"time": ts})

    return time_to_doy_and_hour(res)


# area-weighting. Adapted from https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py
def _assert_increasing(x: np.ndarray):
    if not (np.diff(x) > 0).all():
        raise ValueError(f"array is not increasing: {x}")


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
    pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
    return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
    """Calculate the area overlap as a function of latitude."""
    bounds = _latitude_cell_bounds(points)
    _assert_increasing(bounds)
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    return np.sin(upper) - np.sin(lower)


def get_lat_weights(ds: xr.Dataset) -> xr.DataArray:
    """Computes latitude/area weights from latitude coordinate of dataset."""
    weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
    weights /= np.mean(weights)
    weights = ds.latitude.copy(data=weights)
    return weights


# written myself.
def rescaled_weights(lat_weights: xr.DataArray, longitude, R_E=6371):
    """Assumes that latitude and longitude cover entire domain of earth"""
    assert np.isclose(lat_weights.mean(), 1)
    weights = lat_weights.expand_dims({"longitude": longitude}, axis=1)
    return (
        (
            weights
            * (4 * np.pi * R_E**2)
            / len(weights.latitude)
            / len(weights.longitude)
        )
        .assign_attrs(units="km^2")
        .rename("Area")
    )


def spatially_weighted_average(
    array: xr.DataArray, weights: xr.DataArray, mask: xr.DataArray
) -> float:
    """Assume array to have dimensions (latitude, longitude, dayofyear, hour). Not necessarily in this order.

    Args:
    ----
        array (xr.DataArray): Array to be averaged
        weights (xr.DataArray): Precomputed weights (proportional to area covered)
        mask (xr.DataArray): Mask to apply before averaging

    Returns:
    -------
        float: _description_

    """
    return (
        (
            (weights * array).where(mask).sum(dim=("latitude", "longitude"))
            / (weights).where(mask).sum(dim=("latitude", "longitude"))
        )
        .mean(dim=("dayofyear", "hour"))
        .to_numpy()
    )
