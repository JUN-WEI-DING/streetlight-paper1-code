"""
PAR Zarr data reader module.

This module provides functions to load and query PAR data from the
consolidated zarr dataset created by scripts/par_to_zarr.py.

Example:
    >>> from streetlight.sources.par_zarr import load_par_zarr, get_par_at_time
    >>> # Load full dataset
    >>> ds = load_par_zarr()
    >>> print(ds.time.size)
    45341
    >>> # Load specific time range
    >>> ds_subset = load_par_zarr(time_range=("2024-06-01", "2024-06-30"))
    >>> # Get nearest timestamp
    >>> par = get_par_at_time("2024-06-15 12:00")
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from streetlight import paths as sl_paths
from streetlight.config import get_config


def get_par_zarr_path() -> Path:
    """
    Get the default PAR zarr path from configuration.

    Returns:
        Path to PAR zarr dataset
    """
    config = get_config()
    zarr_path = config.get("data.par_zarr_path", "data/zarr/himawari_par_2024.zarr")
    p = Path(zarr_path)
    if not p.is_absolute():
        p = sl_paths.project_root() / p
    return p


def load_par_zarr(
    zarr_path: Optional[Path] = None,
    time_range: Optional[Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Load PAR data from zarr with optional slicing.

    Parameters
    ----------
    zarr_path : Path, optional
        Path to zarr dataset. If None, uses config default.
    time_range : tuple, optional
        (start, end) time range as strings or Timestamps.
        Strings should be ISO format like "2024-06-01" or "2024-06-01 12:00".
    lon_range : tuple, optional
        (min, max) longitude range in degrees.
    lat_range : tuple, optional
        (min, max) latitude range in degrees.
    chunks : dict, optional
        Chunk sizes for dask. If None, uses zarr's internal chunking.

    Returns
    -------
    xr.Dataset
        Dataset with PAR variable and time/latitude/longitude coordinates.

    Raises
    ------
    FileNotFoundError
        If zarr path does not exist.

    Examples
    --------
    >>> ds = load_par_zarr()
    >>> ds_subset = load_par_zarr(
    ...     time_range=("2024-06-01", "2024-06-30"),
    ...     lon_range=(120, 122),
    ...     lat_range=(22, 25),
    ... )
    """
    path = zarr_path or get_par_zarr_path()

    if not path.exists():
        raise FileNotFoundError(f"PAR zarr not found at {path}")

    # Build slice kwargs
    sel_kwargs = {}

    if time_range is not None:
        start, end = time_range
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        sel_kwargs["time"] = slice(start, end)

    if lon_range is not None:
        lon_min, lon_max = lon_range
        sel_kwargs["longitude"] = slice(lon_min, lon_max)

    if lat_range is not None:
        lat_min, lat_max = lat_range
        # Note: latitude is stored in decreasing order (N→S)
        sel_kwargs["latitude"] = slice(lat_max, lat_min)

    # Open zarr with optional chunks
    open_kwargs = {}
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    ds = xr.open_zarr(path, **open_kwargs)

    # Apply slicing if any
    if sel_kwargs:
        ds = ds.sel(**sel_kwargs)

    return ds


def get_par_at_time(
    timestamp: Union[str, pd.Timestamp],
    zarr_path: Optional[Path] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Get PAR data for a specific timestamp.

    Parameters
    ----------
    timestamp : str or pd.Timestamp
        Target timestamp. String should be ISO format.
    zarr_path : Path, optional
        Path to zarr dataset. If None, uses config default.
    lon_range : tuple, optional
        (min, max) longitude range for spatial slicing.
    lat_range : tuple, optional
        (min, max) latitude range for spatial slicing.
    method : str
        Selection method: "nearest" (default), "ffill", or "bfill".

    Returns
    -------
    xr.DataArray
        PAR values at the nearest timestamp.

    Examples
    --------
    >>> par = get_par_at_time("2024-06-15 12:00")
    >>> par.plot()
    """
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)

    # Load with spatial slice first (more efficient)
    ds = load_par_zarr(
        zarr_path=zarr_path,
        lon_range=lon_range,
        lat_range=lat_range,
    )

    # Select time
    par = ds["PAR"].sel(time=timestamp, method=method)

    return par


def get_par_time_series(
    lon: float,
    lat: float,
    zarr_path: Optional[Path] = None,
    time_range: Optional[Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]]] = None,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Get PAR time series at a specific location.

    Parameters
    ----------
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    zarr_path : Path, optional
        Path to zarr dataset. If None, uses config default.
    time_range : tuple, optional
        (start, end) time range for temporal slicing.
    method : str
        Selection method for spatial coordinates.

    Returns
    -------
    xr.DataArray
        Time series of PAR values at the location.

    Examples
    --------
    >>> ts = get_par_time_series(121.0, 24.5, time_range=("2024-06-01", "2024-06-30"))
    >>> ts.plot()
    """
    ds = load_par_zarr(
        zarr_path=zarr_path,
        time_range=time_range,
    )

    # Select location
    par_ts = ds["PAR"].sel(
        longitude=lon,
        latitude=lat,
        method=method,
    )

    return par_ts


def get_daily_par_sum(
    zarr_path: Optional[Path] = None,
    time_range: Optional[Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
) -> xr.DataArray:
    """
    Compute daily sum of PAR (daily light integral proxy).

    Parameters
    ----------
    zarr_path : Path, optional
        Path to zarr dataset.
    time_range : tuple, optional
        Time range for computation.
    lon_range : tuple, optional
        Longitude range.
    lat_range : tuple, optional
        Latitude range.

    Returns
    -------
    xr.DataArray
        Daily PAR sum with dimensions (day, latitude, longitude).

    Notes
    -----
    PAR units are μmol/m²/s. Daily sum approximates daily light integral
    but requires multiplication by time interval for accurate values.

    Examples
    --------
    >>> daily = get_daily_par_sum(time_range=("2024-06-01", "2024-06-30"))
    >>> daily.mean(dim="day").plot()
    """
    ds = load_par_zarr(
        zarr_path=zarr_path,
        time_range=time_range,
        lon_range=lon_range,
        lat_range=lat_range,
    )

    # Resample to daily sum
    # Note: This is approximate since time intervals may vary
    daily_sum = ds["PAR"].resample(time="1D").sum()

    # Rename time to day for clarity
    daily_sum = daily_sum.rename(time="day")

    return daily_sum


def get_par_stats(
    zarr_path: Optional[Path] = None,
    time_range: Optional[Tuple[Union[str, pd.Timestamp], Union[str, pd.Timestamp]]] = None,
) -> dict:
    """
    Get basic statistics for PAR data.

    Parameters
    ----------
    zarr_path : Path, optional
        Path to zarr dataset.
    time_range : tuple, optional
        Time range for statistics.

    Returns
    -------
    dict
        Statistics including time range, spatial coverage, and data summary.

    Examples
    --------
    >>> stats = get_par_stats()
    >>> print(f"Time range: {stats['time_range']}")
    """
    ds = load_par_zarr(zarr_path=zarr_path, time_range=time_range)

    stats = {
        "time_range": f"{ds.time.min().values} to {ds.time.max().values}",
        "time_count": int(ds.time.size),
        "latitude_range": (float(ds.latitude.min()), float(ds.latitude.max())),
        "longitude_range": (float(ds.longitude.min()), float(ds.longitude.max())),
        "spatial_shape": (int(ds.latitude.size), int(ds.longitude.size)),
        "par_attrs": dict(ds.PAR.attrs),
    }

    return stats
