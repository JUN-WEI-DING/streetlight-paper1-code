"""
PAR (Photosynthetically Active Radiation) aggregation module.

This module processes Himawari-8/9 satellite PAR data and aggregates it
to county-level time series using geospatial operations.

Key operations:
1. PAR data loading from compressed NetCDF files
2. Geospatial rasterization of county boundaries
3. Upsampling and vectorized aggregation
4. Parallel processing support for large datasets
"""

from typing import Tuple, Optional, List
from pathlib import Path
import re
import gzip
import shutil
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box as sbox


def make_valid_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix invalid geometries in GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with potentially invalid geometries

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with valid geometries

    Notes
    -----
    Uses shapely.make_valid() with vectorized array input (Shapely 2.0+),
    otherwise falls back to buffer(0) method.
    """
    try:
        from shapely import make_valid
        gdf_out = gdf.copy()
        # Vectorized path: convert GeometryArray to numpy array for shapely 2.0+
        # This avoids the slow .apply() loop
        gdf_out["geometry"] = make_valid(np.asarray(gdf.geometry.values))
        return gdf_out
    except Exception:
        gdf_out = gdf.copy()
        gdf_out["geometry"] = gdf.buffer(0)
        return gdf_out


def open_par_from_gz(
    gz_path: Path,
    temp_dir: Optional[Path] = None
) -> Tuple[xr.DataArray, xr.Dataset]:
    """
    Decompress and load PAR data from gzipped NetCDF file.

    Parameters
    ----------
    gz_path : Path
        Path to .nc.gz file
    temp_dir : Path, optional
        Directory for temporary decompressed file. If None, uses system temp.
        Can be set to a ramdisk path (e.g., /dev/shm on Linux) for faster I/O.

    Returns
    -------
    Tuple[xr.DataArray, xr.Dataset]
        - DataArray containing PAR values
        - Dataset (for cleanup)

    Notes
    -----
    Uses netCDF4 directly to avoid xarray segfault with HDF4 files.
    Ensures latitude is decreasing (N→S) and longitude is increasing (W→E).

    For better performance on Linux, consider setting temp_dir to /dev/shm
    to use a RAM disk for the temporary decompressed file.
    """
    import netCDF4
    import numpy as np
    import os

    # Configure temporary file location
    if temp_dir is not None:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Use a specific temp file in the specified directory
        tmp_path = temp_dir / f"{gz_path.stem}_{os.getpid()}.nc"
        # Write directly to the specified location
        with gzip.open(gz_path, "rb") as fin, open(tmp_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        nc_path = str(tmp_path)
    else:
        # Use system default temp location
        with gzip.open(gz_path, "rb") as fin, NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            shutil.copyfileobj(fin, tmp)
            nc_path = tmp.name

    try:
        # Use netCDF4 directly to avoid xarray segfault
        nc = netCDF4.Dataset(nc_path, "r")

        # Extract PAR data
        par_data = nc.variables['PAR'][:]
        latitude = nc.variables['latitude'][:]
        longitude = nc.variables['longitude'][:]

        # Get attributes
        par_attrs = {}
        if hasattr(nc.variables['PAR'], 'ncattrs'):
            for attr in nc.variables['PAR'].ncattrs():
                try:
                    par_attrs[attr] = nc.variables['PAR'].getncattr(attr)
                except:
                    pass

        nc.close()

        # Create xarray DataArray manually
        da = xr.DataArray(
            par_data,
            dims=('latitude', 'longitude'),
            coords={'latitude': latitude, 'longitude': longitude},
            name='PAR',
            attrs=par_attrs
        )

        # Create a minimal dataset
        ds = xr.Dataset({'PAR': da})

        # Ensure coordinate order
        if not (da.latitude[0] > da.latitude[-1]):
            da = da.sortby("latitude", ascending=False)
        if not (da.longitude[0] < da.longitude[-1]):
            da = da.sortby("longitude", ascending=True)

        return da, ds
    finally:
        try:
            os.remove(nc_path)
        except Exception:
            pass


def clip_to_aoi(
    gdf: gpd.GeoDataFrame,
    minx: float, miny: float, maxx: float, maxy: float
) -> gpd.GeoDataFrame:
    """
    Clip GeoDataFrame to area of interest (AOI).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame (should be in WGS84, crs=4326)
    minx, miny, maxx, maxy : float
        Bounding box coordinates in degrees

    Returns
    -------
    gpd.GeoDataFrame
        Clipped GeoDataFrame
    """
    aoi_poly = sbox(minx, miny, maxx, maxy)
    return gpd.clip(gdf, gpd.GeoDataFrame(geometry=[aoi_poly], crs=4326))


def upsample_par_data(da: xr.DataArray, factor: int = 5) -> xr.DataArray:
    """
    Upsample PAR data by repeating pixels (5km → 1km).

    Parameters
    ----------
    da : xr.DataArray
        Input PAR data
    factor : int
        Upsampling factor (default: 5)

    Returns
    -------
    xr.DataArray
        Upsampled DataArray with new coordinates

    Notes
    -----
    For Taiwan at ~23°N:
    - Original resolution: ~0.05° (≈5km)
    - After factor=5: ~0.01° (≈1km)

    Uses np.repeat for efficient upsampling with dtype preservation.
    Memory usage grows by factor^2, so be cautious with large factors.
    """
    if factor <= 0:
        raise ValueError(f"Upsampling factor must be positive, got {factor}")

    lat = da.latitude.values
    lon = da.longitude.values
    vals = da.values

    # Memory check: warn if upsampled array would be very large
    output_size = vals.nbytes * (factor * factor)
    if output_size > 500_000_000:  # > 500 MB
        import warnings
        warnings.warn(
            f"Upsampled array will be ~{output_size / 1024**2:.1f} MB. "
            f"Consider reducing factor or processing in chunks.",
            ResourceWarning
        )
    # Reject if output would be larger than practical limit (2 GB)
    if output_size > 2_000_000_000:  # > 2 GB
        raise MemoryError(
            f"Upsampled array would be ~{output_size / 1024**2:.1f} MB, "
            f"exceeding safe limit. Reduce factor or process in chunks."
        )

    # Preserve input dtype for memory efficiency
    input_dtype = vals.dtype

    # Handle single-element arrays (assume 0.05 degree resolution for Taiwan)
    if len(lat) == 1:
        dlat = 0.05
    else:
        dlat = float(lat[0] - lat[1])  # Latitude decreasing
    if len(lon) == 1:
        dlon = 0.05
    else:
        dlon = float(lon[1] - lon[0])  # Longitude increasing

    sub_lat = dlat / factor
    sub_lon = dlon / factor

    # Create fine coordinate arrays (use float32 to save memory)
    lat_fine = (lat[0] + 2 * sub_lat) - sub_lat * np.arange(
        len(lat) * factor, dtype=np.float32
    )
    lon_fine = (lon[0] - 2 * sub_lon) + sub_lon * np.arange(
        len(lon) * factor, dtype=np.float32
    )

    # Upsample values using np.repeat (efficient for this use case)
    # Preserves dtype to minimize memory growth
    vals_up = np.repeat(np.repeat(vals, factor, axis=0), factor, axis=1)

    return xr.DataArray(
        vals_up,
        dims=('latitude', 'longitude'),
        coords={'latitude': lat_fine, 'longitude': lon_fine},
        name=getattr(da, 'name', 'PAR'),
        attrs=da.attrs
    )


def rasterize_counties(
    gdf: gpd.GeoDataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    all_touched: bool = False
) -> Tuple[np.ndarray, 'from_origin']:
    """
    Rasterize county boundaries to grid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        County boundaries (must have __cid__ column)
    lat : np.ndarray
        Latitude values (decreasing)
    lon : np.ndarray
        Longitude values (increasing)
    all_touched : bool
        Pixel inclusion rule (default: False = center point only)

    Returns
    -------
    Tuple[np.ndarray, object]
        - Rasterized county ID labels (int32)
        - Transform object for reference

    Notes
    -----
    Creates integer county IDs (1..N) for vectorized aggregation.
    Background (no county) is labeled as 0.
    """
    # Calculate resolution and transform
    xres = float(lon[1] - lon[0])
    yres = float(lat[0] - lat[1])
    ny, nx = len(lat), len(lon)

    transform = from_origin(
        float(lon.min() - xres / 2.0),
        float(lat.max() + yres / 2.0),
        float(xres), float(yres)
    )

    # Create shapes list (geometry, label)
    shapes = list(zip(
        gdf.geometry.values,
        gdf['__cid__'].astype(int).values
    ))

    # Rasterize
    labels = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype='int32',
        all_touched=all_touched
    )

    return labels, transform


def compute_county_par(
    da: xr.DataArray,
    labels: np.ndarray,
    county_ids: np.ndarray
) -> pd.DataFrame:
    """
    Compute mean PAR value for each county using vectorized bincount.

    Parameters
    ----------
    da : xr.DataArray
        PAR data values
    labels : np.ndarray
        Rasterized county labels (from rasterize_counties)
    county_ids : np.ndarray
        County IDs to compute statistics for

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - __cid__: County ID
        - mean_PAR_umol_m2_s: Mean PAR value (μmol/m²/s)
        - n_pixels: Number of pixels used for average

    Notes
    -----
    Uses np.bincount for efficient vectorized aggregation.
    Only includes finite (non-NaN) values in computation.
    """
    # Handle both xarray.DataArray and numpy.ndarray
    vals = da.values if hasattr(da, 'values') else da
    finite = np.isfinite(vals)

    if not np.any(finite):
        # No valid data
        return pd.DataFrame({
            '__cid__': county_ids,
            'mean_PAR_umol_m2_s': np.nan,
            'n_pixels': 0
        })

    # Flatten arrays and filter to finite values
    lab = labels[finite].ravel()
    dat = vals[finite].ravel()

    max_id = int(county_ids.max())
    sum_per_id = np.bincount(lab, weights=dat, minlength=max_id + 1)
    cnt_per_id = np.bincount(lab, minlength=max_id + 1)

    # Compute mean (avoid division by zero)
    mean_per_id = np.divide(
        sum_per_id, cnt_per_id,
        out=np.full_like(sum_per_id, np.nan, dtype=np.float64),
        where=cnt_per_id > 0
    )

    return pd.DataFrame({
        '__cid__': county_ids,
        'mean_PAR_umol_m2_s': mean_per_id[county_ids],
        'n_pixels': cnt_per_id[county_ids].astype(int)
    })


def build_area_weight_table(
    gdf: gpd.GeoDataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    target_crs: str = "EPSG:3826",
) -> pd.DataFrame:
    """
    Build cell-county overlap weights for area-weighted PAR aggregation.

    The input grid is interpreted as cell centers. Overlap areas are computed
    after projecting both grid cells and county polygons to target_crs.
    """
    if "__cid__" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain __cid__ before building weights")
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("At least two latitude and longitude coordinates are required")

    xres = abs(float(lon[1] - lon[0]))
    yres = abs(float(lat[0] - lat[1]))

    cells = []
    for row, y in enumerate(lat):
        for col, x in enumerate(lon):
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "cell_index": row * len(lon) + col,
                    "geometry": sbox(
                        float(x) - xres / 2,
                        float(y) - yres / 2,
                        float(x) + xres / 2,
                        float(y) + yres / 2,
                    ),
                }
            )

    cells_gdf = gpd.GeoDataFrame(cells, geometry="geometry", crs=4326)
    counties = gdf[["__cid__", "geometry"]].copy()
    if counties.crs is None:
        counties = counties.set_crs(4326)
    elif counties.crs.to_epsg() != 4326:
        counties = counties.to_crs(4326)

    cells_eq = cells_gdf.to_crs(target_crs)
    counties_eq = counties.to_crs(target_crs)

    overlap = gpd.overlay(
        cells_eq,
        counties_eq,
        how="intersection",
        keep_geom_type=False,
    )
    if overlap.empty:
        return pd.DataFrame(
            columns=["__cid__", "row", "col", "cell_index", "overlap_area", "cell_area"]
        )

    overlap["overlap_area"] = overlap.geometry.area
    cell_area = cells_eq.set_index("cell_index").geometry.area.rename("cell_area")
    overlap = overlap.join(cell_area, on="cell_index")
    overlap["coverage_fraction"] = overlap["overlap_area"] / overlap["cell_area"]
    return pd.DataFrame(
        overlap[
            [
                "__cid__",
                "row",
                "col",
                "cell_index",
                "overlap_area",
                "cell_area",
                "coverage_fraction",
            ]
        ]
    )


def compute_area_weighted_county_par(
    par: np.ndarray,
    weights: pd.DataFrame,
    county_ids: np.ndarray,
) -> pd.DataFrame:
    """Compute county PAR using precomputed cell overlap-area weights."""
    if weights.empty:
        return pd.DataFrame(
            {
                "__cid__": county_ids,
                "mean_PAR_umol_m2_s": np.nan,
                "n_pixels": 0,
                "weighted_area": 0.0,
            }
        )

    values = np.asarray(par).ravel()
    cell_index = weights["cell_index"].to_numpy(dtype=int)
    county_index = weights["__cid__"].to_numpy(dtype=int)
    overlap_area = weights["overlap_area"].to_numpy(dtype=float)
    par_values = values[cell_index]
    valid = np.isfinite(par_values) & np.isfinite(overlap_area) & (overlap_area > 0)

    if not np.any(valid):
        return pd.DataFrame(
            {
                "__cid__": county_ids,
                "mean_PAR_umol_m2_s": np.nan,
                "n_pixels": 0,
                "weighted_area": 0.0,
            }
        )

    max_id = int(county_ids.max())
    weighted_sum = np.bincount(
        county_index[valid],
        weights=par_values[valid] * overlap_area[valid],
        minlength=max_id + 1,
    )
    weighted_area = np.bincount(
        county_index[valid],
        weights=overlap_area[valid],
        minlength=max_id + 1,
    )
    overlap_count = np.bincount(
        county_index[valid],
        minlength=max_id + 1,
    )
    means = np.divide(
        weighted_sum,
        weighted_area,
        out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
        where=weighted_area > 0,
    )

    return pd.DataFrame(
        {
            "__cid__": county_ids,
            "mean_PAR_umol_m2_s": means[county_ids],
            "n_pixels": overlap_count[county_ids].astype(int),
            "weighted_area": weighted_area[county_ids],
        }
    )


def prepare_county_geoms(
    gdf: gpd.GeoDataFrame,
    id_cols: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Prepare county geometries for PAR aggregation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input county boundaries
    id_cols : list, optional
        Columns to use as identifiers (default: ['COUNTYID', 'COUNTYNAME'])

    Returns
    -------
    gpd.GeoDataFrame
        Prepared GeoDataFrame with:
        - Fixed geometries
        - WGS84 CRS (4326)
        - Integer __cid__ column (1..N)
        - Selected id_cols

    Notes
    -----
    Ensures CRS is WGS84 (EPSG:4326) for compatibility with PAR data.
    Creates sequential integer IDs for efficient rasterization.
    """
    # Handle empty GeoDataFrame
    if len(gdf) == 0:
        gdf_out = gdf.copy()
        if gdf_out.crs is None:
            gdf_out = gdf_out.set_crs(4326)
        elif gdf_out.crs.to_epsg() != 4326:
            gdf_out = gdf_out.to_crs(4326)
        gdf_out['__cid__'] = np.array([], dtype=np.int32)
        return gdf_out[['__cid__', 'geometry']].copy() if 'geometry' in gdf_out.columns else gdf_out

    if id_cols is None:
        id_cols = [c for c in ['COUNTYID', 'COUNTYCODE', 'COUNTYNAME', 'COUNTYENG']
                  if c in gdf.columns]
    if not id_cols:
        id_cols = []

    # Fix geometries
    gdf_out = make_valid_geoms(gdf)

    # Ensure WGS84
    if gdf_out.crs is None:
        gdf_out = gdf_out.set_crs(4326)
    elif gdf_out.crs.to_epsg() != 4326:
        gdf_out = gdf_out.to_crs(4326)

    # Reset index and create sequential IDs
    gdf_out = gdf_out.reset_index(drop=True)
    gdf_out['__cid__'] = np.arange(1, len(gdf_out) + 1, dtype=np.int32)

    return gdf_out[id_cols + ['__cid__', 'geometry']].copy()


def attach_county_metadata(df_result: pd.DataFrame, gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Attach county labels to aggregated PAR output by ``__cid__``.

    Parameters
    ----------
    df_result : pd.DataFrame
        Aggregated PAR result containing ``__cid__``.
    gdf : pd.DataFrame
        County metadata table containing ``__cid__`` and optional county columns.

    Returns
    -------
    pd.DataFrame
        Result with an added ``county`` column.
    """
    if "__cid__" not in df_result.columns:
        return df_result.copy()

    county_col = next(
        (c for c in ["COUNTYNAME", "COUNTYENG", "COUNTYID", "COUNTYCODE"] if c in gdf.columns),
        None,
    )
    if county_col is None:
        county_meta = gdf[["__cid__"]].copy()
        county_meta["county"] = county_meta["__cid__"].astype(str)
    else:
        county_meta = gdf[["__cid__", county_col]].copy().rename(columns={county_col: "county"})

    out = df_result.merge(county_meta, on="__cid__", how="left")
    return out


def extract_timestamp_from_path(filepath: str, pattern: str = r"H09_(\d{8})_(\d{4})") -> pd.Timestamp:
    """
    Extract UTC timestamp from Himawari filename.

    Parameters
    ----------
    filepath : str
        Path to PAR file
    pattern : str
        Regex pattern for timestamp extraction

    Returns
    -------
    pd.Timestamp
        UTC timestamp

    Examples
    --------
    >>> extract_timestamp_from_path("H09_20240101_0300_RFL021.nc.gz")
    Timestamp('2024-01-01 03:00:00')
    """
    m = re.search(pattern, Path(filepath).name)
    if not m:
        raise ValueError(f"Cannot extract timestamp from {filepath}")

    # Handle patterns with different numbers of groups
    # Default pattern: H09_20240101_0300 -> 2 groups (ymd, hm)
    # Custom patterns may have 3 groups: e.g., SAT_2024-01-01_03h00 -> (year-month-day, hour, minute)
    groups = m.groups()
    if len(groups) == 2:
        ymd, hm = groups
        return pd.Timestamp(ymd + hm)
    elif len(groups) == 3:
        # Pattern like r"SAT_(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})"
        # groups: ('2024-01-01', '03', '00')
        date_part, hour, minute = groups
        return pd.Timestamp(f"{date_part} {hour}:{minute}")
    else:
        # Join all groups for flexibility
        return pd.Timestamp(''.join(groups))


def process_par_file(
    gz_path: Path,
    gdf: gpd.GeoDataFrame,
    labels: Optional[np.ndarray] = None,
    lat_bounds: Optional[Tuple[float, float, float, float]] = None,
    factor: int = 5,
    all_touched: bool = False,
    temp_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    Process a single PAR file and aggregate to county level.

    Parameters
    ----------
    gz_path : Path
        Path to gzipped NetCDF file
    gdf : gpd.GeoDataFrame
        County boundaries (with __cid__ column)
    labels : np.ndarray, optional
        Pre-computed rasterized labels (if None, will compute)
    lat_bounds : tuple, optional
        (lon_min, lon_max, lat_max, lat_min) for clipping
    factor : int
        Upsampling factor (default: 5)
    all_touched : bool
        Rasterization pixel inclusion rule
    temp_dir : Path, optional
        Directory for temporary decompressed file (can be ramdisk path)

    Returns
    -------
    pd.DataFrame or None
        County-level PAR means with time_utc column, or None if processing failed
    """
    try:
        # Extract timestamp
        ts_utc = extract_timestamp_from_path(str(gz_path))

        # Load data
        da, ds = open_par_from_gz(gz_path, temp_dir=temp_dir)

        try:
            # Filter valid values
            vmin = float(da.attrs.get("valid_min", 0))
            vmax = float(da.attrs.get("valid_max", np.inf))
            da = da.where((da >= vmin) & (da <= vmax))

            # Clip to AOI if bounds provided
            if lat_bounds:
                lon_min, lon_max, lat_max, lat_min = lat_bounds
                da = da.sel(
                    longitude=slice(lon_min, lon_max),
                    latitude=slice(lat_max, lat_min)
                )

            # Upsample
            if factor > 1:
                da = upsample_par_data(da, factor=factor)

            # Rasterize if not provided
            if labels is None:
                if lat_bounds:
                    # lat_bounds is (lon_min, lon_max, lat_max, lat_min)
                    # clip_to_aoi expects (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
                    lon_min, lon_max, lat_max, lat_min = lat_bounds
                    gdf_aoi = clip_to_aoi(gdf, lon_min, lat_min, lon_max, lat_max)
                else:
                    gdf_aoi = gdf
                labels, _ = rasterize_counties(gdf_aoi, da.latitude.values, da.longitude.values, all_touched)

            # Compute county means
            df_result = compute_county_par(da, labels, gdf['__cid__'].values)
            df_result = attach_county_metadata(df_result, gdf)

            # Add timestamp
            df_result['time_utc'] = ts_utc

            cols = [c for c in ['county', 'mean_PAR_umol_m2_s', 'n_pixels', 'time_utc'] if c in df_result.columns]
            remaining = [c for c in df_result.columns if c not in cols and c != '__cid__']
            return df_result[cols + remaining]

        finally:
            ds.close()

    except Exception as e:
        # Log error but continue processing other files
        print(f"Warning: Failed to process {gz_path}: {e}")
        return None
