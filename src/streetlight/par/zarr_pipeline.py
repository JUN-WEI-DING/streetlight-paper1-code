"""
PAR processing pipeline using Zarr data source.

This module provides a faster alternative to the original PARPipeline by reading
from the pre-processed zarr dataset instead of individual .nc.gz files.

Performance comparison:
- Original pipeline: ~40 minutes for 45K files (includes decompression)
- Zarr pipeline: ~1-2 minutes (direct array operations)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

import numpy as np
import pandas as pd
import xarray as xr

from streetlight.config import get_config
from streetlight.paths import county_shapefile_path
from streetlight.sources.par_zarr import load_par_zarr, get_par_zarr_path

# Optional geospatial dependencies
try:
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from streetlight.par.par_aggregation import (
        build_area_weight_table,
        compute_area_weighted_county_par,
    )
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    gpd = None


logger = logging.getLogger(__name__)


@dataclass
class PARZarrConfig:
    """Configuration for PAR processing from Zarr."""

    # Zarr path (uses config default if None)
    zarr_path: Optional[Path] = None

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("./data/par/processed"))

    # Processing parameters
    upsample_factor: int = 2  # 5km → 2.5km grid

    # AOI bounds (lon_min, lon_max, lat_max, lat_min) for Taiwan
    aoi_bounds: Tuple[float, float, float, float] = (118.0, 124.0, 27.0, 20.0)

    # Time range filter (None = all data)
    time_range: Optional[Tuple[str, str]] = None

    # County boundaries file
    county_boundaries_file: Optional[Path] = None

    # Aggregation method: "rasterized" keeps legacy behavior; "area_weighted"
    # uses polygon-cell overlap areas.
    aggregation_method: str = "rasterized"

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config=None,
        *,
        zarr_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        time_range: Optional[Tuple[str, str]] = None,
        county_boundaries_file: Optional[Path] = None,
    ) -> "PARZarrConfig":
        """Build PAR zarr config from shared runtime config plus optional overrides."""
        runtime_config = runtime_config or get_config()
        return cls(
            zarr_path=zarr_path or Path(runtime_config.get('data.par_zarr_path', "./data/zarr/himawari_par_2024.zarr")),
            output_dir=output_dir or runtime_config.outputs_par_dir,
            upsample_factor=int(runtime_config.get('par.upsample_factor', 2)),
            aoi_bounds=tuple(runtime_config.get('par.aoi_bounds', (118.0, 124.0, 27.0, 20.0))),
            time_range=time_range,
            county_boundaries_file=county_boundaries_file,
        )


class PARZarrPipeline:
    """
    Pipeline for processing PAR data from Zarr.

    This pipeline reads from the consolidated zarr dataset, which is much faster
    than processing individual .nc.gz files.

    Example:
        >>> from streetlight.par.zarr_pipeline import PARZarrPipeline
        >>> pipeline = PARZarrPipeline()
        >>> df = pipeline.run(time_range=("2024-06-01", "2024-06-30"))
    """

    def __init__(
        self,
        config: Optional[PARZarrConfig] = None,
        county_gdf: Optional['gpd.GeoDataFrame'] = None
    ):
        """
        Initialize the PAR Zarr pipeline.

        Args:
            config: Pipeline configuration
            county_gdf: Pre-loaded county boundaries

        Raises:
            ImportError: If geospatial dependencies are not available
            FileNotFoundError: If zarr dataset does not exist
        """
        if not HAS_GEOSPATIAL:
            raise ImportError(
                "Geospatial dependencies (geopandas, rasterio) are required. "
                "Install with: pip install geopandas rasterio"
            )

        self.streetlight_config = get_config()
        self.config = config or PARZarrConfig.from_runtime_config(self.streetlight_config)

        # Resolve zarr path
        self.zarr_path = self.config.zarr_path or get_par_zarr_path()
        if not self.zarr_path.exists():
            raise FileNotFoundError(
                f"PAR zarr not found at {self.zarr_path}. "
                "Run scripts/par_to_zarr.py first."
            )

        # Load county boundaries
        if county_gdf is not None:
            self.county_gdf = county_gdf
        else:
            self.county_gdf = self._load_county_boundaries()

        # Prepare geometries
        self.prepared_gdf = self._prepare_geoms()

    def _load_county_boundaries(self) -> 'gpd.GeoDataFrame':
        """Load county boundary data."""
        attempted_paths = []

        if self.config.county_boundaries_file:
            attempted_paths.append(str(self.config.county_boundaries_file))
            if self.config.county_boundaries_file.exists():
                return gpd.read_file(self.config.county_boundaries_file)

        county_file = county_shapefile_path()
        attempted_paths.append(str(county_file))
        if county_file.exists():
            return gpd.read_file(county_file)

        geo_dir = Path(self.streetlight_config.data_dir) / "geo"
        for ext in ['.geojson', '.shp', '.gpkg']:
            county_file = geo_dir / f"county_boundaries{ext}"
            attempted_paths.append(str(county_file))
            if county_file.exists():
                return gpd.read_file(county_file)

        raise FileNotFoundError(
            f"County boundaries not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in attempted_paths)
        )

    def _prepare_geoms(self) -> 'gpd.GeoDataFrame':
        """Prepare county geometries for rasterization."""
        from shapely import make_valid

        gdf = self.county_gdf.copy()

        # Fix invalid geometries
        try:
            gdf["geometry"] = make_valid(np.asarray(gdf.geometry.values))
        except Exception:
            gdf["geometry"] = gdf.buffer(0)

        # Ensure WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        # Create sequential IDs
        gdf = gdf.reset_index(drop=True)
        gdf['__cid__'] = np.arange(1, len(gdf) + 1, dtype=np.int32)

        # Get county name column
        county_col = next(
            (c for c in ['COUNTYNAME', 'COUNTYENG', 'COUNTYID'] if c in gdf.columns),
            None
        )
        if county_col:
            return gdf[[county_col, '__cid__', 'geometry']].copy()
        return gdf[['__cid__', 'geometry']].copy()

    def _clip_to_aoi(self, gdf: 'gpd.GeoDataFrame') -> 'gpd.GeoDataFrame':
        """Clip GeoDataFrame to AOI."""
        from shapely.geometry import box
        lon_min, lon_max, lat_max, lat_min = self.config.aoi_bounds
        aoi_poly = box(lon_min, lat_min, lon_max, lat_max)
        return gpd.clip(gdf, gpd.GeoDataFrame(geometry=[aoi_poly], crs=4326))

    def _rasterize_counties(
        self,
        gdf: 'gpd.GeoDataFrame',
        lat: np.ndarray,
        lon: np.ndarray
    ) -> np.ndarray:
        """Rasterize county boundaries to grid."""
        # Calculate transform
        xres = float(lon[1] - lon[0]) if len(lon) > 1 else 0.05
        yres = float(lat[0] - lat[1]) if len(lat) > 1 else 0.05
        ny, nx = len(lat), len(lon)

        transform = from_origin(
            float(lon.min() - xres / 2),
            float(lat.max() + yres / 2),
            xres, yres
        )

        # Create shapes
        shapes = list(zip(gdf.geometry.values, gdf['__cid__'].astype(int).values))

        # Rasterize
        labels = rasterize(
            shapes=shapes,
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            dtype='int32',
            all_touched=False
        )

        return labels

    def _upsample_array(self, arr: np.ndarray, factor: int) -> np.ndarray:
        """Upsample array by repeating pixels."""
        return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)

    def _upsample_coords(
        self,
        coords: np.ndarray,
        factor: int,
        is_latitude: bool
    ) -> np.ndarray:
        """Upsample coordinate array."""
        if len(coords) == 1:
            d = 0.05
        else:
            d = abs(float(coords[1] - coords[0])) if len(coords) > 1 else 0.05

        sub = d / factor
        n_new = len(coords) * factor

        if is_latitude:
            # Latitude decreasing (N→S)
            return (coords[0] + 2 * sub) - sub * np.arange(n_new, dtype=np.float32)
        else:
            # Longitude increasing (W→E)
            return (coords[0] - 2 * sub) + sub * np.arange(n_new, dtype=np.float32)

    def _aggregate_to_counties(
        self,
        par: np.ndarray,
        labels: np.ndarray,
        county_ids: np.ndarray
    ) -> pd.DataFrame:
        """Compute mean PAR for each county."""
        finite = np.isfinite(par)

        if not np.any(finite):
            return pd.DataFrame({
                '__cid__': county_ids,
                'mean_PAR_umol_m2_s': np.nan,
                'n_pixels': 0
            })

        lab = labels[finite].ravel()
        dat = par[finite].ravel()

        max_id = int(county_ids.max())
        sums = np.bincount(lab, weights=dat, minlength=max_id + 1)
        counts = np.bincount(lab, minlength=max_id + 1)

        means = np.divide(
            sums, counts,
            out=np.full_like(sums, np.nan, dtype=np.float64),
            where=counts > 0
        )

        return pd.DataFrame({
            '__cid__': county_ids,
            'mean_PAR_umol_m2_s': means[county_ids],
            'n_pixels': counts[county_ids].astype(int)
        })

    def _aggregate_area_weighted(
        self,
        par: np.ndarray,
        weights: pd.DataFrame,
        county_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Compute county PAR with polygon-cell overlap-area weights."""
        return compute_area_weighted_county_par(par, weights, county_ids)

    def process_time_slice(
        self,
        ds: xr.Dataset,
        time_idx: int,
        labels: np.ndarray,
        county_ids: np.ndarray
    ) -> pd.DataFrame:
        """Process a single time slice from the dataset."""
        # Get PAR data for this time
        par = ds['PAR'].isel(time=time_idx).values

        # Upsample if needed
        if self.config.upsample_factor > 1:
            par = self._upsample_array(par, self.config.upsample_factor)

        # Aggregate
        df = self._aggregate_to_counties(par, labels, county_ids)

        # Add timestamp
        df['time_utc'] = pd.Timestamp(ds.time.values[time_idx])

        return df

    def run(
        self,
        time_range: Optional[Tuple[str, str]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        save: bool = False,
        output_path: Optional[Path] = None,
        format: str = "csv",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Run the PAR aggregation pipeline.

        Args:
            time_range: (start, end) time range (e.g., ("2024-06-01", "2024-06-30"))
            lon_range: (min, max) longitude bounds (default: config.aoi_bounds)
            lat_range: (min, max) latitude bounds (default: config.aoi_bounds)
            save: Whether to save results
            output_path: Output file path
            format: Output format ('csv' or 'parquet')
            show_progress: Show progress bar

        Returns:
            DataFrame with county-level PAR time series
        """
        # Use config defaults if not specified
        if lon_range is None:
            lon_range = (self.config.aoi_bounds[0], self.config.aoi_bounds[1])
        if lat_range is None:
            lat_range = (self.config.aoi_bounds[3], self.config.aoi_bounds[2])  # (lat_min, lat_max)
        if time_range is None:
            time_range = self.config.time_range

        logger.info(f"Loading PAR zarr from {self.zarr_path}")
        ds = load_par_zarr(
            zarr_path=self.zarr_path,
            time_range=time_range,
            lon_range=lon_range,
            lat_range=lat_range,
        )

        logger.info(f"Loaded dataset with {ds.time.size} time steps")

        # Get coordinates
        lat = ds.latitude.values
        lon = ds.longitude.values

        # Clip counties to AOI and rasterize at original resolution
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        gdf_aoi = self._clip_to_aoi(self.prepared_gdf)
        labels = self._rasterize_counties(gdf_aoi, lat, lon)
        weights = None

        # Upsample coordinates and labels if needed
        if self.config.upsample_factor > 1:
            lat = self._upsample_coords(lat, self.config.upsample_factor, is_latitude=True)
            lon = self._upsample_coords(lon, self.config.upsample_factor, is_latitude=False)
            labels = self._upsample_array(labels, self.config.upsample_factor)

        county_ids = self.prepared_gdf['__cid__'].values
        if self.config.aggregation_method == "area_weighted":
            logger.info("Building area-weighted county/cell overlap table")
            weights = build_area_weight_table(gdf_aoi, lat, lon)

        # Process all time steps
        results = []
        n_times = ds.time.size

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(n_times), desc="Aggregating PAR")
        else:
            iterator = range(n_times)

        for i in iterator:
            if weights is None:
                df = self.process_time_slice(ds, i, labels, county_ids)
            else:
                par = ds['PAR'].isel(time=i).values
                if self.config.upsample_factor > 1:
                    par = self._upsample_array(par, self.config.upsample_factor)
                df = self._aggregate_area_weighted(par, weights, county_ids)
                df['time_utc'] = pd.Timestamp(ds.time.values[i])
            results.append(df)

        ds.close()

        # Combine results
        combined = pd.concat(results, ignore_index=True)

        # Add county names
        county_col = next(
            (c for c in ['COUNTYNAME', 'COUNTYENG', 'COUNTYID'] if c in self.prepared_gdf.columns),
            None
        )
        if county_col:
            county_meta = self.prepared_gdf[[county_col, '__cid__']].copy()
            county_meta = county_meta.rename(columns={county_col: 'county'})
            combined = combined.merge(county_meta, on='__cid__', how='left')

        # Sort and reorder columns
        combined = combined.sort_values('time_utc')
        cols = ['county', 'mean_PAR_umol_m2_s', 'n_pixels', 'time_utc']
        combined = combined[[c for c in cols if c in combined.columns]]

        logger.info(f"Generated {len(combined)} records")

        # Save if requested
        if save:
            output_path = output_path or self.config.output_dir / f"par_aggregated.{format}"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "csv":
                combined.to_csv(output_path, index=False)
            elif format == "parquet":
                combined.to_parquet(output_path, index=False)

            logger.info(f"Saved to {output_path}")

        return combined


def create_par_zarr_pipeline(
    zarr_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    time_range: Optional[Tuple[str, str]] = None,
    county_boundaries_file: Optional[Path] = None
) -> PARZarrPipeline:
    """
    Convenience function to create a PAR zarr pipeline.

    Args:
        zarr_path: Path to zarr dataset
        output_dir: Output directory
        time_range: Time range filter
        county_boundaries_file: Path to county boundaries

    Returns:
        Configured PARZarrPipeline instance
    """
    runtime_config = get_config()
    config = PARZarrConfig.from_runtime_config(runtime_config)

    if zarr_path is not None:
        config.zarr_path = zarr_path
    if output_dir is not None:
        config.output_dir = output_dir
    if time_range is not None:
        config.time_range = time_range
    if county_boundaries_file is not None:
        config.county_boundaries_file = county_boundaries_file

    return PARZarrPipeline(config)
