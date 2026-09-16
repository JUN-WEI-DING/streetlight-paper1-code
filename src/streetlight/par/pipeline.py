"""
PAR processing pipeline.

This module provides a pipeline for processing Himawari satellite PAR data.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
import pandas as pd

from streetlight.config import get_config
from streetlight.paths import county_shapefile_path

logger = logging.getLogger(__name__)

# Optional geospatial dependencies
try:
    import geopandas as gpd
    from streetlight.par.par_aggregation import (
        make_valid_geoms,
        open_par_from_gz,
        clip_to_aoi,
        upsample_par_data,
        rasterize_counties,
        compute_county_par,
        prepare_county_geoms,
        extract_timestamp_from_path,
        process_par_file,
    )
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    gpd = None


# Process-level shared data for workers (initialized once per worker process)
_worker_gdf: 'gpd.GeoDataFrame' = None
_worker_labels: Optional[np.ndarray] = None
_worker_lat_bounds: Optional[Tuple[float, float, float, float]] = None
_worker_factor: int = 5
_worker_all_touched: bool = False
_worker_temp_dir: Optional[Path] = None


def _worker_initializer(
    gdf: 'gpd.GeoDataFrame',
    lat_bounds: Tuple[float, float, float, float],
    factor: int,
    all_touched: bool,
    labels: Optional[np.ndarray] = None,
    temp_dir: Optional[Path] = None
) -> None:
    """
    Initialize worker process with shared data.

    This function runs once per worker process when the pool is created.
    It stores the shared data in process-level globals to avoid pickling
    large objects on every task submission.

    Args:
        gdf: Prepared GeoDataFrame with county boundaries
        lat_bounds: (lon_min, lon_max, lat_max, lat_min) for clipping
        factor: Upsampling factor
        all_touched: Rasterization pixel rule
        labels: Pre-computed rasterized county labels
        temp_dir: Temporary directory for decompressed files
    """
    global _worker_gdf, _worker_labels, _worker_lat_bounds, _worker_factor, _worker_all_touched, _worker_temp_dir
    _worker_gdf = gdf
    _worker_labels = labels
    _worker_lat_bounds = lat_bounds
    _worker_factor = factor
    _worker_all_touched = all_touched
    _worker_temp_dir = temp_dir


def _process_par_worker_shared(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Worker function that uses process-level shared data.

    This function reads from global variables initialized by _worker_initializer,
    avoiding pickling of large GeoDataFrame objects on every task submission.

    Args:
        file_path: Path to PAR file

    Returns:
        DataFrame with county-level PAR means or None if failed
    """
    global _worker_gdf, _worker_labels, _worker_lat_bounds, _worker_factor, _worker_all_touched, _worker_temp_dir

    return process_par_file(
        file_path, _worker_gdf, labels=_worker_labels,
        lat_bounds=_worker_lat_bounds, factor=_worker_factor, all_touched=_worker_all_touched,
        temp_dir=_worker_temp_dir
    )


@dataclass
class PARConfig:
    """Configuration for PAR processing."""

    # Data directories
    input_dir: Path = field(default_factory=lambda: Path("./himawari/L2/PAR/L2/PAR"))
    output_dir: Path = field(default_factory=lambda: Path("./data/par/processed"))

    # Processing parameters
    upsample_factor: int = 2  # Reduced from 5 to 2 for 6.25x speedup (4x vs 25x pixels)
    all_touched: bool = False

    # AOI bounds (lon_min, lon_max, lat_max, lat_min) for Taiwan
    aoi_bounds: Tuple[float, float, float, float] = (118.0, 124.0, 27.0, 20.0)

    # File pattern
    file_pattern: str = "*.nc.gz"

    # Parallel processing - auto-detect optimal workers
    # Guard against cpu_count=None (some containers) and ensure at least 1
    max_workers: Optional[int] = field(default_factory=lambda: max(1, min(16, int((os.cpu_count() or 1) * 0.5))))

    # County boundaries file
    county_boundaries_file: Optional[Path] = None

    # Temporary directory for decompressed files (can be ramdisk path like /dev/shm)
    temp_dir: Optional[Path] = None

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config=None,
        *,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        county_boundaries_file: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
    ) -> "PARConfig":
        """Build PAR config from shared runtime config plus optional overrides."""
        runtime_config = runtime_config or get_config()
        return cls(
            input_dir=input_dir or Path(runtime_config.get('data.par_raw_dir', "./himawari/L2/PAR/L2/PAR")),
            output_dir=output_dir or runtime_config.outputs_par_dir,
            upsample_factor=int(runtime_config.get('par.upsample_factor', 2)),
            all_touched=bool(runtime_config.get('par.all_touched', False)),
            aoi_bounds=tuple(runtime_config.get('par.aoi_bounds', (118.0, 124.0, 27.0, 20.0))),
            file_pattern=str(runtime_config.get('par.file_pattern', "*.nc.gz")),
            max_workers=runtime_config.get('par.max_workers', None),
            county_boundaries_file=county_boundaries_file,
            temp_dir=temp_dir,
        )


class PARPipeline:
    """
    Pipeline for processing Himawari PAR data.

    Orchestrates loading PAR files, upsampling, rasterizing county boundaries,
    and aggregating to county-level time series.
    """

    def __init__(
        self,
        config: Optional[PARConfig] = None,
        county_gdf: Optional['gpd.GeoDataFrame'] = None
    ):
        """
        Initialize the PAR pipeline.

        Args:
            config: PAR configuration (uses default if None)
            county_gdf: Pre-loaded county boundaries (loads from file if None)

        Raises:
            ImportError: If geospatial dependencies are not available
        """
        if not HAS_GEOSPATIAL:
            raise ImportError(
                "Geospatial dependencies (geopandas, rasterio) are required for PAR processing. "
                "Install them with: pip install geopandas rasterio"
            )

        self.streetlight_config = get_config()
        self.config = config or PARConfig.from_runtime_config(self.streetlight_config)

        # Load county boundaries
        if county_gdf is not None:
            self.county_gdf = county_gdf
        else:
            self.county_gdf = self._load_county_boundaries()

        # Prepare geometries
        self.prepared_gdf = prepare_county_geoms(self.county_gdf)

        # Shared labels for all files (same grid)
        self._cached_labels = None

    def _load_county_boundaries(self) -> 'gpd.GeoDataFrame':
        """
        Load county boundary data.

        Returns:
            GeoDataFrame with county boundaries

        Raises:
            FileNotFoundError: If county boundaries file cannot be found
        """
        attempted_paths = []

        # Try to load from configured path
        if self.config.county_boundaries_file:
            attempted_paths.append(str(self.config.county_boundaries_file))
            if self.config.county_boundaries_file.exists():
                return gpd.read_file(self.config.county_boundaries_file)

        # Try paths module
        county_file = county_shapefile_path()
        attempted_paths.append(str(county_file))
        if county_file.exists():
            return gpd.read_file(county_file)

        # Use default path from config - try multiple formats
        geo_dir = Path(self.streetlight_config.data_dir) / "geo"
        for ext in ['.geojson', '.shp', '.gpkg']:
            county_file = geo_dir / f"county_boundaries{ext}"
            attempted_paths.append(str(county_file))
            if county_file.exists():
                return gpd.read_file(county_file)

        # Fail fast instead of using dummy data
        raise FileNotFoundError(
            f"County boundaries file not found. Attempted paths:\n" +
            "\n".join(f"  - {p}" for p in attempted_paths) +
            "\n\nPlease provide a county boundaries file (GeoJSON/Shapefile/GeoPackage)."
        )

    def find_par_files(
        self,
        input_dir: Optional[Path] = None,
        pattern: Optional[str] = None
    ) -> List[Path]:
        """
        Find PAR files in input directory.

        Args:
            input_dir: Input directory (uses config if None)
            pattern: File pattern (uses config if None)

        Returns:
            List of PAR file paths
        """
        input_dir = input_dir or self.config.input_dir
        pattern = pattern or self.config.file_pattern

        files = list(Path(input_dir).rglob(pattern))
        return sorted(files)

    def process_file(
        self,
        file_path: Path,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Process a single PAR file.

        Args:
            file_path: Path to PAR file
            use_cache: Whether to use cached labels

        Returns:
            DataFrame with county-level PAR means or None if failed
        """
        # Use shared labels cache (all PAR files have same grid)
        labels = self._cached_labels if use_cache else None

        result = process_par_file(
            file_path,
            self.prepared_gdf,
            labels=labels,
            lat_bounds=self.config.aoi_bounds,
            factor=self.config.upsample_factor,
            all_touched=self.config.all_touched,
            temp_dir=self.config.temp_dir
        )

        # Cache labels from first file for reuse
        if use_cache and self._cached_labels is None and result is not None:
            # Create and cache labels for subsequent files
            from streetlight.par.par_aggregation import open_par_from_gz, clip_to_aoi, rasterize_counties, upsample_par_data
            da_sample, _ = open_par_from_gz(file_path)
            lon_min, lon_max, lat_max, lat_min = self.config.aoi_bounds
            da_clipped = da_sample.sel(
                longitude=slice(lon_min, lon_max),
                latitude=slice(lat_max, lat_min)
            )
            da_up = upsample_par_data(da_clipped, factor=self.config.upsample_factor)
            gdf_aoi = clip_to_aoi(self.prepared_gdf, lon_min, lat_min, lon_max, lat_max)
            self._cached_labels, _ = rasterize_counties(
                gdf_aoi, da_up.latitude.values, da_up.longitude.values,
                all_touched=self.config.all_touched
            )

        return result

    def process_files(
        self,
        files: Optional[List[Path]] = None,
        parallel: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process multiple PAR files.

        Args:
            files: List of files to process (finds all if None)
            parallel: Whether to use parallel processing
            show_progress: Whether to show progress (for parallel mode)

        Returns:
            Concatenated DataFrame with all results
        """
        if files is None:
            files = self.find_par_files()

        if not files:
            logger.info(f"No files found matching pattern in {self.config.input_dir}")
            return pd.DataFrame()

        results = []

        if parallel and len(files) > 1:
            results = self._process_parallel(files, show_progress)
        else:
            results = self._process_sequential(files, show_progress)

        # Combine results
        if results:
            combined = pd.concat(results, ignore_index=True)
            # Only sort by time_utc if column exists
            if 'time_utc' in combined.columns:
                return combined.sort_values('time_utc')
            return combined
        return pd.DataFrame()

    def _process_sequential(
        self,
        files: List[Path],
        show_progress: bool
    ) -> List[pd.DataFrame]:
        """Process files sequentially."""
        results = []
        total = len(files)

        for i, file_path in enumerate(files):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing {i+1}/{total}...")

            result = self.process_file(file_path)
            if result is not None:
                results.append(result)

        return results

    def _process_parallel(
        self,
        files: List[Path],
        show_progress: bool
    ) -> List[pd.DataFrame]:
        """Process files in parallel using multiprocessing."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial

        # Cap workers at configured max_workers, but never exceed file count
        # Ensure at least 1 worker even if config is 0 or None
        max_workers = max(1, min(self.config.max_workers or 8, len(files)))

        # Pre-compute labels from first file to avoid re-rasterization in each worker
        # All PAR files share the same grid structure, so labels can be reused
        precomputed_labels = None
        if files:
            sample_file = files[0]
            try:
                from streetlight.par.par_aggregation import (
                    open_par_from_gz, clip_to_aoi, rasterize_counties, upsample_par_data
                )
                da_sample, _ = open_par_from_gz(sample_file)
                lon_min, lon_max, lat_max, lat_min = self.config.aoi_bounds
                da_clipped = da_sample.sel(
                    longitude=slice(lon_min, lon_max),
                    latitude=slice(lat_max, lat_min)
                )
                da_up = upsample_par_data(da_clipped, factor=self.config.upsample_factor)
                gdf_aoi = clip_to_aoi(self.prepared_gdf, lon_min, lat_min, lon_max, lat_max)
                precomputed_labels, _ = rasterize_counties(
                    gdf_aoi, da_up.latitude.values, da_up.longitude.values,
                    all_touched=self.config.all_touched
                )
            except Exception as e:
                logger.warning(f"Failed to pre-compute labels, will compute per-file: {e}")

        results = []
        completed = 0

        # Use initializer to set up shared data in each worker process
        # This avoids pickling large GeoDataFrame objects on every task submission
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.prepared_gdf,
                self.config.aoi_bounds,
                self.config.upsample_factor,
                self.config.all_touched,
                precomputed_labels,
                self.config.temp_dir,
            )
        ) as executor:
            # Submit all tasks using the worker function that reads from shared data
            futures = {
                executor.submit(_process_par_worker_shared, f): f
                for f in files
            }

            # Collect results as they complete
            for future in as_completed(futures):
                completed += 1
                if show_progress and completed % 100 == 0:
                    logger.info(f"Processing {completed}/{len(files)}...")

                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logger.warning(f"Failed to process {file_path.name}: {e}")

        return results

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
        format: str = "csv"
    ) -> None:
        """
        Save PAR aggregation results.

        Args:
            df: Results DataFrame
            output_path: Output file path or directory (uses config if None)
            format: Output format ('csv' or 'parquet')

        Note on output_path:
            - If None: uses config.output_dir / f"par_aggregated.{format}"
            - If has suffix (extension): treated as file path, used directly
            - If no suffix (directory): treated as directory, writes to
              output_path / f"par_aggregated.{format}"
        """
        output_path = output_path or self.config.output_dir / f"par_aggregated.{format}"
        output_path = Path(output_path)

        # Determine if path is directory or file
        if output_path.suffix == "":
            # No suffix = directory
            output_path = output_path / f"par_aggregated.{format}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved results to {output_path}")

    def run(
        self,
        files: Optional[List[Path]] = None,
        save: bool = True,
        output_path: Optional[Path] = None,
        format: str = "csv"
    ) -> pd.DataFrame:
        """
        Run the complete PAR pipeline.

        Args:
            files: List of files to process (finds all if None)
            save: Whether to save results
            output_path: Custom output path
            format: Output format ('csv' or 'parquet')

        Returns:
            DataFrame with county-level PAR time series
        """
        logger.info("Processing PAR data...")

        # Find files if not provided
        if files is None:
            files = self.find_par_files()
            logger.info(f"Found {len(files)} files")

        # Process files
        df = self.process_files(files)
        logger.info(f"Processed {len(df)} records")

        # Save if requested
        if save and not df.empty:
            self.save_results(df, output_path, format=format)

        return df

    def get_county_list(self) -> List[str]:
        """
        Get list of counties in the prepared GeoDataFrame.

        Returns:
            List of county names
        """
        if 'COUNTYNAME' in self.prepared_gdf.columns:
            return self.prepared_gdf['COUNTYNAME'].tolist()
        elif 'COUNTYENG' in self.prepared_gdf.columns:
            return self.prepared_gdf['COUNTYENG'].tolist()
        return []

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for PAR data.

        Args:
            df: PAR results DataFrame

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}

        summary = {
            'n_records': len(df),
            'counties': df['county'].nunique() if 'county' in df.columns else 'N/A',
        }

        # Only add time_range if time_utc column exists
        if 'time_utc' in df.columns:
            summary['time_range'] = (df['time_utc'].min(), df['time_utc'].max())
        else:
            summary['time_range'] = 'N/A'

        if 'mean_PAR_umol_m2_s' in df.columns:
            valid_par = df['mean_PAR_umol_m2_s'].dropna()
            if not valid_par.empty:
                summary['par_mean'] = valid_par.mean()
                summary['par_std'] = valid_par.std()
                summary['par_min'] = valid_par.min()
                summary['par_max'] = valid_par.max()

        return summary


def create_par_pipeline(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    county_boundaries_file: Optional[Path] = None,
    temp_dir: Optional[Path] = None
) -> PARPipeline:
    """
    Convenience function to create a PAR pipeline.

    Args:
        input_dir: Input directory for PAR files
        output_dir: Output directory for processed data
        county_boundaries_file: Path to county boundaries GeoJSON
        temp_dir: Temporary directory for decompressed files (can be ramdisk path)

    Returns:
        Configured PARPipeline instance
    """
    runtime_config = get_config()
    config = PARConfig.from_runtime_config(runtime_config)

    if input_dir is not None:
        config.input_dir = input_dir
    if output_dir is not None:
        config.output_dir = output_dir
    if county_boundaries_file is not None:
        config.county_boundaries_file = county_boundaries_file
    if temp_dir is not None:
        config.temp_dir = temp_dir

    return PARPipeline(config)
