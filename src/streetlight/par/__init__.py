"""
PAR (Photosynthetically Active Radiation) processing pipeline.

This module provides pipelines for processing Himawari satellite PAR data
and aggregating it to county-level time series.

Two pipeline implementations are available:
- PARPipeline: Reads from original .nc.gz files (slower, full resolution)
- PARZarrPipeline: Reads from zarr dataset (faster, pre-cropped)
"""

from streetlight.par.pipeline import (
    PARPipeline,
    PARConfig,
    create_par_pipeline,
    HAS_GEOSPATIAL
)
from streetlight.par.handoff import (
    load_par_aggregated,
    interpolate_short_gaps,
    par_aggregated_to_wide,
    prepare_par_for_simulation,
    save_par_wide,
)
from streetlight.par.zarr_pipeline import (
    PARZarrPipeline,
    PARZarrConfig,
    create_par_zarr_pipeline
)

__all__ = [
    # Original pipeline (from .nc.gz files)
    'PARPipeline',
    'PARConfig',
    'create_par_pipeline',
    'load_par_aggregated',
    'interpolate_short_gaps',
    'par_aggregated_to_wide',
    'prepare_par_for_simulation',
    'save_par_wide',
    # Fast zarr pipeline
    'PARZarrPipeline',
    'PARZarrConfig',
    'create_par_zarr_pipeline',
    # Status
    'HAS_GEOSPATIAL'
]
