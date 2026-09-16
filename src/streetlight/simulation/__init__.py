"""
Streetlight simulation module - PV+battery/grid split and carbon emission analysis.

This module combines PAR data and AEF (Average Emission Factors) to simulate
streetlight PV+battery systems and calculate carbon emission reduction.
"""

from streetlight.simulation.streetlight_simulation import (
    load_aef_by_region,
    load_par_wide,
    load_region_city_map,
    save_simulation_outputs,
    simulate_from_par_and_aef,
    align_time_indices,
)
from streetlight.simulation.storage import (
    storage_dispatch,
    compute_city_emission,
    infer_dt_hours,
)

__all__ = [
    "load_aef_by_region",
    "load_par_wide",
    "load_region_city_map",
    "save_simulation_outputs",
    "simulate_from_par_and_aef",
    "align_time_indices",
    "storage_dispatch",
    "compute_city_emission",
    "infer_dt_hours",
]
