"""Simulation utilities for streetlight PV+storage with PAR and AEF inputs."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from streetlight.simulation.storage import (
    storage_dispatch,
    compute_city_emission,
    _resolve_economic_costs,
    _resolve_financial_params,
)

logger = logging.getLogger(__name__)
TAIWAN_TZ = ZoneInfo("Asia/Taipei")


def load_par_wide(path: Path) -> pd.DataFrame:
    """Load PAR wide table with datetime index and city columns."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if df.index.isna().any():
        raise ValueError("PAR file index must be parseable datetime values")

    if len(df.columns) == 0:
        raise ValueError("PAR file has no city columns")

    return df.sort_index()


def load_region_city_map(path: Path) -> Dict[str, list[str]]:
    """Load region-to-city mapping from JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Region map must be a JSON object")

    parsed: Dict[str, list[str]] = {}
    for region, cities in data.items():
        if not isinstance(cities, list) or not all(isinstance(c, str) for c in cities):
            raise ValueError(f"Region '{region}' must map to a list of city names")
        parsed[region] = cities
    return parsed


def load_aef_by_region(aef_dir: Path, regions: Iterable[str], aef_column: str = "FLOW_UNIT_FINAL_AEF") -> Dict[str, pd.Series]:
    """Load region AEF time series from region CSV files."""
    aef_dir = Path(aef_dir)
    result: Dict[str, pd.Series] = {}

    for region in regions:
        file_path = aef_dir / f"{region}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"AEF file not found: {file_path}")

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if aef_column in df.columns:
            series = df[aef_column]
        elif "AEF" in df.columns:
            series = df["AEF"]
        else:
            raise ValueError(f"AEF column '{aef_column}' not found in {file_path}")

        result[region] = pd.to_numeric(series, errors="coerce").sort_index()

    return result


@lru_cache(maxsize=1)
def _load_city_representative_points() -> Dict[str, tuple[float, float]]:
    """Load representative lon/lat points for Taiwan counties/cities."""
    import geopandas as gpd

    from streetlight.paths import county_shapefile_path

    gdf = gpd.read_file(county_shapefile_path())
    if "COUNTYNAME" not in gdf.columns:
        raise KeyError("County shapefile missing COUNTYNAME column")

    points = gdf[["COUNTYNAME", "geometry"]].copy()
    points["representative_point"] = points.geometry.representative_point()
    return {
        str(row["COUNTYNAME"]): (
            float(row["representative_point"].x),
            float(row["representative_point"].y),
        )
        for _, row in points.iterrows()
    }


def _to_taiwan_local_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert an index to Taiwan local time; naive timestamps are treated as UTC."""
    if index.tz is None:
        return index.tz_localize("UTC").tz_convert(TAIWAN_TZ)
    return index.tz_convert(TAIWAN_TZ)


def _solar_zenith_deg_for_city(
    *,
    index: pd.DatetimeIndex,
    latitude_deg: float,
    longitude_deg: float,
) -> np.ndarray:
    """Approximate solar zenith angle using the NOAA solar position equations."""
    local_index = _to_taiwan_local_index(index)
    day_of_year = local_index.dayofyear.to_numpy(dtype=float)
    minutes = (
        local_index.hour.to_numpy(dtype=float) * 60.0
        + local_index.minute.to_numpy(dtype=float)
        + local_index.second.to_numpy(dtype=float) / 60.0
    )
    fractional_hour = minutes / 60.0
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (fractional_hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude_deg - 60.0 * 8.0
    true_solar_time = minutes + time_offset
    hour_angle_deg = (true_solar_time / 4.0) - 180.0
    hour_angle_deg = ((hour_angle_deg + 180.0) % 360.0) - 180.0

    latitude_rad = np.deg2rad(latitude_deg)
    hour_angle_rad = np.deg2rad(hour_angle_deg)
    cos_zenith = (
        np.sin(latitude_rad) * np.sin(decl)
        + np.cos(latitude_rad) * np.cos(decl) * np.cos(hour_angle_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_zenith))


def _build_city_switching_mask(
    *,
    index: pd.DatetimeIndex,
    city: str,
    load_mode: str,
    lighting_threshold_par: float,
    par_series: pd.Series,
    solar_zenith_deg: float,
) -> pd.Series:
    """Build an on/off mask for the configured streetlight switching rule."""
    if load_mode == "constant":
        return pd.Series(1.0, index=index, dtype=float)
    if load_mode == "par_threshold":
        if lighting_threshold_par < 0:
            raise ValueError("lighting_threshold_par must be non-negative")
        return (par_series <= lighting_threshold_par).astype(float)
    if load_mode == "solar_zenith":
        coords = _load_city_representative_points()
        if city not in coords:
            raise KeyError(f"City '{city}' not found in county representative-point lookup")
        longitude_deg, latitude_deg = coords[city]
        zenith = _solar_zenith_deg_for_city(
            index=index,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
        )
        return pd.Series((zenith >= solar_zenith_deg).astype(float), index=index, dtype=float)
    raise ValueError("load_mode must be one of: par_threshold, solar_zenith, constant")


def align_time_indices(
    par_df: pd.DataFrame,
    aef_by_region: Dict[str, pd.Series],
    strategy: str = "ffill"
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, any]]:
    """
    Align PAR and AEF time indices with specified strategy.

    Parameters
    ----------
    par_df : pd.DataFrame
        PAR data with datetime index
    aef_by_region : Dict[str, pd.Series]
        AEF series by region
    strategy : str
        Alignment strategy: 'intersection', 'strict', 'ffill', 'nearest' (default: 'ffill')
        - intersection: keep only timestamps present in PAR and all regional AEF
          series, then drop timestamps where any regional AEF value is NaN
        - strict: alias for intersection
        - ffill: forward-fill then backward-fill AEF to match PAR index
        - nearest: use nearest-neighbor interpolation

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, any]]
        - Aligned PAR DataFrame (same as input)
        - Aligned AEF dictionary
        - Statistics dictionary with alignment info
    """
    strategy = "intersection" if strategy == "strict" else strategy
    par_index = par_df.index
    source_dt_hours = _infer_dt_hours(par_index)
    stats = {
        "par_start": par_index.min(),
        "par_end": par_index.max(),
        "par_count": len(par_index),
        "strategy": strategy,
        "regions": {},
    }

    aligned_aef: Dict[str, pd.Series] = {}

    if strategy == "intersection":
        common = par_index
        for aef_series in aef_by_region.values():
            common = common.intersection(aef_series.index)
        common = common.sort_values()
        if len(common) == 0:
            aef_ranges = {
                region: (series.index.min(), series.index.max())
                for region, series in aef_by_region.items()
            }
            raise ValueError(
                "No common time overlap between PAR and all AEF regions. "
                f"PAR: {par_index.min()} to {par_index.max()}, AEF ranges: {aef_ranges}"
            )

        aef_panel = pd.concat(
            {
                region: pd.to_numeric(series.reindex(common), errors="coerce")
                for region, series in aef_by_region.items()
            },
            axis=1,
        )
        valid_common = common[~aef_panel.isna().any(axis=1)]
        dropped_nan_count = int(len(common) - len(valid_common))
        if len(valid_common) == 0:
            raise ValueError("No valid common PAR/AEF timestamps remain after dropping AEF NaNs")

        aligned_par = par_df.reindex(valid_common)
        aligned_par.attrs["source_dt_hours"] = source_dt_hours
        for region, aef_series in aef_by_region.items():
            aef_index = aef_series.index
            stats["regions"][region] = {
                "aef_start": aef_index.min(),
                "aef_end": aef_index.max(),
                "aef_count": len(aef_index),
                "overlap_count": int(len(common)),
                "filled_count": 0,
                "dropped_nan_count": dropped_nan_count,
            }
            aligned_aef[region] = pd.to_numeric(aef_series.reindex(valid_common), errors="coerce")

        stats["intersection_count"] = int(len(valid_common))
        stats["dropped_nan_count"] = dropped_nan_count
        logger.info(
            "Time alignment complete: kept %d common PAR/AEF timestamps, dropped %d AEF-NaN timestamps",
            len(valid_common),
            dropped_nan_count,
        )
        return aligned_par, aligned_aef, stats

    for region, aef_series in aef_by_region.items():
        aef_index = aef_series.index

        region_stats = {
            "aef_start": aef_index.min(),
            "aef_end": aef_index.max(),
            "aef_count": len(aef_index),
            "overlap_count": len(par_index.intersection(aef_index)),
            "filled_count": 0,
        }

        if strategy == "ffill":
            aligned = aef_series.reindex(par_index).ffill().bfill()
            filled = aligned.isna().sum()
            if filled > 0:
                raise ValueError(
                    f"AEF for region '{region}' has {filled} unresolved NaN values after ffill/bfill"
                )
            region_stats["filled_count"] = len(par_index) - region_stats["overlap_count"]
            aligned_aef[region] = aligned
        elif strategy == "nearest":
            aligned = aef_series.reindex(par_index, method="nearest")
            aligned_aef[region] = aligned.ffill().bfill()
        else:
            raise ValueError(f"Unknown alignment strategy: {strategy}")

        stats["regions"][region] = region_stats

        # Log alignment info
        logger.info(
            f"Region '{region}': {region_stats['overlap_count']} exact matches, "
            f"{region_stats['filled_count']} filled ({strategy})"
        )

    # Log summary
    logger.info(
        f"Time alignment complete: PAR [{stats['par_start']} to {stats['par_end']}] "
        f"with strategy '{strategy}'"
    )

    par_df.attrs["source_dt_hours"] = source_dt_hours
    return par_df, aligned_aef, stats


def _resolve_installation_load_kw(
    *,
    n_lights: int,
    light_power_kw: float,
) -> float:
    """Resolve installation-level load from the standardized light-count definition."""
    if n_lights <= 0:
        raise ValueError("n_lights must be positive")
    if light_power_kw <= 0:
        raise ValueError("light_power_kw must be positive")
    return float(n_lights * light_power_kw)


def _build_load_series_kw(
    *,
    par_series: pd.Series,
    city: str,
    installation_load_kw: float,
    load_mode: str,
    lighting_threshold_par: float,
    solar_zenith_deg: float,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Build load series from PAR using the configured lighting rule."""
    on_mask = _build_city_switching_mask(
        index=index,
        city=city,
        load_mode=load_mode,
        lighting_threshold_par=lighting_threshold_par,
        par_series=par_series,
        solar_zenith_deg=solar_zenith_deg,
    )
    return on_mask * installation_load_kw


def simulate_from_par_and_aef(
    par_df: pd.DataFrame,
    aef_by_region: Dict[str, pd.Series],
    region_city_map: Dict[str, list[str]],
    par_to_kw_factor: float,
    storage_params: Dict[str, float],
    load_mode: str = "solar_zenith",
    align_strategy: str = "ffill",
    n_lights: int = 64,
    light_power_kw: float = 0.1,
    analysis_years: Optional[float] = None,
    lighting_threshold_par: float = 1.0,
    solar_zenith_deg: float = 90.833,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Simulate city/region metrics for PV+storage streetlight operation.

    Parameters
    ----------
    par_df : pd.DataFrame
        PAR time series with datetime index and city columns
    aef_by_region : Dict[str, pd.Series]
        AEF time series by region
    region_city_map : Dict[str, list[str]]
        Mapping from region names to city lists
    par_to_kw_factor : float
        Conversion factor from PAR values to kW
    storage_params : Dict[str, float]
        Battery storage parameters
    load_mode : str
        'solar_zenith' (load when solar zenith >= threshold),
        'par_threshold' (load when PAR <= threshold), or 'constant' (always on)
    align_strategy : str
        Time alignment strategy: 'strict', 'ffill', or 'nearest'
    n_lights : int
        Number of lights in the standardized installation
    light_power_kw : float
        Rated power per light in kW
    analysis_years : float | None
        If provided, scale modeled energy/emission totals to the specified
        analysis horizon in years. If None, totals reflect only the modeled
        time window in par_df.
    lighting_threshold_par : float
        PAR threshold below which the lights are considered on when
        load_mode = 'par_threshold'.
    solar_zenith_deg : float
        Solar zenith threshold in degrees used when load_mode = 'solar_zenith'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]
        - City-level metrics DataFrame
        - Region-level aggregated DataFrame
        - Summary statistics dictionary
    """
    if par_to_kw_factor <= 0:
        raise ValueError("par_to_kw_factor must be positive")
    if load_mode not in {"par_threshold", "solar_zenith", "constant"}:
        raise ValueError("load_mode must be one of: par_threshold, solar_zenith, constant")
    if analysis_years is not None and analysis_years <= 0:
        raise ValueError("analysis_years must be positive when provided")
    if lighting_threshold_par < 0:
        raise ValueError("lighting_threshold_par must be non-negative")
    if solar_zenith_deg <= 0 or solar_zenith_deg >= 180:
        raise ValueError("solar_zenith_deg must be between 0 and 180")

    installation_load_kw = _resolve_installation_load_kw(
        n_lights=n_lights,
        light_power_kw=light_power_kw,
    )

    # Align time indices
    par_df, aef_by_region, align_stats = align_time_indices(
        par_df, aef_by_region, strategy=align_strategy
    )

    dt_h, modeled_hours, analysis_scaling_factor = _compute_analysis_scaling_factor(
        par_df.index,
        analysis_years,
        dt_hours_override=par_df.attrs.get("source_dt_hours"),
    )
    records: list[dict] = []

    for region, cities in region_city_map.items():
        if region not in aef_by_region:
            raise ValueError(f"Missing AEF series for region: {region}")

        aef = aef_by_region[region]
        if aef.isna().any():
            raise ValueError(f"AEF for region '{region}' contains unresolved NaN values")

        for city in cities:
            if city not in par_df.columns:
                raise ValueError(f"City '{city}' not found in PAR columns")

            par_numeric = pd.to_numeric(par_df[city], errors="coerce")
            n_missing = par_numeric.isna().sum()
            if n_missing > 0:
                logger.warning(
                    "City '%s': %d/%d PAR timesteps are NaN and will be treated as zero irradiance",
                    city, n_missing, len(par_numeric),
                )
            par_series = par_numeric.fillna(0.0)
            pv_kw = par_series * par_to_kw_factor
            load_series_kw = _build_load_series_kw(
                par_series=par_series,
                city=city,
                installation_load_kw=installation_load_kw,
                load_mode=load_mode,
                lighting_threshold_par=lighting_threshold_par,
                solar_zenith_deg=solar_zenith_deg,
                index=par_df.index,
            )
            net_kw = pv_kw - load_series_kw

            dispatch = storage_dispatch(net_kw=net_kw, dt_hours=dt_h, **storage_params)
            grid_import_kw = dispatch["grid_import_kw"].clip(lower=0.0)
            deficit_kw = (-net_kw).clip(lower=0.0)
            battery_discharge_kw = (deficit_kw - grid_import_kw).clip(lower=0.0)
            pv_direct_kw = (load_series_kw - battery_discharge_kw - grid_import_kw).clip(lower=0.0)

            total_load_kwh = float((load_series_kw * dt_h).sum()) * analysis_scaling_factor
            pv_generation_kwh = float((pv_kw * dt_h).sum()) * analysis_scaling_factor
            pv_direct_kwh = float((pv_direct_kw * dt_h).sum()) * analysis_scaling_factor
            battery_discharge_kwh = float((battery_discharge_kw * dt_h).sum()) * analysis_scaling_factor
            grid_import_kwh = float((grid_import_kw * dt_h).sum()) * analysis_scaling_factor

            baseline_emission_kg = float(((load_series_kw * dt_h) * aef).sum()) * analysis_scaling_factor
            scenario_emission_kg = float(((grid_import_kw * dt_h) * aef).sum()) * analysis_scaling_factor
            emission_reduction_kg = baseline_emission_kg - scenario_emission_kg

            if total_load_kwh > 0:
                self_supply_ratio = (pv_direct_kwh + battery_discharge_kwh) / total_load_kwh
                grid_ratio = grid_import_kwh / total_load_kwh
            else:
                self_supply_ratio = 0.0
                grid_ratio = 0.0

            records.append(
                {
                    "region": region,
                    "city": city,
                    "dt_hours": dt_h,
                    "functional_unit_lights": int(n_lights),
                    "light_power_kw": float(light_power_kw),
                    "installation_load_kw": installation_load_kw,
                    "par_to_kw_factor": float(par_to_kw_factor),
                    "load_mode": load_mode,
                    "lighting_threshold_par": float(lighting_threshold_par),
                    "solar_zenith_deg": float(solar_zenith_deg),
                    "analysis_years": float(analysis_years) if analysis_years is not None else np.nan,
                    "modeled_hours": modeled_hours,
                    "analysis_scaling_factor": analysis_scaling_factor,
                    "deployment_units": 1,
                    "battery_capacity_kwh": float(storage_params.get("capacity_kwh", np.nan)),
                    "battery_power_kw": float(storage_params.get("power_kw", np.nan)),
                    "total_load_kwh": total_load_kwh,
                    "pv_generation_kwh": pv_generation_kwh,
                    "pv_direct_kwh": pv_direct_kwh,
                    "battery_discharge_kwh": battery_discharge_kwh,
                    "grid_import_kwh": grid_import_kwh,
                    "self_supply_ratio": self_supply_ratio,
                    "grid_ratio": grid_ratio,
                    "baseline_emission_kg": baseline_emission_kg,
                    "scenario_emission_kg": scenario_emission_kg,
                    "emission_reduction_kg": emission_reduction_kg,
                }
            )

    city_df = pd.DataFrame.from_records(records).sort_values(["region", "city"]).reset_index(drop=True)

    region_df = (
        city_df.groupby("region", as_index=False)[
            [
                "total_load_kwh",
                "pv_generation_kwh",
                "pv_direct_kwh",
                "battery_discharge_kwh",
                "grid_import_kwh",
                "baseline_emission_kg",
                "scenario_emission_kg",
                "emission_reduction_kg",
            ]
        ]
        .sum()
    )
    region_df["self_supply_ratio"] = (
        (region_df["pv_direct_kwh"] + region_df["battery_discharge_kwh"]) / region_df["total_load_kwh"]
    )
    region_df["grid_ratio"] = region_df["grid_import_kwh"] / region_df["total_load_kwh"]
    region_df["functional_unit_lights"] = int(n_lights)
    region_df["light_power_kw"] = float(light_power_kw)
    region_df["installation_load_kw"] = installation_load_kw
    region_df["par_to_kw_factor"] = float(par_to_kw_factor)
    region_df["load_mode"] = load_mode
    region_df["lighting_threshold_par"] = float(lighting_threshold_par)
    region_df["solar_zenith_deg"] = float(solar_zenith_deg)
    region_df["analysis_years"] = float(analysis_years) if analysis_years is not None else np.nan
    region_df["modeled_hours"] = modeled_hours
    region_df["analysis_scaling_factor"] = analysis_scaling_factor
    region_df["deployment_units"] = 1
    region_df["battery_capacity_kwh"] = float(storage_params.get("capacity_kwh", np.nan))
    region_df["battery_power_kw"] = float(storage_params.get("power_kw", np.nan))

    summary = {
        "n_regions": int(region_df.shape[0]),
        "n_cities": int(city_df.shape[0]),
        "functional_unit_lights": int(n_lights),
        "light_power_kw": float(light_power_kw),
        "installation_load_kw": installation_load_kw,
        "par_to_kw_factor": float(par_to_kw_factor),
        "load_mode": load_mode,
        "lighting_threshold_par": float(lighting_threshold_par),
        "solar_zenith_deg": float(solar_zenith_deg),
        "analysis_years": float(analysis_years) if analysis_years is not None else np.nan,
        "modeled_hours": modeled_hours,
        "analysis_scaling_factor": analysis_scaling_factor,
        "deployment_units": 1,
        "battery_capacity_kwh": float(storage_params.get("capacity_kwh", np.nan)),
        "battery_power_kw": float(storage_params.get("power_kw", np.nan)),
        "total_load_kwh": float(city_df["total_load_kwh"].sum()),
        "total_grid_import_kwh": float(city_df["grid_import_kwh"].sum()),
        "total_baseline_emission_kg": float(city_df["baseline_emission_kg"].sum()),
        "total_scenario_emission_kg": float(city_df["scenario_emission_kg"].sum()),
        "total_emission_reduction_kg": float(city_df["emission_reduction_kg"].sum()),
    }

    return city_df, region_df, summary


def save_simulation_outputs(city_df: pd.DataFrame, region_df: pd.DataFrame, summary: Dict[str, float], output_dir: Path) -> None:
    """Save simulation outputs to CSV/JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    city_df.to_csv(output_dir / "city_metrics.csv", index=False)
    region_df.to_csv(output_dir / "region_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _infer_dt_hours(index: pd.DatetimeIndex) -> float:
    """Infer timestep (hours) from datetime index using timedelta arithmetic."""
    if len(index) <= 1:
        return 1.0
    deltas = index.to_series().diff().dropna()
    dt_h = deltas.median().total_seconds() / 3600.0
    if dt_h <= 0:
        raise ValueError("PAR datetime index must be strictly increasing")
    return float(dt_h)


def _compute_analysis_scaling_factor(
    index: pd.DatetimeIndex,
    analysis_years: Optional[float],
    dt_hours_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return timestep hours, modeled hours, and scaling factor to target horizon."""
    dt_h = float(dt_hours_override) if dt_hours_override is not None else _infer_dt_hours(index)
    modeled_hours = float(len(index) * dt_h)
    if modeled_hours <= 0:
        raise ValueError("Modeled time span must be positive")
    scaling_factor = 1.0
    if analysis_years is not None:
        scaling_factor = (float(analysis_years) * 365.0 * 24.0) / modeled_hours
    return dt_h, modeled_hours, scaling_factor


def run_parameter_sweep(
    par_df: pd.DataFrame,
    aef_by_region: Dict[str, pd.Series],
    region_city_map: Dict[str, list[str]],
    par_to_kw_factor: float,
    solar_range: Optional[List[float]] = None,
    battery_range: Optional[List[float]] = None,
    load_mode: str = "solar_zenith",
    analysis_years: float = 20.0,
    electricity_price: float = 3.7556,
    n_lights: int = 64,
    light_power_kw: float = 0.1,
    lighting_threshold_par: float = 1.0,
    solar_zenith_deg: float = 90.833,
    align_strategy: str = "ffill",
    base_capacity_kwh: float = 10.0,
    base_power_kw: float = 5.0,
    eta_roundtrip: float = 0.90,
    economic_costs: Optional[dict] = None,
    financial_params: Optional[dict] = None,
    battery_power_cap_multiplier_of_load: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run parameter sweep over storage and PV capacities for Pareto analysis.

    Uses the same cost model as compute_city_emission from pareto.py.

    Args:
        par_df: PAR time series DataFrame
        aef_by_region: Region AEF series mapping
        region_city_map: Region to city name mapping
        par_to_kw_factor: Base conversion factor from PAR to kW
        solar_range: List of solar panel factors to test
        battery_range: List of battery factors to test
        load_mode: Load calculation mode ("solar_zenith", "par_threshold", or "constant")
        analysis_years: Analysis period in years
        electricity_price: Grid electricity price (NTD/kWh)
        n_lights: Number of lights in the standardized installation
        light_power_kw: Rated power per light in kW
        lighting_threshold_par: PAR threshold for 'par_threshold' load mode
        solar_zenith_deg: Solar zenith threshold for 'solar_zenith' load mode
        align_strategy: Time alignment strategy for AEF-to-PAR matching
        base_capacity_kwh: Base battery capacity at battery_factor = 1.0
        base_power_kw: Base battery power at battery_factor = 1.0
        eta_roundtrip: Battery round-trip efficiency
        economic_costs: Optional cost override mapping for scenario analysis
        financial_params: Optional financial override mapping for discounting/salvage
        battery_power_cap_multiplier_of_load: Optional cap on battery power,
            expressed as a multiple of installation_load_kw

    Returns:
        DataFrame with all configuration results for Pareto/MACC analysis
    """
    if solar_range is None:
        solar_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    if battery_range is None:
        battery_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]

    installation_load_kw = _resolve_installation_load_kw(
        n_lights=n_lights,
        light_power_kw=light_power_kw,
    )
    if (
        battery_power_cap_multiplier_of_load is not None
        and float(battery_power_cap_multiplier_of_load) <= 0
    ):
        raise ValueError("battery_power_cap_multiplier_of_load must be positive when provided")
    power_cap_kw = (
        None
        if battery_power_cap_multiplier_of_load is None
        else float(battery_power_cap_multiplier_of_load) * installation_load_kw
    )

    par_df, aef_by_region, _ = align_time_indices(
        par_df,
        aef_by_region,
        strategy=align_strategy,
    )
    dt_h, modeled_hours, analysis_scaling_factor = _compute_analysis_scaling_factor(
        par_df.index,
        analysis_years,
        dt_hours_override=par_df.attrs.get("source_dt_hours"),
    )
    resolved_costs = _resolve_economic_costs(economic_costs)
    resolved_financial = _resolve_financial_params(financial_params)
    city_contexts: List[tuple[str, str, pd.Series, pd.Series, pd.Series]] = []

    for region, cities in region_city_map.items():
        if region not in aef_by_region:
            raise ValueError(f"Missing AEF series for region: {region}")

        aef = aef_by_region[region]
        if aef.isna().any():
            raise ValueError(f"AEF for region '{region}' contains unresolved NaN values")

        for city in cities:
            if city not in par_df.columns:
                raise ValueError(f"City '{city}' not found in PAR columns")

            par_numeric = pd.to_numeric(par_df[city], errors="coerce")
            n_missing = par_numeric.isna().sum()
            if n_missing > 0:
                logger.warning(
                    "City '%s': %d/%d PAR timesteps are NaN and will be treated as zero irradiance",
                    city, n_missing, len(par_numeric),
                )
            par_series = par_numeric.fillna(0.0)
            base_pv_kw = par_series * par_to_kw_factor
            load_series_kw = _build_load_series_kw(
                par_series=par_series,
                city=city,
                installation_load_kw=installation_load_kw,
                load_mode=load_mode,
                lighting_threshold_par=lighting_threshold_par,
                solar_zenith_deg=solar_zenith_deg,
                index=par_df.index,
            )
            city_contexts.append((region, city, aef, base_pv_kw, load_series_kw))

    results: List[dict] = []

    for solar_panel_factor in solar_range:
        for battery_factor in battery_range:
            # Scale storage parameters based on battery_factor
            requested_power_kw = base_power_kw * battery_factor
            storage_params = {
                "capacity_kwh": base_capacity_kwh * battery_factor,
                "power_kw": requested_power_kw if power_cap_kw is None else min(requested_power_kw, power_cap_kw),
                "eta_roundtrip": eta_roundtrip,
            }

            for region, city, aef, base_pv_kw, load_series_kw in city_contexts:
                net_kw = (base_pv_kw * solar_panel_factor) - load_series_kw

                result = compute_city_emission(
                    city_power_kw=net_kw,
                    aef=aef,
                    dt_hours=dt_h,
                    years=analysis_years,
                    solar_panel_factor=solar_panel_factor,
                    battery_factor=battery_factor,
                    elec_price=electricity_price,
                    params=storage_params,
                    analysis_scaling_factor=analysis_scaling_factor,
                    _resolved_costs=resolved_costs,
                    _resolved_financial=resolved_financial,
                )

                result["region"] = region
                result["city"] = city
                result["dt_hours"] = dt_h
                result["functional_unit_lights"] = int(n_lights)
                result["light_power_kw"] = float(light_power_kw)
                result["installation_load_kw"] = installation_load_kw
                result["par_to_kw_factor"] = float(par_to_kw_factor)
                result["load_mode"] = load_mode
                result["lighting_threshold_par"] = float(lighting_threshold_par)
                result["solar_zenith_deg"] = float(solar_zenith_deg)
                result["modeled_hours"] = modeled_hours
                result["analysis_scaling_factor"] = analysis_scaling_factor
                result["deployment_units"] = 1
                result["analysis_years"] = float(analysis_years)
                result["solar_panel_factor"] = solar_panel_factor
                result["battery_factor"] = battery_factor
                result["battery_capacity_kwh"] = storage_params["capacity_kwh"]
                result["battery_power_kw"] = storage_params["power_kw"]
                result["requested_battery_power_kw"] = requested_power_kw
                result["battery_power_cap_multiplier_of_load"] = (
                    float(battery_power_cap_multiplier_of_load)
                    if battery_power_cap_multiplier_of_load is not None
                    else np.nan
                )
                result["abatement_t"] = -result["delta_t"]  # Positive = reduction

                results.append(result)

    return pd.DataFrame(results)
