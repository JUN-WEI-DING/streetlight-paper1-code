"""Future-grid scenarios v2 — generation-by-fuel scaling under official policy targets.

This script supersedes the methodology of ``run_future_grid_scenarios.py`` (v1).
v1 multiplies the *current* regional AEF time series by a scalar to hit a
target mean AEF; that approach silently flattens out the structural fuel-mix
shift and breaks the time-shape integrity of the storage / inter-regional
flow physics.

v2 instead:

1. Loads unit-level generation per region from ``data/power/<region>_unit_generation.csv``
2. Aggregates to fuel categories via ``AEFCalculator.aggregate_by_fuel``,
   including the wind unit classification table that separates onshore
   ``Wind`` from ``Offshore Wind``.
3. Computes a *system-wide* per-fuel scaling factor ``x_f = t_f / s_f``
   where ``t_f`` is the target annual share and ``s_f`` the baseline share.
4. Applies that scalar uniformly to every region & every timestep —
   preserving the per-fuel diurnal/seasonal shape AND the regional
   distribution shape. NO post-scaling per-region renormalisation is
   performed: with ``Sum(s_f * x_f) = Sum(t_f) = 1`` the scaled total
   already matches baseline demand within numerical noise.
5. Differentiates the renewable bucket into Solar / Offshore Wind /
   Onshore Wind / Hydro / Biomass+Geothermal sub-targets per Taiwan policy
   (2030 MOEA "532" + 2050 net-zero pathway).
6. Phase-out fuels (Coal in 2050, Nuclear and Oil in 2030 & 2050) get
   ``x_f = 0`` — their freed share is reallocated to the *growing*
   renewables and LNG via the explicit target shares, not by ad-hoc
   redistribution.
7. Splits combined Biomass+Geothermal targets back to the separate model
   columns using their 2024 generation weights, and injects only fuels not
   present in the baseline (Hydrogen for 2050, plus fallback Offshore Wind only
   if no classified 2024 offshore series exists). Time-shape proportional to a
   chosen carrier (current onshore Wind for fallback offshore wind, total demand
   for injected biomass/hydrogen). Regional distribution proportional to a
   sensible carrier (current Wind for fallback offshore wind, current Hydro for
   injected biomass, current LNG for hydrogen — gas peaker replacement
   assumption).
8. For 2050 only, overrides the LNG emission factor to a CCUS lifecycle
   value (UNECE 2021 LCA midpoint).
9. Runs the existing ``AEFPipeline`` end-to-end (storage pool +
   distributed fuel allocator + inter-regional flow) on the modified
   generation data, producing recomputed regional AEF CSVs.
10. Re-runs the streetlight Pareto sweep against the new AEF series.

Renewable sub-targets (% of total system generation):

* **2030** (renewable bucket = 30%):
    Solar 15.0, Offshore Wind 7.5, Onshore Wind 3.0, Hydro 3.0,
    Biomass+Geothermal 1.5.
* **2050** (renewable bucket = 65%):
    Solar 29.25, Offshore Wind 24.7, Onshore Wind 3.25, Hydro 4.55,
    Biomass+Geothermal 3.25.

Citations:

* ``taiwan_2030_532_target``: Taiwan MOEA "532" plan — 30% renewables,
  30 GW solar, 13.1 GW offshore wind by 2030.
* ``taiwan_2050_netzero_pathway``: Taiwan 2050 net-zero pathway —
  60-70% renewables, 40-80 GW solar, 40-55 GW offshore wind by 2050.
* ``unece2021lca``: UNECE 2021 LCA — NGCC + CCS lifecycle EF 0.092-0.220
  kg CO2e/kWh; midpoint 0.156 used.
* ``ipcc_ar6_wg3_ch6``: IPCC AR6 WG3 Ch6 sectoral framing.

Outputs::

    outputs/paper_baseline/future_grid_scenarios/
        official_2030/
            aef/{north,central,south,east,island_penghu,island_kinmen,island_lienchiang}.csv
            aef/storage_pools.csv
            results.csv, portfolio.csv, frontier.csv, macc.csv,
            knee.json, city_metrics.csv
        pathway_2050/  (same shape)
        scenario_summary.csv
        scenario_metadata.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from streetlight.aef.calculator import AEFCalculator
from streetlight.aef.pipeline import AEFPipeline
from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    load_aef_by_region,
    load_par_wide,
    load_region_city_map,
    run_parameter_sweep,
)
from streetlight.simulation.storage import (
    build_city_metrics,
    compute_macc_from_frontier,
    find_knee_point,
    pareto_frontier_min_cost_max_abatement,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


MAINLAND_REGIONS = ("north", "central", "south", "east")
TRANSFER_COLS = {"BESS", "PHS", "load", "pumping_gen", "pumping_load",
                 "storage", "storage_load"}
FLOW_PREFIX = "F_"

# ---- Policy targets (annual fuel share, % of total system generation) ----
# Renewable bucket is decomposed into per-fuel sub-targets so that the
# realised mix reflects Taiwan policy emphasis (Solar + Offshore Wind),
# not the baseline split (Hydro/Wind dominant, Solar tiny).

POLICY_2030 = {
    "name": "official_2030_grid_target",
    "label": "Taiwan MOEA '532' 2030 target (Solar + Offshore Wind led)",
    "citations": ["taiwan_2030_532_target"],
    # Per-fuel ANNUAL system-wide share targets (sum = 100%).
    "shares_pct": {
        "Coal":          20.0,
        "LNG":           50.0,
        "Solar":         15.0,
        "Offshore Wind":  7.5,
        "Wind":           3.0,   # baseline "Wind" treated as onshore wind
        "Hydro":          3.0,
        # Phase-outs — explicit zero target.
        "Nuclear":        0.0,
        "Oil":            0.0,
        "Diesel":         0.0,
    },
    "combined_targets_pct": {
        "Biomass+Geothermal": 1.5,
    },
    "ef_overrides": {},
    "introduce_hydrogen_share_pct": 0.0,
    # Sub-target metadata for transparency / metadata JSON.
    "renewable_sub_targets_pct_of_total": {
        "Solar": 15.0,
        "Offshore Wind": 7.5,
        "Wind (onshore)": 3.0,
        "Hydro": 3.0,
        "Biomass+Geothermal": 1.5,
    },
}

POLICY_2050 = {
    "name": "pathway_2050_midpoint_proxy",
    "label": "Taiwan 2050 net-zero pathway midpoint (Solar + Offshore Wind led)",
    "citations": ["taiwan_2050_netzero_pathway", "unece2021lca", "ipcc_ar6_wg3_ch6"],
    "shares_pct": {
        # Coal/Nuclear/Oil/Diesel all phased out.
        "Coal":           0.0,
        "LNG":           24.5,   # LNG+CCUS — see note below for budget
        "Solar":         29.25,
        "Offshore Wind": 24.7,
        "Wind":           3.25,
        "Hydro":          4.55,
        "Nuclear":        0.0,
        "Oil":            0.0,
        "Diesel":         0.0,
    },
    "combined_targets_pct": {
        "Biomass+Geothermal": 3.25,
    },
    "ef_overrides": {
        # LNG+CCUS lifecycle EF — UNECE 2021 LCA midpoint of 0.092-0.220.
        "LNG": 0.156,
    },
    # Hydrogen is NOT a primary fuel in baseline; injected as a new column
    # with target share. EF = 0 (imported green hydrogen assumption).
    "introduce_hydrogen_share_pct": 10.5,
    "renewable_sub_targets_pct_of_total": {
        "Solar": 29.25,
        "Offshore Wind": 24.7,
        "Wind (onshore)": 3.25,
        "Hydro": 4.55,
        "Biomass+Geothermal": 3.25,
    },
}
# 2050 budget: Renewables 65 + LNG 24.5 + Hydrogen 10.5 = 100%.

# Fuels we treat as "new injections" — not present in current Taiwan
# unit-generation data, must be created with a chosen time- and regional
# shape rather than scaled.
NEW_FUELS_DEFAULT = ("Biomass",)
NEW_FUELS_2050_EXTRA = ("Hydrogen",)
COMBINED_TARGET_MEMBERS = {
    "Biomass+Geothermal": ("Biomass", "Geothermal"),
}


# ---- Loading helpers --------------------------------------------------------

def _load_raw_unit_generation(power_dir: Path, region: str) -> pd.DataFrame:
    path = power_dir / f"{region}_unit_generation.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def _split_unit_df(df: pd.DataFrame, calc: AEFCalculator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a raw unit-level CSV into (fuel-aggregated generation, transfer/storage)."""
    transfer_cols = [c for c in df.columns
                     if c.startswith(FLOW_PREFIX) or c in TRANSFER_COLS]
    gen_cols = [c for c in df.columns if c not in transfer_cols]
    gen_df = df[gen_cols].copy()
    transfer_df = df[transfer_cols].copy() if transfer_cols else pd.DataFrame(index=df.index)
    # Aggregate units to fuel categories (e.g. "LNG-Datan-CC#1" -> "LNG")
    fuel_df = calc.aggregate_by_fuel(gen_df)
    return fuel_df, transfer_df


# ---- Bookkeeping ------------------------------------------------------------

def _annual_fuel_totals(per_region_fuel: Dict[str, pd.DataFrame],
                        ef_known: Dict[str, float]) -> Dict[str, float]:
    """System-wide annual generation totals, restricted to EF-known fuels.

    The scenario builder preserves the AEF pipeline's 10-minute average-power
    sample scale. Fuel shares, scaling factors, AEFs, and retention results are
    invariant to multiplying every generation series by the constant timestep
    factor; legacy metadata keys keep ``*_mwh`` names for downstream readers.
    """
    totals: Dict[str, float] = {}
    for fuel_df in per_region_fuel.values():
        for col in fuel_df.columns:
            if col not in ef_known:
                continue
            v = float(pd.to_numeric(fuel_df[col], errors="coerce").clip(lower=0).sum())
            totals[col] = totals.get(col, 0.0) + v
    return totals


def _system_total(totals: Dict[str, float]) -> float:
    return float(sum(totals.values()))


def _per_region_fuel_totals(
    per_region_fuel: Dict[str, pd.DataFrame],
    fuel: str,
) -> Dict[str, float]:
    """Per-region annual total for a fuel (zero if absent)."""
    out: Dict[str, float] = {}
    for region, df in per_region_fuel.items():
        if fuel in df.columns:
            v = float(pd.to_numeric(df[fuel], errors="coerce").clip(lower=0).sum())
        else:
            v = 0.0
        out[region] = v
    return out


# ---- Core scaling logic -----------------------------------------------------

def _per_fuel_scaling_factors(
    base_totals: Dict[str, float],
    target_shares_pct: Dict[str, float],
) -> Dict[str, float]:
    """Compute x_f = t_f / s_f for each fuel that exists in baseline.

    Fuels in ``target_shares_pct`` that have zero baseline generation are
    flagged as new-fuel injections (returned with NaN factor) — caller
    handles these via :func:`_inject_new_fuel`.

    Fuels in baseline but absent from ``target_shares_pct`` get factor 0.0.
    The policy dictionary is treated as the full annual fuel budget for the
    scenario, so legacy residual categories such as Co-Gen are not carried
    forward unless they are explicitly assigned a target share.

    Phase-out targets (target_pct = 0) yield x_f = 0.
    """
    total = _system_total(base_totals)
    if total <= 0:
        raise ValueError("Total baseline generation must be positive")

    factors: Dict[str, float] = {}

    # Baseline-present fuels: standard x = t/s.
    for fuel, baseline in base_totals.items():
        if fuel not in target_shares_pct:
            factors[fuel] = 0.0
            continue
        target_pct = float(target_shares_pct[fuel])
        if baseline <= 0:
            # Should not happen since fuel is in base_totals; guard anyway.
            factors[fuel] = float("nan")
            continue
        cur_pct = 100.0 * baseline / total
        if cur_pct <= 0:
            factors[fuel] = float("nan")
        else:
            factors[fuel] = target_pct / cur_pct

    # Targets that aren't in baseline → mark for new-fuel injection.
    for fuel in target_shares_pct:
        if fuel not in factors and target_shares_pct[fuel] > 0:
            factors[fuel] = float("nan")

    return factors


def _expand_combined_targets(
    target_shares_pct: Dict[str, float],
    combined_targets_pct: Dict[str, float],
    base_totals: Dict[str, float],
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Split combined policy buckets back to model fuel columns.

    Policy reports biomass and geothermal as a combined renewable sub-target.
    The AEF model keeps separate emission factors and observed profiles, so the
    target is split by the baseline generation ratio among member fuels.
    """
    expanded = dict(target_shares_pct)
    records: Dict[str, Dict[str, float]] = {}
    for combined_name, combined_share in combined_targets_pct.items():
        members = COMBINED_TARGET_MEMBERS.get(combined_name)
        if not members:
            raise KeyError(f"Unknown combined target bucket: {combined_name}")

        baseline_sum = sum(max(float(base_totals.get(fuel, 0.0)), 0.0) for fuel in members)
        if baseline_sum > 0:
            weights = {
                fuel: max(float(base_totals.get(fuel, 0.0)), 0.0) / baseline_sum
                for fuel in members
            }
        else:
            weights = {fuel: 1.0 / len(members) for fuel in members}

        split = {fuel: float(combined_share) * weights[fuel] for fuel in members}
        expanded.update(split)
        records[combined_name] = {
            "combined_share_pct": float(combined_share),
            **{f"{fuel}_share_pct": split[fuel] for fuel in members},
            **{f"{fuel}_baseline_weight": weights[fuel] for fuel in members},
        }
    return expanded, records


def _inject_new_fuel(
    fuel: str,
    target_share_pct: float,
    baseline_total_demand_mwh: float,
    target_total_per_t: pd.Series,
    per_region_fuel: Dict[str, pd.DataFrame],
    time_shape_carrier: Optional[str],
    region_dist_carrier: str,
) -> Dict[str, pd.Series]:
    """Inject a new fuel that is not present in baseline generation.

    Parameters
    ----------
    fuel : str
        New fuel column name (e.g. "Offshore Wind", "Hydrogen", "Biomass").
    target_share_pct : float
        Annual share target (% of baseline total demand). New fuel total =
        ``share/100 * baseline_total_demand_mwh``.
    baseline_total_demand_mwh : float
        System-wide annual demand (MWh) — used as the total energy budget
        for the new fuel.
    target_total_per_t : pd.Series
        Per-timestep system demand. Used as fallback time-shape carrier.
    per_region_fuel : dict[str, pd.DataFrame]
        Per-region fuel-aggregated generation (baseline). Source for
        regional-share carrier and time-shape carrier.
    time_shape_carrier : str | None
        Existing fuel whose system-wide diurnal shape the new fuel should
        mimic (e.g. "Wind" for fallback Offshore Wind). If None or not present,
        fall back to ``target_total_per_t`` (flat-ish demand shape).
    region_dist_carrier : str
        Existing fuel whose regional distribution the new fuel inherits
        (e.g. "Wind" for offshore wind, "LNG" for hydrogen). If absent in
        every region, fall back to uniform-by-baseline-demand.

    Returns
    -------
    dict[region, pd.Series]
        Per-region time series of the injected fuel (MWh per timestep).
    """
    target_total_mwh = (target_share_pct / 100.0) * baseline_total_demand_mwh

    # ---- Region weights ----
    carrier_totals = _per_region_fuel_totals(per_region_fuel, region_dist_carrier)
    carrier_sum = sum(carrier_totals.values())
    if carrier_sum > 0:
        region_weights = {r: v / carrier_sum for r, v in carrier_totals.items()}
    else:
        # Fallback: weight by region's annual demand.
        demand_totals = {r: float(df.clip(lower=0).sum().sum())
                         for r, df in per_region_fuel.items()}
        demand_sum = sum(demand_totals.values())
        if demand_sum > 0:
            region_weights = {r: v / demand_sum for r, v in demand_totals.items()}
        else:
            n = max(len(per_region_fuel), 1)
            region_weights = {r: 1.0 / n for r in per_region_fuel}

    # ---- Time shape (system-wide normalised series, sums to 1) ----
    if time_shape_carrier is not None:
        sys_shape = pd.Series(0.0, index=target_total_per_t.index)
        for df in per_region_fuel.values():
            if time_shape_carrier in df.columns:
                s = pd.to_numeric(df[time_shape_carrier], errors="coerce").fillna(0.0).clip(lower=0.0)
                sys_shape = sys_shape.add(s.reindex(sys_shape.index).fillna(0.0), fill_value=0.0)
        if float(sys_shape.sum()) <= 0:
            sys_shape = target_total_per_t.clip(lower=0.0).copy()
    else:
        sys_shape = target_total_per_t.clip(lower=0.0).copy()

    shape_sum = float(sys_shape.sum())
    if shape_sum <= 0:
        # Last resort: uniform.
        sys_shape = pd.Series(1.0, index=target_total_per_t.index)
        shape_sum = float(sys_shape.sum())
    sys_shape_norm = sys_shape / shape_sum  # sums to 1.0

    # System-wide per-timestep injection (MWh).
    sys_inject = sys_shape_norm * target_total_mwh

    # Distribute to regions per region_weights.
    out: Dict[str, pd.Series] = {}
    for region, w in region_weights.items():
        out[region] = (sys_inject * w).rename(fuel)
    return out


def _apply_scaling(
    per_region_fuel: Dict[str, pd.DataFrame],
    scale_factors: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    """Apply x_f uniformly to every (region, timestep) for fuel f.

    NO renormalisation. With Sum_f s_f * x_f = Sum_f t_f = 1, scaled total
    auto-matches baseline total within numerical noise.
    """
    out: Dict[str, pd.DataFrame] = {}
    for region, fuel_df in per_region_fuel.items():
        new_df = fuel_df.copy()
        for col in new_df.columns:
            x = scale_factors.get(col)
            if x is None or (isinstance(x, float) and np.isnan(x)):
                # No directive: keep as-is (factor = 1.0).
                continue
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0.0) * float(x)
        out[region] = new_df
    return out


def _add_injected_fuel(
    per_region_fuel: Dict[str, pd.DataFrame],
    fuel: str,
    region_series: Dict[str, pd.Series],
) -> None:
    """Add a new fuel column to each region's DataFrame in-place."""
    for region, s in region_series.items():
        if region not in per_region_fuel:
            continue
        df = per_region_fuel[region]
        if fuel in df.columns:
            df[fuel] = pd.to_numeric(df[fuel], errors="coerce").fillna(0.0) + s.reindex(df.index).fillna(0.0)
        else:
            df[fuel] = s.reindex(df.index).fillna(0.0)
        per_region_fuel[region] = df


def _normalise_generation_budget(
    per_region_fuel: Dict[str, pd.DataFrame],
    target_total_mwh: float,
    ef_known: Dict[str, float],
) -> float:
    """Scale fuel generation to the annual energy budget after local repairs.

    Per-fuel target scaling preserves the annual system budget by construction,
    but disconnected-zone remediation can add a small amount of local generation
    after that step. A single system-wide scalar restores the original annual
    generation budget without changing realised fuel shares.
    """
    realised_total = 0.0
    for df in per_region_fuel.values():
        for col in df.columns:
            if col in ef_known:
                realised_total += float(pd.to_numeric(df[col], errors="coerce").clip(lower=0).sum())
    if target_total_mwh <= 0 or realised_total <= 0:
        return 1.0
    factor = target_total_mwh / realised_total
    for region, df in per_region_fuel.items():
        for col in df.columns:
            if col in ef_known:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) * factor
        per_region_fuel[region] = df
    return float(factor)


def _realised_shares(
    per_region_fuel: Dict[str, pd.DataFrame],
    ef_known: Dict[str, float],
) -> Dict[str, float]:
    """Compute realised system-wide annual share per fuel (%)."""
    totals: Dict[str, float] = {}
    grand_total = 0.0
    for df in per_region_fuel.values():
        for col in df.columns:
            if col not in ef_known:
                continue
            v = float(pd.to_numeric(df[col], errors="coerce").clip(lower=0).sum())
            totals[col] = totals.get(col, 0.0) + v
            grand_total += v
    if grand_total <= 0:
        return {f: 0.0 for f in totals}
    return {f: 100.0 * v / grand_total for f, v in totals.items()}


# ---- Pipeline invocation ----------------------------------------------------

def _build_modified_unit_dfs(
    raw_unit: Dict[str, pd.DataFrame],
    transfer_dfs: Dict[str, pd.DataFrame],
    scaled_fuel: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Reconstruct per-region unit-level DataFrames with scaled per-fuel totals."""
    out: Dict[str, pd.DataFrame] = {}
    for region in raw_unit:
        fuel_df = scaled_fuel[region]
        transfer_df = transfer_dfs.get(region, pd.DataFrame(index=fuel_df.index))
        merged = pd.concat([fuel_df, transfer_df], axis=1)
        out[region] = merged
    return out


def _set_pipeline_region_data(
    pipeline: AEFPipeline,
    modified_unit: Dict[str, pd.DataFrame],
    calc: AEFCalculator,
    ef_overrides: Dict[str, float],
) -> Dict[str, pd.DataFrame]:
    """Mimic AEFPipeline.load_region_data() but with custom EF + pre-aggregated fuels."""
    ef_effective = dict(calc.default_emission_factors())
    ef_effective.update(ef_overrides)
    ef_effective.setdefault("Hydrogen", 0.0)

    pipeline.calculator = AEFCalculator(ef_effective)

    region_data: Dict[str, pd.DataFrame] = {}
    for region, df in modified_unit.items():
        transfer_cols = [c for c in df.columns
                         if c.startswith(FLOW_PREFIX) or c in TRANSFER_COLS]
        gen_cols = [c for c in df.columns if c not in transfer_cols]
        gen_df = df[gen_cols].copy()
        aef_result = pipeline.calculator.compute_aef_dataframe(gen_df, ef_dict=ef_effective)
        transfer_df = df[transfer_cols].copy() if transfer_cols else pd.DataFrame(index=df.index)
        for col in transfer_df.columns:
            if col.startswith(FLOW_PREFIX):
                transfer_df[col] = transfer_df[col].clip(lower=0)
        region_df = pd.concat([aef_result, transfer_df], axis=1)
        if "BESS" not in region_df.columns:
            region_df["BESS"] = 0.0
        else:
            region_df["BESS"] = region_df["BESS"].fillna(0)
        if "PHS" not in region_df.columns:
            region_df["PHS"] = 0.0
        else:
            region_df["PHS"] = region_df["PHS"].fillna(0)
        region_data[region] = region_df

    pipeline.set_region_data(region_data)
    return region_data


# ---- Scenario summary helpers ----------------------------------------------

def _portfolio_df(results_df: pd.DataFrame) -> pd.DataFrame:
    total = results_df.groupby(["solar_panel_factor", "battery_factor"]).agg(
        {
            "grid_only_emission_t": "sum",
            "pv_storage_emission_t": "sum",
            "delta_t": "sum",
            "grid_only_cost": "sum",
            "pv_storage_cost": "sum",
            "delta_cost": "sum",
        }
    ).reset_index()
    total["abatement_t"] = -total["delta_t"]
    return total


def _summarize_frontier(
    scenario: str, frontier: pd.DataFrame, portfolio_df: pd.DataFrame
) -> Dict[str, Any]:
    knee_idx, knee_abate, knee_cost = find_knee_point(frontier)
    if knee_idx is None:
        knee_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    else:
        knee_row = frontier.iloc[int(knee_idx)]
    min_cost_row = frontier.sort_values("delta_cost", ascending=True).iloc[0]
    max_abate_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    return {
        "scenario": scenario,
        "frontier_points": int(len(frontier)),
        "negative_delta_cost_count": int((portfolio_df["delta_cost"] < 0).sum()),
        "positive_delta_cost_count": int((portfolio_df["delta_cost"] > 0).sum()),
        "knee_index": None if knee_idx is None else int(knee_idx),
        "knee_solar_panel_factor": float(knee_row["solar_panel_factor"]),
        "knee_battery_factor": float(knee_row["battery_factor"]),
        "knee_abatement_t": None if knee_abate is None else float(knee_abate),
        "knee_delta_cost": None if knee_cost is None else float(knee_cost),
        "min_cost_delta_cost": float(min_cost_row["delta_cost"]),
        "max_abatement_t": float(max_abate_row["abatement_t"]),
    }


def _regional_panel_mean(aef_by_region: Dict[str, pd.Series]) -> float:
    panel = pd.concat(
        [pd.to_numeric(s, errors="coerce").rename(r) for r, s in aef_by_region.items()],
        axis=1,
    )
    return float(panel.mean().mean())


def _fixed_allocation_abatement_per_streetlight(
    results_df: pd.DataFrame,
    solar_panel_factor: float,
    battery_factor: float,
    n_lights: int,
) -> float:
    selected = results_df[
        np.isclose(results_df["solar_panel_factor"].astype(float), solar_panel_factor)
        & np.isclose(results_df["battery_factor"].astype(float), battery_factor)
    ]
    if selected.empty:
        raise ValueError(
            "Selected allocation not found in results: "
            f"solar_panel_factor={solar_panel_factor}, battery_factor={battery_factor}"
        )
    reporting_lights = selected["city"].nunique() * int(n_lights)
    if reporting_lights <= 0:
        raise ValueError("Could not infer reporting lights for fixed-allocation retention")
    return float(selected["abatement_t"].sum() / reporting_lights)


def _build_factor_range(section: dict) -> List[float]:
    minimum = float(section.get("min", 0.5))
    maximum = float(section.get("max", 3.0))
    step = float(section.get("step", 0.5))
    count = int(round((maximum - minimum) / step)) + 1
    return [round(minimum + i * step, 10) for i in range(count)]


# ---- Main scenario runner ---------------------------------------------------

def _run_scenario(
    policy: Dict[str, Any],
    raw_unit: Dict[str, pd.DataFrame],
    fuel_dfs: Dict[str, pd.DataFrame],
    transfer_dfs: Dict[str, pd.DataFrame],
    target_total_per_t: pd.Series,
    base_totals: Dict[str, float],
    scenario_root: Path,
    cfg,
    par_df: pd.DataFrame,
    region_city_map: Dict[str, list[str]],
    aef_column: str,
    align_strategy: str,
    par_to_kw_factor: float,
    economic_costs: Dict[str, Any],
    financial_params: Dict[str, Any],
) -> Dict[str, Any]:
    name = policy["name"]
    out_dir = scenario_root / name
    aef_out_dir = out_dir / "aef"
    aef_out_dir.mkdir(parents=True, exist_ok=True)

    target_shares_pct, combined_target_records = _expand_combined_targets(
        dict(policy["shares_pct"]),
        dict(policy.get("combined_targets_pct", {})),
        base_totals,
    )
    h2_share_pct = float(policy.get("introduce_hydrogen_share_pct", 0.0))

    baseline_total_demand_mwh = _system_total(base_totals)
    ef_known = AEFCalculator().default_emission_factors()
    ef_known_with_h2 = dict(ef_known)
    ef_known_with_h2.setdefault("Hydrogen", 0.0)

    # ---- 1. Per-fuel scaling factors (system-wide, t_f / s_f) ----
    scale_factors = _per_fuel_scaling_factors(base_totals, target_shares_pct)
    logger.info("[%s] per-fuel scaling factors: %s", name,
                {k: (round(v, 5) if not (isinstance(v, float) and np.isnan(v)) else "INJECT")
                 for k, v in scale_factors.items()})

    # ---- 2. Apply scaling uniformly across regions and timesteps ----
    scaled_fuel = _apply_scaling(fuel_dfs, scale_factors)

    # ---- 3. Inject new fuels (fallback Offshore Wind, Biomass if absent, Hydrogen) ----
    new_fuel_records: Dict[str, Dict[str, float]] = {}

    def _inject(fuel: str,
                share_pct: float,
                time_carrier: Optional[str],
                region_carrier: str,
                target_total_demand_mwh: float) -> None:
        injected = _inject_new_fuel(
            fuel=fuel,
            target_share_pct=share_pct,
            baseline_total_demand_mwh=target_total_demand_mwh,
            target_total_per_t=target_total_per_t,
            per_region_fuel=fuel_dfs,
            time_shape_carrier=time_carrier,
            region_dist_carrier=region_carrier,
        )
        _add_injected_fuel(scaled_fuel, fuel, injected)
        new_fuel_records[fuel] = {
            "share_pct": share_pct,
            "time_shape_carrier": time_carrier or "<system_demand>",
            "region_dist_carrier": region_carrier,
            "energy_mwh": (share_pct / 100.0) * target_total_demand_mwh,
        }

    # Offshore Wind: scale the classified 2024 offshore profile when present.
    # Only inject a fallback profile from onshore Wind if the baseline has no
    # offshore wind time series.
    if (
        target_shares_pct.get("Offshore Wind", 0.0) > 0
        and base_totals.get("Offshore Wind", 0.0) <= 0
    ):
        _inject(
            fuel="Offshore Wind",
            share_pct=float(target_shares_pct["Offshore Wind"]),
            time_carrier="Wind",
            region_carrier="Wind",
            target_total_demand_mwh=baseline_total_demand_mwh,
        )

    # Biomass+Geothermal: policy combines the bucket, but the model keeps
    # Biomass and Geothermal as separate EF columns. The target split is set by
    # the baseline Biomass/Geothermal generation ratio above. If Biomass were
    # absent, inject it with a demand-shaped Hydro regional proxy.
    if target_shares_pct.get("Biomass", 0.0) > 0 and base_totals.get("Biomass", 0.0) <= 0:
        _inject(
            fuel="Biomass",
            share_pct=float(target_shares_pct["Biomass"]),
            time_carrier=None,  # follows total demand
            region_carrier="Hydro",
            target_total_demand_mwh=baseline_total_demand_mwh,
        )

    # Hydrogen (2050 only): regional split from current LNG (gas-peaker
    # replacement). Shape from total demand (dispatchable).
    if h2_share_pct > 0:
        _inject(
            fuel="Hydrogen",
            share_pct=h2_share_pct,
            time_carrier=None,
            region_carrier="LNG",
            target_total_demand_mwh=baseline_total_demand_mwh,
        )

    # ---- 3b. Disconnected-region remediation ----
    # Per-fuel system-wide scaling assumes inter-regional flows can balance
    # per-region energy. Disconnected outlying-island zones have no flow column
    # and no carrier-based allocation for the surviving/growing fuels, so
    # without remediation their scaled+injected generation can collapse to zero
    # -> AEF NaN. Remediation serves each disconnected zone's baseline
    # generation by allocating its local demand across the system target fuel
    # mix. These zones are a small share of system demand, so the perturbation
    # is removed by the following annual normalisation.
    island_remediation: Dict[str, float] = {}
    disconnected_region_remediation: Dict[str, Dict[str, float]] = {}
    disconnected_regions = [r for r in region_city_map if r not in MAINLAND_REGIONS]
    for disconnected_region in disconnected_regions:
        if disconnected_region not in scaled_fuel or disconnected_region not in fuel_dfs:
            continue

        scaled_region = scaled_fuel[disconnected_region]
        fuel_region = fuel_dfs[disconnected_region]
        gen_cols = [c for c in scaled_region.columns if c in ef_known_with_h2]
        scaled_region_gen = (
            scaled_region[gen_cols].clip(lower=0).sum().sum()
            if gen_cols
            else 0.0
        )
        baseline_region_gen = float(
            sum(
                pd.to_numeric(fuel_region[c], errors="coerce").clip(lower=0).sum()
                for c in fuel_region.columns if c in ef_known_with_h2
            )
        )
        region_gap = baseline_region_gen - float(scaled_region_gen)
        if region_gap > 1.0:
            # Allocate gap by system target share among remaining/growing fuels.
            allocate_targets: Dict[str, float] = {}
            for fuel, pct in target_shares_pct.items():
                if pct > 0:
                    allocate_targets[fuel] = pct
            if h2_share_pct > 0:
                allocate_targets["Hydrogen"] = h2_share_pct
            tgt_sum = sum(allocate_targets.values())
            if tgt_sum > 0:
                # Per-timestep disconnected-zone demand shape from baseline.
                base_region_gen_per_t = pd.Series(0.0, index=scaled_region.index)
                for c in fuel_region.columns:
                    if c in ef_known_with_h2:
                        base_region_gen_per_t = base_region_gen_per_t.add(
                            pd.to_numeric(fuel_region[c], errors="coerce")
                            .reindex(base_region_gen_per_t.index).fillna(0.0).clip(lower=0.0),
                            fill_value=0.0,
                        )
                # Normalise to gap.
                shape_sum = float(base_region_gen_per_t.sum())
                if shape_sum > 0:
                    region_per_t = base_region_gen_per_t * (region_gap / shape_sum)
                else:
                    # Flat fallback.
                    region_per_t = pd.Series(
                        region_gap / max(len(base_region_gen_per_t), 1),
                        index=base_region_gen_per_t.index,
                    )
                region_remediation: Dict[str, float] = {}
                for fuel, pct in allocate_targets.items():
                    weight = pct / tgt_sum
                    add_series = region_per_t * weight
                    if fuel in scaled_region.columns:
                        scaled_region[fuel] = (
                            pd.to_numeric(scaled_region[fuel], errors="coerce").fillna(0.0)
                            + add_series.reindex(scaled_region.index).fillna(0.0)
                        )
                    else:
                        scaled_region[fuel] = add_series.reindex(scaled_region.index).fillna(0.0)
                    amount = float(weight * region_gap)
                    region_remediation[fuel] = amount
                    island_remediation[fuel] = island_remediation.get(fuel, 0.0) + amount
                disconnected_region_remediation[disconnected_region] = region_remediation
                logger.info(
                    "[%s] %s remediation: gap=%.1f MWh (%.4f%% of system) "
                    "redistributed across %d fuels by target share",
                    name,
                    disconnected_region,
                    region_gap,
                    100.0 * region_gap / baseline_total_demand_mwh,
                    len(allocate_targets),
                )

    energy_normalization_factor = _normalise_generation_budget(
        scaled_fuel,
        baseline_total_demand_mwh,
        ef_known_with_h2,
    )
    if abs(energy_normalization_factor - 1.0) > 1e-6:
        logger.info(
            "[%s] applied annual energy normalisation factor %.9f after remediation",
            name,
            energy_normalization_factor,
        )

    # ---- 4. Validate realised shares vs targets ----
    realised = _realised_shares(scaled_fuel, ef_known_with_h2)

    realised_total = 0.0
    for df in scaled_fuel.values():
        for col in df.columns:
            if col not in ef_known_with_h2:
                continue
            realised_total += float(pd.to_numeric(df[col], errors="coerce").clip(lower=0).sum())
    energy_balance_ratio = realised_total / baseline_total_demand_mwh if baseline_total_demand_mwh > 0 else 0.0

    # Combined targets including hydrogen for ±1% sanity check.
    full_targets = dict(target_shares_pct)
    if h2_share_pct > 0:
        full_targets["Hydrogen"] = h2_share_pct

    deviations: Dict[str, float] = {}
    flagged: List[str] = []
    for fuel, target_pct in full_targets.items():
        got = realised.get(fuel, 0.0)
        deviations[fuel] = got - target_pct
        if abs(got - target_pct) > 1.0:
            flagged.append(f"{fuel}: target={target_pct:.2f}% realised={got:.3f}% (delta={got-target_pct:+.3f})")

    logger.info("[%s] energy_balance_ratio = %.6f", name, energy_balance_ratio)
    logger.info("[%s] realised shares (top 12 by share):", name)
    for fuel, share in sorted(realised.items(), key=lambda kv: -kv[1])[:12]:
        tgt = full_targets.get(fuel, None)
        suffix = f" (target {tgt:.2f}%)" if tgt is not None else ""
        logger.info("    %-18s %7.3f%%%s", fuel, share, suffix)
    if flagged:
        logger.warning("[%s] shares OUTSIDE +/-1%% tolerance:", name)
        for line in flagged:
            logger.warning("    %s", line)
    else:
        logger.info("[%s] all targeted shares within +/-1%% tolerance", name)

    # ---- 5. Reconstruct unit-level DataFrames ----
    modified_unit = _build_modified_unit_dfs(raw_unit, transfer_dfs, scaled_fuel)

    # ---- 6. Run AEF pipeline ----
    pipeline = AEFPipeline(config=cfg)
    unit_fuel_map = AEFCalculator.load_unit_fuel_map(cfg.wind_unit_classification_path)
    calc = AEFCalculator(unit_fuel_map=unit_fuel_map)
    _set_pipeline_region_data(pipeline, modified_unit, calc, policy["ef_overrides"])
    _, pool_df = pipeline.run(region_data=pipeline.region_data)
    pipeline.save_results(aef_out_dir, pool_df=pool_df)

    # ---- 7. Reload AEF for simulation ----
    new_aef_by_region = load_aef_by_region(aef_out_dir, region_city_map.keys(),
                                            aef_column=aef_column)
    new_mean = _regional_panel_mean(new_aef_by_region)
    logger.info("[%s] regional panel mean AEF (kg CO2/kWh) = %.6f", name, new_mean)

    # ---- 8. Pareto sweep ----
    solar_cfg = cfg.pareto_solar_cfg
    battery_cfg = cfg.pareto_battery_cfg
    battery_power_cap_multiplier = battery_cfg.get("power_cap_multiplier_of_load")

    results_df = run_parameter_sweep(
        par_df=par_df,
        aef_by_region=new_aef_by_region,
        region_city_map=region_city_map,
        par_to_kw_factor=par_to_kw_factor,
        solar_range=_build_factor_range(solar_cfg),
        battery_range=_build_factor_range(battery_cfg),
        load_mode=cfg.load_mode,
        analysis_years=cfg.analysis_years,
        electricity_price=cfg.electricity_price,
        n_lights=cfg.n_lights,
        light_power_kw=cfg.light_power_kw,
        lighting_threshold_par=cfg.lighting_threshold_par,
        base_capacity_kwh=float(cfg.storage_capacity_kwh),
        base_power_kw=float(cfg.storage_power_kw),
        eta_roundtrip=float(cfg.storage_eta_roundtrip),
        economic_costs=economic_costs,
        financial_params=financial_params,
        battery_power_cap_multiplier_of_load=(
            None if battery_power_cap_multiplier is None
            else float(battery_power_cap_multiplier)
        ),
        align_strategy=align_strategy,
    )
    results_df.to_csv(out_dir / "results.csv", index=False)

    portfolio_df = _portfolio_df(results_df)
    portfolio_df.to_csv(out_dir / "portfolio.csv", index=False)

    frontier = pareto_frontier_min_cost_max_abatement(portfolio_df)
    frontier.to_csv(out_dir / "frontier.csv", index=False)

    macc = compute_macc_from_frontier(frontier)
    if not macc.empty:
        macc.to_csv(out_dir / "macc.csv", index=False)

    knee_idx, knee_abate, knee_cost = find_knee_point(frontier)
    (out_dir / "knee.json").write_text(
        json.dumps({"index": knee_idx, "abatement_t": knee_abate, "cost": knee_cost}, indent=2),
        encoding="utf-8",
    )
    city_metrics = build_city_metrics(results_df)
    city_metrics.to_csv(out_dir / "city_metrics.csv", index=False)

    summary = _summarize_frontier(name, frontier, portfolio_df)
    summary["regional_panel_mean_kgco2e_per_kwh"] = new_mean
    summary["realized_total_demand_mwh"] = realised_total
    summary["target_total_demand_mwh"] = baseline_total_demand_mwh
    summary["energy_balance_ratio"] = energy_balance_ratio
    summary["scale_factors"] = {k: (None if isinstance(v, float) and np.isnan(v) else float(v))
                                 for k, v in scale_factors.items()}
    summary["realized_shares_pct"] = realised
    summary["target_shares_pct"] = full_targets
    summary["combined_target_split_pct"] = combined_target_records
    summary["share_deviations_pct_points"] = deviations
    summary["new_fuels_injected"] = new_fuel_records
    summary["island_remediation_mwh_by_fuel"] = island_remediation
    summary["disconnected_region_remediation_mwh_by_region_fuel"] = (
        disconnected_region_remediation
    )
    summary["energy_normalization_factor"] = energy_normalization_factor
    summary["all_targets_within_1pct"] = (len(flagged) == 0)
    return summary


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    scenario_root = output_dir / "future_grid_scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)

    par_df = load_par_wide(cfg.paper_par_path)
    region_city_map = load_region_city_map(cfg.paper_region_map_path)
    aef_column = cfg.paper_aef_column
    align_strategy = cfg.paper_align_strategy
    economic_costs = dict(cfg.economic_costs)
    financial_params = dict(cfg.financial_params)

    contract_path = output_dir / "paper_contract.json"
    if contract_path.exists():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        par_to_kw_factor = float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))
    else:
        par_to_kw_factor = float(cfg.par_to_kw_factor)
    logger.info("Effective par_to_kw_factor=%.6f", par_to_kw_factor)

    current_aef = load_aef_by_region(cfg.paper_aef_dir, region_city_map.keys(),
                                      aef_column=aef_column)
    current_mean = _regional_panel_mean(current_aef)
    logger.info("Current regional panel mean AEF = %.6f kg/kWh", current_mean)

    # Load raw unit-level generation per region
    power_dir = Path(cfg.power_dir)
    unit_fuel_map = AEFCalculator.load_unit_fuel_map(cfg.wind_unit_classification_path)
    calc = AEFCalculator(unit_fuel_map=unit_fuel_map)
    ef_known = calc.default_emission_factors()
    raw_unit: Dict[str, pd.DataFrame] = {}
    fuel_dfs: Dict[str, pd.DataFrame] = {}
    transfer_dfs: Dict[str, pd.DataFrame] = {}
    for region in region_city_map.keys():
        df = _load_raw_unit_generation(power_dir, region)
        fuel_df, transfer_df = _split_unit_df(df, calc)
        raw_unit[region] = df
        fuel_dfs[region] = fuel_df
        transfer_dfs[region] = transfer_df

    # Per-timestep system demand (sum of EF-known fuel-aggregated generation).
    total_per_t = pd.Series(0.0, index=fuel_dfs["central"].index)
    for region, df in fuel_dfs.items():
        valid_cols = [c for c in df.columns if c in ef_known]
        total_per_t = total_per_t.add(
            df[valid_cols].sum(axis=1).reindex(total_per_t.index).fillna(0.0),
            fill_value=0.0,
        )

    base_totals = _annual_fuel_totals(fuel_dfs, ef_known)
    logger.info(
        "Baseline annual totals (10-minute sample scale; legacy MWh metadata): %s",
        {k: round(v, 1) for k, v in base_totals.items()},
    )
    baseline_total = _system_total(base_totals)
    logger.info("Baseline shares (%%): %s",
                {k: round(100.0 * v / baseline_total, 4) for k, v in base_totals.items()})

    # ---- Run both scenarios ----
    summaries = []
    for policy in (POLICY_2030, POLICY_2050):
        s = _run_scenario(
            policy=policy,
            raw_unit=raw_unit,
            fuel_dfs=fuel_dfs,
            transfer_dfs=transfer_dfs,
            target_total_per_t=total_per_t,
            base_totals=base_totals,
            scenario_root=scenario_root,
            cfg=cfg,
            par_df=par_df,
            region_city_map=region_city_map,
            aef_column=aef_column,
            align_strategy=align_strategy,
            par_to_kw_factor=par_to_kw_factor,
            economic_costs=economic_costs,
            financial_params=financial_params,
        )
        summaries.append(s)

    # ---- Summary CSV ----
    summary_df = pd.DataFrame([
        {k: v for k, v in s.items()
         if k not in ("scale_factors", "realized_shares_pct", "target_shares_pct",
                       "share_deviations_pct_points", "new_fuels_injected",
                       "island_remediation_mwh_by_fuel",
                       "disconnected_region_remediation_mwh_by_region_fuel",
                       "combined_target_split_pct")}
        for s in summaries
    ])
    summary_df["regional_panel_mean_vs_current"] = (
        summary_df["regional_panel_mean_kgco2e_per_kwh"] - current_mean
    )
    summary_df["regional_panel_mean_pct_change"] = (
        100.0 * (summary_df["regional_panel_mean_kgco2e_per_kwh"] - current_mean) / current_mean
    )
    summary_df.to_csv(scenario_root / "scenario_summary.csv", index=False)
    logger.info("Wrote %s", scenario_root / "scenario_summary.csv")
    print("\n=== scenario_summary.csv ===")
    print(summary_df.to_string(index=False))

    current_results_path = output_dir / "pareto" / "results.csv"
    if current_results_path.exists():
        selected_solar = 1.85
        selected_battery = 8.45
        current_results = pd.read_csv(current_results_path)
        current_fixed = _fixed_allocation_abatement_per_streetlight(
            current_results, selected_solar, selected_battery, int(cfg.n_lights)
        )
        retention_rows = []
        for s in summaries:
            scen_name = s["scenario"]
            scen_results = pd.read_csv(scenario_root / scen_name / "results.csv")
            fixed_abatement = _fixed_allocation_abatement_per_streetlight(
                scen_results, selected_solar, selected_battery, int(cfg.n_lights)
            )
            scale_factor = s["regional_panel_mean_kgco2e_per_kwh"] / current_mean
            uniform_abatement = current_fixed * scale_factor
            n_reporting_cities = sum(len(cities) for cities in region_city_map.values())
            reselected_abatement = (
                float(s["knee_abatement_t"]) / (n_reporting_cities * int(cfg.n_lights))
            )
            retention_rows.append({
                "scenario": scen_name,
                "method": (
                    "uniform AEF scaling: scale each region's 2024 "
                    "FLOW_UNIT_FINAL_AEF by panel_canonical/panel_2024"
                ),
                "panel_mean_2024": current_mean,
                "panel_mean_canonical_scenario": s["regional_panel_mean_kgco2e_per_kwh"],
                "scale_factor": scale_factor,
                "uniform_panel_mean_check": s["regional_panel_mean_kgco2e_per_kwh"],
                "knee_abatement_per_pole_2024_t": current_fixed,
                "uniform_aef_knee_abatement_per_pole_t": uniform_abatement,
                "generation_by_fuel_knee_abatement_per_pole_t": fixed_abatement,
                "reselected_knee_abatement_per_pole_t": reselected_abatement,
                "reselected_knee_solar_panel_factor": s["knee_solar_panel_factor"],
                "reselected_knee_battery_factor": s["knee_battery_factor"],
                "uniform_retention_pct": 100.0 * uniform_abatement / current_fixed,
                "generation_by_fuel_retention_pct": 100.0 * fixed_abatement / current_fixed,
                "reselected_knee_retention_pct": 100.0 * reselected_abatement / current_fixed,
                "diurnal_asymmetry_gap_pp": (
                    100.0 * fixed_abatement / current_fixed
                    - 100.0 * uniform_abatement / current_fixed
                ),
                "interpretation": (
                    "Uniform-AEF scaling assumes abatement is proportional to "
                    "panel-mean AEF; generation-by-fuel scaling preserves "
                    "diurnal and regional structure."
                ),
            })
        retention_df = pd.DataFrame(retention_rows)
        retention_df.to_csv(scenario_root / "uniform_aef_summary.csv", index=False)
        retention_df.to_csv(scenario_root / "fixed_representative_retention.csv", index=False)
        logger.info("Wrote %s", scenario_root / "fixed_representative_retention.csv")
    else:
        logger.warning("Current results file missing; skipped fixed-allocation retention CSV")

    # Backward-compatible flat aliases at scenario_root level for downstream
    # consumers that read `<scenario_name>_results.csv` directly.
    for s in summaries:
        scen_name = s["scenario"]
        scen_dir = scenario_root / scen_name
        for fname in ("results.csv", "frontier.csv", "portfolio.csv"):
            src = scen_dir / fname
            dst = scenario_root / f"{scen_name}_{fname}"
            if src.exists():
                pd.read_csv(src).to_csv(dst, index=False)

    # ---- Metadata JSON ----
    metadata = {
        "method": (
            "Per-fuel system-wide scaling x_f = t_f / s_f applied uniformly "
            "across regions and timesteps; phase-out fuels get x_f = 0; "
            "combined Biomass+Geothermal targets are split back to Biomass and "
            "Geothermal using 2024 generation weights; fuels not in baseline "
            "(Hydrogen, Biomass if absent, and only fallback Offshore Wind if "
            "no classified 2024 offshore series is available) are injected as "
            "new generation series with policy-target share, time-shape carrier "
            "(current onshore Wind for fallback offshore wind, total demand for "
            "injected biomass/hydrogen) and regional distribution carrier "
            "(current Wind for fallback offshore wind, current Hydro for "
            "injected biomass, current LNG for hydrogen). Disconnected "
            "outlying-island load closure allocates any phase-out shortfall "
            "across remaining target-mix fuels by target-share weights while "
            "preserving each zone's 2024 total-generation time profile. After "
            "disconnected-zone load closure, a "
            "system-wide annual normalisation preserves the 2024 "
            "generation budget while retaining the target shares. "
            "Renewable bucket decomposed into Solar/Offshore Wind/Onshore "
            "Wind/Hydro/Biomass+Geothermal sub-targets per Taiwan policy. "
            "AEFPipeline (storage pool + distributed-fuel allocator + "
            "inter-regional flow) re-run on modified generation; time-shape "
            "and physics preserved."
        ),
        "current_regional_panel_mean_kgco2e_per_kwh": current_mean,
        "generation_total_unit_note": (
            "Fuel-total metadata uses the AEF pipeline's common 10-minute "
            "average-power sample scale. The constant timestep factor cancels "
            "for scenario fuel shares, scale factors, AEF reconstruction, and "
            "abatement-retention results; legacy *_mwh field names are retained "
            "for downstream compatibility and should not be interpreted as "
            "audited absolute annual electricity totals."
        ),
        "baseline_fuel_totals_mwh": base_totals,
        "baseline_fuel_shares_pct": {
            f: 100.0 * v / baseline_total for f, v in base_totals.items()
        },
        "baseline_total_demand_mwh": baseline_total,
        "scenarios": {
            policy["name"]: {
                "label": policy["label"],
                "citations": policy["citations"],
                "target_shares_pct": s["target_shares_pct"],
                "combined_target_split_pct": s["combined_target_split_pct"],
                "renewable_sub_targets_pct_of_total":
                    policy.get("renewable_sub_targets_pct_of_total", {}),
                "ef_overrides_kgco2e_per_kwh": policy["ef_overrides"],
                "introduce_hydrogen_share_pct": policy["introduce_hydrogen_share_pct"],
                "scale_factors_applied": s["scale_factors"],
                "realized_shares_pct": s["realized_shares_pct"],
                "share_deviations_pct_points": s["share_deviations_pct_points"],
                "all_targets_within_1pct": s["all_targets_within_1pct"],
                "new_fuels_injected": s["new_fuels_injected"],
                "island_remediation_mwh_by_fuel": s["island_remediation_mwh_by_fuel"],
                "disconnected_region_remediation_mwh_by_region_fuel": (
                    s["disconnected_region_remediation_mwh_by_region_fuel"]
                ),
                "regional_panel_mean_kgco2e_per_kwh": s["regional_panel_mean_kgco2e_per_kwh"],
                "energy_balance_ratio": s["energy_balance_ratio"],
                "knee_abatement_t": s["knee_abatement_t"],
                "knee_delta_cost": s["knee_delta_cost"],
            }
            for policy, s in zip((POLICY_2030, POLICY_2050), summaries)
        },
        "citations": {
            "taiwan_2030_532_target": (
                "Taiwan MOEA '532' 2030 fuel-share target — 50% LNG, "
                "30% renewable (30 GW solar + 13.1 GW offshore wind), "
                "20% coal. "
                "Renewable sub-mix per MOEA installed-capacity guidance: "
                "Solar 50%, Offshore Wind 25%, Onshore Wind 10%, Hydro 10%, "
                "Biomass+Geothermal 5% of the renewable bucket."
            ),
            "taiwan_2050_netzero_pathway": (
                "Taiwan 2050 net-zero pathway midpoint — 60-70% renewables "
                "(40-80 GW solar + 40-55 GW offshore wind), LNG+CCUS bridge, "
                "10.5% green hydrogen. Renewable sub-mix: Solar 45%, Offshore "
                "Wind 38%, Onshore Wind 5%, Hydro 7%, Biomass+Geothermal 5%."
            ),
            "unece2021lca": (
                "UNECE 2021 LCA — NGCC with CCS lifecycle EF range "
                "0.092-0.220 kg CO2e/kWh; midpoint 0.156 used here."
            ),
            "ipcc_ar6_wg3_ch6": (
                "IPCC AR6 WG3 Ch6 — sectoral electricity decarbonisation framing."
            ),
        },
    }
    (scenario_root / "scenario_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    logger.info("Wrote %s", scenario_root / "scenario_metadata.json")


if __name__ == "__main__":
    main()
