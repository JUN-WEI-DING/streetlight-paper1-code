from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    load_aef_by_region,
    load_par_wide,
    load_region_city_map,
    run_parameter_sweep,
)
from streetlight.simulation.storage import find_knee_point, pareto_frontier_min_cost_max_abatement


def _build_factor_range(section: dict) -> list[float]:
    minimum = float(section.get("min", 0.5))
    maximum = float(section.get("max", 3.0))
    step = float(section.get("step", 0.5))
    count = int(round((maximum - minimum) / step)) + 1
    return [round(minimum + i * step, 10) for i in range(count)]


def _load_effective_par_to_kw_factor(output_dir: Path, cfg) -> float:
    contract_path = output_dir / "paper_contract.json"
    if not contract_path.exists():
        return cfg.par_to_kw_factor
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))


def _portfolio_results(results_df: pd.DataFrame) -> pd.DataFrame:
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


def _scenario_summary_row(scenario: str, portfolio_df: pd.DataFrame, frontier: pd.DataFrame) -> dict[str, Any]:
    negative_delta_cost_count = int((portfolio_df["delta_cost"] < 0).sum())
    positive_delta_cost_count = int((portfolio_df["delta_cost"] > 0).sum())
    knee_idx, knee_abatement_t, knee_delta_cost = find_knee_point(frontier)
    if knee_idx is None:
        knee_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    else:
        knee_row = frontier.iloc[int(knee_idx)]
    min_cost_row = frontier.sort_values("delta_cost", ascending=True).iloc[0]
    max_abatement_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    return {
        "scenario": scenario,
        "negative_delta_cost_count": negative_delta_cost_count,
        "positive_delta_cost_count": positive_delta_cost_count,
        "total_design_points": int(len(portfolio_df)),
        "frontier_points": int(len(frontier)),
        "knee_index": None if knee_idx is None else int(knee_idx),
        "knee_abatement_t": None if knee_abatement_t is None else float(knee_abatement_t),
        "knee_delta_cost": None if knee_delta_cost is None else float(knee_delta_cost),
        "knee_solar_panel_factor": float(knee_row["solar_panel_factor"]),
        "knee_battery_factor": float(knee_row["battery_factor"]),
        "min_cost_delta_cost": float(min_cost_row["delta_cost"]),
        "min_cost_abatement_t": float(min_cost_row["abatement_t"]),
        "max_abatement_delta_cost": float(max_abatement_row["delta_cost"]),
        "max_abatement_t": float(max_abatement_row["abatement_t"]),
    }


def _static_average_aef(aef_by_region: dict[str, pd.Series]) -> tuple[dict[str, pd.Series], float]:
    panel = pd.concat(
        [pd.to_numeric(series, errors="coerce").rename(region) for region, series in aef_by_region.items()],
        axis=1,
    )
    scalar_mean = float(panel.mean().mean())
    return (
        {
            region: pd.Series(scalar_mean, index=series.index, name=series.name)
            for region, series in aef_by_region.items()
        },
        scalar_mean,
    )


def _uniform_city_par(par_df: pd.DataFrame) -> pd.DataFrame:
    mean_profile = par_df.apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return pd.DataFrame({column: mean_profile for column in par_df.columns}, index=par_df.index)


def _electricity_only_costs(base_costs: dict[str, Any]) -> dict[str, Any]:
    costs = dict(base_costs)
    zero_keys = [
        "pv_capex_ntd_per_kw",
        "pv_om_ntd_per_kw_year",
        "pv_eol_ntd_per_kw",
        "battery_capex_ntd_per_kwh",
        "battery_power_capex_ntd_per_kw",
        "battery_om_fraction_of_capex_per_year",
        "battery_replacement_year",
        "battery_replacement_energy_capex_ntd_per_kwh",
        "battery_replacement_power_capex_ntd_per_kw",
        "battery_eol",
        "other_capex",
        "other_om",
        "other_eol",
        "grid_fixed_cost",
    ]
    for key in zero_keys:
        costs[key] = 0.0
    return costs


def _apply_static_average_aef(results_df: pd.DataFrame, scalar_mean: float) -> pd.DataFrame:
    out = results_df.copy()
    out["grid_only_emission_t"] = out["grid_energy_kwh_20y"].astype(float) * scalar_mean / 1000.0
    out["pv_storage_emission_t"] = out["pv_storage_energy_kwh_20y"].astype(float) * scalar_mean / 1000.0
    out["delta_t"] = out["pv_storage_emission_t"] - out["grid_only_emission_t"]
    out["abatement_t"] = -out["delta_t"]
    return out


def _apply_electricity_only_cost(results_df: pd.DataFrame) -> pd.DataFrame:
    out = results_df.copy()
    out["grid_only_fixed_cost"] = 0.0
    out["grid_only_cost"] = out["grid_only_energy_cost"].astype(float)
    out["pv_storage_capex_cost"] = 0.0
    out["pv_storage_om_cost"] = 0.0
    out["pv_storage_eol_cost"] = 0.0
    out["pv_storage_cost"] = out["pv_storage_energy_cost"].astype(float)
    out["delta_cost"] = out["pv_storage_cost"] - out["grid_only_cost"]
    return out


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    simpl_output_dir = output_dir / "method_simplifications"
    simpl_output_dir.mkdir(parents=True, exist_ok=True)

    par_path = cfg.paper_par_path
    aef_dir = cfg.paper_aef_dir
    region_map_path = cfg.paper_region_map_path
    align_strategy = cfg.paper_align_strategy

    par_df = load_par_wide(par_path)
    region_city_map = load_region_city_map(region_map_path)
    aef_by_region = load_aef_by_region(aef_dir, region_city_map.keys())
    effective_par_to_kw_factor = _load_effective_par_to_kw_factor(output_dir, cfg)
    baseline_results = pd.read_csv(output_dir / "pareto" / "results.csv")

    solar_cfg = cfg.pareto_solar_cfg
    battery_cfg = cfg.pareto_battery_cfg
    solar_range = _build_factor_range(solar_cfg)
    battery_range = _build_factor_range(battery_cfg)
    battery_power_cap_multiplier = battery_cfg.get("power_cap_multiplier_of_load")
    storage_params = dict(cfg.storage_params)
    base_economic_costs = dict(cfg.economic_costs)
    base_financial_params = dict(cfg.financial_params)

    static_aef_by_region, static_aef_value = _static_average_aef(aef_by_region)
    uniform_par_df = _uniform_city_par(par_df)

    scenarios: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "description": "Dynamic regional AEF + city-specific PAR + discounted lifecycle cost",
            "derived_results_df": baseline_results,
        },
        {
            "name": "static_average_aef",
            "description": "Single annual average AEF applied to all regions and all timestamps",
            "derived_results_df": _apply_static_average_aef(baseline_results, static_aef_value),
            "static_aef_value": static_aef_value,
        },
        {
            "name": "uniform_solar_yield",
            "description": "City-specific PAR replaced by a common mean PAR time series across all cities",
            "par_df": uniform_par_df,
            "aef_by_region": aef_by_region,
            "economic_costs": base_economic_costs,
            "financial_params": base_financial_params,
        },
        {
            "name": "electricity_only_cost",
            "description": "Discounted electricity expenditure only; all PV/battery/grid fixed lifecycle costs removed",
            "derived_results_df": _apply_electricity_only_cost(baseline_results),
        },
        # Combined scenario — all three simplifications stacked together.
        # This is the "typical existing-streetlight-LCA-literature setup" baseline
        # (Liu 2020, Allwyn 2022, Chaianong 2019, Duman 2019 all use this combo).
        # Pipeline: uniform-PAR sweep -> static-AEF override -> electricity-only cost.
        # Resolved later in the loop because uniform-PAR sweep depends on
        # par_df/aef_by_region inputs not yet known here.
        {
            "name": "combined_typical_simplifications",
            "description": (
                "Typical streetlight-LCA-literature setup: uniform PAR + static average AEF + "
                "electricity-only cost simultaneously applied. Quantifies the cumulative "
                "distortion of the canonical baseline."
            ),
            "par_df": uniform_par_df,
            "aef_by_region": aef_by_region,
            "economic_costs": base_economic_costs,
            "financial_params": base_financial_params,
            "post_pipeline_transforms": [
                ("static_average_aef", static_aef_value),
                ("electricity_only_cost", None),
            ],
            "static_aef_value": static_aef_value,
        },
    ]

    summary_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        if "derived_results_df" in scenario:
            results_df = scenario["derived_results_df"].copy()
        else:
            results_df = run_parameter_sweep(
                par_df=scenario["par_df"],
                aef_by_region=scenario["aef_by_region"],
                region_city_map=region_city_map,
                par_to_kw_factor=effective_par_to_kw_factor,
                solar_range=solar_range,
                battery_range=battery_range,
                load_mode=cfg.load_mode,
                analysis_years=cfg.analysis_years,
                electricity_price=cfg.electricity_price,
                n_lights=cfg.n_lights,
                light_power_kw=cfg.light_power_kw,
                lighting_threshold_par=cfg.lighting_threshold_par,
                base_capacity_kwh=float(cfg.storage_capacity_kwh),
                base_power_kw=float(cfg.storage_power_kw),
                eta_roundtrip=float(cfg.storage_eta_roundtrip),
                battery_power_cap_multiplier_of_load=(
                    None if battery_power_cap_multiplier is None else float(battery_power_cap_multiplier)
                ),
                align_strategy=align_strategy,
            )
        # Apply post-pipeline transforms in order (e.g., for combined scenario:
        # static-AEF override then electricity-only cost stripping).
        for transform_name, transform_arg in scenario.get("post_pipeline_transforms", []):
            if transform_name == "static_average_aef":
                results_df = _apply_static_average_aef(results_df, float(transform_arg))
            elif transform_name == "electricity_only_cost":
                results_df = _apply_electricity_only_cost(results_df)
            else:
                raise ValueError(f"Unknown post-pipeline transform: {transform_name}")
        results_df.to_csv(simpl_output_dir / f"{scenario['name']}_results.csv", index=False)

        portfolio_df = _portfolio_results(results_df)
        portfolio_df.to_csv(simpl_output_dir / f"{scenario['name']}_portfolio.csv", index=False)

        frontier = pareto_frontier_min_cost_max_abatement(portfolio_df)
        frontier.to_csv(simpl_output_dir / f"{scenario['name']}_frontier.csv", index=False)

        row = _scenario_summary_row(scenario["name"], portfolio_df, frontier)
        row["description"] = scenario["description"]
        if "static_aef_value" in scenario:
            row["static_aef_value"] = float(scenario["static_aef_value"])
        summary_rows.append(row)
        metadata_rows.append(
            {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "static_aef_value": scenario.get("static_aef_value"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df[summary_df["scenario"] == "baseline"].iloc[0]
    for col in ["knee_abatement_t", "knee_delta_cost", "min_cost_delta_cost", "max_abatement_t"]:
        summary_df[f"{col}_vs_baseline"] = summary_df[col] - float(baseline_row[col])
    summary_df.to_csv(simpl_output_dir / "scenario_summary.csv", index=False)

    (simpl_output_dir / "scenario_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    (simpl_output_dir / "scenario_metadata.json").write_text(
        json.dumps(metadata_rows, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
