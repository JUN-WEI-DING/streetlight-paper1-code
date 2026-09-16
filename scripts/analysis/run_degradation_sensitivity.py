from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from streetlight.config import get_config
from streetlight.simulation.storage import find_knee_point, pareto_frontier_min_cost_max_abatement, recompute_costs_from_results

from research_analysis_common import aggregate_design_from_results, deep_merge, load_raw_paper_settings, portfolio_results


def _summarize_frontier(frontier: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict[str, Any]:
    knee_idx, knee_abatement_t, knee_delta_cost = find_knee_point(frontier)
    if knee_idx is None:
        knee_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    else:
        knee_row = frontier.iloc[int(knee_idx)]
    min_cost_row = frontier.sort_values("delta_cost", ascending=True).iloc[0]
    max_abatement_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    return {
        "frontier_points": int(len(frontier)),
        "negative_delta_cost_count": int((portfolio_df["delta_cost"] < 0).sum()),
        "positive_delta_cost_count": int((portfolio_df["delta_cost"] > 0).sum()),
        "knee_index": None if knee_idx is None else int(knee_idx),
        "knee_solar_panel_factor": float(knee_row["solar_panel_factor"]),
        "knee_battery_factor": float(knee_row["battery_factor"]),
        "knee_abatement_t": None if knee_abatement_t is None else float(knee_abatement_t),
        "knee_delta_cost": None if knee_delta_cost is None else float(knee_delta_cost),
        "min_cost_delta_cost": float(min_cost_row["delta_cost"]),
        "max_abatement_t": float(max_abatement_row["abatement_t"]),
    }


def _average_pv_output_multiplier(rate: float, years: float) -> float:
    rounded_years = max(1, int(round(float(years))))
    annual_multipliers = [(1.0 - float(rate)) ** year for year in range(rounded_years)]
    return float(sum(annual_multipliers) / float(len(annual_multipliers)))


def _apply_degradation_proxy(
    results_df: pd.DataFrame,
    *,
    years: float,
    pv_degradation_rate_per_year: float,
    battery_usable_capacity_multiplier: float,
    representative_night_hours: float,
) -> pd.DataFrame:
    out = results_df.copy()
    total_load = out["grid_energy_kwh_20y"].astype(float)
    delivered_local_supply = (total_load - out["pv_storage_energy_kwh_20y"].astype(float)).clip(lower=0.0)
    avoided_carbon_intensity = (
        out["abatement_t"].astype(float) / delivered_local_supply.where(delivered_local_supply > 1e-9, other=1.0)
    )
    avoided_carbon_intensity = avoided_carbon_intensity.where(delivered_local_supply > 1e-9, other=0.0)

    pv_output_multiplier = _average_pv_output_multiplier(pv_degradation_rate_per_year, years)
    representative_night_load_kwh = out["installation_load_kw"].astype(float) * float(representative_night_hours)
    storage_dependence_proxy = (
        out["battery_capacity_kwh"].astype(float)
        / (out["battery_capacity_kwh"].astype(float) + representative_night_load_kwh)
    ).fillna(0.0)
    storage_effect_multiplier = 1.0 - storage_dependence_proxy * (1.0 - float(battery_usable_capacity_multiplier))
    combined_local_supply_multiplier = float(pv_output_multiplier) * storage_effect_multiplier

    degraded_local_supply = delivered_local_supply * combined_local_supply_multiplier
    out["pv_storage_energy_kwh_20y"] = (total_load - degraded_local_supply).clip(lower=0.0)
    out["abatement_t"] = degraded_local_supply * avoided_carbon_intensity
    out["delta_t"] = -out["abatement_t"]
    out["pv_storage_emission_t"] = out["grid_only_emission_t"].astype(float) - out["abatement_t"]
    out["pv_degradation_rate_per_year"] = float(pv_degradation_rate_per_year)
    out["battery_usable_capacity_multiplier"] = float(battery_usable_capacity_multiplier)
    out["pv_output_multiplier"] = float(pv_output_multiplier)
    out["storage_dependence_proxy"] = storage_dependence_proxy
    out["combined_local_supply_multiplier"] = combined_local_supply_multiplier
    return out


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    analysis_dir = output_dir / "degradation_sensitivity"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = pd.read_csv(output_dir / "pareto" / "results.csv")
    raw_paper = load_raw_paper_settings()
    base_cost = dict(cfg.economic_costs)
    base_financial = dict(cfg.financial_params)
    degradation_section = dict(raw_paper.get("degradation_sensitivity", {}))
    representative_night_hours = float(degradation_section.get("representative_night_hours", 12.0))
    scenario_defs = dict(
        degradation_section.get(
            "scenarios",
            {
                "low_degradation": {
                    "pv_degradation_per_year": 0.003,
                    "battery_usable_capacity_multiplier": 0.95,
                },
                "base_degradation": {
                    "pv_degradation_per_year": 0.005,
                    "battery_usable_capacity_multiplier": 0.9,
                },
                "high_degradation": {
                    "pv_degradation_per_year": 0.008,
                    "battery_usable_capacity_multiplier": 0.85,
                    "battery_replacement_year_override": 10,
                },
            },
        )
    )

    summary_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []

    for scenario_name, scenario in scenario_defs.items():
        degraded_results = _apply_degradation_proxy(
            baseline_results,
            years=float(cfg.analysis_years),
            pv_degradation_rate_per_year=float(
                scenario.get("pv_degradation_per_year", scenario.get("pv_degradation_rate_per_year", 0.005))
            ),
            battery_usable_capacity_multiplier=float(scenario.get("battery_usable_capacity_multiplier", 0.9)),
            representative_night_hours=representative_night_hours,
        )
        cost_override = deep_merge(base_cost, {})
        replacement_year = scenario.get("battery_replacement_year_override", scenario.get("battery_replacement_year"))
        if replacement_year is not None:
            cost_override["battery_replacement_year"] = float(replacement_year)
        if "battery_replacement_energy_capex_ntd_per_kwh" in scenario:
            cost_override["battery_replacement_energy_capex_ntd_per_kwh"] = float(
                scenario["battery_replacement_energy_capex_ntd_per_kwh"]
            )
        results_df = recompute_costs_from_results(
            degraded_results,
            years=float(cfg.analysis_years),
            elec_price=float(cfg.electricity_price),
            economic_costs=cost_override,
            financial_params=base_financial,
        )
        results_df.to_csv(analysis_dir / f"{scenario_name}_results.csv", index=False)
        portfolio_df = portfolio_results(results_df)
        portfolio_df.to_csv(analysis_dir / f"{scenario_name}_portfolio.csv", index=False)
        frontier = pareto_frontier_min_cost_max_abatement(portfolio_df)
        frontier.to_csv(analysis_dir / f"{scenario_name}_frontier.csv", index=False)

        scenario_row = {
            "scenario": scenario_name,
            "pv_degradation_rate_per_year": float(
                scenario.get("pv_degradation_per_year", scenario.get("pv_degradation_rate_per_year", 0.005))
            ),
            "battery_usable_capacity_multiplier": float(scenario.get("battery_usable_capacity_multiplier", 0.9)),
            "battery_replacement_year": float(replacement_year or base_cost.get("battery_replacement_year", 0.0)),
        }
        scenario_row.update(_summarize_frontier(frontier, portfolio_df))
        summary_rows.append(scenario_row)

        for design_label in ("min_cost", "knee", "max_abatement"):
            design_row = aggregate_design_from_results(results_df, design_label, output_dir)
            subset = results_df[
                (results_df["solar_panel_factor"] == float(design_row["solar_panel_factor"]))
                & (results_df["battery_factor"] == float(design_row["battery_factor"]))
            ]
            design_row.update(
                {
                    "scenario": scenario_name,
                    "pv_output_multiplier": float(subset["pv_output_multiplier"].mean()),
                    "battery_usable_capacity_multiplier": float(subset["battery_usable_capacity_multiplier"].mean()),
                    "combined_local_supply_multiplier": float(subset["combined_local_supply_multiplier"].mean()),
                    "storage_dependence_proxy": float(subset["storage_dependence_proxy"].mean()),
                }
            )
            design_rows.append(design_row)

    summary_df = pd.DataFrame(summary_rows)
    baseline_like = summary_df[summary_df["scenario"] == "base_degradation"]
    if not baseline_like.empty:
        baseline_row = baseline_like.iloc[0]
        for col in ("knee_abatement_t", "knee_delta_cost", "min_cost_delta_cost", "max_abatement_t"):
            summary_df[f"{col}_vs_base_degradation"] = summary_df[col] - float(baseline_row[col])
    summary_df.to_csv(analysis_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(design_rows).to_csv(analysis_dir / "representative_designs.csv", index=False)
    (analysis_dir / "scenario_metadata.json").write_text(
        json.dumps(
            {
                "interpretation": (
                    "Degradation sensitivity uses a proxy performance multiplier applied to delivered "
                    "local supply rather than a year-by-year dispatch rerun. PV degradation reduces "
                    "long-run solar yield and battery fade reduces the usable storage contribution via "
                    "a night-load-based dependence proxy."
                ),
                "representative_night_hours": representative_night_hours,
                "scenario_definitions": scenario_defs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
