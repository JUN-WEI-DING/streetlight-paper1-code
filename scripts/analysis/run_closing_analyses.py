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
    simulate_from_par_and_aef,
)


def _load_effective_par_to_kw_factor(output_dir: Path, cfg) -> float:
    contract_path = output_dir / "paper_contract.json"
    if not contract_path.exists():
        return cfg.par_to_kw_factor
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))


def _load_design_points(output_dir: Path) -> dict[str, dict[str, float]]:
    frontier = pd.read_csv(output_dir / "pareto" / "frontier.csv")
    knee = json.loads((output_dir / "pareto" / "knee.json").read_text(encoding="utf-8"))

    min_cost_row = frontier.sort_values("delta_cost", ascending=True).iloc[0]
    max_abatement_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    if knee.get("index") is None:
        knee_row = max_abatement_row
    else:
        knee_row = frontier.iloc[int(knee["index"])]

    return {
        "min_cost": {
            "solar_panel_factor": float(min_cost_row["solar_panel_factor"]),
            "battery_factor": float(min_cost_row["battery_factor"]),
        },
        "knee": {
            "solar_panel_factor": float(knee_row["solar_panel_factor"]),
            "battery_factor": float(knee_row["battery_factor"]),
        },
        "max_abatement": {
            "solar_panel_factor": float(max_abatement_row["solar_panel_factor"]),
            "battery_factor": float(max_abatement_row["battery_factor"]),
        },
    }


def _city_ranking_shift(method_output_dir: Path, closing_dir: Path, design_points: dict[str, dict[str, float]]) -> None:
    baseline = pd.read_csv(method_output_dir / "baseline_results.csv")
    static = pd.read_csv(method_output_dir / "static_average_aef_results.csv")

    knee = design_points["knee"]
    mask = (
        (baseline["solar_panel_factor"] == knee["solar_panel_factor"])
        & (baseline["battery_factor"] == knee["battery_factor"])
    )
    base_knee = baseline.loc[mask, ["city", "region", "abatement_t"]].rename(
        columns={"abatement_t": "baseline_abatement_t"}
    )
    static_knee = static.loc[mask, ["city", "region", "abatement_t"]].rename(
        columns={"abatement_t": "static_average_aef_abatement_t"}
    )

    ranking = base_knee.merge(static_knee, on=["city", "region"], how="inner")
    ranking["baseline_rank"] = ranking["baseline_abatement_t"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["static_average_aef_rank"] = ranking["static_average_aef_abatement_t"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["rank_shift"] = ranking["static_average_aef_rank"] - ranking["baseline_rank"]
    ranking["abs_rank_shift"] = ranking["rank_shift"].abs()
    ranking["abatement_delta_t"] = (
        ranking["static_average_aef_abatement_t"] - ranking["baseline_abatement_t"]
    )
    ranking = ranking.sort_values(
        ["abs_rank_shift", "abatement_delta_t"], ascending=[False, False]
    ).reset_index(drop=True)
    ranking.to_csv(closing_dir / "city_ranking_shift.csv", index=False)

    summary = {
        "design_point": knee,
        "city_count": int(len(ranking)),
        "mean_abs_rank_shift": float(ranking["abs_rank_shift"].mean()),
        "max_abs_rank_shift": int(ranking["abs_rank_shift"].max()),
        "spearman_rank_correlation": float(
            ranking["baseline_rank"].corr(ranking["static_average_aef_rank"], method="spearman")
        ),
        "top_shifted_cities": ranking.head(10).to_dict(orient="records"),
    }
    (closing_dir / "city_ranking_shift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _aggregate_design_costs(results_df: pd.DataFrame, solar: float, battery: float) -> dict[str, float]:
    subset = results_df[
        (results_df["solar_panel_factor"] == solar)
        & (results_df["battery_factor"] == battery)
    ]
    totals = subset[
        [
            "grid_only_emission_t",
            "pv_storage_emission_t",
            "abatement_t",
            "grid_energy_kwh_20y",
            "pv_storage_energy_kwh_20y",
            "grid_only_cost",
            "pv_storage_cost",
            "grid_only_energy_cost",
            "grid_only_fixed_cost",
            "pv_storage_energy_cost",
            "pv_storage_capex_cost",
            "pv_storage_om_cost",
            "pv_storage_eol_cost",
            "delta_cost",
        ]
    ].sum()
    abatement_t = float(totals["abatement_t"])
    delta_cost = float(totals["delta_cost"])
    macc_ntd_per_tco2e = (delta_cost / abatement_t) if abatement_t != 0 else None
    return {
        "grid_only_emission_t": float(totals["grid_only_emission_t"]),
        "pv_storage_emission_t": float(totals["pv_storage_emission_t"]),
        "abatement_t": abatement_t,
        "grid_energy_kwh_20y": float(totals["grid_energy_kwh_20y"]),
        "pv_storage_energy_kwh_20y": float(totals["pv_storage_energy_kwh_20y"]),
        "grid_only_cost": float(totals["grid_only_cost"]),
        "pv_storage_cost": float(totals["pv_storage_cost"]),
        "grid_only_energy_cost": float(totals["grid_only_energy_cost"]),
        "grid_only_fixed_cost": float(totals["grid_only_fixed_cost"]),
        "pv_storage_energy_cost": float(totals["pv_storage_energy_cost"]),
        "pv_storage_capex_cost": float(totals["pv_storage_capex_cost"]),
        "pv_storage_om_cost": float(totals["pv_storage_om_cost"]),
        "pv_storage_eol_cost": float(totals["pv_storage_eol_cost"]),
        "delta_cost": delta_cost,
        "macc": macc_ntd_per_tco2e,
        "delta_cost_ntd": delta_cost,
        "macc_ntd_per_tco2e": macc_ntd_per_tco2e,
    }


def _design_decision_table(
    output_dir: Path,
    closing_dir: Path,
    cfg,
    effective_par_to_kw_factor: float,
    design_points: dict[str, dict[str, float]],
) -> None:
    results_df = pd.read_csv(output_dir / "pareto" / "results.csv")
    par_df = load_par_wide(cfg.paper_par_path)
    aef_dir = cfg.paper_aef_dir
    region_map_path = cfg.paper_region_map_path
    region_city_map = load_region_city_map(region_map_path)
    aef_by_region = load_aef_by_region(aef_dir, region_city_map.keys())
    align_strategy = cfg.paper_align_strategy

    storage_params = dict(cfg.storage_params)
    base_capacity_kwh = float(cfg.storage_capacity_kwh)
    base_power_kw = float(cfg.storage_power_kw)
    eta_roundtrip = float(cfg.storage_eta_roundtrip)
    battery_power_cap_multiplier = cfg.battery_power_cap_multiplier_of_load
    init_soc = storage_params.pop("init_soc", None)
    if init_soc is not None and "soc0_kwh" not in storage_params:
        storage_params["soc0_kwh"] = float(init_soc) * base_capacity_kwh
    installation_load_kw = float(cfg.n_lights) * float(cfg.light_power_kw)
    power_cap_kw = (
        None
        if battery_power_cap_multiplier is None
        else float(battery_power_cap_multiplier) * installation_load_kw
    )

    rows: list[dict[str, Any]] = []
    for label, point in design_points.items():
        solar = point["solar_panel_factor"]
        battery = point["battery_factor"]
        sim_storage_params = {
            **storage_params,
            "capacity_kwh": base_capacity_kwh * battery,
            "power_kw": (
                base_power_kw * battery
                if power_cap_kw is None
                else min(base_power_kw * battery, power_cap_kw)
            ),
            "eta_roundtrip": eta_roundtrip,
        }
        city_df, _, _ = simulate_from_par_and_aef(
            par_df=par_df,
            aef_by_region=aef_by_region,
            region_city_map=region_city_map,
            par_to_kw_factor=effective_par_to_kw_factor * solar,
            storage_params=sim_storage_params,
            load_mode=cfg.load_mode,
            n_lights=cfg.n_lights,
            light_power_kw=cfg.light_power_kw,
            analysis_years=cfg.analysis_years,
            lighting_threshold_par=cfg.lighting_threshold_par,
            align_strategy=align_strategy,
        )
        energy_totals = city_df[
            [
                "total_load_kwh",
                "pv_generation_kwh",
                "pv_direct_kwh",
                "battery_discharge_kwh",
                "grid_import_kwh",
            ]
        ].sum()
        total_load = float(energy_totals["total_load_kwh"])
        grid_import = float(energy_totals["grid_import_kwh"])
        self_supply_ratio = (
            float((energy_totals["pv_direct_kwh"] + energy_totals["battery_discharge_kwh"]) / total_load)
            if total_load > 0
            else 0.0
        )
        grid_ratio = float(grid_import / total_load) if total_load > 0 else 0.0

        row = {
            "design_label": label,
            "solar_panel_factor": solar,
            "battery_factor": battery,
            "battery_capacity_kwh": base_capacity_kwh * battery,
            "battery_power_kw": sim_storage_params["power_kw"],
            "requested_battery_power_kw": base_power_kw * battery,
            "total_load_kwh": total_load,
            "pv_generation_kwh": float(energy_totals["pv_generation_kwh"]),
            "pv_direct_kwh": float(energy_totals["pv_direct_kwh"]),
            "battery_discharge_kwh": float(energy_totals["battery_discharge_kwh"]),
            "grid_import_kwh": grid_import,
            "self_supply_ratio": self_supply_ratio,
            "grid_ratio": grid_ratio,
        }
        row.update(_aggregate_design_costs(results_df, solar, battery))
        rows.append(row)

    design_df = pd.DataFrame(rows)
    order = {"min_cost": 0, "knee": 1, "max_abatement": 2}
    design_df["sort_order"] = design_df["design_label"].map(order)
    design_df = design_df.sort_values("sort_order").drop(columns=["sort_order"])
    design_df.to_csv(closing_dir / "design_decision_table.csv", index=False)
    (closing_dir / "design_decision_table.json").write_text(
        design_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    method_output_dir = output_dir / "method_simplifications"
    closing_dir = output_dir / "closing_analyses"
    closing_dir.mkdir(parents=True, exist_ok=True)

    effective_par_to_kw_factor = _load_effective_par_to_kw_factor(output_dir, cfg)
    design_points = _load_design_points(output_dir)

    _city_ranking_shift(method_output_dir, closing_dir, design_points)
    _design_decision_table(output_dir, closing_dir, cfg, effective_par_to_kw_factor, design_points)

    (closing_dir / "design_points.json").write_text(
        json.dumps(design_points, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
