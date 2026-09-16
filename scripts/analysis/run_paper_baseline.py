from __future__ import annotations

import json

from calibrate_par_to_kw_factor import calibrate_par_to_kw_factor
from paper1_provenance import normalize_repo_paths

from streetlight.config import get_config
from streetlight.simulation.storage import (
    build_city_metrics,
    compute_macc_from_frontier,
    find_knee_point,
    pareto_frontier_min_cost_max_abatement,
)
from streetlight.simulation.streetlight_simulation import (
    load_aef_by_region,
    load_par_wide,
    load_region_city_map,
    run_parameter_sweep,
    save_simulation_outputs,
    simulate_from_par_and_aef,
)


def main() -> None:
    cfg = get_config()
    standardized = cfg.standardized_installation
    economic_costs = dict(cfg.economic_costs)
    financial_params = dict(cfg.financial_params)

    par_path = cfg.paper_par_path
    aef_dir = cfg.paper_aef_dir
    region_map_path = cfg.paper_region_map_path
    output_dir = cfg.paper_output_dir
    aef_column = cfg.paper_aef_column
    align_strategy = cfg.paper_align_strategy
    sim_output_dir = output_dir / "streetlight_sim"
    pareto_output_dir = output_dir / "pareto"
    sim_output_dir.mkdir(parents=True, exist_ok=True)
    pareto_output_dir.mkdir(parents=True, exist_ok=True)

    calibration_summary = calibrate_par_to_kw_factor(par_path=par_path, output_dir=output_dir / "calibration")
    effective_par_to_kw_factor = float(
        calibration_summary.get("rounded_paper_baseline", cfg.par_to_kw_factor)
    )
    contract = normalize_repo_paths(
        {
            "analysis_years": cfg.analysis_years,
            "electricity_price": cfg.electricity_price,
            "standardized_installation": standardized,
            "calibration": calibration_summary,
            "effective_par_to_kw_factor": effective_par_to_kw_factor,
            "paper_inputs": {
                "par_path": par_path,
                "aef_dir": aef_dir,
                "region_map": region_map_path,
                "aef_column": aef_column,
                "align_strategy": align_strategy,
            },
            "economic_costs": economic_costs,
            "financial_params": financial_params,
            "economic_boundary": "discounted_lifecycle_cost_with_battery_replacement",
        }
    )
    (output_dir / "paper_contract.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )

    par_df = load_par_wide(par_path)
    region_city_map = load_region_city_map(region_map_path)
    aef_by_region = load_aef_by_region(aef_dir, region_city_map.keys(), aef_column=aef_column)

    storage_params = dict(cfg.storage_params)
    init_soc = storage_params.pop("init_soc", None)
    if init_soc is not None and "soc0_kwh" not in storage_params:
        capacity_kwh = float(cfg.storage_capacity_kwh)
        storage_params["soc0_kwh"] = float(init_soc) * capacity_kwh

    city_df, region_df, summary = simulate_from_par_and_aef(
        par_df=par_df,
        aef_by_region=aef_by_region,
        region_city_map=region_city_map,
        par_to_kw_factor=effective_par_to_kw_factor,
        storage_params=storage_params,
        load_mode=cfg.load_mode,
        n_lights=cfg.n_lights,
        light_power_kw=cfg.light_power_kw,
        analysis_years=cfg.analysis_years,
        lighting_threshold_par=cfg.lighting_threshold_par,
        align_strategy=align_strategy,
    )
    save_simulation_outputs(city_df, region_df, summary, sim_output_dir)

    solar_cfg = cfg.pareto_solar_cfg
    battery_cfg = cfg.pareto_battery_cfg
    battery_power_cap_multiplier = battery_cfg.get("power_cap_multiplier_of_load")

    def build_factor_range(section: dict) -> list[float]:
        minimum = float(section.get("min", 0.5))
        maximum = float(section.get("max", 3.0))
        step = float(section.get("step", 0.5))
        count = int(round((maximum - minimum) / step)) + 1
        return [round(minimum + i * step, 10) for i in range(count)]

    results_df = run_parameter_sweep(
        par_df=par_df,
        aef_by_region=aef_by_region,
        region_city_map=region_city_map,
        par_to_kw_factor=effective_par_to_kw_factor,
        solar_range=build_factor_range(solar_cfg),
        battery_range=build_factor_range(battery_cfg),
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
            None if battery_power_cap_multiplier is None else float(battery_power_cap_multiplier)
        ),
        align_strategy=align_strategy,
    )
    results_df.to_csv(pareto_output_dir / "results.csv", index=False)

    total_results = results_df.groupby(["solar_panel_factor", "battery_factor"]).agg(
        {
            "grid_only_emission_t": "sum",
            "pv_storage_emission_t": "sum",
            "delta_t": "sum",
            "grid_only_cost": "sum",
            "pv_storage_cost": "sum",
            "delta_cost": "sum",
        }
    ).reset_index()
    total_results["abatement_t"] = -total_results["delta_t"]

    frontier = pareto_frontier_min_cost_max_abatement(total_results)
    frontier.to_csv(pareto_output_dir / "frontier.csv", index=False)

    macc = compute_macc_from_frontier(frontier)
    if not macc.empty:
        macc.to_csv(pareto_output_dir / "macc.csv", index=False)

    knee_idx, knee_abate, knee_cost = find_knee_point(frontier)
    (pareto_output_dir / "knee.json").write_text(
        json.dumps({"index": knee_idx, "abatement_t": knee_abate, "cost": knee_cost}, indent=2),
        encoding="utf-8",
    )

    city_metrics = build_city_metrics(results_df)
    city_metrics.to_csv(pareto_output_dir / "city_metrics.csv", index=False)

    if knee_idx is not None:
        knee_row = frontier.iloc[int(knee_idx)]
        selected_city = results_df[
            (results_df["solar_panel_factor"] == float(knee_row["solar_panel_factor"]))
            & (results_df["battery_factor"] == float(knee_row["battery_factor"]))
        ].copy()
        selected_city["functional_unit_lights"] = cfg.n_lights
        selected_city["abatement_t_per_streetlight"] = (
            selected_city["abatement_t"] / selected_city["functional_unit_lights"]
        )
        selected_city["delta_cost_ntd_per_streetlight"] = (
            selected_city["delta_cost"] / selected_city["functional_unit_lights"]
        )
        keep = [
            "city",
            "region",
            "solar_panel_factor",
            "battery_factor",
            "functional_unit_lights",
            "grid_only_emission_t",
            "pv_storage_emission_t",
            "abatement_t",
            "abatement_t_per_streetlight",
            "grid_energy_kwh_20y",
            "pv_storage_energy_kwh_20y",
            "delta_cost",
            "delta_cost_ntd_per_streetlight",
        ]
        selected_city[keep].to_csv(
            pareto_output_dir / "selected_allocation_city_metrics.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
