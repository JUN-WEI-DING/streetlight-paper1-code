from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    load_aef_by_region,
    load_par_wide,
    load_region_city_map,
    run_parameter_sweep,
)
from streetlight.simulation.storage import recompute_costs_from_results


def _load_effective_par_to_kw_factor(output_dir: Path, cfg) -> float:
    contract_path = output_dir / "paper_contract.json"
    if not contract_path.exists():
        return cfg.par_to_kw_factor
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))


def _load_knee_design(output_dir: Path) -> tuple[float, float]:
    frontier_path = output_dir / "pareto" / "frontier.csv"
    knee_path = output_dir / "pareto" / "knee.json"
    frontier = pd.read_csv(frontier_path)
    if frontier.empty:
        raise ValueError(f"Empty frontier: {frontier_path}")

    knee_idx = None
    if knee_path.exists():
        knee = json.loads(knee_path.read_text(encoding="utf-8"))
        knee_idx = knee.get("index")

    if knee_idx is None:
        row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    else:
        row = frontier.iloc[int(knee_idx)]

    return float(row["solar_panel_factor"]), float(row["battery_factor"])


def _portfolio_row(results_df: pd.DataFrame) -> dict:
    total = results_df.agg(
        {
            "grid_only_emission_t": "sum",
            "pv_storage_emission_t": "sum",
            "delta_t": "sum",
            "grid_only_cost": "sum",
            "pv_storage_cost": "sum",
            "delta_cost": "sum",
        }
    )
    abatement_t = float(-total["delta_t"])
    delta_cost = float(total["delta_cost"])
    return {
        "grid_only_emission_t": float(total["grid_only_emission_t"]),
        "pv_storage_emission_t": float(total["pv_storage_emission_t"]),
        "abatement_t": abatement_t,
        "delta_cost": delta_cost,
        "macc": (delta_cost / abatement_t) if abatement_t != 0 else None,
    }


def _run_case(
    *,
    par_df: pd.DataFrame,
    aef_by_region: dict[str, pd.Series],
    region_city_map: dict[str, list[str]],
    par_to_kw_factor: float,
    electricity_price: float,
    n_lights: int,
    light_power_kw: float,
    load_mode: str,
    lighting_threshold_par: float,
    analysis_years: float,
    base_capacity_kwh: float,
    base_power_kw: float,
    eta_roundtrip: float,
    knee_solar: float,
    knee_battery: float,
    economic_costs: dict | None = None,
    financial_params: dict | None = None,
    battery_power_cap_multiplier_of_load: float | None = None,
    align_strategy: str = "ffill",
) -> dict:
    results_df = run_parameter_sweep(
        par_df=par_df,
        aef_by_region=aef_by_region,
        region_city_map=region_city_map,
        par_to_kw_factor=par_to_kw_factor,
        solar_range=[knee_solar],
        battery_range=[knee_battery],
        load_mode=load_mode,
        analysis_years=analysis_years,
        electricity_price=electricity_price,
        n_lights=n_lights,
        light_power_kw=light_power_kw,
        lighting_threshold_par=lighting_threshold_par,
        base_capacity_kwh=base_capacity_kwh,
        base_power_kw=base_power_kw,
        eta_roundtrip=eta_roundtrip,
        economic_costs=economic_costs,
        financial_params=financial_params,
        battery_power_cap_multiplier_of_load=battery_power_cap_multiplier_of_load,
        align_strategy=align_strategy,
    )
    out = _portfolio_row(results_df)
    out["solar_panel_factor"] = knee_solar
    out["battery_factor"] = knee_battery
    return out


def _resolve_n_jobs(arg_n_jobs: int | None) -> int:
    """Resolve n_jobs from CLI flag, env var, or default.

    Order: explicit CLI flag > STREETLIGHT_N_JOBS env var > default.
    Default: min(cpu_count, 8) to keep aggregate memory under the project's
    52 GB virtual cap (each ``run_parameter_sweep`` worker uses ~2 GB).
    """
    if arg_n_jobs is not None:
        return int(arg_n_jobs)
    env_val = os.environ.get("STREETLIGHT_N_JOBS")
    if env_val is not None and env_val.strip() != "":
        return int(env_val)
    n_cpu = os.cpu_count() or 1
    return min(n_cpu, 8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-way sensitivity analysis for the paper design point. "
            "Recommended invocation:\n"
            "  ulimit -v $((52*1024*1024)) && "
            "python scripts/analysis/run_paper_sensitivity.py --n-jobs 8"
        )
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for joblib. "
            "-1 = all cores. Defaults to env STREETLIGHT_N_JOBS or min(cpu_count, 8)."
        ),
    )
    args = parser.parse_args()
    n_jobs = _resolve_n_jobs(args.n_jobs)

    cfg = get_config()
    output_dir = cfg.paper_output_dir
    sensitivity_cfg = dict(cfg.paper_settings("sensitivity", {}))
    sensitivity_output_dir = cfg.resolve_path(
        sensitivity_cfg.get("output_dir", output_dir / "sensitivity")
    )
    sensitivity_output_dir.mkdir(parents=True, exist_ok=True)

    par_path = cfg.paper_par_path
    aef_dir = cfg.paper_aef_dir
    region_map_path = cfg.paper_region_map_path
    align_strategy = cfg.paper_align_strategy

    par_df = load_par_wide(par_path)
    region_city_map = load_region_city_map(region_map_path)
    aef_by_region = load_aef_by_region(aef_dir, region_city_map.keys())

    effective_par_to_kw_factor = _load_effective_par_to_kw_factor(output_dir, cfg)
    knee_solar, knee_battery = _load_knee_design(output_dir)
    baseline_results_all = pd.read_csv(output_dir / "pareto" / "results.csv")
    baseline_design_results = baseline_results_all[
        (baseline_results_all["solar_panel_factor"] == knee_solar)
        & (baseline_results_all["battery_factor"] == knee_battery)
    ].copy()

    storage_params = dict(cfg.storage_params)
    base_capacity_kwh = float(cfg.storage_capacity_kwh)
    base_power_kw = float(cfg.storage_power_kw)
    base_eta_roundtrip = float(cfg.storage_eta_roundtrip)
    battery_power_cap_multiplier = cfg.battery_power_cap_multiplier_of_load
    base_economic_costs = dict(cfg.economic_costs)
    base_financial_params = dict(cfg.financial_params)

    baseline = _portfolio_row(baseline_design_results)
    baseline["solar_panel_factor"] = knee_solar
    baseline["battery_factor"] = knee_battery

    # ---- Build the list of cases ----------------------------------------
    # Each item is a tuple (order_idx, kind, meta, payload) describing how to
    # produce the row. We split into "heavy" cases (full run_parameter_sweep —
    # parallelized) and "light" cases (recompute_costs_from_results — done
    # inline; the heavy lifting was already done in the baseline sweep).
    heavy_cases: list[tuple[int, dict, dict]] = []  # (order_idx, meta, kwargs)
    light_cases: list[tuple[int, dict]] = []  # (order_idx, fully-formed row)

    rows_template: list[dict | None] = [None]  # index 0 reserved for baseline
    rows_template[0] = {
        "parameter": "baseline",
        "scenario_label": "baseline",
        "parameter_value": None,
        "multiplier": 1.0,
        **baseline,
    }
    next_idx = 1

    common_run_kwargs = dict(
        par_df=par_df,
        aef_by_region=aef_by_region,
        region_city_map=region_city_map,
        electricity_price=cfg.electricity_price,
        n_lights=cfg.n_lights,
        light_power_kw=cfg.light_power_kw,
        load_mode=cfg.load_mode,
        lighting_threshold_par=cfg.lighting_threshold_par,
        analysis_years=cfg.analysis_years,
        base_capacity_kwh=base_capacity_kwh,
        base_power_kw=base_power_kw,
        eta_roundtrip=base_eta_roundtrip,
        knee_solar=knee_solar,
        knee_battery=knee_battery,
        economic_costs=base_economic_costs,
        financial_params=base_financial_params,
        battery_power_cap_multiplier_of_load=(
            None if battery_power_cap_multiplier is None else float(battery_power_cap_multiplier)
        ),
        align_strategy=align_strategy,
    )

    for multiplier in sensitivity_cfg.get("par_to_kw_factor_multipliers", [0.8, 1.0, 1.2]):
        value = effective_par_to_kw_factor * float(multiplier)
        meta = {
            "parameter": "par_to_kw_factor",
            "scenario_label": f"x{float(multiplier):.2f}",
            "parameter_value": value,
            "multiplier": float(multiplier),
        }
        kwargs = dict(common_run_kwargs)
        kwargs["par_to_kw_factor"] = value
        heavy_cases.append((next_idx, meta, kwargs))
        rows_template.append(None)
        next_idx += 1

    for value in sensitivity_cfg.get("lighting_threshold_par_values", [0.5, 1.0, 2.0]):
        value = float(value)
        meta = {
            "parameter": "lighting_threshold_par",
            "scenario_label": f"{value:g}",
            "parameter_value": value,
            "multiplier": None,
        }
        kwargs = dict(common_run_kwargs)
        kwargs["par_to_kw_factor"] = effective_par_to_kw_factor
        kwargs["lighting_threshold_par"] = value
        heavy_cases.append((next_idx, meta, kwargs))
        rows_template.append(None)
        next_idx += 1

    for multiplier in sensitivity_cfg.get("electricity_price_multipliers", [0.8, 1.0, 1.2]):
        value = cfg.electricity_price * float(multiplier)
        case_df = recompute_costs_from_results(
            baseline_design_results,
            years=float(cfg.analysis_years),
            elec_price=float(value),
            economic_costs=base_economic_costs,
            financial_params=base_financial_params,
        )
        light_row = {
            "parameter": "electricity_price",
            "scenario_label": f"x{float(multiplier):.2f}",
            "parameter_value": value,
            "multiplier": float(multiplier),
            **{
                **_portfolio_row(case_df),
                "solar_panel_factor": knee_solar,
                "battery_factor": knee_battery,
            },
        }
        light_cases.append((next_idx, light_row))
        rows_template.append(None)
        next_idx += 1

    for multiplier in sensitivity_cfg.get("battery_capex_multipliers", [0.8, 1.0, 1.2]):
        if "battery_capex_ntd_per_kwh" in base_economic_costs:
            value = base_economic_costs["battery_capex_ntd_per_kwh"] * float(multiplier)
        else:
            value = base_economic_costs["battery_capex"] * float(multiplier)
        economic_costs = dict(base_economic_costs)
        if "battery_capex_ntd_per_kwh" in economic_costs:
            economic_costs["battery_capex_ntd_per_kwh"] = (
                economic_costs["battery_capex_ntd_per_kwh"] * float(multiplier)
            )
            economic_costs["battery_power_capex_ntd_per_kw"] = (
                economic_costs["battery_power_capex_ntd_per_kw"] * float(multiplier)
            )
            if "battery_replacement_energy_capex_ntd_per_kwh" in economic_costs:
                economic_costs["battery_replacement_energy_capex_ntd_per_kwh"] = (
                    economic_costs["battery_replacement_energy_capex_ntd_per_kwh"] * float(multiplier)
                )
            if "battery_replacement_power_capex_ntd_per_kw" in economic_costs:
                economic_costs["battery_replacement_power_capex_ntd_per_kw"] = (
                    economic_costs["battery_replacement_power_capex_ntd_per_kw"] * float(multiplier)
                )
        else:
            economic_costs["battery_capex"] = value
        light_row = {
            "parameter": "battery_capex",
            "scenario_label": f"x{float(multiplier):.2f}",
            "parameter_value": value,
            "multiplier": float(multiplier),
            **{
                **_portfolio_row(
                    recompute_costs_from_results(
                        baseline_design_results,
                        years=float(cfg.analysis_years),
                        elec_price=float(cfg.electricity_price),
                        economic_costs=economic_costs,
                        financial_params=base_financial_params,
                    )
                ),
                "solar_panel_factor": knee_solar,
                "battery_factor": knee_battery,
            },
        }
        light_cases.append((next_idx, light_row))
        rows_template.append(None)
        next_idx += 1

    for multiplier in sensitivity_cfg.get("pv_capex_multipliers", [0.8, 1.0, 1.2]):
        if "pv_capex_ntd_per_kw" in base_economic_costs:
            value = base_economic_costs["pv_capex_ntd_per_kw"] * float(multiplier)
        else:
            value = base_economic_costs["solar_capex"] * float(multiplier)
        economic_costs = dict(base_economic_costs)
        if "pv_capex_ntd_per_kw" in economic_costs:
            economic_costs["pv_capex_ntd_per_kw"] = (
                economic_costs["pv_capex_ntd_per_kw"] * float(multiplier)
            )
        else:
            economic_costs["solar_capex"] = value
        light_row = {
            "parameter": "pv_capex",
            "scenario_label": f"x{float(multiplier):.2f}",
            "parameter_value": value,
            "multiplier": float(multiplier),
            **{
                **_portfolio_row(
                    recompute_costs_from_results(
                        baseline_design_results,
                        years=float(cfg.analysis_years),
                        elec_price=float(cfg.electricity_price),
                        economic_costs=economic_costs,
                        financial_params=base_financial_params,
                    )
                ),
                "solar_panel_factor": knee_solar,
                "battery_factor": knee_battery,
            },
        }
        light_cases.append((next_idx, light_row))
        rows_template.append(None)
        next_idx += 1

    for value in sensitivity_cfg.get("light_power_kw_values", [0.08, 0.1, 0.12]):
        value = float(value)
        meta = {
            "parameter": "light_power_kw",
            "scenario_label": f"{value:.2f}",
            "parameter_value": value,
            "multiplier": value / cfg.light_power_kw,
        }
        kwargs = dict(common_run_kwargs)
        kwargs["par_to_kw_factor"] = effective_par_to_kw_factor
        kwargs["light_power_kw"] = value
        # battery_power_cap_multiplier_of_load not propagated in the original
        # branch; preserve that behaviour.
        kwargs["battery_power_cap_multiplier_of_load"] = None
        heavy_cases.append((next_idx, meta, kwargs))
        rows_template.append(None)
        next_idx += 1

    for value in sensitivity_cfg.get("eta_roundtrip_values", [0.85, 0.90, 0.95]):
        value = float(value)
        meta = {
            "parameter": "eta_roundtrip",
            "scenario_label": f"{value:.2f}",
            "parameter_value": value,
            "multiplier": value / base_eta_roundtrip,
        }
        kwargs = dict(common_run_kwargs)
        kwargs["par_to_kw_factor"] = effective_par_to_kw_factor
        kwargs["eta_roundtrip"] = value
        # Preserve original behaviour: this branch did not pass
        # battery_power_cap_multiplier_of_load.
        kwargs["battery_power_cap_multiplier_of_load"] = None
        heavy_cases.append((next_idx, meta, kwargs))
        rows_template.append(None)
        next_idx += 1

    for value in sensitivity_cfg.get("discount_rate_values", [0.03, 0.05, 0.08]):
        value = float(value)
        financial_params = dict(base_financial_params)
        financial_params["discount_rate"] = value
        light_row = {
            "parameter": "discount_rate",
            "scenario_label": f"{value:.2f}",
            "parameter_value": value,
            "multiplier": (value / base_financial_params["discount_rate"])
            if base_financial_params.get("discount_rate", 0.0) not in (0.0, None)
            else None,
            **{
                **_portfolio_row(
                    recompute_costs_from_results(
                        baseline_design_results,
                        years=float(cfg.analysis_years),
                        elec_price=float(cfg.electricity_price),
                        economic_costs=base_economic_costs,
                        financial_params=financial_params,
                    )
                ),
                "solar_panel_factor": knee_solar,
                "battery_factor": knee_battery,
            },
        }
        light_cases.append((next_idx, light_row))
        rows_template.append(None)
        next_idx += 1

    # ---- Place light cases (already computed) ---------------------------
    for idx, row in light_cases:
        rows_template[idx] = row

    # ---- Run heavy cases in parallel ------------------------------------
    print(
        f"[sensitivity] heavy_cases={len(heavy_cases)} light_cases={len(light_cases)} "
        f"n_jobs={n_jobs}"
    )

    if heavy_cases:
        heavy_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_case)(**kwargs) for _idx, _meta, kwargs in heavy_cases
        )
        # Sorted assembly: zip preserves the case order; we then place each
        # result at its original output index for deterministic ordering.
        for (idx, meta, _kwargs), result in zip(heavy_cases, heavy_results):
            rows_template[idx] = {**meta, **result}

    rows = [r for r in rows_template if r is not None]

    df = pd.DataFrame(rows)
    df["delta_abatement_vs_baseline_t"] = df["abatement_t"] - baseline["abatement_t"]
    df["delta_cost_vs_baseline"] = df["delta_cost"] - baseline["delta_cost"]

    df.to_csv(sensitivity_output_dir / "one_way_sensitivity.csv", index=False)
    (sensitivity_output_dir / "summary.json").write_text(
        json.dumps(
            {
                "knee_design": {
                    "solar_panel_factor": knee_solar,
                    "battery_factor": knee_battery,
                },
                "effective_par_to_kw_factor": effective_par_to_kw_factor,
                "baseline": baseline,
                "financial_params": base_financial_params,
                "cases": len(df) - 1,
                "n_jobs": n_jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
