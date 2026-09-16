"""Conditional economic quantiles and hardware break-even thresholds at fixed design.

Reuses compact dispatch results; no hourly dispatch or design search is run.
PV cases and replacement intervals are conditions, with no probability weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from paper1_config import ROOT, load_paper1_config
from streetlight.lca import hardware

PV_CASES = ("calibrated", "horizontal", "south_tilt20")
LIFETIMES = (8, 10, 12, 15)
SAMPLE_SIZES = (2000, 10000, 50000)
SEED = 20260914
CHECK_SEED = 20260915
INPUT_NAMES = ("electricity_price_ntd_per_kwh", "discount_rate",
               "pv_capex_ntd_per_kw", "battery_cost_multiplier")


def replacement_years(lifetime: float, horizon: float) -> list[float]:
    if lifetime <= 0 or horizon <= 0:
        raise ValueError("Lifetime and horizon must be positive")
    return [lifetime * j for j in range(1, math.floor((horizon - 1e-9) / lifetime) + 1)]


def triangular_parameters(config) -> dict[str, list[float]]:
    costs = config.economic_costs
    scenarios = config.raw["economics"]["scenarios"]
    low, high = scenarios["optimistic"], scenarios["conservative"]
    tariff = config.electricity_price_ntd_per_kwh
    multipliers = config.raw["paper"]["sensitivity"]["electricity_price_multipliers"]
    battery = costs["battery_capex_ntd_per_kwh"]
    return dict(zip(INPUT_NAMES, (
        [tariff * m for m in multipliers],
        [low["financial"]["discount_rate"], config.discount_rate, high["financial"]["discount_rate"]],
        [low["cost"]["pv_capex_ntd_per_kw"], costs["pv_capex_ntd_per_kw"], high["cost"]["pv_capex_ntd_per_kw"]],
        [low["cost"]["battery_capex_ntd_per_kwh"] / battery, 1.0,
         high["cost"]["battery_capex_ntd_per_kwh"] / battery],
    )))


def sample_inputs(parameters: dict, n: int, seed: int) -> np.ndarray:
    """Rows are draws; drawing (n,4) uniforms preserves nested sample prefixes."""
    if n <= 0:
        raise ValueError("Sample count must be positive")
    uniforms = np.random.default_rng(seed).random((n, len(INPUT_NAMES)))
    draws = np.empty_like(uniforms)
    for column, name in enumerate(INPUT_NAMES):
        low, mode, high = parameters[name]
        if not low <= mode <= high or low == high:
            raise ValueError(f"Invalid triangular support for {name}")
        u = uniforms[:, column]
        draws[:, column] = np.where(
            u <= (mode - low) / (high - low),
            low + np.sqrt(u * (high - low) * (mode - low)),
            high - np.sqrt((1 - u) * (high - low) * (high - mode)),
        )
    return draws


def incremental_cost_usd(draws: np.ndarray, *, annual_avoided_kwh: float,
                         lifetime: float, selected: dict, config) -> np.ndarray:
    """Vectorized per-light equivalent of storage.recompute_costs_from_results."""
    tariff, rate, pv_capex, battery_multiplier = draws.T
    years, lights = config.analysis_years, config.lights_per_city
    costs, financial = config.economic_costs, config.financial_params
    pw = np.full_like(rate, years)
    np.divide(1 - (1 + rate) ** -years, rate, out=pw, where=np.abs(rate) >= 1e-12)
    pv_kw = selected["pv_kw_per_streetlight"]
    battery_kwh = selected["battery_kwh_per_streetlight"]
    battery_kw = min(battery_kwh * config.storage_power_kw_per_kwh,
                     config.battery_power_cap_kw_per_city / lights)
    battery_initial = battery_multiplier * (
        battery_kwh * costs["battery_capex_ntd_per_kwh"]
        + battery_kw * costs["battery_power_capex_ntd_per_kw"])
    pv_initial = pv_kw * pv_capex
    other_initial = costs["other_capex"] / lights
    annual = (pv_kw * costs["pv_om_ntd_per_kw_year"]
              + battery_initial * costs["battery_om_fraction_of_capex_per_year"]
              + (costs["other_om"] - costs["grid_fixed_cost"]) / lights / years
              - annual_avoided_kwh * tariff)
    replacement_unit = battery_multiplier * (
        battery_kwh * costs["battery_replacement_energy_capex_ntd_per_kwh"]
        + battery_kw * costs["battery_replacement_power_capex_ntd_per_kw"])
    replacement = sum(replacement_unit / (1 + rate) ** year
                      for year in replacement_years(lifetime, years))
    terminal = (pv_kw * costs["pv_eol_ntd_per_kw"]
                + (costs["battery_eol"] + costs["other_eol"]) / lights
                - pv_initial * financial["pv_salvage_fraction"]
                - battery_initial * financial["battery_salvage_fraction"]
                - other_initial * financial["other_salvage_fraction"])
    return (pv_initial + battery_initial + other_initial + annual * pw + replacement
            + terminal / (1 + rate) ** years) / config.ntd_per_usd


def hardware_components(selected: dict, count: int) -> dict:
    """Decompose the unchanged capacity proxy into scaled production and residual."""
    pv, battery = selected["solar_panel_factor"], selected["battery_factor"]
    stages = hardware.compute_lca_at_knee(pv, battery, battery_replacements=count)
    modules = pv * hardware.MODULES_PER_POLE_PER_FACTOR
    cells = battery * hardware.CELLS_PER_POLE_PER_FACTOR
    pv_production = modules * hardware.G_PV_MODULE / 1000
    battery_initial = cells * hardware.G_BATTERY_PER_UNIT / 1000
    production = pv_production + (1 + count) * battery_initial
    increment = (sum(stages["SOLAR"].values()) - sum(stages["TRAD"].values())) / 1000
    return {
        "pv_modules_per_streetlight": modules,
        "battery_cell_bms_units_per_streetlight_initial": cells,
        "battery_cell_bms_units_per_streetlight_replacements": count * cells,
        "proxy_pv_mass_kg_per_streetlight": modules * hardware.PV_MODULE_MASS_KG,
        "proxy_battery_system_mass_kg_per_streetlight_initial": cells * hardware.BATT_SYSTEM_MASS_KG,
        "proxy_battery_system_mass_kg_per_streetlight_replacements": count * cells * hardware.BATT_SYSTEM_MASS_KG,
        "pv_module_production_t_per_streetlight": pv_production,
        "battery_initial_production_t_per_streetlight": battery_initial,
        "battery_replacement_production_t_per_streetlight": count * battery_initial,
        "scaled_production_t_per_streetlight": production,
        "other_incremental_hardware_t_per_streetlight": increment - production,
        "hardware_addition_t_per_streetlight": increment,
        "stage_increment_t_per_streetlight": {
            stage: (stages["SOLAR"][stage] - stages["TRAD"][stage]) / 1000
            for stage in stages["SOLAR"]},
        "coefficients_kg_co2e_per_unit": {
            "pv_module": hardware.G_PV_MODULE, "battery_cell_bms": hardware.G_BATTERY_PER_UNIT},
        "proxy_mass_kg_per_unit": {"pv_module": hardware.PV_MODULE_MASS_KG,
                                   "battery_system_including_enclosure": hardware.BATT_SYSTEM_MASS_KG},
    }


def break_even_multiplier(operational_t: float, components: dict) -> float:
    return ((operational_t - components["other_incremental_hardware_t_per_streetlight"])
            / components["scaled_production_t_per_streetlight"])


def prepare_conditions(values: dict, pv_comparison: dict, config) -> list[dict]:
    selected = values["frontier"]["knee"]
    baseline = pv_comparison["baseline_dispatch"]
    energy = values["discount_rate"]["energy"]
    avoided = (energy["grid_only_kwh_20y_per_streetlight"]
               - energy["retrofit_grid_kwh_20y_per_streetlight"]) / config.analysis_years
    rate = config.discount_rate
    pw = sum((1 + rate) ** -year for year in range(1, int(config.analysis_years) + 1))
    baseline_cost = baseline["summary"]["incremental_cost_usd_per_streetlight"]
    deterministic_draw = np.array([[v[1] for v in triangular_parameters(config).values()]])
    lifetimes = {row["battery_lifetime_years"]: row for row in values["battery_lifetime"]["scenarios"]}
    conditions = []
    for name in PV_CASES:
        dispatch = baseline if name == "calibrated" else pv_comparison["cases"][name]["dispatch"]
        summary = dispatch["summary"]
        # Fixed hardware/financial assumptions mean the PV-case cost delta is
        # exactly the change in the present value of imported electricity.
        case_avoided = avoided - (summary["incremental_cost_usd_per_streetlight"] - baseline_cost) * config.ntd_per_usd / (config.electricity_price_ntd_per_kwh * pw)
        city_min = min(dispatch["cities"], key=lambda r: r["operational_abatement_t_per_streetlight"])
        operational = summary["operational_abatement_t_per_streetlight"]
        for lifetime in LIFETIMES:
            years = replacement_years(lifetime, config.analysis_years)
            components = hardware_components(selected, len(years))
            cost = float(incremental_cost_usd(deterministic_draw, annual_avoided_kwh=case_avoided,
                         lifetime=lifetime, selected=selected, config=config)[0])
            reference = (lifetimes[lifetime]["delta_cost_usd_per_streetlight"]
                         + summary["incremental_cost_usd_per_streetlight"] - baseline_cost)
            if not math.isclose(cost, reference, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Deterministic cost differs from existing results: {name}/{lifetime}")
            net = operational - components["hardware_addition_t_per_streetlight"]
            minimum_city_net = city_min["operational_abatement_t_per_streetlight"] - components["hardware_addition_t_per_streetlight"]
            if min(net, minimum_city_net) <= 0:
                raise ValueError("MAC denominators require positive fixed life-cycle abatement")
            conditions.append({
                "pv_case": name, "battery_lifetime_years": lifetime,
                "replacement_years": years, "replacement_count": len(years),
                "annual_avoided_grid_kwh_per_streetlight": case_avoided,
                "deterministic": {
                    "incremental_cost_usd_per_streetlight": cost,
                    "operational_abatement_t_per_streetlight": operational,
                    "net_lifecycle_abatement_t_per_streetlight": net,
                    "minimum_city_net_lifecycle_abatement_t_per_streetlight": minimum_city_net,
                    "operational_mac_usd_per_t": cost / operational,
                    "lifecycle_mac_usd_per_t": cost / net,
                    "existing_cost_absolute_error_usd": abs(cost - reference),
                },
                "hardware_components": components,
                "hardware_break_even": {
                    "mean_city_k": break_even_multiplier(operational, components),
                    "minimum_city_k": break_even_multiplier(city_min["operational_abatement_t_per_streetlight"], components),
                    "minimum_city": city_min["city"],
                },
            })
    return conditions


def distribution_summary(costs: np.ndarray, deterministic: dict) -> dict:
    def quantiles(samples):
        q = np.quantile(samples, [.05, .50, .95], method="linear")
        return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]),
                "sample_minimum": float(np.min(samples))}
    return {
        "incremental_cost_usd_per_streetlight": quantiles(costs),
        "operational_mac_usd_per_t": quantiles(costs / deterministic["operational_abatement_t_per_streetlight"]),
        "lifecycle_mac_usd_per_t": quantiles(costs / deterministic["net_lifecycle_abatement_t_per_streetlight"]),
        "cost_positive_sample_fraction": float(np.mean(costs > 0)),
    }


def build_conditional_uncertainty(values: dict, pv_comparison: dict, config) -> dict:
    parameters = triangular_parameters(config)
    primary = sample_inputs(parameters, SAMPLE_SIZES[-1], SEED)
    independent = sample_inputs(parameters, SAMPLE_SIZES[-1], CHECK_SEED)
    conditions = prepare_conditions(values, pv_comparison, config)
    selected = values["frontier"]["knee"]
    for condition in conditions:
        kwargs = dict(annual_avoided_kwh=condition["annual_avoided_grid_kwh_per_streetlight"],
                      lifetime=condition["battery_lifetime_years"], selected=selected, config=config)
        costs = incremental_cost_usd(primary, **kwargs)
        check_costs = incremental_cost_usd(independent, **kwargs)
        condition["conditional_quantiles"] = distribution_summary(costs, condition["deterministic"])
        condition["convergence"] = {
            str(n): distribution_summary(costs[:n], condition["deterministic"]) for n in SAMPLE_SIZES}
        condition["independent_seed_check"] = distribution_summary(check_costs, condition["deterministic"])
    # Hash only consumed sections: embedding this result into values must not
    # create a self-referential whole-file provenance dependency.
    consumed_values = {"selected": selected, "discount_energy": values["discount_rate"]["energy"],
                       "battery_lifetime_scenarios": values["battery_lifetime"]["scenarios"]}
    semantic_hash = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    source_paths = (Path(__file__), Path(hardware.__file__), ROOT / "src/streetlight/simulation/storage.py",
                    ROOT / "scripts/analysis/paper1_config.py", config.config_path)
    return {
        "schema_version": 1,
        "scope": "Economic uncertainty conditional on each fixed PV model and battery replacement interval; equal-city mean per streetlight at the selected allocation.",
        "parameters": {
            "analysis_years": config.analysis_years, "ntd_per_usd": config.ntd_per_usd,
            "selected_design": selected, "pv_cases": list(PV_CASES), "battery_lifetime_years": list(LIFETIMES),
            "sample_sizes": list(SAMPLE_SIZES), "seed": SEED, "independent_check_seed": CHECK_SEED,
            "sampler": "NumPy PCG64 (N,4) independent uniforms, inverse triangular CDF; common draws across all conditions",
            "numpy_version": np.__version__, "input_column_order": list(INPUT_NAMES),
            "triangular_low_mode_high": parameters, "quantile_method": "linear",
        },
        "assumptions": {
            "interpretation": "p05/p50/p95 are conditional economic scenario quantiles, not an overall confidence interval; no probability is assigned to PV models, replacement intervals, or hardware coefficients.",
            "economics": "Four independent triangular inputs; a shared battery cost multiplier scales initial energy/power CAPEX, associated annual O&M, and replacement energy/power purchases. Other costs and salvage fractions retain the existing engine conventions.",
            "dispatch": "Stored dispatch for each PV model; selected capacities and allocation fixed. Nominal battery capacity and efficiency unchanged; replacements strictly before year 20.",
            "energy_recovery": "Baseline from values.discount_rate.energy; alternative annual avoided imports recovered from PV-case incremental-cost difference divided by baseline tariff and present-worth factor.",
            "hardware_k": "Common multiplier k on PV-module and battery cell/BMS production (initial and every replacement). All remaining incremental hardware terms, including freight, enclosure, MPPT, installation and calibrated end-of-life treatment, fixed within each condition. Break-even k=(operational abatement-other incremental hardware)/scaled production; k=1 is the existing proxy.",
            "carbon": "Physical operational and hardware emissions are undiscounted; no hardware probability distribution or probabilistic claim about positive life-cycle benefit.",
            "minimum": "Sample minima and positive-cost fractions describe simulated economic draws only.",
        },
        "provenance": {
            "semantic_input_sha256": {"consumed_values_sections": semantic_hash(consumed_values),
                                      "pv_model_comparison": semantic_hash(pv_comparison)},
            "consumed_values_sections": ["values.frontier.knee", "values.discount_rate.energy", "values.battery_lifetime.scenarios"],
            "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        },
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_paper1_config(args.config)
    values = json.loads(config.manuscript_values_path.read_text(encoding="utf-8"))["values"]
    pv = json.loads((config.canonical_results_dir / "pv_model_comparison.json").read_text(encoding="utf-8"))
    result = build_conditional_uncertainty(values, pv, config)
    output = args.output or config.canonical_results_dir / "conditional_uncertainty.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['conditions'])} conditions with {SAMPLE_SIZES[-1]} draws and independent seed check: {output}")


if __name__ == "__main__":
    main()
