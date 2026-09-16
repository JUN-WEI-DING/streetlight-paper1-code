from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import load_region_city_map
from streetlight.simulation.storage import _discount_lump, _present_worth_factor


def _sample_triangular(rng: np.random.Generator, low: float, mode: float, high: float, size: int) -> np.ndarray:
    return rng.triangular(low, mode, high, size=size)


def _sample_replacement_year(
    rng: np.random.Generator,
    values: list[float],
    probabilities: list[float],
    size: int,
) -> np.ndarray:
    return rng.choice(values, size=size, p=probabilities)


def _compute_cost_distribution(
    *,
    row: pd.Series,
    years: float,
    pv_capacity_kw_per_factor: float,
    n_deployments: int,
    n_samples: int,
    seed: int,
    economic_costs: dict | None = None,
    financial_params: dict | None = None,
    economic_scenarios: dict | None = None,
    sensitivity_cfg: dict | None = None,
    uncertainty_cfg: dict | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if (
        economic_costs is None
        or financial_params is None
        or economic_scenarios is None
        or sensitivity_cfg is None
        or uncertainty_cfg is None
    ):
        cfg = get_config()
        economic_costs = dict(cfg.economic_costs)
        financial_params = dict(cfg.financial_params)
        economic_scenarios = dict(cfg.get("economics.scenarios", {}))
        sensitivity_cfg = dict(cfg.paper_settings("sensitivity", {}))
        uncertainty_cfg = dict(cfg.paper_settings("uncertainty", {}))
        uncertainty_cfg["base_electricity_price"] = float(cfg.electricity_price)

    conservative = dict(economic_scenarios.get("conservative", {}))
    optimistic = dict(economic_scenarios.get("optimistic", {}))
    conservative_cost = dict(conservative.get("cost", {}))
    optimistic_cost = dict(optimistic.get("cost", {}))
    conservative_financial = dict(conservative.get("financial", {}))
    optimistic_financial = dict(optimistic.get("financial", {}))

    electricity_multipliers = [
        float(v) for v in sensitivity_cfg.get("electricity_price_multipliers", [0.8, 1.0, 1.2])
    ]
    base_electricity_price = float(uncertainty_cfg["base_electricity_price"])
    electricity_price = _sample_triangular(
        rng,
        base_electricity_price * electricity_multipliers[0],
        base_electricity_price * electricity_multipliers[1],
        base_electricity_price * electricity_multipliers[2],
        n_samples,
    )
    discount_rate = _sample_triangular(
        rng,
        float(optimistic_financial.get("discount_rate", 0.03)),
        float(financial_params["discount_rate"]),
        float(conservative_financial.get("discount_rate", 0.08)),
        n_samples,
    )
    pv_capex_ntd_per_kw = _sample_triangular(
        rng,
        float(optimistic_cost.get("pv_capex_ntd_per_kw", economic_costs["pv_capex_ntd_per_kw"])),
        float(economic_costs["pv_capex_ntd_per_kw"]),
        float(conservative_cost.get("pv_capex_ntd_per_kw", economic_costs["pv_capex_ntd_per_kw"])),
        n_samples,
    )
    base_battery_capex = float(economic_costs["battery_capex_ntd_per_kwh"])
    battery_multiplier = _sample_triangular(
        rng,
        float(optimistic_cost.get("battery_capex_ntd_per_kwh", base_battery_capex)) / base_battery_capex,
        1.0,
        float(conservative_cost.get("battery_capex_ntd_per_kwh", base_battery_capex)) / base_battery_capex,
        n_samples,
    )
    replacement_year_values = [
        float(v) for v in uncertainty_cfg.get("battery_replacement_year_values", [10, 12, 14])
    ]
    replacement_year_probabilities = [
        float(v)
        for v in uncertainty_cfg.get("battery_replacement_year_probabilities", [0.25, 0.5, 0.25])
    ]
    if len(inspect.signature(_sample_replacement_year).parameters) <= 2:
        # Older tests monkeypatch this sampler with the pre-config signature.
        battery_replacement_year = _sample_replacement_year(rng, n_samples)
    else:
        battery_replacement_year = _sample_replacement_year(
            rng,
            replacement_year_values,
            replacement_year_probabilities,
            n_samples,
        )

    pv_om_ntd_per_kw_year = np.full(n_samples, float(economic_costs["pv_om_ntd_per_kw_year"]))
    battery_om_fraction = np.full(
        n_samples,
        float(economic_costs["battery_om_fraction_of_capex_per_year"]),
    )
    replacement_energy_capex = (
        float(economic_costs["battery_replacement_energy_capex_ntd_per_kwh"]) * battery_multiplier
    )
    replacement_power_capex = np.zeros(n_samples)
    battery_capex_ntd_per_kwh = float(economic_costs["battery_capex_ntd_per_kwh"]) * battery_multiplier
    battery_power_capex_ntd_per_kw = (
        float(economic_costs["battery_power_capex_ntd_per_kw"]) * battery_multiplier
    )

    pv_capacity_kw = pv_capacity_kw_per_factor * float(row["solar_panel_factor"])
    battery_capacity_kwh = float(row["battery_capacity_kwh"])
    battery_power_kw = float(row["battery_power_kw"])

    grid_energy_kwh_20y = float(row["grid_energy_kwh_20y"])
    pv_storage_energy_kwh_20y = float(row["pv_storage_energy_kwh_20y"])

    other_capex = float(economic_costs["other_capex"]) * n_deployments
    other_om = float(economic_costs["other_om"]) * n_deployments
    other_eol = float(economic_costs["other_eol"]) * n_deployments
    grid_fixed_cost = float(economic_costs["grid_fixed_cost"]) * n_deployments

    rows = []
    for i in range(n_samples):
        pw_factor = _present_worth_factor(years, float(discount_rate[i]))

        annual_grid_energy = grid_energy_kwh_20y / years
        annual_pv_grid_energy = pv_storage_energy_kwh_20y / years

        grid_only_energy_cost = annual_grid_energy * float(electricity_price[i]) * pw_factor
        grid_only_fixed_cost = (grid_fixed_cost / years) * pw_factor
        grid_only_cost = grid_only_energy_cost + grid_only_fixed_cost

        battery_init_cost = n_deployments * (
            battery_capacity_kwh * float(battery_capex_ntd_per_kwh[i])
            + battery_power_kw * float(battery_power_capex_ntd_per_kw[i])
        )
        pv_init_cost = n_deployments * pv_capacity_kw * float(pv_capex_ntd_per_kw[i])
        init_cost = pv_init_cost + battery_init_cost + other_capex

        om_cost = (
            n_deployments * pv_capacity_kw * float(pv_om_ntd_per_kw_year[i]) * pw_factor
            + battery_init_cost * float(battery_om_fraction[i]) * pw_factor
            + (other_om / years) * pw_factor
        )

        replacement_cost = 0.0
        replacement_year = float(battery_replacement_year[i])
        n_repl = int(np.floor((years - 1e-9) / replacement_year))
        for j in range(1, n_repl + 1):
            replacement_cost += _discount_lump(
                n_deployments
                * (
                    battery_capacity_kwh * float(replacement_energy_capex[i])
                    + battery_power_kw * float(replacement_power_capex[i])
                ),
                replacement_year * j,
                float(discount_rate[i]),
            )

        eol_cost = _discount_lump(other_eol, years, float(discount_rate[i])) + replacement_cost
        pv_storage_energy_cost = annual_pv_grid_energy * float(electricity_price[i]) * pw_factor
        pv_storage_cost = pv_storage_energy_cost + init_cost + om_cost + eol_cost
        delta_cost = pv_storage_cost - grid_only_cost
        abatement_t = float(row["abatement_t"])

        rows.append(
            {
                "electricity_price": float(electricity_price[i]),
                "discount_rate": float(discount_rate[i]),
                "pv_capex_ntd_per_kw": float(pv_capex_ntd_per_kw[i]),
                "battery_multiplier": float(battery_multiplier[i]),
                "battery_replacement_year": replacement_year,
                "grid_only_cost": grid_only_cost,
                "pv_storage_cost": pv_storage_cost,
                "delta_cost": delta_cost,
                "abatement_t": abatement_t,
                "macc": (delta_cost / abatement_t) if abatement_t != 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _process_design(
    idx: int,
    row: pd.Series,
    *,
    years: float,
    pv_capacity_kw_per_factor: float,
    n_deployments: int,
    n_samples: int,
    base_seed: int,
    uncertainty_dir: Path,
    economic_costs: dict,
    financial_params: dict,
    economic_scenarios: dict,
    sensitivity_cfg: dict,
    uncertainty_cfg: dict,
) -> dict:
    """Compute distribution + summary for a single design point. Worker-safe."""
    dist = _compute_cost_distribution(
        row=row,
        years=years,
        pv_capacity_kw_per_factor=pv_capacity_kw_per_factor,
        n_deployments=n_deployments,
        n_samples=n_samples,
        seed=base_seed + idx,
        economic_costs=economic_costs,
        financial_params=financial_params,
        economic_scenarios=economic_scenarios,
        sensitivity_cfg=sensitivity_cfg,
        uncertainty_cfg=uncertainty_cfg,
    )
    label = str(row["design_label"])
    # Each worker writes to its own file — no race condition.
    dist.to_csv(uncertainty_dir / f"{label}_samples.csv", index=False)

    return {
        "idx": idx,
        "design_label": label,
        "n_samples": n_samples,
        "delta_cost_p05": float(dist["delta_cost"].quantile(0.05)),
        "delta_cost_p50": float(dist["delta_cost"].quantile(0.50)),
        "delta_cost_p95": float(dist["delta_cost"].quantile(0.95)),
        "macc_p05": float(dist["macc"].quantile(0.05)),
        "macc_p50": float(dist["macc"].quantile(0.50)),
        "macc_p95": float(dist["macc"].quantile(0.95)),
        "probability_cost_saving": float((dist["delta_cost"] < 0).mean()),
    }


def _resolve_n_jobs(arg_n_jobs: int | None) -> int:
    """Resolve n_jobs from CLI flag, env var, or default.

    Order: explicit CLI flag > STREETLIGHT_N_JOBS env var > default (-1, all cores).
    A safety cap of 8 is applied unless the user explicitly requests more,
    to keep total memory under the project's 52 GB virtual cap.
    """
    if arg_n_jobs is not None:
        return int(arg_n_jobs)
    env_val = os.environ.get("STREETLIGHT_N_JOBS")
    if env_val is not None and env_val.strip() != "":
        return int(env_val)
    # Default: use all cores but cap at 8 for memory safety (~2 GB/worker).
    n_cpu = os.cpu_count() or 1
    return min(n_cpu, 8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run probabilistic Monte Carlo uncertainty analysis for the paper "
            "design points. Recommended invocation:\n"
            "  ulimit -v $((52*1024*1024)) && "
            "python scripts/analysis/run_probabilistic_uncertainty.py --n-jobs 8"
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
    closing_dir = output_dir / "closing_analyses"
    uncertainty_dir = output_dir / "probabilistic_uncertainty"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    design_df = pd.read_csv(closing_dir / "design_decision_table.csv")
    years = float(cfg.analysis_years)
    pv_capacity_kw_per_factor = float(cfg.economic_costs["pv_capacity_kw_per_factor"])
    region_map_path = cfg.paper_region_map_path
    region_city_map = load_region_city_map(region_map_path)
    n_deployments = int(sum(len(v) for v in region_city_map.values()))
    uncertainty_cfg = dict(cfg.paper_settings("uncertainty", {"n_samples": 2000, "seed": 20260330}))
    uncertainty_cfg["base_electricity_price"] = float(cfg.electricity_price)
    sensitivity_cfg = dict(cfg.paper_settings("sensitivity", {}))
    economic_scenarios = dict(cfg.get("economics.scenarios", {}))
    n_samples = int(uncertainty_cfg.get("n_samples", 2000))
    seed = int(uncertainty_cfg.get("seed", 20260330))

    print(
        f"[uncertainty] n_designs={len(design_df)} n_samples={n_samples} "
        f"n_jobs={n_jobs}"
    )

    # Each design point uses a unique seed (base + idx), so MC samples are
    # deterministic and independent of execution order. Workers write
    # per-design CSVs to separate files; the summary is aggregated here.
    summary_rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_design)(
            idx,
            row,
            years=years,
            pv_capacity_kw_per_factor=pv_capacity_kw_per_factor,
            n_deployments=n_deployments,
            n_samples=n_samples,
            base_seed=seed,
            uncertainty_dir=uncertainty_dir,
            economic_costs=dict(cfg.economic_costs),
            financial_params=dict(cfg.financial_params),
            economic_scenarios=economic_scenarios,
            sensitivity_cfg=sensitivity_cfg,
            uncertainty_cfg=uncertainty_cfg,
        )
        for idx, row in design_df.iterrows()
    )

    # Sort by original idx for deterministic output ordering, then drop idx.
    summary_rows = sorted(summary_rows, key=lambda r: r["idx"])
    for r in summary_rows:
        r.pop("idx", None)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(uncertainty_dir / "summary.csv", index=False)
    (uncertainty_dir / "summary.json").write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "type": "engineering_probabilistic_uncertainty",
        "scope": "economic uncertainty for representative design points",
        "n_samples": n_samples,
        "seed": seed,
        "n_deployments": n_deployments,
        "n_jobs": n_jobs,
        "sampled_parameters": {
            "electricity_price": "triangular(0.8x, 1.0x, 1.2x baseline)",
            "discount_rate": "triangular(config optimistic, base, conservative discount_rate)",
            "pv_capex_ntd_per_kw": "triangular(config optimistic, base, conservative pv_capex_ntd_per_kw)",
            "battery_cost_multiplier": "triangular(config optimistic/base/conservative battery_capex ratios) applied to energy, power, and replacement energy costs",
            "battery_replacement_year": "categorical values/probabilities from paper.uncertainty",
        },
    }
    (uncertainty_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
