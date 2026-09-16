from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_effective_par_to_kw_factor(output_dir: Path, cfg) -> float:
    contract_path = output_dir / "paper_contract.json"
    if not contract_path.exists():
        return float(cfg.par_to_kw_factor)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))


def load_design_decision_table(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "closing_analyses" / "design_decision_table.csv")


def load_design_points(output_dir: Path) -> dict[str, dict[str, float]]:
    design_df = load_design_decision_table(output_dir)
    return {
        str(row["design_label"]): {
            "solar_panel_factor": float(row["solar_panel_factor"]),
            "battery_factor": float(row["battery_factor"]),
        }
        for row in design_df.to_dict(orient="records")
    }


def aggregate_design_point(
    results_df: pd.DataFrame,
    *,
    solar_panel_factor: float,
    battery_factor: float,
    design_label: str,
) -> dict[str, float | str]:
    subset = results_df[
        (results_df["solar_panel_factor"] == float(solar_panel_factor))
        & (results_df["battery_factor"] == float(battery_factor))
    ]
    totals = subset[
        [
            "grid_only_emission_t",
            "pv_storage_emission_t",
            "abatement_t",
            "grid_only_cost",
            "pv_storage_cost",
            "delta_cost",
            "grid_energy_kwh_20y",
            "pv_storage_energy_kwh_20y",
        ]
    ].sum()
    return {
        "design_label": design_label,
        "solar_panel_factor": float(solar_panel_factor),
        "battery_factor": float(battery_factor),
        "grid_only_emission_t": float(totals["grid_only_emission_t"]),
        "pv_storage_emission_t": float(totals["pv_storage_emission_t"]),
        "abatement_t": float(totals["abatement_t"]),
        "grid_only_cost": float(totals["grid_only_cost"]),
        "pv_storage_cost": float(totals["pv_storage_cost"]),
        "delta_cost": float(totals["delta_cost"]),
        "grid_energy_kwh_20y": float(totals["grid_energy_kwh_20y"]),
        "pv_storage_energy_kwh_20y": float(totals["pv_storage_energy_kwh_20y"]),
    }


def aggregate_design_from_results(results_df: pd.DataFrame, design_label: str, output_dir: Path) -> dict[str, float | str]:
    point = load_design_points(output_dir)[design_label]
    return aggregate_design_point(
        results_df,
        solar_panel_factor=float(point["solar_panel_factor"]),
        battery_factor=float(point["battery_factor"]),
        design_label=design_label,
    )


def portfolio_results(results_df: pd.DataFrame, *, delta_cost_col: str = "delta_cost") -> pd.DataFrame:
    total = results_df.groupby(["solar_panel_factor", "battery_factor"]).agg(
        {
            "grid_only_emission_t": "sum",
            "pv_storage_emission_t": "sum",
            "delta_t": "sum",
            "grid_only_cost": "sum",
            "pv_storage_cost": "sum",
            delta_cost_col: "sum",
        }
    ).reset_index()
    total["abatement_t"] = -total["delta_t"]
    return total


def capital_recovery_factor(years: float, discount_rate: float) -> float:
    if abs(discount_rate) < 1e-12:
        return 1.0 / float(years)
    growth = (1.0 + float(discount_rate)) ** float(years)
    return float(discount_rate) * growth / (growth - 1.0)


def load_raw_paper_settings(default_config_path: str = "config/paper_baseline.yaml") -> dict[str, Any]:
    config_path = Path(os.environ.get("STREETLIGHT_CONFIG", default_config_path))
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(loaded.get("paper", {}))
