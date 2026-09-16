"""Deterministic artificial example; no observations or paper-result assertions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import yaml
from streetlight.simulation.storage import compute_city_emission, storage_dispatch


def run_demo() -> tuple[pd.DataFrame, dict]:
    # Explicit settings avoid dependence on a caller's config environment.
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/paper_baseline.yaml").read_text())
    index = pd.date_range("2024-01-01", periods=288, freq="10min", tz="UTC")
    local = index.tz_convert("Asia/Taipei")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    # Illustrative standardized 64-light installation, with fictional solar shape.
    pv = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) * 12.0
    load = np.where((hour < 6) | (hour >= 18), 6.4, 0.0)
    aef = pd.Series(0.5 + 0.05 * np.cos(2 * np.pi * hour / 24), index=index)
    net = pd.Series(pv - load, index=index)
    params = dict(capacity_kwh=20.0, power_kw=8.0, eta_roundtrip=0.9, soc0_kwh=0.0)
    dispatch = storage_dispatch(net, dt_hours=1/6, **params)
    metrics = compute_city_emission(net, aef, dt_hours=1/6, years=20,
        solar_panel_factor=1.0, battery_factor=2.0, params=params,
        economic_costs=config["economics"]["cost"], financial_params=config["economics"]["financial"],
        elec_price=3.7556, analysis_scaling_factor=20 * 8760 / 48)
    series = pd.DataFrame({"pv_kw": pv, "load_kw": load, "net_kw": net,
        "aef_kg_co2e_per_kwh": aef, **dispatch})
    summary = {"input_kind": "synthetic; not paper observations", "steps": len(index),
        "dt_hours": 1/6, "comparison": "pre-storage net deficit versus post-storage import",
        "scaling": "48 artificial hours repeated to 20 * 365 days; illustrative only",
        "metrics_installation_64_lights": metrics}
    return series, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    series, summary = run_demo()
    args.output.mkdir(parents=True, exist_ok=True)
    series.to_csv(args.output / "dispatch.csv", index_label="timestamp_utc")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
