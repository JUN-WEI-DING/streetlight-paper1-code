from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from paper1_provenance import repo_display_path

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    _build_city_switching_mask,
    load_par_wide,
)


def calibrate_par_to_kw_factor(par_path: Path | None = None, output_dir: Path | None = None) -> dict:
    cfg = get_config()
    standardized = cfg.standardized_installation

    if par_path is None:
        par_path = cfg.paper_par_path
    if output_dir is None:
        output_dir = cfg.paper_output_dir / "calibration"

    output_dir.mkdir(parents=True, exist_ok=True)

    par_df = load_par_wide(par_path)
    dt_h = par_df.index.to_series().diff().dropna().median().total_seconds() / 3600.0

    n_lights = int(standardized["n_lights"])
    light_power_kw = float(standardized["light_power_kw"])
    load_mode = str(standardized["load_mode"])
    solar_zenith_deg = float(standardized.get("solar_zenith_deg", 90.833))
    lighting_threshold_par = float(standardized["lighting_threshold_par"])
    installation_load_kw = n_lights * light_power_kw

    on_mask = pd.DataFrame(index=par_df.index)
    for city in par_df.columns:
        par_series = pd.to_numeric(par_df[city], errors="coerce").fillna(0.0)
        on_mask[city] = _build_city_switching_mask(
            index=par_df.index,
            city=city,
            load_mode=load_mode,
            lighting_threshold_par=lighting_threshold_par,
            par_series=par_series,
            solar_zenith_deg=solar_zenith_deg,
        )

    annual_load_kwh = (on_mask * installation_load_kw * dt_h).sum()
    annual_par_integral = (par_df * dt_h).sum()
    parity_factor = annual_load_kwh / annual_par_integral

    per_city = pd.DataFrame(
        {
            "annual_load_kwh": annual_load_kwh,
            "annual_par_integral": annual_par_integral,
            "k_par_to_pv_parity": parity_factor,
        }
    ).sort_index()
    per_city.index.name = "city"
    per_city.to_csv(output_dir / "per_city_calibration.csv")

    threshold_rows = []
    for threshold in [0.0, 0.5, 1.0, 5.0, 10.0]:
        hours_on = ((par_df <= threshold).astype(float) * dt_h).sum().mean()
        threshold_rows.append(
            {
                "lighting_threshold_par": threshold,
                "mean_annual_lighting_hours": float(hours_on),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    calibrated = round(float(parity_factor.mean()), 4)
    summary = {
        "method": "annual_energy_parity_mean_city",
        "reference_year": 2024,
        "par_path": repo_display_path(par_path),
        "load_mode": load_mode,
        "solar_zenith_deg": solar_zenith_deg,
        "lighting_threshold_par": lighting_threshold_par,
        "n_lights": n_lights,
        "light_power_kw": light_power_kw,
        "installation_load_kw": installation_load_kw,
        "mean_city_parity_factor": float(parity_factor.mean()),
        "median_city_parity_factor": float(parity_factor.median()),
        "rounded_paper_baseline": calibrated,
        "min_city_parity_factor": float(parity_factor.min()),
        "max_city_parity_factor": float(parity_factor.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = calibrate_par_to_kw_factor()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
