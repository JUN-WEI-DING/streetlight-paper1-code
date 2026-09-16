from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from calibrate_par_to_kw_factor import calibrate_par_to_kw_factor
from paper1_provenance import resolve_repo_display_path

from streetlight.config import get_config


def _resolve_paths() -> tuple[Path, Path]:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    calibration_dir = output_dir / "calibration"
    robustness_dir = output_dir / "calibration_robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    if not (calibration_dir / "per_city_calibration.csv").exists():
        calibrate_par_to_kw_factor(output_dir=calibration_dir)
    return calibration_dir, robustness_dir


def _threshold_factor_sensitivity(summary: dict, thresholds: list[float]) -> pd.DataFrame:
    par_path = resolve_repo_display_path(summary["par_path"])
    par_wide = pd.read_csv(par_path, parse_dates=[0], index_col=0)
    par_wide = par_wide.apply(pd.to_numeric, errors="coerce")
    index = pd.to_datetime(par_wide.index)
    if len(index) < 2:
        raise ValueError("par_wide.csv must contain at least two timestamps for calibration robustness")
    dt_hours = np.median(np.diff(index.view("int64"))) / 3.6e12
    annual_par_integral = par_wide.fillna(0.0).sum(axis=0) * dt_hours
    n_lights = float(summary["n_lights"])
    light_power_kw = float(summary["light_power_kw"])

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        on_mask = par_wide.le(threshold).fillna(False)
        annual_load_kwh = on_mask.sum(axis=0) * dt_hours * n_lights * light_power_kw
        per_city_factor = annual_load_kwh / annual_par_integral.replace(0.0, np.nan)
        modeled_hours = on_mask.sum(axis=0) * dt_hours
        rows.append(
            {
                "lighting_threshold_par": float(threshold),
                "mean_city_parity_factor": float(per_city_factor.mean()),
                "median_city_parity_factor": float(per_city_factor.median()),
                "std_city_parity_factor": float(per_city_factor.std(ddof=0)),
                "mean_annual_lighting_hours": float(modeled_hours.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    calibration_dir, robustness_dir = _resolve_paths()
    per_city = pd.read_csv(calibration_dir / "per_city_calibration.csv")
    threshold_hours = pd.read_csv(calibration_dir / "threshold_sensitivity.csv")
    summary = json.loads((calibration_dir / "summary.json").read_text(encoding="utf-8"))

    baseline_factor = float(summary["mean_city_parity_factor"])
    per_city = per_city.sort_values("k_par_to_pv_parity").reset_index(drop=True)
    per_city["delta_vs_baseline"] = per_city["k_par_to_pv_parity"] - baseline_factor
    per_city["pct_delta_vs_baseline"] = per_city["delta_vs_baseline"] / baseline_factor * 100.0
    per_city.to_csv(robustness_dir / "city_factor_distribution.csv", index=False)

    loo_rows: list[dict[str, float | str]] = []
    for _, row in per_city.iterrows():
        leave_out_city = row["city"]
        retained = per_city.loc[per_city["city"] != leave_out_city, "k_par_to_pv_parity"]
        loo_factor = float(retained.mean())
        loo_rows.append(
            {
                "left_out_city": leave_out_city,
                "leave_one_out_factor": loo_factor,
                "delta_vs_baseline": loo_factor - baseline_factor,
                "pct_delta_vs_baseline": (loo_factor - baseline_factor) / baseline_factor * 100.0,
            }
        )
    loo = pd.DataFrame(loo_rows).sort_values("pct_delta_vs_baseline", key=lambda s: s.abs(), ascending=False)
    loo.to_csv(robustness_dir / "leave_one_city_out.csv", index=False)

    thresholds = sorted({float(x) for x in threshold_hours["lighting_threshold_par"].tolist()} | {2.0})
    threshold_factor = _threshold_factor_sensitivity(summary, thresholds)
    threshold_factor["delta_vs_baseline_factor"] = threshold_factor["mean_city_parity_factor"] - baseline_factor
    threshold_factor["pct_delta_vs_baseline_factor"] = (
        threshold_factor["delta_vs_baseline_factor"] / baseline_factor * 100.0
    )
    threshold_factor.to_csv(robustness_dir / "threshold_factor_sensitivity.csv", index=False)

    robustness_summary = {
        "baseline_mean_city_parity_factor": baseline_factor,
        "baseline_median_city_parity_factor": float(summary["median_city_parity_factor"]),
        "city_factor_min": float(per_city["k_par_to_pv_parity"].min()),
        "city_factor_max": float(per_city["k_par_to_pv_parity"].max()),
        "city_factor_cv": float(per_city["k_par_to_pv_parity"].std(ddof=0) / baseline_factor),
        "max_abs_city_pct_delta": float(per_city["pct_delta_vs_baseline"].abs().max()),
        "max_abs_leave_one_city_pct_delta": float(loo["pct_delta_vs_baseline"].abs().max()),
        "max_abs_threshold_pct_delta": float(threshold_factor["pct_delta_vs_baseline_factor"].abs().max()),
        "rounded_paper_baseline": float(summary["rounded_paper_baseline"]),
    }
    (robustness_dir / "summary.json").write_text(
        json.dumps(robustness_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
