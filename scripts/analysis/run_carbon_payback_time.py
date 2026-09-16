from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from streetlight.config import get_config
from streetlight.lca import compute_lca_at_knee


ROOT = Path(__file__).resolve().parents[2]


def _hardware_delta_t_per_streetlight(pv_factor: float, battery_factor: float) -> tuple[float, float]:
    """Return initial and non-operational hardware deltas per streetlight.

    The same SimaPro-calibrated helper used by Figure 3 supplies the hardware
    stages, so carbon-payback evidence stays aligned with the manuscript's
    life-cycle balance.
    """
    lca = compute_lca_at_knee(pv_factor, battery_factor)
    solar = lca["SOLAR"]
    trad = lca["TRAD"]
    initial_stages = ("A1_A3", "A4", "A5")
    non_operational_stages = ("A1_A3", "A4", "A5", "B", "C")
    initial_delta_kg = sum(solar[s] - trad[s] for s in initial_stages)
    non_operational_delta_kg = sum(solar[s] - trad[s] for s in non_operational_stages)
    return initial_delta_kg / 1000.0, non_operational_delta_kg / 1000.0


def _lookup_design_abatement(results_path: Path, design_row: pd.Series) -> float:
    results_df = pd.read_csv(results_path)
    subset = results_df[
        (results_df["solar_panel_factor"] == float(design_row["solar_panel_factor"]))
        & (results_df["battery_factor"] == float(design_row["battery_factor"]))
    ]
    return float(subset["abatement_t"].sum())


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    analysis_dir = output_dir / "carbon_payback"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    design_df = pd.read_csv(output_dir / "closing_analyses" / "design_decision_table.csv")
    future_grid_dir = output_dir / "future_grid_scenarios"
    region_city_map = json.loads((ROOT / "config" / "region_city_map.json").read_text(encoding="utf-8"))
    n_reporting_lights = sum(len(cities) for cities in region_city_map.values()) * int(cfg.n_lights)

    rows = []
    for _, row in design_df.iterrows():
        additional_initial_t, additional_non_operational_t = _hardware_delta_t_per_streetlight(
            float(row["solar_panel_factor"]),
            float(row["battery_factor"]),
        )
        current_abatement_t = float(row["abatement_t"]) / n_reporting_lights
        annual_current = current_abatement_t / float(cfg.analysis_years)
        annual_2030 = (
            _lookup_design_abatement(future_grid_dir / "official_2030_grid_target_results.csv", row)
            / n_reporting_lights
            / float(cfg.analysis_years)
        )
        annual_2050 = (
            _lookup_design_abatement(future_grid_dir / "pathway_2050_midpoint_proxy_results.csv", row)
            / n_reporting_lights
            / float(cfg.analysis_years)
        )

        def _payback(additional_t: float, annual_abatement_t: float):
            if annual_abatement_t <= 0:
                return None
            if additional_t <= 0:
                return 0.0
            return additional_t / annual_abatement_t

        rows.append(
            {
                "design_label": row["design_label"],
                "solar_panel_factor": float(row["solar_panel_factor"]),
                "battery_factor": float(row["battery_factor"]),
                "additional_initial_carbon_tco2e_per_streetlight": additional_initial_t,
                "additional_non_operational_carbon_tco2e_per_streetlight": additional_non_operational_t,
                "annual_abatement_current_tco2e_per_year": annual_current,
                "annual_abatement_2030_tco2e_per_year": annual_2030,
                "annual_abatement_2050_proxy_tco2e_per_year": annual_2050,
                "initial_carbon_payback_years_current": _payback(additional_initial_t, annual_current),
                "non_operational_carbon_payback_years_current": _payback(additional_non_operational_t, annual_current),
                "initial_carbon_payback_years_2030": _payback(additional_initial_t, annual_2030),
                "non_operational_carbon_payback_years_2030": _payback(additional_non_operational_t, annual_2030),
                "initial_carbon_payback_years_2050_proxy": _payback(additional_initial_t, annual_2050),
                "non_operational_carbon_payback_years_2050_proxy": _payback(additional_non_operational_t, annual_2050),
                "payback_within_horizon_current": _payback(additional_non_operational_t, annual_current) is not None
                and _payback(additional_non_operational_t, annual_current) <= float(cfg.analysis_years),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(analysis_dir / "representative_designs.csv", index=False)
    (analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "interpretation": (
                    "Carbon payback uses the SimaPro-calibrated hardware-stage difference "
                    "from streetlight.lca.compute_lca_at_knee, divided by annual per-streetlight "
                    "operational abatement under current and future-grid states. The payback basis uses "
                    "additional upfront or non-operational carbon only, so operational "
                    "electricity impacts are not double-counted."
                ),
                "reporting_lights": n_reporting_lights,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
