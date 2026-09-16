from __future__ import annotations

import json

import pandas as pd

from streetlight.config import get_config


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "manuscript_results_snapshot.json"

    contract = json.loads((output_dir / "paper_contract.json").read_text(encoding="utf-8"))
    streetlight_summary = json.loads((output_dir / "streetlight_sim" / "summary.json").read_text(encoding="utf-8"))
    frontier = pd.read_csv(output_dir / "pareto" / "frontier.csv")
    knee_json = json.loads((output_dir / "pareto" / "knee.json").read_text(encoding="utf-8"))
    designs = pd.read_csv(output_dir / "closing_analyses" / "design_decision_table.csv")

    design_lookup = {
        str(row["design_label"]): row
        for row in designs.to_dict(orient="records")
    }

    knee_row = design_lookup["knee"]
    snapshot = {
        "paper1": {
            "load_mode": str(contract["standardized_installation"]["load_mode"]),
            "lighting_threshold_par": float(contract["standardized_installation"]["lighting_threshold_par"]),
            "solar_zenith_deg": float(contract["standardized_installation"]["solar_zenith_deg"]),
            "effective_par_to_kw_factor": float(contract["effective_par_to_kw_factor"]),
            "frontier_points": int(len(frontier)),
            "fixed_baseline": {
                "total_load_kwh": float(streetlight_summary["total_load_kwh"]),
                "total_grid_import_kwh": float(streetlight_summary["total_grid_import_kwh"]),
                "baseline_grid_only_emission_t": float(streetlight_summary["total_baseline_emission_kg"]) / 1000.0,
                "pv_storage_emission_t": float(streetlight_summary["total_scenario_emission_kg"]) / 1000.0,
                "abatement_t": float(streetlight_summary["total_emission_reduction_kg"]) / 1000.0,
            },
            "representative_designs": {
                label: {
                    "solar_panel_factor": float(row["solar_panel_factor"]),
                    "battery_factor": float(row["battery_factor"]),
                    "abatement_t": float(row["abatement_t"]),
                    "delta_cost_ntd": float(row["delta_cost"]),
                }
                for label, row in design_lookup.items()
            },
            "knee": {
                "frontier_index": int(knee_json["index"]),
                "solar_panel_factor": float(knee_row["solar_panel_factor"]),
                "battery_factor": float(knee_row["battery_factor"]),
                "abatement_t": float(knee_row["abatement_t"]),
                "delta_cost_ntd": float(knee_row["delta_cost"]),
            },
        },
    }

    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
