"""Sensitivity sweep for AEF-pipeline subjective inputs.

This script perturbs the AEF-pipeline inputs whose values are subjective
(literature-bracketed rather than directly observed):

1. PHS round-trip efficiency
   - Values: [0.75, 0.78 (current), 0.82]
   - Source treatment: scenario bracket for storage-loss sensitivity
   - Each of charge/discharge efficiency = sqrt(RTE)

2. Per-fuel lifecycle EFs
   - Uniform ±20% perturbation on each fuel's median EF
   - Fuels: Coal, LNG, Nuclear, Solar, onshore wind, offshore wind, Hydro,
     Geothermal, Biomass, Co-Gen, Oil, Diesel
   - Source bracketed in ``fuel_perturbations`` block

For each perturbation the script:
  * Re-runs the AEF pipeline with the perturbed input
  * Computes the regional panel mean AEF (kg CO2e/kWh) using the
    ``FLOW_UNIT_FINAL_AEF`` column
  * Reports the absolute and percentage delta vs the in-script baseline run
    (default settings, run once for self-consistency)

Outputs:
  outputs/paper_baseline/subjective_sensitivity/summary.csv
  outputs/paper_baseline/subjective_sensitivity/metadata.json

Usage::

    ulimit -v $((52*1024*1024)) && PYTHONPATH=src \
        STREETLIGHT_CONFIG=config/paper_baseline.yaml \
        python scripts/analysis/run_subjective_sensitivity.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from streetlight.aef.calculator import AEFCalculator
from streetlight.aef.pipeline import AEFPipeline
from streetlight.aef.storage import StorageConfig, StorageManager
from streetlight.config import get_config


AEF_COLUMN = "FLOW_UNIT_FINAL_AEF"


def _build_pipeline(
    *,
    nuclear_ef: float | None = None,
    phs_rte: float | None = None,
    ef_overrides: Dict[str, float] | None = None,
) -> AEFPipeline:
    """Build a fresh AEFPipeline with optional EF / RTE overrides.

    ``nuclear_ef`` is kept as a named arg for backwards compatibility with the
    nuclear-only sweep. ``ef_overrides`` accepts arbitrary fuel-key → EF
    overrides for the per-fuel symmetric sensitivity sweep.
    """
    cfg = get_config()
    ef = AEFCalculator.default_emission_factors()
    if nuclear_ef is not None:
        ef["Nuclear"] = float(nuclear_ef)
    if ef_overrides:
        for k, v in ef_overrides.items():
            ef[k] = float(v)
    pipeline = AEFPipeline(config=cfg, emission_factors=ef)
    if phs_rte is not None:
        eta = math.sqrt(float(phs_rte))
        pipeline.storage_manager = StorageManager(
            phs_config=StorageConfig(charge_efficiency=eta, discharge_efficiency=eta),
        )
    return pipeline


def _panel_mean_aef(region_data: Dict[str, pd.DataFrame]) -> float:
    """Configured-region panel mean of FLOW_UNIT_FINAL_AEF."""
    means = []
    for r in get_config().regions:
        if r in region_data and AEF_COLUMN in region_data[r].columns:
            s = region_data[r][AEF_COLUMN].dropna()
            if not s.empty:
                means.append(float(s.mean()))
    if not means:
        return float("nan")
    return float(np.mean(means))


def _run_with_data_dir(
    data_dir: Path,
    *,
    nuclear_ef: float | None = None,
    phs_rte: float | None = None,
    ef_overrides: Dict[str, float] | None = None,
) -> float:
    """Run AEF pipeline against a given data_dir and return panel mean AEF."""
    pipeline = _build_pipeline(
        nuclear_ef=nuclear_ef, phs_rte=phs_rte, ef_overrides=ef_overrides,
    )
    region_data, _pool = pipeline.run(data_dir=data_dir)
    return _panel_mean_aef(region_data)


def _execute_task(task: dict, baseline_panel_mean: float) -> dict:
    """Generic task runner — runs one AEF pipeline perturbation and returns row dict.

    Task spec keys:
      - parameter, value, notes : pass-through to output row
      - data_dir : Path for AEF pipeline input
      - nuclear_ef, phs_rte, ef_overrides : optional pipeline kwargs
    """
    mean = _run_with_data_dir(
        task["data_dir"],
        nuclear_ef=task.get("nuclear_ef"),
        phs_rte=task.get("phs_rte"),
        ef_overrides=task.get("ef_overrides"),
    )
    delta = mean - baseline_panel_mean
    return {
        "parameter": task["parameter"],
        "value": task["value"],
        "regional_panel_mean_aef": mean,
        "delta_vs_baseline_kgkwh": delta,
        "delta_vs_baseline_pct": (delta / baseline_panel_mean) * 100.0,
        "notes": task["notes"],
    }


def _fuel_param(label: str) -> str:
    return f"fuel_ef_{label.lower().replace(' ', '_').replace('-', '_')}"


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir / "subjective_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    src_data_dir = Path(cfg.power_dir)

    # ---- 1. Establish baseline (sequential — needed for delta calc) ------
    print("[subjective] running baseline...", flush=True)
    baseline_panel_mean = _run_with_data_dir(src_data_dir)
    print(
        f"[subjective] baseline panel mean AEF = "
        f"{baseline_panel_mean:.6f} kg CO2e/kWh",
        flush=True,
    )

    # ---- 2. Build full task list (all parallel-safe) ---------------------
    central_df = pd.read_csv(src_data_dir / "central_unit_generation.csv", nrows=1)
    phs_active = "PHS" in central_df.columns

    phs_rte_values = [0.75, 0.78, 0.82]

    # Uniform ±20% perturbation on each fuel's lifecycle EF median, so every
    # row in the F8 tornado is comparable on identical perturbation magnitude
    # (the standard tornado-chart convention). The wider IPCC AR5 / ecoinvent
    # literature ranges are documented separately in SI Table S10 as a
    # reality-check view.
    PERT = 0.20
    fuel_perturbations = [
        ("Coal", {"Coal": 0.820 * (1 - PERT), "IPP-Coal": 0.820 * (1 - PERT)},
                 {"Coal": 0.820 * (1 + PERT), "IPP-Coal": 0.820 * (1 + PERT)},
                 "±20% on IPCC AR5 pulverised-coal median 0.820"),
        ("LNG", {"LNG": 0.490 * (1 - PERT), "IPP-LNG": 0.490 * (1 - PERT)},
                {"LNG": 0.490 * (1 + PERT), "IPP-LNG": 0.490 * (1 + PERT)},
                "±20% on IPCC AR5 gas-CC median 0.490"),
        ("Nuclear", {"Nuclear": 0.012 * (1 - PERT)},
                    {"Nuclear": 0.012 * (1 + PERT)},
                    "±20% on IPCC AR5 nuclear lifecycle median 0.012"),
        ("Solar", {"Solar": 0.048 * (1 - PERT)},
                  {"Solar": 0.048 * (1 + PERT)},
                  "±20% on IPCC AR5 utility solar PV median 0.048"),
        ("Wind", {"Wind": 0.011 * (1 - PERT)},
                 {"Wind": 0.011 * (1 + PERT)},
                 "±20% on IPCC AR5 onshore wind median 0.011"),
        ("Offshore Wind", {"Offshore Wind": 0.012 * (1 - PERT)},
                          {"Offshore Wind": 0.012 * (1 + PERT)},
                          "±20% on IPCC AR5 offshore wind median 0.012"),
        ("Hydro", {"Hydro": 0.024 * (1 - PERT)},
                  {"Hydro": 0.024 * (1 + PERT)},
                  "±20% on IPCC AR5 hydropower median 0.024"),
        ("Geothermal", {"Geothermal": 0.038 * (1 - PERT)},
                       {"Geothermal": 0.038 * (1 + PERT)},
                       "±20% on IPCC AR5 geothermal median 0.038"),
        ("Biomass", {"Biomass": 0.230 * (1 - PERT)},
                    {"Biomass": 0.230 * (1 + PERT)},
                    "±20% on IPCC AR5 dedicated biomass median 0.230"),
        ("Co-Gen", {"Co-Gen": 0.460 * (1 - PERT)},
                   {"Co-Gen": 0.460 * (1 + PERT)},
                   "±20% on ecoinvent CCGT CHP median 0.460"),
        ("Oil", {"Oil": 0.815 * (1 - PERT)},
                {"Oil": 0.815 * (1 + PERT)},
                "±20% on ecoinvent oil-fired median 0.815"),
        ("Diesel", {"Diesel": 0.880 * (1 - PERT)},
                   {"Diesel": 0.880 * (1 + PERT)},
                   "±20% on ecoinvent diesel-fired median 0.880"),
    ]

    tasks: List[dict] = []

    # PHS RTE (skip 0.78 — equals baseline)
    for rte in phs_rte_values:
        if abs(rte - 0.78) < 1e-9:
            continue
        note = f"PHS RTE perturbed; charge=discharge=sqrt({rte:.2f})={math.sqrt(rte):.4f}"
        if not phs_active:
            note += ("; CAVEAT: PHS aggregate column missing in input data, "
                     "so storage dispatch is inactive and delta is structurally 0")
        tasks.append({
            "parameter": "phs_round_trip_efficiency",
            "value": rte,
            "data_dir": src_data_dir,
            "phs_rte": rte,
            "notes": note,
        })

    # Per-fuel EF (lo + hi for each)
    fuel_perturbation_summary: List[dict] = []
    for label, lo_overrides, hi_overrides, source in fuel_perturbations:
        for endpoint_label, overrides in (("lo", lo_overrides), ("hi", hi_overrides)):
            ef_value = next(iter(overrides.values()))
            tasks.append({
                "parameter": _fuel_param(label),
                "value": f"{endpoint_label}={ef_value:.4f}",
                "data_dir": src_data_dir,
                "ef_overrides": overrides,
                "notes": source,
            })
            fuel_perturbation_summary.append({
                "fuel": label,
                "endpoint": endpoint_label,
                "ef_overrides": overrides,
                "source": source,
            })

    # ---- 4. Run all tasks in parallel ------------------------------------
    n_jobs = min(8, len(tasks))
    print(f"[subjective] dispatching {len(tasks)} perturbation tasks "
          f"across {n_jobs} parallel workers...", flush=True)
    parallel_rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_execute_task)(t, baseline_panel_mean) for t in tasks
    )

    # ---- 5. Add baseline + identity rows + sort --------------------------
    rows: List[dict] = [
        {
            "parameter": "baseline",
            "value": "default",
            "regional_panel_mean_aef": baseline_panel_mean,
            "delta_vs_baseline_kgkwh": 0.0,
            "delta_vs_baseline_pct": 0.0,
            "notes": "default emission factors, default storage RTE",
        },
        # PHS RTE identity row (0.78 = baseline; skipped in parallel sweep)
        {
            "parameter": "phs_round_trip_efficiency",
            "value": 0.78,
            "regional_panel_mean_aef": baseline_panel_mean,
            "delta_vs_baseline_kgkwh": 0.0,
            "delta_vs_baseline_pct": 0.0,
            "notes": "current default (charge=discharge=sqrt(0.78)=0.8832); identical to baseline",
        },
    ]
    rows.extend(parallel_rows)

    # Sort by parameter family + value for stable readable order
    family_order = {"baseline": 0, "phs_round_trip_efficiency": 1}
    def _sort_key(r: dict) -> tuple:
        fam = r["parameter"]
        primary = family_order.get(fam, 4 if fam.startswith("fuel_ef_") else 5)
        return (primary, fam, str(r["value"]))
    rows.sort(key=_sort_key)

    # ---- 6. Print one-line summary per row -------------------------------
    print("\n[subjective] results:")
    for r in rows:
        print(f"  {r['parameter']:30s} {str(r['value']):20s} "
              f"mean={r['regional_panel_mean_aef']:.6f} "
              f"delta={r['delta_vs_baseline_pct']:+.3f}%")

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # ---- Worst-case statistics ------------------------------------------
    non_baseline = summary[summary["parameter"] != "baseline"]
    worst_abs = non_baseline["delta_vs_baseline_kgkwh"].abs().max()
    worst_pct = non_baseline["delta_vs_baseline_pct"].abs().max()
    claim_holds = bool(worst_pct < 1.5)

    metadata = {
        "baseline_panel_mean_aef_kgkwh": baseline_panel_mean,
        "aef_column": AEF_COLUMN,
        "regions": list(cfg.regions),
        "panel_mean_definition": (
            "mean across regions of each region's mean FLOW_UNIT_FINAL_AEF "
            "over all valid timestamps"
        ),
        "perturbations": {
            "phs_round_trip_efficiency_values": phs_rte_values,
            "fuel_ef_perturbations": fuel_perturbation_summary,
        },
        "citations": {
            "ipcc_ar6_wg3_ch6": (
                "IPCC AR6 WG3 Ch.6 — lifecycle GHG range for nuclear power "
                "(median 0.012 kg CO2e/kWh; range 0.0 - 0.025)"
            ),
            "phs_round_trip_efficiency_scenario_bracket": (
                "Pumped-hydro round-trip efficiency scenario bracket "
                "(0.75 - 0.82), used to test storage-loss sensitivity; "
                "not treated as a separate bibliographic citation"
            ),
        },
        "worst_case_absolute_delta_kgkwh": worst_abs,
        "worst_case_percentage_delta": worst_pct,
        "claim_under_1_5_pct_holds": claim_holds,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ---- Final report ----------------------------------------------------
    print("\n" + "=" * 60)
    print(f"summary.csv: {summary_path}")
    print(f"metadata.json: {metadata_path}")
    print(f"Worst-case absolute delta: {worst_abs:.6f} kg CO2e/kWh")
    print(f"Worst-case percentage delta: {worst_pct:.3f} %")
    print(
        f"Claim '<1.5% AEF delta worst case' "
        f"{'HOLDS' if claim_holds else 'DOES NOT HOLD'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
