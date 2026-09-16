"""Paper 1 analysis suite orchestrator.

Runs the analyses required to produce the Paper 1 manuscript
(``docs/paper1/manuscript/prose.md``) and publication figures. The default
suite is intentionally limited to manuscript-facing Paper 1 outputs:
baseline design space, calibration and method checks, future-grid stress
tests, uncertainty/sensitivity views, marginal-cost lens, carbon payback,
degradation sensitivity, and the compact manuscript snapshot consumed by
Figure 2. The outlying-island split is part of the reference AEF workflow.

Usage:
    PYTHONPATH=src STREETLIGHT_CONFIG=config/paper_baseline.yaml \
        python scripts/analysis/run_paper1_suite.py

With ``config/paper_baseline.yaml`` this writes the canonical result bundle to
``outputs/final_runs/paper1_canonical_results/``.  Manuscript and SI rendering
consume that bundle rather than hand-maintained numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

from build_manuscript_results_snapshot import main as build_manuscript_results_snapshot
from paper1_provenance import repo_display_path
from run_calibration_robustness import main as run_calibration_robustness
from run_carbon_payback_time import main as run_carbon_payback_time
from run_closing_analyses import main as run_closing_analyses
from run_degradation_sensitivity import main as run_degradation_sensitivity
from run_future_grid_scenarios import main as run_future_grid_scenarios
from run_marginal_cost_economic_lens import main as run_marginal_cost_economic_lens
from run_merit_order_marginal_carbon import main as run_merit_order_marginal_carbon
from run_method_simplifications import main as run_method_simplifications
from run_paper_baseline import main as run_paper_baseline
from run_paper_sensitivity import main as run_paper_sensitivity
from run_probabilistic_uncertainty import main as run_probabilistic_uncertainty
from run_subjective_sensitivity import main as run_subjective_sensitivity

from streetlight.config import get_config


def _run_step(name: str, fn: Callable[[], None]) -> dict[str, float | str]:
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    return {
        "step": name,
        "elapsed_seconds": round(elapsed, 6),
    }


def main(*, start_at: str | None = None) -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Order is load-bearing:
    # 1. paper_baseline produces the canonical design space + frontier that all
    #    downstream Paper 1 analyses read.
    # 2. calibration_robustness needs the calibration directory written by
    #    paper_baseline (via calibrate_par_to_kw_factor).
    # 3. method_simplifications, closing_analyses, paper_sensitivity, and
    #    probabilistic_uncertainty all consume the baseline frontier/contract.
    # 4. future_grid_scenarios and degradation_sensitivity feed §3.8 of Paper 1.
    # 5. merit_order_marginal produces the marginal-cost/carbon proxy consumed
    #    by marginal_cost_economic_lens; both feed §3.5.
    # 6. carbon_payback_time consumes the closing_analyses design table.
    # 7. manuscript_results_snapshot writes only the compact Paper 1 snapshot
    #    required by Figure 2 and archive/readiness checks.
    steps: list[tuple[str, Callable[[], None]]] = [
        ("paper_baseline", run_paper_baseline),
        ("calibration_robustness", run_calibration_robustness),
        ("method_simplifications", run_method_simplifications),
        ("closing_analyses", run_closing_analyses),
        ("future_grid_scenarios", run_future_grid_scenarios),
        ("probabilistic_uncertainty", run_probabilistic_uncertainty),
        ("paper_sensitivity", run_paper_sensitivity),
        ("subjective_sensitivity", run_subjective_sensitivity),
        ("merit_order_marginal", run_merit_order_marginal_carbon),
        ("marginal_cost_economic_lens", run_marginal_cost_economic_lens),
        ("carbon_payback", run_carbon_payback_time),
        ("degradation_sensitivity", run_degradation_sensitivity),
        ("manuscript_results_snapshot", build_manuscript_results_snapshot),
    ]

    names = [name for name, _ in steps]
    if start_at is not None and start_at not in names:
        raise ValueError(f'Unknown step {start_at}; choose from {names}')
    start = names.index(start_at) if start_at else 0
    step_results = []
    for name, fn in steps[start:]:
        step_results.append(_run_step(name, fn))
        manifest = {
            "runner": "run_paper1_suite.py",
            "paper_output_dir": repo_display_path(output_dir),
            "start_at": start_at,
            "complete": name == steps[-1][0],
            "steps": step_results,
        }
        (output_dir / "paper1_suite_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", help="Resume at a named step; retain previously generated upstream outputs")
    main(start_at=parser.parse_args().start_at)
