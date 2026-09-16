from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from streetlight.config import get_config
from streetlight.simulation.storage import find_knee_point, pareto_frontier_min_cost_max_abatement, recompute_costs_from_results

from research_analysis_common import aggregate_design_from_results, portfolio_results


def _summarize_frontier(frontier: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict[str, Any]:
    knee_idx, knee_abatement_t, knee_delta_cost = find_knee_point(frontier)
    if knee_idx is None:
        knee_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    else:
        knee_row = frontier.iloc[int(knee_idx)]
    min_cost_row = frontier.sort_values("delta_cost", ascending=True).iloc[0]
    max_abatement_row = frontier.sort_values("abatement_t", ascending=False).iloc[0]
    return {
        "frontier_points": int(len(frontier)),
        "negative_delta_cost_count": int((portfolio_df["delta_cost"] < 0).sum()),
        "positive_delta_cost_count": int((portfolio_df["delta_cost"] > 0).sum()),
        "knee_index": None if knee_idx is None else int(knee_idx),
        "knee_solar_panel_factor": float(knee_row["solar_panel_factor"]),
        "knee_battery_factor": float(knee_row["battery_factor"]),
        "knee_abatement_t": None if knee_abatement_t is None else float(knee_abatement_t),
        "knee_delta_cost": None if knee_delta_cost is None else float(knee_delta_cost),
        "min_cost_delta_cost": float(min_cost_row["delta_cost"]),
        "max_abatement_t": float(max_abatement_row["abatement_t"]),
    }


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    analysis_dir = output_dir / "marginal_cost_economic_lens"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = pd.read_csv(output_dir / "pareto" / "results.csv")
    merit_df = pd.read_csv(output_dir / "merit_order_marginal" / "marginal_carbon_timeseries.csv")
    avg_marginal_cost = float(pd.to_numeric(merit_df["marginal_cost_ntd_per_kwh"], errors="coerce").mean())
    retail_price = float(cfg.electricity_price)

    results_df = recompute_costs_from_results(
        baseline_results,
        years=float(cfg.analysis_years),
        elec_price=avg_marginal_cost,
        economic_costs=dict(cfg.economic_costs),
        financial_params=dict(cfg.financial_params),
    )
    results_df.to_csv(analysis_dir / "mean_marginal_cost_basis_results.csv", index=False)

    portfolio_df = portfolio_results(results_df)
    portfolio_df.to_csv(analysis_dir / "mean_marginal_cost_basis_portfolio.csv", index=False)

    frontier = pareto_frontier_min_cost_max_abatement(portfolio_df)
    frontier.to_csv(analysis_dir / "frontier_marginal_cost_basis.csv", index=False)

    comparison_rows = []
    for label in ("min_cost", "knee", "max_abatement"):
        baseline_row = aggregate_design_from_results(baseline_results, label, output_dir)
        marginal_row = aggregate_design_from_results(results_df, label, output_dir)
        comparison_rows.append(
            {
                "design_label": label,
                "solar_panel_factor": baseline_row["solar_panel_factor"],
                "battery_factor": baseline_row["battery_factor"],
                "retail_delta_cost": baseline_row["delta_cost"],
                "marginal_cost_basis_delta_cost": marginal_row["delta_cost"],
                "delta_cost_shift": float(marginal_row["delta_cost"]) - float(baseline_row["delta_cost"]),
                "retail_grid_only_cost": baseline_row["grid_only_cost"],
                "marginal_cost_basis_grid_only_cost": marginal_row["grid_only_cost"],
                "retail_pv_storage_cost": baseline_row["pv_storage_cost"],
                "marginal_cost_basis_pv_storage_cost": marginal_row["pv_storage_cost"],
                "abatement_t": baseline_row["abatement_t"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(analysis_dir / "representative_design_marginal_cost_comparison.csv", index=False)

    summary = {
        "interpretation": (
            "Alternative economic lens using the mean Taipower-informed merit-order marginal cost "
            "proxy instead of the retail tariff. This is a system-cost cross-check and does not "
            "replace the private tariff-based baseline."
        ),
        "retail_electricity_price_ntd_per_kwh": retail_price,
        "mean_proxy_marginal_cost_ntd_per_kwh": avg_marginal_cost,
        "marginal_over_retail_ratio": avg_marginal_cost / retail_price if retail_price > 0 else None,
        **_summarize_frontier(frontier, portfolio_df),
        "representative_designs": comparison_rows,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
