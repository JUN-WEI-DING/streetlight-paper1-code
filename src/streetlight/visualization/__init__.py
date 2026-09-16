"""
Visualization module for streetlight analysis.

Provides plotting functions for:
- Pareto frontier
- MACC curve
- City metrics bar chart
- Cost-abatement scatter
- Taiwan heatmap
"""

from streetlight.visualization.pareto_plots import (
    plot_pareto_frontier,
    plot_macc_curve,
    plot_city_metrics_bar,
    plot_cost_abatement_scatter,
    generate_all_pareto_plots,
)

__all__ = [
    "plot_pareto_frontier",
    "plot_macc_curve",
    "plot_city_metrics_bar",
    "plot_cost_abatement_scatter",
    "generate_all_pareto_plots",
]
