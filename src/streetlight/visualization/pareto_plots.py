"""
Pareto analysis visualization functions.

Generates publication-quality plots for Pareto frontier, MACC curve,
and city-level metrics.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects

# Region colors for plots
REGION_COLORS = {
    "north": "#E6973F",
    "central": "#6CA6D9",
    "south": "#6FB96F",
    "east": "#D96A6A",
    "island": "#C2A3D6",
    "island_penghu": "#C2A3D6",
    "island_kinmen": "#B39AC9",
    "island_lienchiang": "#A58CBC",
    "北區": "#f28e2b",
    "中區": "#4e79a7",
    "南區": "#59a14f",
    "東區": "#e15759",
    "離島": "#b07aa1",
}

REGION_EN_TO_ZH = {
    "north": "北部",
    "central": "中部",
    "south": "南部",
    "east": "東部",
    "island": "離島",
    "island_penghu": "澎湖",
    "island_kinmen": "金門",
    "island_lienchiang": "連江",
}


def _thousands(x, pos=None):
    """Format numbers with thousands separator."""
    try:
        if float(x).is_integer():
            return f"{int(x):,}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def _get_chinese_font():
    """Try to load Chinese font, return None if not available."""
    from matplotlib import font_manager
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',  # macOS
    ]
    for path in font_paths:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path)
    return None


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    frontier: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Plot Pareto frontier with all configuration points.

    Parameters
    ----------
    results_df : pd.DataFrame
        All configuration results with abatement_t and delta_cost columns
        Can be individual city data or aggregated data
    frontier : pd.DataFrame
        Pareto frontier points (should be aggregated totals)
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    """
    df = results_df.copy()

    # Ensure abatement column exists
    if "abatement_t" not in df.columns and "delta_t" in df.columns:
        df["abatement_t"] = -df["delta_t"]

    # Remove inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_cost", "abatement_t"])
    df = df[df["abatement_t"] > 0].copy()

    if df.empty:
        print("[Skip] No data points for Pareto plot")
        return

    # Aggregate by configuration (solar_panel_factor, battery_factor)
    # This ensures scatter points match frontier scale (both aggregated totals)
    if "city" in df.columns:
        # Data is per-city, need to aggregate
        agg_df = df.groupby(["solar_panel_factor", "battery_factor"]).agg({
            "abatement_t": "sum",
            "delta_cost": "sum",
        }).reset_index()
    else:
        # Data is already aggregated
        agg_df = df

    chinese_font = _get_chinese_font()
    font_context = {'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False} if chinese_font else {}

    with plt.rc_context(font_context):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        # All points colored by PV capacity (use aggregated data to match frontier scale)
        scatter = ax.scatter(
            agg_df["abatement_t"], agg_df["delta_cost"],
            c=agg_df["solar_panel_factor"], cmap='viridis',
            s=30, alpha=0.6, edgecolors="none",
            label="All Configurations"
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('PV Capacity Factor', fontsize=10)

        if not frontier.empty:
            # Frontier line
            ax.plot(
                frontier["abatement_t"], frontier["delta_cost"],
                color='red', linewidth=2.5, marker="o", markersize=8,
                label="Pareto Frontier", markerfacecolor="yellow",
                markeredgecolor="black", zorder=5
            )

            # Max abatement point
            x_max = frontier["abatement_t"].max()
            y_at_xmax = frontier.loc[frontier["abatement_t"].idxmax(), "delta_cost"]
            ax.axvline(x=x_max, color="red", linestyle="--", alpha=0.3, linewidth=1.2)
            ax.scatter(
                [x_max], [y_at_xmax], s=100, color="red", edgecolor="black",
                zorder=6, label="Max Reduction"
            )

            # Knee point if enough points
            if len(frontier) >= 3:
                from streetlight.simulation.storage import find_knee_point
                idx_knee, x_knee, y_knee = find_knee_point(frontier)
                if x_knee is not None:
                    ax.scatter(
                        [x_knee], [y_knee], s=120, color="blue",
                        label='Knee Point', edgecolor="black", zorder=7
                    )
                    ax.annotate(
                        f'Knee: {x_knee:.0f} t, ${y_knee/1e6:.1f}M',
                        xy=(x_knee, y_knee), xytext=(10, 10),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7)
                    )

        ax.set_xlabel(r"Emission Reduction (t CO$_2$e)", fontsize=13)
        ax.set_ylabel("Cost Difference (NTD)", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.yaxis.set_major_formatter(lambda x, p: f'{x/1e6:.1f}M' if abs(x) >= 1e6 else f'{x:,.0f}')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10, frameon=False, loc='upper left')

        # Set X-axis to start from 0 for better visualization
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if title:
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()


def plot_macc_curve(
    macc: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Plot MACC (Marginal Abatement Cost Curve).

    Parameters
    ----------
    macc : pd.DataFrame
        MACC data with abate_left, abate_right, delta_abate, mac columns
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    """
    chinese_font = _get_chinese_font()
    font_context = {'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False} if chinese_font else {}

    with plt.rc_context(font_context):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

        if macc.empty or len(macc) < 1:
            ax.text(
                0.5, 0.5, "Insufficient data for MACC curve",
                ha='center', va='center', transform=ax.transAxes, fontsize=12
            )
        else:
            xs = macc["abate_left"].values
            widths = macc["delta_abate"].values
            heights = macc["mac"].values

            # Color bars by PV capacity
            pv_col = "solar_panel_factor" if "solar_panel_factor" in macc.columns else "pv_mult"
            if pv_col in macc.columns and macc[pv_col].max() > 0:
                colors = plt.cm.viridis(macc[pv_col] / macc[pv_col].max())
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, len(macc)))

            bars = ax.bar(
                xs, heights, width=widths, align="edge", alpha=0.8,
                edgecolor="black", color=colors
            )

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

            # Annotate bars
            for i, (_, row) in enumerate(macc.iterrows()):
                if i % max(1, len(macc) // 5) == 0:
                    pv_val = row.get("solar_panel_factor", row.get("pv_mult", "?"))
                    bat_val = row.get("battery_factor", row.get("batt_cap", "?"))
                    ax.text(
                        row["abate_right"], row["mac"],
                        f"PV:{pv_val:.1f}x",
                        fontsize=7, ha='left', va='bottom' if row["mac"] > 0 else 'top'
                    )

        ax.set_xlabel(r"Cumulative Emission Reduction (t CO$_2$e)", fontsize=13)
        ax.set_ylabel(r"Marginal Abatement Cost (NTD/t CO$_2$e)", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.25, axis='y')

        if title:
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()


def plot_city_metrics_bar(
    city_metrics: pd.DataFrame,
    output_path: Path,
    metric: str = "max_abatement_t",
    title: Optional[str] = None,
):
    """
    Plot city metrics as horizontal bar chart.

    Parameters
    ----------
    city_metrics : pd.DataFrame
        City metrics with city, region, and metric columns
    output_path : Path
        Output file path
    metric : str
        Column name to plot (default: max_abatement_t)
    title : str, optional
        Plot title
    """
    df = city_metrics.copy()

    if metric not in df.columns:
        print(f"[Skip] Column '{metric}' not found in city_metrics")
        return

    # Sort by metric
    df = df.sort_values(metric, ascending=True).reset_index(drop=True)

    # Assign colors by region
    df["color"] = df["region"].map(REGION_COLORS).fillna("gray")

    chinese_font = _get_chinese_font()
    font_context = {'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False} if chinese_font else {}

    with plt.rc_context(font_context):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        bars = ax.barh(
            df["city"], df[metric],
            color=df["color"], edgecolor="black", linewidth=0.5
        )

        ax.set_xlabel(r"Max Emission Reduction (t CO$_2$e)", fontsize=12)
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.25, axis='x')

        # Add legend for regions
        regions = df["region"].unique()
        handles = [
            Patch(facecolor=REGION_COLORS.get(r, "gray"), edgecolor="black", label=r)
            for r in regions if r in REGION_COLORS
        ]
        if handles:
            ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")

        if title:
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()


def plot_cost_abatement_scatter(
    results_df: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Plot cost vs abatement scatter with bubble sizes.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with abatement_t and delta_cost columns
    output_path : Path
        Output file path
    title : str, optional
        Plot title
    """
    df = results_df.copy()

    # Ensure abatement column
    if "abatement_t" not in df.columns and "delta_t" in df.columns:
        df["abatement_t"] = -df["delta_t"]

    # Remove inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_cost", "abatement_t"])
    df = df[df["abatement_t"] > 0].copy()

    if df.empty:
        print("[Skip] No data for cost-abatement scatter")
        return

    # Aggregate by configuration
    agg_df = df.groupby(["solar_panel_factor", "battery_factor"]).agg({
        "abatement_t": "sum",
        "delta_cost": "sum",
    }).reset_index()

    # Bubble size based on PV capacity
    agg_df["bubble_size"] = (10 + agg_df["solar_panel_factor"] * 10) ** 2

    chinese_font = _get_chinese_font()
    font_context = {'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False} if chinese_font else {}

    with plt.rc_context(font_context):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        scatter = ax.scatter(
            agg_df["abatement_t"], agg_df["delta_cost"],
            s=agg_df["bubble_size"],
            c=agg_df["solar_panel_factor"], cmap='viridis',
            alpha=0.7, edgecolors="black", linewidths=0.8
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('PV Capacity Factor', fontsize=10)

        ax.set_xlabel(r"Emission Reduction (t CO$_2$e)", fontsize=13)
        ax.set_ylabel("Cost Difference (NTD)", fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.yaxis.set_major_formatter(lambda x, p: f'{x/1e6:.1f}M' if abs(x) >= 1e6 else f'{x:,.0f}')
        ax.grid(True, alpha=0.25)

        if title:
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()


def generate_all_pareto_plots(
    results_df: pd.DataFrame,
    frontier: pd.DataFrame,
    macc: pd.DataFrame,
    city_metrics: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all Pareto analysis plots.

    Parameters
    ----------
    results_df : pd.DataFrame
        All configuration results
    frontier : pd.DataFrame
        Pareto frontier points
    macc : pd.DataFrame
        MACC curve data
    city_metrics : pd.DataFrame
        City-level metrics
    output_dir : Path
        Output directory

    Returns
    -------
    Dict[str, Path]
        Mapping of plot name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Pareto frontier
    pareto_path = output_dir / "pareto_frontier.png"
    plot_pareto_frontier(results_df, frontier, pareto_path, title="Pareto Frontier")
    paths["pareto_frontier"] = pareto_path

    # 2. MACC curve
    macc_path = output_dir / "macc_curve.png"
    plot_macc_curve(macc, macc_path, title="Marginal Abatement Cost Curve")
    paths["macc_curve"] = macc_path

    # 3. City metrics bar
    city_path = output_dir / "city_metrics.png"
    plot_city_metrics_bar(city_metrics, city_path, title="City-level Emission Reduction Potential")
    paths["city_metrics"] = city_path

    # 4. Cost-abatement scatter
    scatter_path = output_dir / "cost_abatement_scatter.png"
    plot_cost_abatement_scatter(results_df, scatter_path, title="Cost vs Emission Reduction")
    paths["cost_abatement_scatter"] = scatter_path

    return paths
