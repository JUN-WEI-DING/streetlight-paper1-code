"""Plot numerical reproductions; these are not the manuscript's editorial layouts."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from paper1_config import load_paper1_config


def main():
    cfg = load_paper1_config()
    data = cfg.figure_data_dir
    output = Path('outputs/reproduction/figures')
    output.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=180)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)
    f = pd.read_csv(data / 'f2_pareto_frontier_source.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(f.abatement_t_per_streetlight, f.delta_cost_usd_per_streetlight, s=2, c='lightgray')
    front = f[f.is_frontier].sort_values('abatement_t_per_streetlight')
    ax.plot(front.abatement_t_per_streetlight, front.delta_cost_usd_per_streetlight, color='black')
    knee = f[f.representative_label == 'knee']
    ax.scatter(knee.abatement_t_per_streetlight, knee.delta_cost_usd_per_streetlight, marker='*', s=90, c='firebrick')
    ax.set(xlabel='Operational abatement (t CO2e / streetlight)', ylabel='Incremental lifecycle cost (USD / streetlight)')
    save(fig, 'figure2_pareto')
    f = pd.read_csv(data / 'f3_lca_balance_components.csv')
    table = f.pivot(index='system', columns='stage', values='t_co2e_per_streetlight')
    ax = table.plot.barh(stacked=True, figsize=(7, 3))
    ax.set(xlabel='Lifecycle emissions (t CO2e / streetlight)', ylabel='')
    save(ax.figure, 'figure3_lifecycle')
    f = pd.read_csv(data / 'f4_input_output_maps_source.csv')
    # No external map service or decorative raster dependency.
    import geopandas as gpd
    geo = gpd.read_file('data/geo/county_boundaries.shp').merge(f, left_on='COUNTYNAME', right_on='city', validate='one_to_one')
    ax = geo.plot(column='region', categorical=True, legend=True, figsize=(7, 6), edgecolor='white')
    ax.set(xlim=(117.9, 122.3), ylim=(21.6, 26.5))
    ax.set_axis_off()
    ax.set_title('Municipal accounting regions (simplified study-area map)')
    save(ax.figure, 'figure1_accounting_regions')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, col, title in zip(axes, ['mean_par', 'aef', 'abatement_t'], ['PAR (µmol m⁻² s⁻¹)', 'AEF (kg CO2e/kWh)', 'Abatement (t CO2e/light)']):
        geo.plot(column=col, ax=ax, legend=True, cmap='viridis', edgecolor='white', linewidth=.2,
                 legend_kwds={'shrink': .6})
        ax.set(xlim=(117.9, 122.3), ylim=(21.6, 26.5))
        ax.set_title(title); ax.set_axis_off()
    save(fig, 'figure4_municipal_inputs_outputs')
    ranks = pd.read_csv(data / 'f5_rank_shift_bump_data.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    columns = ['r_baseline', 'r_static', 'r_uniform', 'r_combined']
    for _, row in ranks.iterrows():
        ax.plot(range(4), row[columns].astype(float), marker='o', alpha=.5, linewidth=.8)
    ax.set_xticks(range(4), ['Reference', 'Static AEF', 'Uniform solar', 'Combined'])
    ax.set(ylabel='Municipality rank (1 = highest abatement)', ylim=(22.5, .5))
    save(fig, 'figure5_rank_shifts')
    points = pd.read_csv(data / 'f6_future_grid_retention_points.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    regions = sorted(points.region.unique())
    colors = dict(zip(regions, plt.get_cmap('tab10').colors))
    markers = dict(zip(sorted(points.year.unique()), ['o', '^', 's']))
    for (region, year), group in points.groupby(['region', 'year']):
        ax.scatter(group.aef, group.abate, color=colors[region], marker=markers[year], label=f'{region}, {year}')
    ax.set(xlabel='Regional AEF (kg CO2e/kWh)', ylabel='Operational abatement (t CO2e / streetlight)')
    ax.legend(fontsize=6, ncol=2)
    save(fig, 'figure6_static_grid_states')
    print(f'Wrote 6 numerical figures (PNG/PDF) to {output}; see docs/data-bundle.md for layout scope')

if __name__ == '__main__':
    main()
