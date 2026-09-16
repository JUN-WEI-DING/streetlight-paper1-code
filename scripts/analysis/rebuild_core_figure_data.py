"""Regenerate core numerical figure tables from the frozen candidate panel.

Matches the research workflow's numerical transformations without importing its
manuscript renderer, map decoration, or editorial figure layout.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paper1_config import load_paper1_config, CITY_LABELS
from streetlight.lca import compute_lca_at_knee
from streetlight.simulation.storage import pareto_frontier_min_cost_max_abatement, find_knee_point


def main():
    cfg = load_paper1_config()
    results, out = cfg.canonical_results_dir, cfg.figure_data_dir
    panel = pd.read_csv(results / 'pareto/results.csv')
    total = panel.groupby(['solar_panel_factor', 'battery_factor'], as_index=False).agg(
        abatement_t=('abatement_t', 'sum'), delta_cost=('delta_cost', 'sum'))
    frontier = pareto_frontier_min_cost_max_abatement(total)
    idx, _, _ = find_knee_point(frontier)
    representatives = {'min_cost': frontier.loc[frontier.delta_cost.idxmin()],
                       'knee': frontier.iloc[int(idx)],
                       'max_abatement': frontier.loc[frontier.abatement_t.idxmax()]}
    knee = representatives['knee']
    if not (np.isclose(knee.solar_panel_factor, cfg.selected_solar_panel_factor)
            and np.isclose(knee.battery_factor, cfg.selected_battery_factor)):
        raise ValueError('Recomputed knee differs from configured design')
    keys = set(zip(frontier.solar_panel_factor, frontier.battery_factor))
    total['is_frontier'] = [(s, b) in keys for s, b in zip(total.solar_panel_factor, total.battery_factor)]
    total['abatement_t_per_streetlight'] = total.abatement_t / cfg.study_light_count
    total['delta_cost_usd_per_streetlight'] = total.delta_cost / cfg.ntd_per_usd / cfg.study_light_count
    total['deployment_units'] = cfg.study_light_count
    total['abatement_t_scope'] = f'total across {cfg.study_light_count} study streetlights'
    total['delta_cost_currency'] = 'NTD'
    total['delta_cost_scope'] = total.abatement_t_scope
    total['representative_label'] = ''
    for label, row in representatives.items():
        mask = np.isclose(total.solar_panel_factor, row.solar_panel_factor) & np.isclose(total.battery_factor, row.battery_factor)
        total.loc[mask, 'representative_label'] = label
    out.mkdir(parents=True, exist_ok=True)
    total.to_csv(out / 'f2_pareto_frontier_source.csv', index=False)
    selected = panel[np.isclose(panel.solar_panel_factor, knee.solar_panel_factor)
                     & np.isclose(panel.battery_factor, knee.battery_factor)].copy()
    if len(selected) != cfg.study_city_count or selected.city.nunique() != cfg.study_city_count:
        raise ValueError('Selected allocation must have exactly one row per city')
    stages = compute_lca_at_knee(knee.solar_panel_factor, knee.battery_factor)
    labels = {'A1-A3': 'A1-A3 manufacturing', 'A4': 'A4 transport', 'A5': 'A5 installation',
              'B': 'B4 year-12 replacement', 'C': 'C1-C4 end-of-life', 'B6 (operational)': 'B6 operational electricity'}
    rows = []
    for system, key, op in [('grid_only_led', 'TRAD', 'grid_only_emission_t'), ('pv_battery_led', 'SOLAR', 'pv_storage_emission_t')]:
        for stage, label in labels.items():
            kg = (selected[op].sum() * 1000 / cfg.study_light_count if stage == 'B6 (operational)'
                  else stages[key][stage.replace('-', '_')])
            rows.append({'system': system, 'stage': stage, 'stage_label': label,
                         'kg_co2e_per_streetlight': kg, 't_co2e_per_streetlight': kg / 1000})
    pd.DataFrame(rows).to_csv(out / 'f3_lca_balance_components.csv', index=False)
    calibration = pd.read_csv(results / 'calibration/per_city_calibration.csv')
    region_map = cfg.region_city_map
    city_regions = {city: region for region, cities in region_map.items() for city in cities}
    aef = {region: pd.read_csv(cfg.canonical_inputs_dir / 'aef' / f'{region}.csv')['FLOW_UNIT_FINAL_AEF'].mean() for region in region_map}
    selected['abatement_t'] /= selected.functional_unit_lights
    selected['grid_only_t_per_streetlight'] = selected.grid_only_emission_t / selected.functional_unit_lights
    selected['pv_battery_t_per_streetlight'] = selected.pv_storage_emission_t / selected.functional_unit_lights
    selected['delta_cost_usd_per_streetlight'] = selected.delta_cost / cfg.ntd_per_usd / selected.functional_unit_lights
    selected['grid_import_reduction_pct'] = (1 - selected.pv_storage_energy_kwh_20y / selected.grid_energy_kwh_20y) * 100
    calibration['mean_par'] = calibration.annual_par_integral / 8784
    calibration['region'] = calibration.city.map(city_regions)
    calibration['aef'] = calibration.region.map(aef)
    labels = dict(CITY_LABELS, **{'新竹縣': 'Hsinchu Co.', '嘉義縣': 'Chiayi Co.'})
    calibration['city_en'] = calibration.city.map(labels)
    selected_columns = ['city', 'abatement_t', 'grid_only_t_per_streetlight',
                        'pv_battery_t_per_streetlight', 'delta_cost_usd_per_streetlight',
                        'grid_import_reduction_pct']
    metrics = calibration.merge(selected[selected_columns], on='city', validate='one_to_one')
    columns = ['city', 'city_en', 'region', 'mean_par', 'aef', 'abatement_t',
               'grid_only_t_per_streetlight', 'pv_battery_t_per_streetlight',
               'delta_cost_usd_per_streetlight', 'grid_import_reduction_pct']
    metrics[columns].to_csv(out / 'f4_input_output_maps_source.csv', index=False)
    print(json.dumps({'candidate_allocations': len(total), 'frontier_points': len(frontier),
                      'selected_solar_factor': knee.solar_panel_factor, 'selected_battery_factor': knee.battery_factor}))

if __name__ == '__main__':
    main()
