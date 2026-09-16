"""Deterministic, total-preserving distributed-fuel allocation stress test.

Change one mainland share of one allocated fuel bucket by +/-20% relative,
redistribute its remaining share proportionally, and retain all other inputs.
This is an accounting sensitivity with reference transfer inputs fixed, not a forecast
of a physically redispatched power system or a probabilistic uncertainty band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from streetlight.aef.pipeline import AEFPipeline, MAINLAND_REGIONS
from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    load_par_wide, load_region_city_map, run_parameter_sweep,
)

FUELS = ('Co-Gen', 'Biomass')
AEF_COLUMN = 'FLOW_UNIT_FINAL_AEF'


def perturb_shares(shares: pd.Series, region: str, relative_change: float) -> pd.Series:
    """Change one share while preserving the total and other-region ratios."""
    if not np.isclose(shares.sum(), 1.0) or (shares < 0).any():
        raise ValueError('Shares must be nonnegative and sum to one')
    old = float(shares[region])
    new = old * (1.0 + relative_change)
    if not 0 < old < 1 or not 0 <= new <= 1:
        raise ValueError('Requested target share is outside its feasible range')
    result = shares * ((1.0 - new) / (1.0 - old))
    result[region] = new
    return result


def allocated_buckets(raw: dict[str, pd.DataFrame], fuel: str):
    """Recover the allocated national bucket and verify its fixed input shares."""
    col = fuel + '-_other_allocated'
    panel = pd.concat({r: raw[r][col] for r in MAINLAND_REGIONS}, axis=1)
    if panel.isna().any().any() or (panel < 0).any().any():
        raise ValueError(f'Invalid allocated bucket: {fuel}')
    total = panel.sum(axis=1)
    shares = panel.sum() / total.sum()
    np.testing.assert_allclose(panel.to_numpy(), total.to_numpy()[:, None] * shares.to_numpy(),
                               rtol=1e-10, atol=1e-9)
    return total, shares


def region_inputs(raw: dict[str, pd.DataFrame], pipeline: AEFPipeline):
    """Match AEFPipeline.load_region_data without intermediate on-disk CSVs."""
    result = {}
    for region, raw_frame in raw.items():
        frame = pipeline.calculator.aggregate_by_fuel(raw_frame)
        transfers = [c for c in frame if c.startswith('F_') or c in ('BESS', 'PHS', 'load')]
        values = pipeline.calculator.compute_aef_dataframe(frame.drop(columns=transfers))
        for col in transfers:
            values[col] = frame[col].clip(lower=0) if col.startswith('F_') else frame[col]
        for col in ('BESS', 'PHS'):
            values[col] = values[col].fillna(0) if col in values else 0.0
        result[region] = values
    return result


def economic_summary(cities: pd.DataFrame, n_lights: int):
    """Equal municipality mean per light and exact fixed-cost/abatement ratio."""
    if len(cities) != 22 or cities.city.nunique() != 22:
        raise ValueError('Expected exactly one selected-design result for each of 22 cities')
    abatement = float(cities.abatement_t.mean() / n_lights)
    cost = float(cities.delta_cost.mean() / n_lights)
    return {'abatement_t_per_light': abatement, 'delta_cost_ntd_per_light': cost,
            'mac_ntd_per_t': cost / abatement}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    cfg = get_config()
    root = cfg.paper_output_dir
    out = args.output_dir or root / 'allocation_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads((root / 'paper_contract.json').read_text())
    frontier = pd.read_csv(root / 'pareto/frontier.csv')
    knee = json.loads((root / 'pareto/knee.json').read_text())
    selected = frontier.iloc[int(knee['index'])]
    solar, battery = float(selected.solar_panel_factor), float(selected.battery_factor)
    reference_results = pd.read_csv(root / 'pareto/results.csv')
    reference_results = reference_results[
        np.isclose(reference_results.solar_panel_factor, solar)
        & np.isclose(reference_results.battery_factor, battery)].set_index('city').sort_index()
    region_map = load_region_city_map(cfg.paper_region_map_path)
    par = load_par_wide(cfg.paper_par_path)
    raw = {r: pd.read_csv(Path(cfg.power_dir) / f'{r}_unit_generation.csv',
                          index_col=0, parse_dates=True) for r in region_map}
    frozen = {r: pd.read_csv(Path(cfg.paper_aef_dir) / f'{r}.csv', index_col=0,
                             parse_dates=True)[AEF_COLUMN] for r in region_map}
    buckets = {fuel: allocated_buckets(raw, fuel) for fuel in FUELS}
    cases = [('baseline', None, None, 0.0)] + [
        (f'{fuel}_{region}_{"plus" if change > 0 else "minus"}20', fuel, region, change)
        for fuel in FUELS for region in MAINLAND_REGIONS for change in (-0.2, 0.2)]
    scenario_rows, regional_rows, city_rows, share_rows = [], [], [], []
    baseline_cities = baseline_panel = baseline_inputs = None
    max_baseline_error = 0.0
    max_conservation_error = 0.0
    for name, fuel, target, change in cases:
        print(f'Allocation sensitivity: {name}', flush=True)
        modified = dict(raw)
        if fuel is not None:
            total, shares = buckets[fuel]
            scenario_shares = perturb_shares(shares, target, change)
            changed = {}
            for region in MAINLAND_REGIONS:
                frame = raw[region].copy()
                frame[fuel + '-_other_allocated'] = total * scenario_shares[region]
                modified[region] = frame
                changed[region] = frame[fuel + '-_other_allocated']
            error = float((pd.DataFrame(changed).sum(axis=1) - total).abs().max())
            max_conservation_error = max(max_conservation_error, error)
            np.testing.assert_allclose(pd.DataFrame(changed).sum(axis=1), total, rtol=1e-12, atol=1e-9)
        for current_fuel in FUELS:
            _, shares = buckets[current_fuel]
            if current_fuel == fuel:
                shares = scenario_shares
            share_rows.extend({'scenario': name, 'fuel': current_fuel, 'region': r, 'share': float(s)}
                              for r, s in shares.items())
        pipeline = AEFPipeline(config=cfg)
        inputs = region_inputs(modified, pipeline)
        # power_adapter exports used_mw snapshots and this run uses scale=1.
        # Total_Gen (MWh) is a legacy label: these residual differences are MW.
        generation = {r: frame['Total_Gen (MWh)'].copy() for r, frame in inputs.items()}
        results, _ = pipeline.run(region_data=inputs)
        aef_by_region = {r: frame[AEF_COLUMN] for r, frame in results.items()}
        # Islands retain 55 additional timestamps; compare every native regional
        # series first, then use the common intersection for the seven-region mean.
        for region, values in aef_by_region.items():
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(f'Nonfinite regional AEF: {name}/{region}')
        panel = pd.concat(aef_by_region, axis=1, join='inner')
        if baseline_panel is None:
            for region, values in aef_by_region.items():
                pd.testing.assert_index_equal(values.index, frozen[region].index)
                np.testing.assert_allclose(values, frozen[region], rtol=1e-10, atol=1e-12)
                max_baseline_error = max(max_baseline_error, float((values - frozen[region]).abs().max()))
            baseline_panel, baseline_inputs = panel.copy(), generation
        else:
            pd.testing.assert_index_equal(panel.index, baseline_panel.index)
        # Existing sweep with one fixed design retains its original zero-SOC dispatch.
        cities = run_parameter_sweep(
            par_df=par, aef_by_region=aef_by_region, region_city_map=region_map,
            par_to_kw_factor=float(contract['effective_par_to_kw_factor']),
            solar_range=[solar], battery_range=[battery], load_mode=cfg.load_mode,
            analysis_years=cfg.analysis_years, electricity_price=cfg.electricity_price,
            n_lights=cfg.n_lights, light_power_kw=cfg.light_power_kw,
            lighting_threshold_par=cfg.lighting_threshold_par,
            base_capacity_kwh=float(cfg.storage_capacity_kwh), base_power_kw=float(cfg.storage_power_kw),
            eta_roundtrip=float(cfg.storage_eta_roundtrip), economic_costs=dict(cfg.economic_costs),
            financial_params=dict(cfg.financial_params), align_strategy=cfg.paper_align_strategy,
            battery_power_cap_multiplier_of_load=cfg.pareto_battery_cfg.get('power_cap_multiplier_of_load'))
        current = cities.set_index('city').sort_index()
        if baseline_cities is None:
            for col in ('abatement_t', 'delta_cost', 'grid_only_emission_t', 'pv_storage_emission_t'):
                np.testing.assert_allclose(current[col], reference_results[col], rtol=1e-10, atol=1e-8)
            baseline_cities = current.copy()
        np.testing.assert_allclose(current.delta_cost, baseline_cities.delta_cost, rtol=0, atol=1e-8)
        metrics = economic_summary(cities, cfg.n_lights)
        mean_aef = float(panel.to_numpy().mean())
        scenario_rows.append({'scenario': name, 'perturbed_fuel': fuel or 'none',
                              'target_region': target or 'none', 'relative_share_change': change,
                              'panel_mean_aef_kg_per_kwh': mean_aef, **metrics})
        for region in region_map:
            # With reference load/transfers/storage fixed, delta generation is exactly the
            # change in the regional power-balance residual. Absolute load is not staged.
            residual_delta = generation[region] - baseline_inputs[region]
            regional_rows.append({'scenario': name, 'region': region,
                'mean_aef_kg_per_kwh': float(panel[region].mean()),
                'aef_change_pct': 100 * (float(panel[region].mean() / baseline_panel[region].mean()) - 1),
                'balance_residual_change_mean_mw': float(residual_delta.mean()),
                'balance_residual_change_max_abs_mw': float(residual_delta.abs().max())})
        city_rows.append(cities[['region', 'city', 'abatement_t', 'delta_cost',
                                'grid_only_emission_t', 'pv_storage_emission_t']].assign(scenario=name))
    summary = pd.DataFrame(scenario_rows)
    for col, label in [('panel_mean_aef_kg_per_kwh', 'panel_aef_change_pct'),
                       ('abatement_t_per_light', 'abatement_change_pct'), ('mac_ntd_per_t', 'mac_change_pct')]:
        summary[label] = 100 * (summary[col] / summary.iloc[0][col] - 1)
    summary.to_csv(out / 'scenario_summary.csv', index=False)
    pd.DataFrame(regional_rows).to_csv(out / 'regional_summary.csv', index=False)
    pd.concat(city_rows, ignore_index=True).to_csv(out / 'city_results.csv', index=False)
    pd.DataFrame(share_rows).to_csv(out / 'allocation_shares.csv', index=False)
    perturbed = summary.iloc[1:]
    metadata = {'scenario_count': len(summary), 'perturbed_scenario_count': len(perturbed),
        'relative_share_stress': 0.2, 'solar_panel_factor': solar, 'battery_factor': battery,
        'initial_soc_kwh': 0.0, 'municipality_count': 22, 'region_count': len(panel.columns),
        'common_aef_timestamps': len(panel), 'analysis_years': cfg.analysis_years,
        'baseline_max_aef_abs_error': max_baseline_error,
        'max_national_bucket_conservation_error_mw': max_conservation_error,
        'baseline_shares': {fuel: shares.to_dict() for fuel, (_, shares) in buckets.items()},
        'range_definition': 'Minimum and maximum across 16 one-at-a-time deterministic scenarios; not confidence intervals.',
        'aef_mean_definition': 'Equal mean over seven regions and their common valid timestamps.',
        'abatement_definition': 'Equal mean across 22 municipalities, per light; original operational comparator and fixed selected design.',
        'mac_definition': 'Fixed incremental NTD cost divided by scenario operational abatement; no AEF sign-inversion proxy.',
        'balance_diagnostic': 'Change in regional residual = change in generation with reference transfer inputs, storage and load fixed. Absolute residual cannot be computed without staged load; no redispatch or rebalancing is assumed.',
        'scope': 'Distributed-fuel allocation accounting stress test, not probabilistic regional-AEF uncertainty.',
        'ranges': {col: {'min': float(perturbed[col].min()), 'max': float(perturbed[col].max()),
                        'min_scenario': str(perturbed.loc[perturbed[col].idxmin(), 'scenario']),
                        'max_scenario': str(perturbed.loc[perturbed[col].idxmax(), 'scenario'])}
                   for col in ('panel_aef_change_pct', 'abatement_change_pct', 'mac_change_pct')}}
    (out / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['ranges'], indent=2), flush=True)


if __name__ == '__main__':
    main()
