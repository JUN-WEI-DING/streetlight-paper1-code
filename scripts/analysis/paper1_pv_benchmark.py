"""Compare the calibrated PAR proxy with pvlib PVWatts at the fixed allocation.

This is a recognized-model sensitivity comparison, not field validation. POWER
supplies meteorology only; both PV estimates use the same municipal PAR forcing.
Run with STREETLIGHT_CONFIG=config/paper_baseline.yaml and pvlib==0.13.1.
Network access occurs only with --fetch-weather; existing caches are validated.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    _load_city_representative_points, align_time_indices, load_aef_by_region,
    load_par_wide, load_region_city_map, run_parameter_sweep,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'outputs/final_runs/paper1_canonical_results'
WEATHER_CACHE = ROOT / 'outputs/final_runs/paper1_pv_benchmark_inputs'
POWER_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
SOURCES = [
    'https://pvlib-python.readthedocs.io/en/v0.13.1/reference/generated/pvlib.modelchain.ModelChain.with_pvwatts.html',
    'https://docs.nrel.gov/docs/fy14osti/62641.pdf',
    'https://power.larc.nasa.gov/docs/services/api/temporal/hourly/',
    'https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/',
]


def utc_index(index):
    index = pd.DatetimeIndex(index)
    return index.tz_localize('UTC') if index.tz is None else index.tz_convert('UTC')


def weather_grid_point(longitude, latitude):
    return (0.5 * round(latitude / 0.5), 0.625 * round(longitude / 0.625))


def weather_path(cache, latitude, longitude):
    return Path(cache) / f'met_{latitude:.3f}_{longitude:.3f}.json'


def weather_url(latitude, longitude):
    return POWER_URL + '?' + urlencode({
        'parameters': 'T2M,WS10M', 'community': 'RE', 'longitude': longitude,
        'latitude': latitude, 'start': '20240101', 'end': '20241231',
        'format': 'JSON', 'time-standard': 'UTC',
    })


def parse_weather(payload):
    """Require the complete leap year in UTC, declared units, and no fill values."""
    if payload['header']['time_standard'] != 'UTC':
        raise ValueError('POWER weather must use UTC')
    units = {'T2M': 'C', 'WS10M': 'm/s'}
    for key, unit in units.items():
        if payload['parameters'][key]['units'] != unit:
            raise ValueError(f'Unexpected POWER units for {key}')
    frame = pd.DataFrame({key: payload['properties']['parameter'][key] for key in units})
    frame.index = pd.to_datetime(frame.index, format='%Y%m%d%H', utc=True)
    frame = frame.sort_index().astype(float)
    expected = pd.date_range('2024-01-01', '2025-01-01', freq='h', inclusive='left', tz='UTC')
    if not frame.index.equals(expected):
        raise ValueError('POWER weather must contain all 8784 distinct 2024 UTC hours')
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values == payload['header']['fill_value']).any():
        raise ValueError('POWER weather contains nonfinite or fill values')
    if (frame.WS10M < 0).any():
        raise ValueError('POWER wind speed cannot be negative')
    return frame.rename(columns={'T2M': 'temp_air', 'WS10M': 'wind_speed'})


def load_weather(points, cache=WEATHER_CACHE, fetch=False):
    """One request per meteorological grid cell; no interpolation or gap filling."""
    result = {}
    for latitude, longitude in sorted({weather_grid_point(*point) for point in points.values()}):
        path = weather_path(cache, latitude, longitude)
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f'{path}; use --fetch-weather to download meteorology')
            with urlopen(weather_url(latitude, longitude), timeout=60) as response:
                raw = response.read()
            parse_weather(json.loads(raw))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        result[(latitude, longitude)] = parse_weather(json.loads(path.read_bytes()))
    return result


def map_hourly_weather(hourly, index):
    """Use an hour's mean for its six contained ten-minute interval starts."""
    target = utc_index(index)
    frame = hourly.reindex(target.floor('h'))
    frame.index = target
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Weather does not cover all requested intervals')
    return frame


def par_to_ghi(par):
    """Photon energy / PAR fraction gives broadband irradiance, not PV output."""
    return par * (0.219 / 0.46)


def enforce_horizon(ac, zenith):
    values = np.asarray(ac, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite PVWatts AC output')
    return np.where(np.asarray(zenith) >= 90.0, 0.0, np.maximum(values, 0.0))


def model_ac_per_kwp(par, weather, longitude, latitude, tilt):
    """PVWatts AC kW per DC kWp, including temperature, AOI and system losses."""
    import pvlib

    index = utc_index(par.index)
    location = pvlib.location.Location(latitude, longitude, tz='UTC', altitude=0)
    position = location.get_solarposition(index)
    ghi_raw = par_to_ghi(par.to_numpy(dtype=float))
    zenith = position['zenith'].to_numpy()
    ghi = np.where(zenith >= 90.0, 0.0, ghi_raw)
    irradiance = pvlib.irradiance.erbs(ghi, position['zenith'], index)
    forcing = map_hourly_weather(weather, index)
    forcing['ghi'] = ghi
    forcing['dni'] = irradiance['dni']
    forcing['dhi'] = irradiance['dhi']
    system = pvlib.pvsystem.PVSystem(
        surface_tilt=tilt, surface_azimuth=180, albedo=0.2,
        module_parameters={'pdc0': 1000.0, 'gamma_pdc': -0.0047},
        # Inverter pdc0 is its DC input limit; rated AC is eta_nom * pdc0.
        inverter_parameters={'pdc0': 1000.0 / 0.96, 'eta_inv_nom': 0.96},
        temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            'sapm']['open_rack_glass_glass'],
    )
    chain = pvlib.modelchain.ModelChain.with_pvwatts(system, location)
    chain.run_model(forcing)
    ac = enforce_horizon(chain.results.ac, zenith) / 1000.0
    removed = float(ghi_raw[zenith >= 90.0].sum() / ghi_raw.sum())
    return pd.Series(ac, index=par.index), removed


def compare_profiles(simple, benchmark, ghi):
    """Bias is simple/model - 1; shape compares equal annual energy profiles."""
    simple, benchmark, ghi = map(lambda x: np.asarray(x, dtype=float), (simple, benchmark, ghi))
    if not all(np.isfinite(x).all() for x in (simple, benchmark, ghi)):
        raise ValueError('Nonfinite profile')
    mask = ghi > 20.0
    mean = benchmark[mask].mean()
    if not mask.any() or mean <= 0 or simple.sum() <= 0 or benchmark.sum() <= 0:
        raise ValueError('Comparison needs positive annual and daylight energy')
    normalized = simple * benchmark.sum() / simple.sum()
    return {
        'simple_yield_kwh_per_kwp': float(simple.sum() / 6),
        'benchmark_yield_kwh_per_kwp': float(benchmark.sum() / 6),
        'simple_over_benchmark_bias_fraction': float(simple.sum() / benchmark.sum() - 1),
        'daylight_nrmse': float(np.sqrt(np.mean((simple[mask] - benchmark[mask]) ** 2)) / mean),
        'daylight_correlation': float(np.corrcoef(simple[mask], benchmark[mask])[0, 1]),
        'annual_energy_normalized_daylight_shape_nrmse': float(
            np.sqrt(np.mean((normalized[mask] - benchmark[mask]) ** 2)) / mean),
    }


def selected_dispatch(par, aef, region_map, cfg, factor):
    """Retain physical sizing and costs when PAR is an equivalent AC forcing."""
    if cfg.load_mode != 'solar_zenith':
        raise ValueError('Equivalent-PAR dispatch requires solar_zenith switching')
    return run_parameter_sweep(
        par_df=par, aef_by_region=aef, region_city_map=region_map,
        par_to_kw_factor=factor, solar_range=[1.85], battery_range=[8.45],
        load_mode=cfg.load_mode, solar_zenith_deg=cfg.solar_zenith_deg,
        analysis_years=cfg.analysis_years, electricity_price=cfg.electricity_price,
        n_lights=cfg.n_lights, light_power_kw=cfg.light_power_kw,
        lighting_threshold_par=cfg.lighting_threshold_par, align_strategy='intersection',
        base_capacity_kwh=cfg.storage_capacity_kwh, base_power_kw=cfg.storage_power_kw,
        eta_roundtrip=cfg.storage_eta_roundtrip, economic_costs=dict(cfg.economic_costs),
        financial_params=dict(cfg.financial_params),
        battery_power_cap_multiplier_of_load=cfg.pareto_battery_cfg.get('power_cap_multiplier_of_load'),
    ).set_index('city').sort_index()


def summarize_dispatch(frame, hardware_t, n_lights, fx):
    """Equal municipal weight; MAC is ratio of mean cost to mean abatement."""
    rows = []
    for city, row in frame.iterrows():
        abatement = float(row.abatement_t / n_lights)
        cost = float(row.delta_cost / n_lights / fx)
        net = abatement - hardware_t
        rows.append({'city': city, 'operational_abatement_t_per_streetlight': abatement,
                     'incremental_cost_usd_per_streetlight': cost,
                     'operational_mac_usd_per_t': cost / abatement,
                     'hardware_addition_t_per_streetlight': hardware_t,
                     'net_lifecycle_abatement_t_per_streetlight': net,
                     'lifecycle_mac_usd_per_t': cost / net})
    means = pd.DataFrame(rows).drop(columns='city').mean().to_dict()
    means['operational_mac_usd_per_t'] = means['incremental_cost_usd_per_streetlight'] / means['operational_abatement_t_per_streetlight']
    means['lifecycle_mac_usd_per_t'] = means['incremental_cost_usd_per_streetlight'] / means['net_lifecycle_abatement_t_per_streetlight']
    return {'summary': means, 'cities': rows}


def compare_dispatch_ranks(baseline, benchmark):
    """Compare municipal operational-abatement priorities at the fixed design."""
    original = baseline['abatement_t'].sort_index()
    changed = benchmark['abatement_t'].reindex(original.index)
    ranks = pd.DataFrame({'baseline': original.round(12).rank(ascending=False, method='average'),
                          'benchmark': changed.round(12).rank(ascending=False, method='average')})
    result = {
        'operational_abatement_fraction_change': float(changed.sum() / original.sum() - 1),
        'maximum_absolute_rank_shift': float((ranks.benchmark - ranks.baseline).abs().max()),
        'spearman_rank_correlation': float(ranks.baseline.corr(ranks.benchmark)),
        'baseline_top_five_cities': original.sort_values(ascending=False, kind='stable').index[:5].tolist(),
        'benchmark_top_five_cities': changed.sort_values(ascending=False, kind='stable').index[:5].tolist(),
        'cities': [{'city': city, 'baseline_rank': float(row.baseline), 'benchmark_rank': float(row.benchmark),
                    'rank_shift': float(row.benchmark - row.baseline),
                    'operational_abatement_fraction_change': float(changed[city] / original[city] - 1)}
                   for city, row in ranks.iterrows()],
    }
    for column in ('grid_only_emission_t', 'grid_only_cost'):
        if column in baseline:
            result[f'maximum_absolute_{column}_difference_per_installation'] = float(
                (benchmark[column] - baseline[column]).abs().max())
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_comparison(cache=WEATHER_CACHE, fetch=False):
    import pvlib
    import scipy

    config_path = ROOT / 'config/paper_baseline.yaml'
    if Path(os.environ.get('STREETLIGHT_CONFIG', '')).resolve() != config_path:
        raise ValueError('Set STREETLIGHT_CONFIG=config/paper_baseline.yaml')
    if pvlib.__version__ != '0.13.1':
        raise ValueError('This reproducible benchmark requires pvlib==0.13.1')
    cfg = get_config()
    contract_path = RESULTS / 'paper_contract.json'
    contract = json.loads(contract_path.read_text())
    factor = float(contract['effective_par_to_kw_factor'])
    if (factor != .0077 or cfg.n_lights != 64 or cfg.storage_capacity_kwh != 10
            or cfg.economic_costs['pv_capacity_kw_per_factor'] != 18):
        raise ValueError('Canonical calibration/capacity anchor changed')
    if dict(cfg.economic_costs) != contract['economic_costs'] or dict(cfg.financial_params) != contract['financial_params']:
        raise ValueError('Configuration differs from frozen economic contract')
    alpha_cal = factor / 18.0
    par = load_par_wide(cfg.paper_par_path)
    expected = pd.date_range('2024-01-01', '2025-01-01', freq='10min', inclusive='left', tz='UTC')
    if not utc_index(par.index).equals(expected) or not np.isfinite(par.to_numpy()).all() or (par.to_numpy() < 0).any():
        raise ValueError('PAR must contain all 52704 valid nonnegative 2024 UTC intervals')
    region_map = load_region_city_map(cfg.paper_region_map_path)
    points = {city: _load_city_representative_points()[city] for city in par.columns}
    aef = load_aef_by_region(cfg.paper_aef_dir, region_map, aef_column=cfg.paper_aef_column)
    aligned, _, _ = align_time_indices(par, aef, strategy='intersection')
    if len(aligned) != 50892:
        raise ValueError('Canonical common AEF sample changed')
    baseline = selected_dispatch(par, aef, region_map, cfg, factor)
    selected_path = RESULTS / 'pareto/selected_allocation_city_metrics.csv'
    frozen = pd.read_csv(selected_path).set_index('city').sort_index()
    if not baseline.index.equals(frozen.index):
        raise ValueError('Selected allocation municipality mismatch')
    errors = {}
    for col in ('abatement_t', 'delta_cost'):
        errors[col] = float(np.abs(baseline[col] - frozen[col]).max())
        if not np.allclose(baseline[col], frozen[col], rtol=1e-10, atol=1e-8):
            raise ValueError(f'Canonical selected allocation does not reproduce: {col}')
    weather = load_weather(points, cache, fetch)
    values = json.loads((RESULTS / 'paper1_manuscript_values.json').read_text())['values']
    hardware = float(values['lca']['hardware_addition_t'])
    fx = float(cfg.get('paper.currency.ntd_per_usd'))
    cases = {}
    for name, tilt in [('horizontal', 0.0), ('south_tilt20', 20.0)]:
        ac = pd.DataFrame(index=par.index)
        city_rows, monthly_rows = [], []
        for city in par.columns:
            longitude, latitude = points[city]
            ac[city], removed = model_ac_per_kwp(par[city], weather[weather_grid_point(longitude, latitude)], longitude, latitude, tilt)
            city_rows.append({'city': city, 'par_energy_removed_below_horizon_fraction': removed,
                              **compare_profiles(par[city] * alpha_cal, ac[city], par_to_ghi(par[city]))})
            months = pd.DataFrame({'simple_kwh_per_kwp': par[city] * alpha_cal / 6,
                                   'benchmark_kwh_per_kwp': ac[city] / 6}).groupby(utc_index(par.index).month).sum()
            monthly_rows.extend({'city': city, 'month_utc': int(month), **row.to_dict()} for month, row in months.iterrows())
        dispatch = selected_dispatch(ac / alpha_cal, aef, region_map, cfg, factor)
        for col in ('battery_capacity_kwh', 'battery_power_kw', 'pv_storage_capex_cost', 'pv_storage_om_cost', 'pv_storage_eol_cost'):
            if not np.array_equal(dispatch[col].to_numpy(), baseline[col].to_numpy()):
                raise ValueError(f'Fixed-allocation cost/capacity changed: {col}')
        cases[name] = {'tilt_deg': tilt, 'summary': pd.DataFrame(city_rows).drop(columns='city').mean().to_dict(),
                       'cities': city_rows, 'monthly_energy': monthly_rows,
                       'dispatch': summarize_dispatch(dispatch, hardware, cfg.n_lights, fx),
                       'dispatch_comparison': compare_dispatch_ranks(baseline, dispatch)}
    input_paths = [cfg.paper_par_path, cfg.paper_region_map_path, config_path, contract_path, selected_path,
                   *[cfg.paper_aef_dir / f'{region}.csv' for region in region_map]]
    return {
        'scope': 'Recognized-model comparison under shared PAR forcing and reanalysis weather; not field validation or a new optimum.',
        'provenance': {'pvlib_version': pvlib.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
                       'scipy_version': scipy.__version__,
                       'sources': SOURCES, 'input_sha256': {str(Path(p).relative_to(ROOT)): sha256(p) for p in input_paths},
                       'city_representative_points_lon_lat': points,
                       'weather_sha256': {weather_path(cache, *cell).name: sha256(weather_path(cache, *cell)) for cell in sorted(weather)},
                       'weather_grid_cells': [{'latitude': lat, 'longitude': lon, 'url': weather_url(lat, lon),
                                               'hour_count': len(weather[(lat, lon)]),
                                               'api': json.loads(weather_path(cache, lat, lon).read_bytes())['header']['api']}
                                              for lat, lon in sorted(weather)]},
        'parameters': {'par_to_ghi_w_m2_per_umol_m2_s': .219 / .46, 'alpha_cal': alpha_cal,
                       'dc_nameplate_w': 1000, 'inverter_pdc0_w': 1000 / .96, 'inverter_eta_nom': .96,
                       'dc_ac_nameplate_ratio': 1.0, 'gamma_pdc_per_c': -.0047,
                       'losses_percent': float(pvlib.pvsystem.pvwatts_losses()),
                       'loss_components_percent': {name: parameter.default for name, parameter in
                                                   inspect.signature(pvlib.pvsystem.pvwatts_losses).parameters.items()},
                       'temperature_model': 'SAPM open_rack_glass_glass',
                       'temperature_parameters': pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'],
                       'aoi_model': 'physical',
                       'spectral_model': 'no_loss', 'transposition_model': 'Perez', 'decomposition': 'Erbs',
                       'surface_azimuth_deg': 180, 'albedo': .2, 'site_altitude_m': 0,
                       'meteorology': 'NASA POWER T2M (C), WS10M (m/s), hourly UTC means held within each hour',
                       'solar_position': 'pvlib nrel_numpy at original PAR timestamps; geometric zenith >=90 forces zero irradiance and AC',
                       'metric_daylight_threshold_ghi_w_m2': 20, 'monthly_time_standard': 'UTC',
                       'full_year_intervals': len(par), 'common_dispatch_intervals': len(aligned),
                       'pv_kwp_per_streetlight': 18 * 1.85 / 64, 'battery_kwh_per_streetlight': 10 * 8.45 / 64,
                       'analysis_years': cfg.analysis_years, 'ntd_per_usd': fx},
        'baseline_replay_max_absolute_error': errors,
        'baseline_dispatch': summarize_dispatch(baseline, hardware, cfg.n_lights, fx), 'cases': cases,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS / 'pv_model_comparison.json')
    parser.add_argument('--weather-cache', type=Path, default=WEATHER_CACHE)
    parser.add_argument('--fetch-weather', action='store_true')
    args = parser.parse_args()
    result = build_comparison(args.weather_cache, args.fetch_weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({name: case['summary'] for name, case in result['cases'].items()}, indent=2))
