"""Physical units, time alignment, comparison signs, and dispatch invariants."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from paper1_pv_benchmark import (
    compare_dispatch_ranks, compare_profiles, enforce_horizon, map_hourly_weather, model_ac_per_kwp,
    par_to_ghi, parse_weather, selected_dispatch, weather_grid_point,
)
from streetlight.simulation import streetlight_simulation as simulation


def weather_payload():
    index = pd.date_range('2024-01-01', '2025-01-01', freq='h', inclusive='left')
    keys = index.strftime('%Y%m%d%H')
    return {'header': {'time_standard': 'UTC', 'fill_value': -999.0},
            'parameters': {'T2M': {'units': 'C'}, 'WS10M': {'units': 'm/s'}},
            'properties': {'parameter': {'T2M': dict(zip(keys, np.arange(len(keys)) / 1000)),
                                         'WS10M': dict.fromkeys(keys, 2.0)}}}


def test_weather_leap_year_and_contained_hour_mapping():
    frame = parse_weather(weather_payload())
    assert len(frame) == 8784
    index = pd.DatetimeIndex(['2024-02-29 07:50', '2024-02-29 08:00', '2024-02-29 08:10'], tz='Asia/Taipei')
    mapped = map_hourly_weather(frame, index)
    assert mapped.temp_air.iloc[0] == frame.loc['2024-02-28 23:00+00:00', 'temp_air']
    assert mapped.temp_air.iloc[1] == frame.loc['2024-02-29 00:00+00:00', 'temp_air']
    assert mapped.temp_air.iloc[1] == mapped.temp_air.iloc[2]
    with pytest.raises(ValueError, match='cover'):
        map_hourly_weather(frame, pd.DatetimeIndex(['2025-01-01']))
    assert weather_grid_point(121.54, 25.02) == (25.0, 121.25)


@pytest.mark.parametrize('failure', ['missing', 'fill', 'nonfinite', 'units', 'time_standard'])
def test_invalid_weather_rejected_without_filling(failure):
    payload = weather_payload()
    if failure == 'missing':
        del payload['properties']['parameter']['T2M']['2024022900']
    elif failure in ('fill', 'nonfinite'):
        payload['properties']['parameter']['T2M']['2024022900'] = -999 if failure == 'fill' else float('nan')
    elif failure == 'units':
        payload['parameters']['T2M']['units'] = 'K'
    else:
        payload['header']['time_standard'] = 'LST'
    with pytest.raises(ValueError):
        parse_weather(payload)


def test_bias_direction_and_annual_energy_normalized_shape():
    benchmark = np.array([0., .2, .7, .9, .1, 0.])
    ghi = np.array([0., 100., 400., 900., 80., 0.])
    result = compare_profiles(benchmark * 1.2, benchmark, ghi)
    assert result['simple_over_benchmark_bias_fraction'] == pytest.approx(.2)
    assert result['daylight_nrmse'] > .2
    assert result['daylight_correlation'] == pytest.approx(1)
    assert result['annual_energy_normalized_daylight_shape_nrmse'] == pytest.approx(0, abs=1e-12)
    shifted = compare_profiles(np.roll(benchmark, 1), benchmark, ghi)
    assert shifted['simple_over_benchmark_bias_fraction'] == pytest.approx(0)
    assert shifted['annual_energy_normalized_daylight_shape_nrmse'] > .5


def test_photon_conversion_and_nameplate_do_not_apply_module_efficiency_twice():
    pvlib = pytest.importorskip('pvlib')
    assert par_to_ghi(1000) == pytest.approx(476.0869565217391)
    assert pvlib.pvsystem.pvwatts_dc(par_to_ghi(1000), 25, 1000, -.0047) == pytest.approx(476.0869565217391)
    index = pd.date_range('2024-03-20 04:00', periods=2, freq='12h')
    par = pd.Series(1000 / (.219 / .46), index=index)
    ac, removed = model_ac_per_kwp(par, parse_weather(weather_payload()), 121.0, 24.0, 20)
    assert .65 < ac.iloc[0] <= 1.0
    assert ac.iloc[1] == 0.0
    assert removed == pytest.approx(.5)


def test_horizon_is_nonnegative_and_zero_at_and_below_horizon():
    np.testing.assert_array_equal(enforce_horizon([5, 6, -1, 3], [90, 95, 40, 80]), [0, 0, 0, 3])
    with pytest.raises(ValueError, match='Nonfinite'):
        enforce_horizon([float('nan')], [90])


def test_dispatch_rank_shifts_and_abatement_change():
    original = pd.DataFrame({'abatement_t': [4., 3., 2.]}, index=['a', 'b', 'c'])
    changed = pd.DataFrame({'abatement_t': [2., 3., 1.]}, index=['a', 'b', 'c'])
    result = compare_dispatch_ranks(original, changed)
    assert result['operational_abatement_fraction_change'] == pytest.approx(-1 / 3)
    assert result['maximum_absolute_rank_shift'] == 1
    assert result['baseline_top_five_cities'][0] == 'a'
    assert result['benchmark_top_five_cities'][0] == 'b'
    assert result['spearman_rank_correlation'] == pytest.approx(.5)


def test_equivalent_par_preserves_capacity_cost_and_lighting(monkeypatch):
    monkeypatch.setattr(simulation, '_load_city_representative_points', lambda: {'city': (121., 24.)})
    cfg = SimpleNamespace(load_mode='solar_zenith', solar_zenith_deg=90.833,
                          analysis_years=20., electricity_price=3.7556, n_lights=64,
                          light_power_kw=.1, lighting_threshold_par=1., storage_capacity_kwh=10.,
                          storage_power_kw=5., storage_eta_roundtrip=.9,
                          economic_costs={'pv_capacity_kw_per_factor': 18., 'pv_capex_ntd_per_kw': 42880.,
                                          'pv_om_ntd_per_kw_year': 704., 'battery_capex_ntd_per_kwh': 8096.,
                                          'battery_power_capex_ntd_per_kw': 30976.,
                                          'battery_om_fraction_of_capex_per_year': .025,
                                          'other_capex': 44583., 'other_om': 46097.,
                                          'other_eol': 6300., 'grid_fixed_cost': 95000.},
                          financial_params={'discount_rate': .05},
                          pareto_battery_cfg={'power_cap_multiplier_of_load': 1.5})
    index = pd.date_range('2024-03-20', periods=144, freq='10min')
    zenith = simulation._solar_zenith_deg_for_city(index=index, longitude_deg=121., latitude_deg=24.)
    par = pd.DataFrame({'city': np.maximum(np.cos(np.deg2rad(zenith)), 0) * 1500}, index=index)
    aef = {'region': pd.Series(.5, index=index)}
    region_map = {'region': ['city']}
    original = selected_dispatch(par, aef, region_map, cfg, .0077)
    model = par * (.0077 / 18) * .8
    equivalent = model / (.0077 / 18)
    np.testing.assert_allclose(equivalent * .0077 * 1.85, model * 18 * 1.85)
    changed = selected_dispatch(equivalent, aef, region_map, cfg, .0077)
    for column in ('battery_capacity_kwh', 'battery_power_kw', 'grid_only_emission_t',
                   'pv_storage_capex_cost', 'pv_storage_om_cost', 'pv_storage_eol_cost'):
        assert changed.loc['city', column] == original.loc['city', column]
    assert changed.loc['city', 'battery_capacity_kwh'] / 64 == pytest.approx(1.3203125)
    assert original.loc['city', 'solar_panel_factor'] * 18 / 64 == pytest.approx(.5203125)
    cfg.load_mode = 'par_threshold'
    with pytest.raises(ValueError, match='solar_zenith'):
        selected_dispatch(equivalent, aef, region_map, cfg, .0077)
