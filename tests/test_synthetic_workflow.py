"""Physical and accounting checks independent of private result snapshots."""
import numpy as np
import pandas as pd
import pytest

from run_demo import run_demo
from streetlight.simulation.storage import storage_dispatch, recompute_costs_from_results
from paper1_config import load_paper1_config
from paper1_conditional_uncertainty import (
    INPUT_NAMES, SEED, triangular_parameters, sample_inputs, incremental_cost_usd,
)


def test_lossless_dispatch_conserves_energy_and_respects_capacity():
    net = pd.Series([4., 4., -2., -4.], index=pd.date_range('2024-01-01', periods=4, freq='h'))
    result = storage_dispatch(net, capacity_kwh=5, power_kw=4, eta_roundtrip=1, dt_hours=1)
    np.testing.assert_allclose(result['soc_kwh'], [4, 5, 3, 0])
    np.testing.assert_allclose(result['grid_import_kw'], [0, 0, 0, 1])
    # 8 kWh surplus = 5 kWh stored plus 3 kWh remaining; 6 deficit = 5 discharge + 1 grid.
    assert result['net_with_storage_kw'].clip(lower=0).sum() == pytest.approx(3)


def test_demo_integrals_match_reported_emissions_and_are_deterministic():
    frame, summary = run_demo()
    again, repeated = run_demo()
    pd.testing.assert_frame_equal(frame, again)
    assert summary == repeated
    metrics = summary['metrics_installation_64_lights']
    assert frame['soc_kwh'].between(-1e-12, 20 + 1e-12).all()
    assert (frame['grid_import_kw'] >= 0).all()
    scale = 20 * 8760 / 48
    expected = (frame['grid_import_kw'] * frame['aef_kg_co2e_per_kwh']).sum() / 6 * scale / 1000
    assert metrics['pv_storage_emission_t'] == pytest.approx(expected)
    assert metrics['pv_storage_energy_kwh_20y'] < metrics['grid_energy_kwh_20y']


@pytest.mark.parametrize('lifetime', [8, 10, 12, 15])
def test_conditional_cost_matches_scalar_accounting_on_artificial_energy(lifetime):
    config = load_paper1_config()
    lights = config.lights_per_city
    selected = {'pv_kw_per_streetlight': 18 / lights, 'battery_kwh_per_streetlight': 20 / lights}
    power = min(20 * config.storage_power_kw_per_kwh, config.battery_power_cap_kw_per_city)
    parameters = triangular_parameters(config)
    draws = np.vstack([sample_inputs(parameters, 5, SEED), [parameters[n][1] for n in INPUT_NAMES]])
    draws[-1, 1] = 0
    actual = incremental_cost_usd(draws, annual_avoided_kwh=100, lifetime=lifetime,
                                  selected=selected, config=config)
    frame = pd.DataFrame([{'solar_panel_factor': 1., 'battery_capacity_kwh': 20.,
        'battery_power_kw': power, 'grid_energy_kwh_20y': 8000 * lights,
        'pv_storage_energy_kwh_20y': 6000 * lights}])
    for i, (tariff, rate, pv_capex, multiplier) in enumerate(draws):
        costs = {**config.economic_costs, 'pv_capex_ntd_per_kw': pv_capex, 'battery_replacement_year': lifetime}
        for key in ('battery_capex_ntd_per_kwh', 'battery_power_capex_ntd_per_kw',
                    'battery_replacement_energy_capex_ntd_per_kwh', 'battery_replacement_power_capex_ntd_per_kw'):
            costs[key] *= multiplier
        scalar = recompute_costs_from_results(frame, years=20, elec_price=tariff,
            economic_costs=costs, financial_params={**config.financial_params, 'discount_rate': rate})
        assert actual[i] == pytest.approx(scalar.iloc[0]['delta_cost'] / lights / config.ntd_per_usd, abs=1e-9)


def test_triangular_draws_have_correct_cdf_and_reproducible_prefix():
    parameters = triangular_parameters(load_paper1_config())
    draws = sample_inputs(parameters, 100, SEED)
    np.testing.assert_array_equal(draws[:7], sample_inputs(parameters, 7, SEED))
    uniforms = np.random.default_rng(SEED).random((100, 4))
    for i, name in enumerate(INPUT_NAMES):
        low, mode, high = parameters[name]
        x = draws[:, i]
        cdf = np.where(x <= mode, (x-low)**2 / ((high-low)*(mode-low)),
                       1 - (high-x)**2 / ((high-low)*(high-mode)))
        np.testing.assert_allclose(cdf, uniforms[:, i], atol=1e-14)
