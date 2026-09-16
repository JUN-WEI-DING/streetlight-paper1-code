"""Focused arithmetic tests using artificial inputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from paper1_aef_storage_audit import (
    _match, _pool_summary, _replay_with_ledger, build_aef_storage_audit,
)
from streetlight.aef.pipeline import AEFPipeline, MAINLAND_REGIONS


def _regional(scale=1.0):
    idx = pd.date_range('2024-01-01', periods=4, freq='10min')
    frames = {}
    for r, a in zip(MAINLAND_REGIONS, [.4, .6, .8, .2]):
        f = pd.DataFrame({'AEF': a, 'Total_Gen (MWh)': 100.0 * scale,
                          'BESS': 0.0, 'PHS': 0.0}, index=idx)
        flows = {'central': ['F_NC', 'F_SC', 'F_EC'], 'north': ['F_CN'],
                 'south': ['F_CS'], 'east': ['F_CE']}[r]
        for c in flows:
            f[c] = 0.0
        frames[r] = f
    frames['central']['BESS'] = np.array([-10., -5., 100., 0.]) * scale
    frames['central']['PHS'] = np.array([-50., 10., 0., -5.]) * scale
    return frames


def test_common_power_scaling_preserves_aef_and_scales_pool():
    outputs, pool, ledger = _replay_with_ledger(AEFPipeline(), _regional())
    scaled, scaled_pool, _ = _replay_with_ledger(AEFPipeline(), _regional(1 / 6))
    for r in MAINLAND_REGIONS:
        np.testing.assert_allclose(outputs[r]['FLOW_UNIT_FINAL_AEF'], scaled[r]['FLOW_UNIT_FINAL_AEF'])
    for col in ['GRID_POOL_E (MWh)', 'GRID_POOL_C (kgCO2)', 'BESS_DEFICIT (MWh)']:
        np.testing.assert_allclose(pool[col] / 6, scaled_pool[col], atol=1e-12)
    # First slot: gross MW snapshots times 1/6 h and 1000 kWh/MWh.
    assert pool['GRID_POOL_E (MWh)'].iloc[0] / 6 == pytest.approx((10 * .9487 + 50 * .8832) / 6)
    assert pool['GRID_POOL_C (kgCO2)'].iloc[0] * 1000 / 6 == pytest.approx(60 * .6 * 1000 / 6)
    assert ledger['charge_carbon'][0] == pytest.approx(60 * .6)


def test_ledger_closes_with_deficit_and_nonempty_terminal_pool():
    frames = _regional()
    pipeline = AEFPipeline()
    original = pipeline.storage_manager
    _, pool, ledger = _replay_with_ledger(pipeline, frames)
    assert pipeline.storage_manager is original
    summary = _pool_summary(pool, ledger,
        np.stack([frames[r]['BESS'].to_numpy() for r in MAINLAND_REGIONS], axis=1),
        frames['central']['PHS'].to_numpy())
    assert summary['terminal_energy_mwh'] == pytest.approx(5 * .8832 / 6)
    assert summary['terminal_carbon_kgco2e'] == pytest.approx(5 * .6 * 1000 / 6)
    assert summary['deficits']['bess']['steps_above_tolerance'] == 1
    assert summary['closure']['max_step_mwh'] < 1e-12
    assert summary['closure']['max_step_kgco2e'] < 1e-10
    assert summary['charging_vs_pre_storage_mix']['bess']['changed_steps_above_tolerance'] == 1


@pytest.mark.parametrize('bad', [pd.Series([1., np.nan]), pd.Series([1., 3.]),
                               pd.Series([1., 2.], index=[1, 2])])
def test_frozen_comparison_rejects_values_masks_and_indexes(bad):
    with pytest.raises(ValueError):
        _match(bad, pd.Series([1., 2.]))
