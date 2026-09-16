"""Conservation and aggregation invariants for the allocation stress test."""
import numpy as np
import pandas as pd
import pytest
from run_allocation_sensitivity import allocated_buckets, economic_summary, perturb_shares


@pytest.mark.parametrize('target', ['north', 'central', 'south', 'east'])
@pytest.mark.parametrize('change', [-0.2, 0.2])
def test_relative_share_change_preserves_total_and_other_ratios(target, change):
    shares = pd.Series({'north': .20, 'central': .56, 'south': .23, 'east': .01})
    result = perturb_shares(shares, target, change)
    assert result.sum() == pytest.approx(1)
    assert result[target] == pytest.approx(shares[target] * (1 + change))
    other = shares.index != target
    assert np.ptp((result[other] / shares[other]).to_numpy()) < 1e-12
    totals = np.array([0., 1., 500., 1930.4])
    np.testing.assert_allclose((totals[:, None] * result.to_numpy()).sum(axis=1), totals)


def test_bucket_rejects_time_varying_shares():
    raw = {r: pd.DataFrame({'Co-Gen-_other_allocated': values}) for r, values in
           {'north': [1., 2.], 'central': [2., 2.], 'south': [2., 2.], 'east': [1., 1.]}.items()}
    with pytest.raises(AssertionError):
        allocated_buckets(raw, 'Co-Gen')


def test_mac_uses_ratio_of_equal_city_means():
    data = pd.DataFrame({'city': [f'city{i}' for i in range(22)],
                         'abatement_t': [2.] * 11 + [4.] * 11, 'delta_cost': [12.] * 22})
    result = economic_summary(data, 2)
    assert result['abatement_t_per_light'] == 1.5
    assert result['delta_cost_ntd_per_light'] == 6
    assert result['mac_ntd_per_t'] == 4
