"""Annual hours must not depend on pandas timestamp storage resolution."""
import pandas as pd
import pytest
import run_calibration_robustness as calibration


@pytest.mark.parametrize('unit', ['ns', 'us'])
def test_threshold_hours_have_physical_units(tmp_path, monkeypatch, unit):
    path = tmp_path / 'par.csv'
    pd.DataFrame({'city': [0., 0., 20., 20.]},
                 index=pd.date_range('2024-01-01', periods=4, freq='10min')).to_csv(path)
    convert = pd.to_datetime
    monkeypatch.setattr(calibration.pd, 'to_datetime', lambda value: convert(value).as_unit(unit))
    monkeypatch.setattr(calibration, 'resolve_repo_display_path', lambda _: path)
    result = calibration._threshold_factor_sensitivity(
        {'par_path': 'par.csv', 'n_lights': 10, 'light_power_kw': 0.1}, [2.0])
    assert result.iloc[0].mean_annual_lighting_hours == pytest.approx(1 / 3)
    assert result.iloc[0].mean_city_parity_factor == pytest.approx(0.05)
