import hashlib
import json

import pytest
import fetch_bundle_weather as weather


def test_metadata_changes_allowed_but_changed_values_rejected(monkeypatch):
    monkeypatch.setattr(weather, 'parse_weather', lambda value: None)
    payload = {'header': {'api': 'old'}, 'properties': {'parameter': {'T2M': {'2024010100': 12.0}}}}
    digest = hashlib.sha256(json.dumps(payload['properties']['parameter'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    payload['header']['api'] = 'new'
    weather.validate(payload, digest)
    payload['properties']['parameter']['T2M']['2024010100'] = 13.0
    with pytest.raises(ValueError, match='differ from the study snapshot'):
        weather.validate(payload, digest)


def test_check_only_never_downloads_missing_weather(tmp_path, monkeypatch):
    row = {'path': 'outputs/final_runs/paper1_pv_benchmark_inputs/met_22.500_120.625.json',
           'latitude': 22.5, 'longitude': 120.625, 'parameter_sha256': 'unused'}
    (tmp_path / 'bundle.json').write_text(json.dumps({'external_weather': [row]}))
    monkeypatch.setattr(weather, 'urlopen', lambda *a, **kw: pytest.fail('unexpected network access'))
    with pytest.raises(FileNotFoundError):
        weather.fetch(tmp_path, check_only=True)
