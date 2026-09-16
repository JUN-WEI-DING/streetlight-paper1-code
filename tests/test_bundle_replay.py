import io
import tarfile
from pathlib import Path
import pytest
import reproduce_from_bundle as replay


def test_comparison_detects_changed_scientific_value_and_missing_condition():
    expected = {'conditions': [{'abatement': 3.56, 'positive': True}], 'provenance': {'source': 'old'}}
    actual = {'conditions': [{'abatement': 3.57, 'positive': True}], 'provenance': {'source': 'new'}}
    errors, count = replay.compare_numeric(expected, actual)
    assert count == 1 and len(errors) == 1
    assert '/conditions/0/abatement' in errors[0]
    assert replay.compare_numeric(expected, {'conditions': []})[0]


def test_comparison_accepts_roundoff_but_not_nonfinite():
    assert replay.compare_numeric({'x': 1.0}, {'x': 1.0 + 1e-10}) == ([], 1)
    assert replay.compare_numeric({'x': 1.0}, {'x': float('nan')})[0]


def test_archive_cannot_write_outside_checkout(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    root.mkdir()
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('../escaped')
        info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    monkeypatch.setattr(replay, 'ROOT', root)
    with pytest.raises(ValueError, match='Unexpected archive entry'):
        replay.stage(archive)
    assert not (tmp_path / 'escaped').exists()


def test_replay_refuses_preloaded_answers(tmp_path, monkeypatch):
    results = tmp_path / replay.RESULTS
    results.mkdir(parents=True)
    (results / 'paper1_manuscript_values.json').write_text('{}')
    monkeypatch.setattr(replay, 'ROOT', tmp_path)
    with pytest.raises(FileExistsError, match='absent generated answers'):
        replay.replay()
