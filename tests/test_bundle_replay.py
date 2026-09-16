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


def test_full_stage_keeps_results_as_references_only(tmp_path, monkeypatch):
    import hashlib
    import json
    root = tmp_path / 'checkout'
    root.mkdir()
    archive = tmp_path / 'bundle.tar.gz'
    paths = [str(replay.RESULTS / 'pareto/results.csv'),
             'outputs/paper_assets/paper1/figure_data/f5.csv',
             'outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv',
             'outputs/qa/cogen_biomass_sensitivity.csv']
    records = []
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths:
            data = b'x\n1\n'
            info = tarfile.TarInfo('reference/' + path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            records.append({'path': info.name, 'source_path': path, 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest(), 'role': 'input_or_intermediate'})
        data = json.dumps({'files': records}).encode()
        info = tarfile.TarInfo('bundle.json')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(replay, 'ROOT', root)
    replay.stage(archive, full=True)
    for path in paths:
        assert (root / 'reference' / path).exists()
    assert not (root / replay.RESULTS).exists()
    assert not (root / 'outputs/paper_assets').exists()
    assert (root / paths[2]).exists()
    assert (root / paths[3]).exists()  # Explicit historical QA exception.


def test_full_rerun_rejects_staged_old_intermediates(tmp_path, monkeypatch):
    (tmp_path / replay.RESULTS / 'pareto').mkdir(parents=True)
    monkeypatch.setattr(replay, 'ROOT', tmp_path)
    with pytest.raises(FileExistsError, match='absent analysis outputs'):
        replay.replay(full=True)
