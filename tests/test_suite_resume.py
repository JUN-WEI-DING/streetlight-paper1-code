"""A failed suite retains progress and resumes without rerunning upstream work."""
import json
from types import SimpleNamespace
import pytest
import run_paper1_suite as suite


def test_resume_preserves_upstream_work_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, 'get_config', lambda: SimpleNamespace(paper_output_dir=tmp_path))
    monkeypatch.setattr(suite, 'repo_display_path', str)
    calls = []
    names = [name for name in vars(suite) if name.startswith('run_') or name == 'build_manuscript_results_snapshot']
    for name in names:
        monkeypatch.setattr(suite, name, lambda *args, name=name: calls.append(name))
    def fail():
        raise RuntimeError('interrupted scenario')
    monkeypatch.setattr(suite, 'run_future_grid_scenarios', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        suite.main()
    manifest = json.loads((tmp_path / 'paper1_suite_manifest.json').read_text())
    assert [s['step'] for s in manifest['steps']] == [
        'paper_baseline', 'allocation_sensitivity', 'calibration_robustness', 'method_simplifications', 'closing_analyses']
    assert not manifest['complete']
    calls.clear()
    monkeypatch.setattr(suite, 'run_future_grid_scenarios', lambda: calls.append('future'))
    suite.main(start_at='future_grid_scenarios')
    assert calls[0] == 'future'
    assert 'run_paper_baseline' not in calls
    manifest = json.loads((tmp_path / 'paper1_suite_manifest.json').read_text())
    assert manifest['complete'] and manifest['start_at'] == 'future_grid_scenarios'
    assert len(manifest['steps']) == 9
