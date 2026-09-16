"""Bundle selection ships reproducible allocation results, not historical QA."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'build_local_data_bundle',
    Path(__file__).resolve().parents[1] / 'scripts/release/build_local_data_bundle.py',
)
bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bundle)


def test_allocation_references_required_and_historical_qa_excluded(tmp_path):
    required = [bundle.INPUTS / 'par/par_wide.csv', bundle.RESULTS / 'pareto/results.csv',
                bundle.RESULTS / 'paper1_manuscript_values.json']
    allocation = bundle.RESULTS / 'allocation_sensitivity'
    required += [allocation / name for name in (
        'summary.json', 'scenario_summary.csv', 'regional_summary.csv',
        'city_results.csv', 'allocation_shares.csv')]
    historical = Path('outputs/qa/cogen_biomass_sensitivity.csv')
    weather = Path('outputs/final_runs/paper1_pv_benchmark_inputs/met_22.500_120.625.json')
    boundary = Path('data/geo/county_boundaries.shp')
    for relative in required + [historical, weather, boundary]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if path.suffix == '.json' else 'value\n1\n')
    selected = bundle.selected_files(tmp_path)
    assert all(tmp_path / relative in selected for relative in required)
    assert tmp_path / historical not in selected
    assert tmp_path / weather not in selected
    assert tmp_path / boundary not in selected
    (tmp_path / allocation / 'scenario_summary.csv').unlink()
    with pytest.raises(FileNotFoundError, match='scenario_summary.csv'):
        bundle.selected_files(tmp_path)


def test_portable_metadata_changes_only_machine_paths():
    import json
    original = {'source_path': '/mnt/example/observations.parquet',
                'nested': [{'path': '/home/example/weekly', 'count': 50892}],
                'url': 'https://example.org/data', 'relative': 'data/inputs.csv'}
    data = json.dumps(original).encode()
    result = json.loads(bundle.portable_metadata(data))
    assert result == {**original, 'source_path': 'external-inputs/observations.parquet',
                      'nested': [{'path': 'external-inputs/weekly', 'count': 50892}]}
    untouched = b'{"value": 3.56, "path": "data/inputs.csv"}\n'
    assert bundle.portable_metadata(untouched) == untouched
