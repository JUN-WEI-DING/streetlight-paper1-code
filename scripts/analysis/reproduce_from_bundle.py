"""Stage and reproduce the local review bundle in an isolated code checkout.

Quick replay starts at frozen scenario intermediates, not at raw observations.
It recomputes the base value analyses, PV dispatch comparison, and conditional
uncertainty, then compares numerical outputs with immutable reference answers.
Use --full to rerun the documented numerical suite from processed inputs.
This scope does not include every manuscript experiment or raw-data preparation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path('outputs/final_runs/paper1_canonical_results')
FINALS = {'paper1_manuscript_values.json', 'pv_model_comparison.json', 'conditional_uncertainty.json'}


def stage(archive, *, full=False):
    if (ROOT / 'reference').exists() or (ROOT / 'outputs/final_runs').exists():
        raise FileExistsError('Use a fresh checkout: reference or final_runs already exists')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for member in members:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or (path.parts[0] != 'reference' and member.name != 'bundle.json')):
                raise ValueError(f'Unexpected archive entry: {member.name}')
        tar.extractall(ROOT, filter='data')
    manifest = json.loads((ROOT / 'bundle.json').read_text())
    for row in manifest['files']:
        source = ROOT / row['path']
        relative = Path(row['source_path'])
        if (relative.is_absolute() or '..' in relative.parts
                or relative.parts[0] not in {'outputs', 'data'}
                or row['path'] != 'reference/' + relative.as_posix()):
            raise ValueError(f'Invalid bundle path: {relative}')
        if (source.stat().st_size != row['bytes']
                or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']):
            raise ValueError(f'Checksum mismatch: {source}')
        if (row['role'] == 'expected_answer' or (full and (
                relative.is_relative_to(RESULTS) or
                relative.is_relative_to('outputs/paper_assets') or
                relative.is_relative_to('outputs/qa')))):
            continue
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    print(f'Verified {len(manifest["files"])} bundled files; expected answers kept in reference/', flush=True)


def compare_numeric(expected, actual, path='', errors=None, counts=None):
    errors = [] if errors is None else errors
    counts = [0] if counts is None else counts
    # Code hashes, original staging paths and dependency versions describe
    # provenance, not numerical agreement. Input bytes are checked separately.
    ignored = {'provenance', 'config', 'contract', 'source_config', 'source_results_dir',
               'source_figure_data_dir', 'source_sha256', 'numpy_version', 'pandas_version',
               'scipy_version', 'pvlib_version', 'source_files', 'tokens', 'blocks'}
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(path + ': expected mapping'); return errors, counts[0]
        for key, value in expected.items():
            if key in ignored:
                continue
            if key not in actual:
                errors.append(path + '/' + key + ': missing')
            else:
                compare_numeric(value, actual[key], path + '/' + key, errors, counts)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(path + ': list length/type differs')
        else:
            for i, (a, b) in enumerate(zip(expected, actual)):
                compare_numeric(a, b, f'{path}/{i}', errors, counts)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        counts[0] += 1
        if not isinstance(actual, (int, float)) or not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8):
            errors.append(f'{path}: {expected!r} != {actual!r}')
    elif isinstance(expected, bool) or expected is None:
        if expected != actual:
            errors.append(path + ': flag/null differs')
    return errors, counts[0]


def run(script, *args):
    env = dict(os.environ)
    # Do not inherit an author's input/output overrides.
    for key in list(env):
        if key.startswith(('PAPER_', 'STREETLIGHT_')):
            del env[key]
    env.update(STREETLIGHT_CONFIG='config/paper_baseline.yaml', STREETLIGHT_ROOT=str(ROOT),
               PYTHONPATH=str(ROOT / 'src'), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(ROOT / 'outputs/reproduction/matplotlib-cache'))
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analysis' / script), *args],
                   cwd=ROOT, env=env, check=True)


def replay(*, full=False, run_suite=True):
    for name in FINALS:
        if (ROOT / RESULTS / name).exists():
            raise FileExistsError(f'Replay needs absent generated answers: {name}')
    manifest_path = ROOT / 'bundle.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('external_weather'):
        from fetch_bundle_weather import fetch
        fetch(ROOT, check_only=True)
    if full and run_suite:
        if (ROOT / RESULTS).exists() or (ROOT / 'outputs/paper_assets').exists():
            raise FileExistsError('Full rerun needs absent analysis outputs; stage with --full in a fresh checkout')
        run('run_paper1_suite.py')
    run('rebuild_core_figure_data.py')
    if full:
        run('rebuild_supplementary_figure_data.py')
    run('build_paper1_manuscript_values.py', '--base-only')
    run('paper1_pv_benchmark.py')
    run('paper1_conditional_uncertainty.py')
    run('build_paper1_manuscript_values.py')
    verify(full=full)


def verify(*, full=False):
    checks, failures = {}, []
    json_names = sorted(FINALS) + (['allocation_sensitivity/summary.json'] if full else [])
    for name in json_names:
        expected = json.loads((ROOT / 'reference' / RESULTS / name).read_text())
        actual = json.loads((ROOT / RESULTS / name).read_text())
        errors, count = compare_numeric(expected, actual)
        checks[name] = {'numeric_values_checked': count, 'differences': errors}
        failures.extend(errors)
    import pandas as pd
    table_checks = {}
    table_names = (sorted(p.name for p in (ROOT / 'reference/outputs/paper_assets/paper1/figure_data').glob('*.csv'))
                   if full else ['f2_pareto_frontier_source.csv', 'f3_lca_balance_components.csv',
                                 'f4_input_output_maps_source.csv'])
    for name in table_names:
        relative = Path('outputs/paper_assets/paper1/figure_data') / name
        expected = pd.read_csv(ROOT / 'reference' / relative)
        actual = pd.read_csv(ROOT / relative)
        try:
            pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-8, atol=1e-8)
            table_checks[name] = {'rows': len(actual), 'passed': True}
        except AssertionError as error:
            table_checks[name] = {'rows': len(actual), 'passed': False, 'difference': str(error)}
            failures.append(f'Table differs: {name}')
    if full:
        for reference in sorted((ROOT / 'reference' / RESULTS).rglob('*.csv')):
            relative = reference.relative_to(ROOT / 'reference')
            actual_path = ROOT / relative
            key = relative.relative_to(RESULTS).as_posix()
            try:
                expected = pd.read_csv(reference)
                actual = pd.read_csv(actual_path)
                pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-8, atol=1e-8)
                table_checks[key] = {'rows': len(actual), 'passed': True}
            except (AssertionError, FileNotFoundError) as error:
                table_checks[key] = {'passed': False, 'difference': str(error)}
                failures.append(f'Table differs or missing: {key}')
    run('plot_reproduction_results.py')
    report = {'scope': ('Documented numerical suite rerun from processed inputs; excludes raw-data preparation and other manuscript experiments'
                        if full else 'Replay from frozen scenario intermediates; PV dispatch is recalculated from processed time series'),
              'documented_suite_from_processed_inputs': full,
              'all_manuscript_experiments_rebuilt': False,
              'frozen_analysis_inputs': [] if full else ['scenario intermediates'],
              'rtol': 1e-8, 'atol': 1e-8, 'checks': checks, 'table_checks': table_checks,
              'passed': not failures}
    if full:
        manifest = json.loads((ROOT / RESULTS / 'paper1_suite_manifest.json').read_text())
        report['suite_elapsed_seconds'] = sum(step['elapsed_seconds'] for step in manifest['steps'])
    target = ROOT / 'outputs/reproduction/verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': not failures, 'numeric_values_checked': sum(v['numeric_values_checked'] for v in checks.values()),
                      'differences': len(failures), 'report': str(target)}))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Verify and stage archive before replay')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--stage-only', action='store_true')
    mode.add_argument('--verify-only', action='store_true', help='Compare already generated outputs without rerunning calculations')
    parser.add_argument('--full', action='store_true', help='Rerun suite and all numerical figure tables from processed inputs')
    args = parser.parse_args()
    if args.archive:
        stage(args.archive.resolve(), full=args.full)
    if args.verify_only:
        verify(full=args.full)
    elif not args.stage_only:
        replay(full=args.full)

if __name__ == '__main__':
    main()
