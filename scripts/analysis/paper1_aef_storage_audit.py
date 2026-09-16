"""Verify frozen AEF/storage arithmetic from staged CSVs, without raw rebuilds.

The historical scale=1 run accumulates MW snapshots as internal pool quantities.
For each retained 10-minute slot, physical energy is internal energy / 6 MWh;
physical carbon is internal carbon * 1000 / 6 kg CO2e. Missing slots receive no
extra duration. This audits the accounting surrogate, not measured grid closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from streetlight.aef.pipeline import AEFPipeline, MAINLAND_REGIONS
from streetlight.aef.storage import StorageManager, StorageState

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-9


def _match(actual: pd.Series, expected: pd.Series, *, atol: float = TOL) -> float:
    """Require identical index/missing masks and finite numerical agreement."""
    if not actual.index.equals(expected.index):
        raise ValueError(f"AEF audit index mismatch: {expected.name}")
    a, b = actual.to_numpy(dtype=float), expected.to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError(f"AEF audit missing-mask mismatch: {expected.name}")
    valid = ~np.isnan(a)
    if not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all():
        raise ValueError(f"AEF audit nonfinite values: {expected.name}")
    error = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
    if error > atol:
        raise ValueError(f"AEF audit mismatch: {expected.name}, max error {error}")
    return error


def _replay_with_ledger(pipeline: AEFPipeline, regional: dict):
    """Instrument only this pipeline instance; retain original storage equations."""
    n = len(next(iter(regional.values())))
    ledger = {key: np.zeros(n) for key in (
        'charge_energy', 'charge_carbon', 'withdraw_energy', 'withdraw_carbon',
        'phs_charge_base_difference', 'bess_charge_base_difference',
    )}

    class AuditedState(StorageState):
        bess_charge_count = 0

        def charge(self, q, aef, efficiency):
            i = len(self.history_energy)
            super().charge(q, aef, efficiency)
            ledger['charge_energy'][i] += q * efficiency
            ledger['charge_carbon'][i] += q * aef
            if efficiency == pipeline.storage_manager.phs_config.charge_efficiency:
                base = pipeline._loop_base['C'][i]
                key = 'phs_charge_base_difference'
            else:
                charging = [r for r in MAINLAND_REGIONS
                            if r in pipeline._loop_in and pipeline._loop_in[r]['BESS'][i] < 0]
                region = charging[self.bess_charge_count]
                self.bess_charge_count += 1
                base = pipeline._loop_base[dict(pipeline.region_codes)[region]][i]
                key = 'bess_charge_base_difference'
            ledger[key][i] += q * (aef - base)

        def request_discharge(self, request, efficiency):
            delivered, aef, deficit = super().request_discharge(request, efficiency)
            i = len(self.history_energy)
            ledger['withdraw_energy'][i] += delivered / efficiency
            ledger['withdraw_carbon'][i] += delivered * aef if aef is not None else 0.0
            return delivered, aef, deficit

        def record_state(self, **kwargs):
            super().record_state(**kwargs)
            self.bess_charge_count = 0

    class AuditedManager(StorageManager):
        def reset(self):
            self.grid_pool_state = AuditedState()

    original = pipeline.storage_manager
    pipeline.storage_manager = AuditedManager(original.bess_config, original.phs_config)
    try:
        result, pool = pipeline.run(region_data={r: df.copy() for r, df in regional.items()})
    finally:
        pipeline.storage_manager = original
    return result, pool, ledger


def _pool_summary(pool: pd.DataFrame, ledger: dict, bess: np.ndarray, phs: np.ndarray):
    energy = pool['GRID_POOL_E (MWh)'].to_numpy()
    carbon = pool['GRID_POOL_C (kgCO2)'].to_numpy()
    if not np.isfinite(energy).all() or not np.isfinite(carbon).all():
        raise ValueError('Nonfinite storage pool')
    if (energy < 0).any() or (carbon < 0).any():
        raise ValueError('Negative storage pool')
    net = bess.sum(axis=1)
    pos, neg = np.maximum(bess, 0).sum(axis=1), np.maximum(-bess, 0).sum(axis=1)
    closure = {}
    for name, stock, factor, unit in [('energy', energy, 1 / 6, 'mwh'),
                                       ('carbon', carbon, 1000 / 6, 'kgco2e')]:
        incoming, outgoing = ledger[f'charge_{name}'], ledger[f'withdraw_{name}']
        residual = np.diff(np.r_[0.0, stock]) - incoming + outgoing
        maximum = float(np.max(np.abs(residual)))
        if maximum > TOL:
            raise ValueError(f'Storage {name} ledger does not close: {maximum}')
        closure[f'max_step_{unit}'] = maximum * factor
        closure[f'whole_period_{unit}'] = float(stock[-1] - incoming.sum() + outgoing.sum()) * factor
    deficits = {}
    for actor, requests in [('phs', np.maximum(phs, 0)),
                            ('bess', np.where(net >= 0, pos, 0))]:
        deficit = pool[f'{actor.upper()}_DEFICIT (MWh)'].to_numpy()
        deficits[actor] = {
            'steps_above_tolerance': int((deficit > TOL).sum()),
            'requested_mwh': float(requests.sum() / 6),
            'deficit_mwh': float(deficit.sum() / 6),
            'deficit_fraction': float(deficit.sum() / requests.sum()) if requests.sum() else 0.0,
        }
    return {
        'zero_initial_energy_and_carbon': True,
        'terminal_energy_mwh': float(energy[-1] / 6),
        'terminal_carbon_kgco2e': float(carbon[-1] * 1000 / 6),
        'maximum_energy_mwh': float(energy.max() / 6),
        'empty_end_of_step_count': int((energy <= 1e-12).sum()),
        'negative_pool_count': 0,
        'deficits': deficits,
        'closure': closure,
        'mixed_bess_sign_steps': int(((pos > 0) & (neg > 0)).sum()),
        'ignored_opposite_direction_bess_charge_mwh': float(np.where(net >= 0, neg, 0).sum() / 6),
        'ignored_opposite_direction_bess_discharge_mwh': float(np.where(net < 0, pos, 0).sum() / 6),
        'charging_vs_pre_storage_mix': {
            actor: {'changed_steps_above_tolerance': int((np.abs(ledger[f'{actor}_charge_base_difference']) > TOL).sum()),
                    'charge_carbon_difference_kgco2e': float(ledger[f'{actor}_charge_base_difference'].sum() * 1000 / 6)}
            for actor in ('phs', 'bess')
        },
    }


def build_aef_storage_audit(canonical_inputs_dir: Path) -> dict:
    """Return deterministic, compact verification and physical-unit diagnostics."""
    canonical_inputs_dir = Path(canonical_inputs_dir)
    pipeline = AEFPipeline()
    def read(path):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f'Invalid staged index: {path.name}')
        return frame
    frozen = {r: read(canonical_inputs_dir / 'aef' / f'{r}.csv') for r in pipeline.regions}
    frozen_pool = read(canonical_inputs_dir / 'aef/storage_pools.csv')
    common_index = next(iter(frozen.values())).index
    for frame in frozen.values():
        common_index = common_index.intersection(frame.index)
    if not common_index.equals(frozen_pool.index):
        raise ValueError('Common regional/pool index mismatch')
    base_error, base_aef_error = 0.0, 0.0
    for region, expected in frozen.items():
        raw = read(canonical_inputs_dir / 'power' / f'{region}_unit_generation.csv')
        agg = pipeline.calculator.aggregate_by_fuel(raw)
        fuels = [c for c in agg if c in pipeline.calculator.emission_factors and c not in ('BESS', 'PHS')]
        generation = agg[fuels].clip(lower=0).fillna(0)
        total = generation.sum(axis=1)
        emissions = generation.mul(pd.Series(pipeline.calculator.emission_factors)).sum(axis=1)
        aef = emissions / total
        if region not in MAINLAND_REGIONS:
            aef = aef.fillna(aef[total > 1e-12].median())
        base_error = max(base_error, _match(total.where(total > 0), expected['Total_Gen (MWh)']),
                         _match(emissions.where(total > 0), expected['Total_Emis']))
        base_aef_error = max(base_aef_error, _match(aef, expected['AEF'], atol=1e-12))
        for column in [c for c in expected if c in ('BESS', 'PHS') or c.startswith('F_')]:
            source = agg[column] if column in agg else pd.Series(0.0, index=agg.index)
            source = source.clip(lower=0) if column.startswith('F_') else source.fillna(0)
            base_error = max(base_error, _match(source, expected[column]))
    replayed, pool, ledger = _replay_with_ledger(pipeline, frozen)
    pool_error = max(_match(pool[c], frozen_pool[c]) for c in frozen_pool)
    output_error = 0.0
    regional_summary = {}
    for region, expected in frozen.items():
        # Compare every stored column, including all storage/flow intermediates.
        output_error = max(output_error, *[_match(replayed[region][c], expected[c]) for c in expected])
        values = expected['FLOW_UNIT_FINAL_AEF']
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite final AEF: {region}')
        regional_summary[region] = {'rows': len(values), 'minimum_kgco2e_per_kwh': float(values.min()),
                                   'mean_kgco2e_per_kwh': float(values.mean()),
                                   'maximum_kgco2e_per_kwh': float(values.max())}
    bess = np.stack([frozen[r]['BESS'].fillna(0).to_numpy() for r in MAINLAND_REGIONS], axis=1)
    phs = frozen['central']['PHS'].fillna(0).to_numpy()
    return {
        'scope': 'Staged base-input and storage arithmetic replay; not observed network energy/carbon closure.',
        'rows': len(pool),
        'internal_absolute_tolerance': TOL,
        'unit_convention': {'energy_internal_to_mwh': 1 / 6, 'carbon_internal_to_kgco2e': 1000 / 6,
                            'retained_interval_minutes': 10, 'missing_intervals_integrated': False,
                            'aef_ratio_unit': 'kg CO2e/kWh'},
        'verification': {'maximum_base_quantity_error_internal': base_error,
                         'maximum_base_aef_error_kgco2e_per_kwh': base_aef_error,
                         'maximum_replayed_regional_column_error': output_error,
                         'maximum_replayed_pool_column_error_internal': pool_error,
                         'index_and_missing_masks_match': True},
        'storage': _pool_summary(pool, ledger, bess, phs),
        'regional_aef': regional_summary,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            'src/streetlight/aef/pipeline.py', 'src/streetlight/aef/storage.py',
            'src/streetlight/aef/calculator.py', 'config/config.yaml')},
        'efficiencies': {'bess_charge': pipeline.storage_manager.bess_config.charge_efficiency,
                         'bess_discharge': pipeline.storage_manager.bess_config.discharge_efficiency,
                         'phs_charge': pipeline.storage_manager.phs_config.charge_efficiency,
                         'phs_discharge': pipeline.storage_manager.phs_config.discharge_efficiency},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('canonical_inputs_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_aef_storage_audit(args.canonical_inputs_dir), indent=2, allow_nan=False))
