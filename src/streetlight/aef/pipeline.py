"""
Main AEF computation pipeline.

This module provides a complete pipeline for computing Average Emission Factors
from power generation data, considering storage effects and cross-regional flows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from streetlight.aef.calculator import AEFCalculator
from streetlight.aef.storage import StorageManager
from streetlight.aef.flow import FlowCalculator
from streetlight.config import get_config
from streetlight.paths import region_generation_file


MAINLAND_REGION_CODES = (
    ('north', 'N'),
    ('central', 'C'),
    ('south', 'S'),
    ('east', 'E'),
)
MAINLAND_REGIONS = tuple(region for region, _ in MAINLAND_REGION_CODES)


class AEFPipeline:
    """
    Complete AEF computation pipeline.

    Orchestrates the entire workflow from loading regional generation data
    to computing final AEF values considering storage and flows.
    """

    EPS = 1e-12

    def __init__(
        self,
        config=None,
        emission_factors: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the AEF pipeline.

        Args:
            config: Configuration object (uses default if None)
            emission_factors: Custom emission factors (uses default if None)
        """
        self.config = config or get_config()
        unit_fuel_map = AEFCalculator.load_unit_fuel_map(
            getattr(self.config, "wind_unit_classification_path", None)
        )
        self.calculator = AEFCalculator(emission_factors, unit_fuel_map=unit_fuel_map)
        self.storage_manager = StorageManager()
        self.flow_calculator = FlowCalculator()

        self.regions = self.config.regions
        region_mapping = self.config.region_mapping
        self.region_codes = [
            (region, region_mapping.get(region, region))
            for region in self.regions
        ]
        self.region_data: Dict[str, pd.DataFrame] = {}

    def v(self, x) -> float:
        """Convert value to float, returning 0.0 for NaN/None."""
        return self.flow_calculator.safe_value(x)

    def safe_aef(self, aef_value) -> float:
        """
        Safely convert AEF value to float, handling None and invalid types.

        Args:
            aef_value: Value to convert (can be None, str, or any type)

        Returns:
            Float value, or np.nan if conversion fails
        """
        if aef_value is None:
            return np.nan
        try:
            return float(aef_value)
        except (TypeError, ValueError):
            return np.nan

    def load_region_data(
        self,
        data_dir: Optional[Path] = None,
        scale: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Load generation data for all regions.

        Args:
            data_dir: Directory containing regional CSV files
            scale: Scaling factor for generation values (default: 1.0, use 1/6 for 10-min to hourly)

        Returns:
            Dictionary mapping region names to DataFrames
        """
        if data_dir is None:
            data_dir = Path(self.config.power_dir)

        region_data = {}

        for region in self.regions:
            file_path = data_dir / f"{region}_unit_generation.csv"

            if file_path.exists():
                # Read CSV with proper datetime parsing
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

                # Apply scaling factor
                df = df * scale

                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Aggregate by fuel type
                df = self.calculator.aggregate_by_fuel(df)

                # Handle transfer columns separately
                transfer_cols = [col for col in df.columns
                               if col.startswith('F_') or col in ['BESS', 'PHS', 'load']]

                # Get non-transfer columns
                gen_cols = [col for col in df.columns if col not in transfer_cols]

                # Compute AEF for generation columns
                gen_df = df[gen_cols].copy()
                aef_result = self.calculator.compute_aef_dataframe(gen_df)

                # Transfer columns
                transfer_df = df[transfer_cols].copy() if transfer_cols else pd.DataFrame(index=df.index)

                # Clip flows to non-negative
                for col in transfer_df.columns:
                    if col.startswith('F_'):
                        transfer_df[col] = transfer_df[col].clip(lower=0)

                # Combine results
                region_df = pd.concat([aef_result, transfer_df], axis=1)
                region_df['BESS'] = region_df['BESS'].fillna(0) if 'BESS' in region_df.columns else 0
                region_df['PHS'] = region_df['PHS'].fillna(0) if 'PHS' in region_df.columns else 0

                region_data[region] = region_df
            else:
                print(f"Warning: File not found: {file_path}")

        self.region_data = region_data
        return region_data

    def set_region_data(self, region_data: Dict[str, pd.DataFrame]) -> None:
        """
        Set region data directly.

        Args:
            region_data: Dictionary mapping region names to DataFrames
        """
        self.region_data = region_data

    def compute_base_mix_aef(
        self,
        timestamp: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Compute base AEF without storage effects, considering cross-regional flows.

        Args:
            timestamp: Timestamp to compute for

        Returns:
            Dictionary mapping region codes to base AEF values
        """
        return self.flow_calculator.compute_base_mix_aef(timestamp, self.region_data)

    def compute_unit_src_aef(
        self,
        region_aef: float,
        gen_mwh: float,
        bess_mwh: float,
        bess_aef_grid: Optional[float]
    ) -> float:
        """
        Compute unit source AEF (generation + local BESS discharge).

        Args:
            region_aef: Region's own AEF (kg CO2e/kWh)
            gen_mwh: Generation (MWh)
            bess_mwh: BESS discharge to grid (MWh)
            bess_aef_grid: AEF of BESS discharge (kg CO2e/kWh)

        Returns:
            Unit source AEF (kg CO2e/kWh)
        """
        base = region_aef * gen_mwh
        bess_part = 0.0
        if bess_aef_grid is not None and not np.isnan(bess_aef_grid):
            bess_part = bess_aef_grid * bess_mwh

        denominator = gen_mwh + bess_mwh
        if denominator > self.EPS:
            return (base + bess_part) / denominator
        return region_aef

    def run(
        self,
        region_data: Optional[Dict[str, pd.DataFrame]] = None,
        data_dir: Optional[Path] = None,
        scale: float = 1.0,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Run the complete AEF pipeline.

        Args:
            region_data: Pre-loaded region data (loads from files if None)
            data_dir: Directory for loading data (used if region_data is None)
            scale: Scaling factor for generation values

        Returns:
            Tuple of (region_data_dict, pool_df)
                - region_data_dict: Dictionary of regional DataFrames with AEF results
                - pool_df: DataFrame with storage pool states over time
        """
        # Convert data_dir to Path if it's a string
        if data_dir is not None and isinstance(data_dir, str):
            data_dir = Path(data_dir)
        """
        Run the complete AEF pipeline.

        Args:
            region_data: Pre-loaded region data (loads from files if None)
            data_dir: Directory for loading data (used if region_data is None)
            scale: Scaling factor for generation values

        Returns:
            Tuple of (region_data_dict, pool_df)
                - region_data_dict: Dictionary of regional DataFrames with AEF results
                - pool_df: DataFrame with storage pool states over time
        """
        if region_data is not None:
            self.set_region_data(region_data)
        else:
            self.load_region_data(data_dir, scale)

        # Reset storage states
        self.storage_manager.reset()

        # Get reference index - use the intersection of all regions
        all_indices = [df.index for df in self.region_data.values() if not df.empty]
        if not all_indices:
            raise ValueError("No region data available")
        ref_df_index = all_indices[0]
        for idx in all_indices[1:]:
            ref_df_index = ref_df_index.intersection(idx)
        central_df = self.region_data.get('central')
        if central_df is not None and not central_df.empty:
            ref_df = central_df
        else:
            ref_df = next(iter(self.region_data.values()))

        # Storage efficiencies are baked into StorageManager.charge_*/discharge_*
        # Both PHS and BESS share the same grid_pool_state.

        common_index = ref_df_index

        # ----- Vectorized prefetch: pull every read-only column we need -----
        # into numpy arrays aligned to common_index. Subsequent loop accesses
        # each scalar via integer position rather than label-based .at lookup,
        # eliminating ~1.5M pandas indexing calls per full-year run.
        n = len(common_index)

        def _fetch(region: str, col: str) -> np.ndarray:
            if region in self.region_data and col in self.region_data[region].columns:
                arr = self.region_data[region][col].reindex(common_index).to_numpy(dtype=float)
                return np.nan_to_num(arr, nan=0.0)
            return np.zeros(n)

        IN = {
            r: {
                'AEF': _fetch(r, 'AEF'),
                'Total_Gen (MWh)': _fetch(r, 'Total_Gen (MWh)'),
                'BESS': _fetch(r, 'BESS'),
                'PHS': _fetch(r, 'PHS'),
            }
            for r in self.region_data
        }
        if 'north' in IN:
            IN['north']['F_CN'] = _fetch('north', 'F_CN')
        if 'south' in IN:
            IN['south']['F_CS'] = _fetch('south', 'F_CS')
        if 'east' in IN:
            IN['east']['F_CE'] = _fetch('east', 'F_CE')
        if 'central' in IN:
            for fc in ('F_NC', 'F_SC', 'F_EC'):
                IN['central'][fc] = _fetch('central', fc)

        # ----- Vectorized precompute: base_mix_aef for all timesteps at once -----
        # Independent of storage state; depends only on regional AEF + flows.
        # Used by both _case_both_charge and _case_phs_charge_bess_discharge.
        eps = self.EPS
        base_arr: Dict[str, np.ndarray] = {}
        if 'north' in IN:
            num = IN['north']['AEF'] * IN['north']['Total_Gen (MWh)'] + \
                  IN.get('central', {}).get('AEF', np.zeros(n)) * IN['north'].get('F_CN', np.zeros(n))
            den = IN['north']['Total_Gen (MWh)'] + IN['north'].get('F_CN', np.zeros(n))
            base_arr['N'] = np.where(den > eps, num / np.where(den > eps, den, 1.0), 0.0)
        if 'central' in IN:
            num = IN['central']['AEF'] * IN['central']['Total_Gen (MWh)']
            den = IN['central']['Total_Gen (MWh)'].copy()
            for fc, src_region in (('F_NC', 'north'), ('F_SC', 'south'), ('F_EC', 'east')):
                f_arr = IN['central'].get(fc, np.zeros(n))
                src_aef = IN.get(src_region, {}).get('AEF', np.zeros(n))
                num = num + src_aef * f_arr
                den = den + f_arr
            base_arr['C'] = np.where(den > eps, num / np.where(den > eps, den, 1.0), 0.0)
        if 'south' in IN:
            num = IN['south']['AEF'] * IN['south']['Total_Gen (MWh)'] + \
                  IN.get('central', {}).get('AEF', np.zeros(n)) * IN['south'].get('F_CS', np.zeros(n))
            den = IN['south']['Total_Gen (MWh)'] + IN['south'].get('F_CS', np.zeros(n))
            base_arr['S'] = np.where(den > eps, num / np.where(den > eps, den, 1.0), 0.0)
        if 'east' in IN:
            num = IN['east']['AEF'] * IN['east']['Total_Gen (MWh)'] + \
                  IN.get('central', {}).get('AEF', np.zeros(n)) * IN['east'].get('F_CE', np.zeros(n))
            den = IN['east']['Total_Gen (MWh)'] + IN['east'].get('F_CE', np.zeros(n))
            base_arr['E'] = np.where(den > eps, num / np.where(den > eps, den, 1.0), 0.0)
        region_code_map = dict(self.region_codes)
        for region in IN:
            if region not in MAINLAND_REGIONS:
                base_arr[region_code_map.get(region, region)] = IN[region]['AEF']

        # Pre-allocate output arrays for write-only columns
        out_cols = ('BESS_AEF', 'PHS_AEF', 'UNIT_SRC_AEF', 'FLOW_UNIT_FINAL_AEF',
                    'Flow_AEF_base', 'Flow_UNIT_AEF_base', 'Flow_AEF',
                    'Flow_N_AEF', 'Flow_S_AEF', 'Flow_E_AEF',
                    'Flow_N_AEF_base', 'Flow_S_AEF_base', 'Flow_E_AEF_base')
        OUT: Dict[str, Dict[str, np.ndarray]] = {}
        for r in self.region_data:
            OUT[r] = {col: np.full(n, np.nan) for col in out_cols}

        # Stash on instance for case-method access (replaces .at[t, col] reads/writes)
        self._loop_in = IN
        self._loop_out = OUT
        self._loop_index_lookup = {ts: i for i, ts in enumerate(common_index)}
        self._loop_base = base_arr

        # ----- Main loop (array-only; no pandas indexing in hot path) -----
        for i in range(n):
            bess_schedule = {
                'north': IN['north']['BESS'][i] if 'north' in IN else 0.0,
                'central': IN['central']['BESS'][i] if 'central' in IN else 0.0,
                'south': IN['south']['BESS'][i] if 'south' in IN else 0.0,
                'east': IN['east']['BESS'][i] if 'east' in IN else 0.0,
            }
            phs_schedule = IN['central']['PHS'][i] if 'central' in IN else 0.0

            sum_bess = sum(bess_schedule.values())
            bess_discharge = sum_bess >= 0
            bess_charge = sum_bess < 0
            phs_discharge = phs_schedule >= 0
            phs_charge = phs_schedule < 0

            phs_deficit_step = 0.0
            bess_deficit_step = 0.0
            phs_aef_step = None
            bess_aef_step = None

            if phs_charge and bess_charge:
                self._case_both_charge_arr(i, phs_schedule)
            elif phs_charge and bess_discharge:
                bess_aef_step, bess_deficit_step = self._case_phs_charge_bess_discharge_arr(
                    i, bess_schedule, phs_schedule
                )
            elif phs_discharge and bess_charge:
                phs_aef_step, phs_deficit_step = self._case_phs_discharge_bess_charge_arr(
                    i, bess_schedule, phs_schedule
                )
            else:  # both discharge
                phs_aef_step, bess_aef_step, phs_deficit_step, bess_deficit_step = (
                    self._case_both_discharge_arr(i, bess_schedule, phs_schedule)
                )

            self.storage_manager.grid_pool_state.record_state(
                phs_deficit=phs_deficit_step,
                bess_deficit=bess_deficit_step,
                phs_discharge_aef=phs_aef_step,
                bess_discharge_aef=bess_aef_step,
            )

        # ----- Attach output arrays back to DataFrames as columns -----
        for region, df in self.region_data.items():
            for col, arr in OUT[region].items():
                df_aligned = pd.Series(arr, index=common_index)
                df[col] = df_aligned.reindex(df.index)

        # For regions without flow connections, use local AEF for
        # timesteps that no storage case explicitly wrote.
        for region, df in self.region_data.items():
            if region not in MAINLAND_REGIONS:
                local_aef = pd.to_numeric(df['AEF'], errors='coerce')
                total_gen = pd.to_numeric(
                    df.get('Total_Gen (MWh)', pd.Series(np.nan, index=df.index)),
                    errors='coerce',
                )
                local_aef = local_aef.mask(total_gen <= self.EPS)
                valid_local_aef = local_aef.dropna()
                fallback_aef = (
                    float(valid_local_aef.median())
                    if not valid_local_aef.empty
                    else 0.0
                )
                local_aef = local_aef.fillna(fallback_aef)
                df['AEF'] = local_aef
                for col in ('FLOW_UNIT_FINAL_AEF', 'Flow_AEF_base',
                            'Flow_UNIT_AEF_base', 'UNIT_SRC_AEF'):
                    df[col] = local_aef

        # Clear loop scratchpad
        self._loop_in = self._loop_out = self._loop_index_lookup = self._loop_base = None

        # Create pool DataFrame
        pool_df = self.storage_manager.get_pool_dataframe(ref_df.index)

        return self.region_data, pool_df

    # ----- Array-based case methods (used by run()'s main loop) -----

    def _write_charge_only_base_outputs_arr(self, i: int) -> None:
        """Write grid-serving AEF for storage charge-only timesteps."""
        IN, OUT, base = self._loop_in, self._loop_out, self._loop_base

        for r, code in self.region_codes:
            if r not in IN or r not in OUT:
                continue
            final_aef = base[code][i] if code in base else IN[r]['AEF'][i]
            OUT[r]['UNIT_SRC_AEF'][i] = IN[r]['AEF'][i]
            OUT[r]['FLOW_UNIT_FINAL_AEF'][i] = final_aef
            OUT[r]['Flow_UNIT_AEF_base'][i] = final_aef

        if 'central' in IN:
            central_aef = IN['central']['AEF'][i]
            for r in ('north', 'south', 'east'):
                if r in OUT:
                    OUT[r]['Flow_AEF'][i] = central_aef
                    OUT[r]['Flow_AEF_base'][i] = central_aef

        if 'central' in OUT:
            for col, src_region in (
                ('Flow_N_AEF', 'north'),
                ('Flow_S_AEF', 'south'),
                ('Flow_E_AEF', 'east'),
            ):
                if src_region in IN:
                    OUT['central'][col][i] = IN[src_region]['AEF'][i]
                    OUT['central'][f'{col}_base'][i] = IN[src_region]['AEF'][i]

    def _case_both_charge_arr(self, i, phs_schedule):
        """PHS + BESS both charging: both feed the shared pool with base AEF.

        Storage is consuming rather than producing, so storage discharge AEF
        remains undefined; the grid-serving regional AEF remains the base
        generation-plus-flow mix and must stay available to downstream annual
        simulations.
        """
        IN, base = self._loop_in, self._loop_base
        for r, code in MAINLAND_REGION_CODES:
            if r in IN:
                b = IN[r]['BESS'][i]
                if b < 0:
                    self.storage_manager.charge_bess(-b, base[code][i])
        if phs_schedule < 0 and 'central' in IN:
            self.storage_manager.charge_phs(-phs_schedule, base['C'][i])
        self._write_charge_only_base_outputs_arr(i)

    def _case_phs_charge_bess_discharge_arr(self, i, bess_schedule, phs_schedule):
        """PHS charging, BESS discharging."""
        IN, OUT = self._loop_in, self._loop_out
        eps = self.EPS

        wish_bess = sum(max(v, 0) for v in bess_schedule.values())
        deliver_bess, bess_aef_to_grid, deficit = self.storage_manager.discharge_bess(wish_bess)
        scale = (deliver_bess / wish_bess) if (deliver_bess > 0 and wish_bess > 0) else 0.0
        eff_b = {r: max(bess_schedule[r], 0) * scale for r in bess_schedule}

        def _src(r):
            G = IN[r]['Total_Gen (MWh)'][i]
            base = IN[r]['AEF'][i] * G
            bess_part = (bess_aef_to_grid * eff_b[r]
                         if bess_aef_to_grid is not None
                         and not np.isnan(self.safe_aef(bess_aef_to_grid))
                         else 0.0)
            den = G + eff_b[r]
            return (base + bess_part) / den if den > eps else IN[r]['AEF'][i]

        UN = _src('north') if 'north' in IN else 0.0
        UC = _src('central') if 'central' in IN else 0.0
        US = _src('south') if 'south' in IN else 0.0
        UE = _src('east') if 'east' in IN else 0.0

        if 'north' in OUT: OUT['north']['Flow_AEF'][i] = UC
        if 'south' in OUT: OUT['south']['Flow_AEF'][i] = UC
        if 'east' in OUT: OUT['east']['Flow_AEF'][i] = UC
        if 'central' in OUT:
            OUT['central']['Flow_N_AEF'][i] = UN
            OUT['central']['Flow_S_AEF'][i] = US
            OUT['central']['Flow_E_AEF'][i] = UE

        # PHS charge with central AEF after flow
        if 'central' in IN:
            Gc = IN['central']['Total_Gen (MWh)'][i]
            f_nc = IN['central'].get('F_NC', np.zeros(0))[i] if 'F_NC' in IN['central'] else 0.0
            f_sc = IN['central'].get('F_SC', np.zeros(0))[i] if 'F_SC' in IN['central'] else 0.0
            f_ec = IN['central'].get('F_EC', np.zeros(0))[i] if 'F_EC' in IN['central'] else 0.0
            flow_num = UN * f_nc + US * f_sc + UE * f_ec
            flow_den = f_nc + f_sc + f_ec
            den = Gc + flow_den
            C_after_flow = (IN['central']['AEF'][i] * Gc + flow_num) / den if den > eps else IN['central']['AEF'][i]
            if phs_schedule < 0:
                self.storage_manager.charge_phs(-phs_schedule, C_after_flow)

        for r, U in (('north', UN), ('south', US), ('east', UE), ('central', UC)):
            if r in OUT:
                OUT[r]['UNIT_SRC_AEF'][i] = U
                OUT[r]['BESS_AEF'][i] = bess_aef_to_grid if bess_aef_to_grid is not None else np.nan

        # FLOW_UNIT_FINAL_AEF for each region
        flow_aef_in = {'north': IN['north'].get('F_CN', np.zeros(0))[i] if 'F_CN' in IN.get('north', {}) else 0.0,
                       'south': IN['south'].get('F_CS', np.zeros(0))[i] if 'F_CS' in IN.get('south', {}) else 0.0,
                       'east': IN['east'].get('F_CE', np.zeros(0))[i] if 'F_CE' in IN.get('east', {}) else 0.0}
        for r in ('north', 'south', 'east'):
            if r in IN:
                G = IN[r]['Total_Gen (MWh)'][i]
                fmwh = flow_aef_in[r]
                bess_part = (bess_aef_to_grid * eff_b[r]
                             if bess_aef_to_grid is not None
                             and not np.isnan(self.safe_aef(bess_aef_to_grid))
                             else 0.0)
                num = IN[r]['AEF'][i] * G + UC * fmwh + bess_part
                den = G + fmwh + eff_b[r]
                OUT[r]['FLOW_UNIT_FINAL_AEF'][i] = num / den if den > eps else IN[r]['AEF'][i]

        if 'central' in IN:
            Gc = IN['central']['Total_Gen (MWh)'][i]
            f_nc = IN['central'].get('F_NC', np.zeros(0))[i] if 'F_NC' in IN['central'] else 0.0
            f_sc = IN['central'].get('F_SC', np.zeros(0))[i] if 'F_SC' in IN['central'] else 0.0
            f_ec = IN['central'].get('F_EC', np.zeros(0))[i] if 'F_EC' in IN['central'] else 0.0
            bess_central = bess_aef_to_grid * eff_b['central'] if bess_aef_to_grid is not None and not np.isnan(self.safe_aef(bess_aef_to_grid)) else 0.0
            num = IN['central']['AEF'][i] * Gc + bess_central + UN * f_nc + US * f_sc + UE * f_ec
            den = Gc + eff_b['central'] + f_nc + f_sc + f_ec
            OUT['central']['FLOW_UNIT_FINAL_AEF'][i] = num / den if den > eps else IN['central']['AEF'][i]

        return bess_aef_to_grid, deficit

    def _case_phs_discharge_bess_charge_arr(self, i, bess_schedule, phs_schedule):
        """PHS discharging, BESS charging."""
        IN, OUT = self._loop_in, self._loop_out
        eps = self.EPS

        wish_phs = max(phs_schedule, 0) if 'central' in IN else 0
        deliver_phs, phs_aef_to_grid, phs_deficit = self.storage_manager.discharge_phs(wish_phs)

        if 'central' in IN:
            Gc = IN['central']['Total_Gen (MWh)'][i]
            phs_part = (phs_aef_to_grid * deliver_phs
                        if phs_aef_to_grid is not None
                        and not np.isnan(self.safe_aef(phs_aef_to_grid))
                        else 0.0)
            den = Gc + deliver_phs
            UC = (IN['central']['AEF'][i] * Gc + phs_part) / den if den > eps else IN['central']['AEF'][i]
        else:
            UC = 0.0
        UN = IN['north']['AEF'][i] if 'north' in IN else 0.0
        US = IN['south']['AEF'][i] if 'south' in IN else 0.0
        UE = IN['east']['AEF'][i] if 'east' in IN else 0.0

        if 'north' in OUT: OUT['north']['Flow_AEF'][i] = UC
        if 'south' in OUT: OUT['south']['Flow_AEF'][i] = UC
        if 'east' in OUT: OUT['east']['Flow_AEF'][i] = UC
        if 'central' in OUT:
            OUT['central']['Flow_N_AEF'][i] = UN
            OUT['central']['Flow_S_AEF'][i] = US
            OUT['central']['Flow_E_AEF'][i] = UE

        # BESS charging — use and retain post-flow AEF per region
        for r, code in MAINLAND_REGION_CODES:
            if r in IN:
                G = IN[r]['Total_Gen (MWh)'][i]
                if r == 'central':
                    f_nc = IN['central'].get('F_NC', np.zeros(0))[i] if 'F_NC' in IN['central'] else 0.0
                    f_sc = IN['central'].get('F_SC', np.zeros(0))[i] if 'F_SC' in IN['central'] else 0.0
                    f_ec = IN['central'].get('F_EC', np.zeros(0))[i] if 'F_EC' in IN['central'] else 0.0
                    flow_aef = ((UN * f_nc + US * f_sc + UE * f_ec) /
                                (f_nc + f_sc + f_ec)) if (f_nc + f_sc + f_ec) > eps else 0.0
                    flow_mwh = f_nc + f_sc + f_ec
                    unit_aef = UC
                    unit_mwh = G + deliver_phs
                else:
                    flow_aef = UC
                    if r == 'north':
                        flow_mwh = IN['north'].get('F_CN', np.zeros(0))[i] if 'F_CN' in IN['north'] else 0.0
                    elif r == 'south':
                        flow_mwh = IN['south'].get('F_CS', np.zeros(0))[i] if 'F_CS' in IN['south'] else 0.0
                    elif r == 'east':
                        flow_mwh = IN['east'].get('F_CE', np.zeros(0))[i] if 'F_CE' in IN['east'] else 0.0
                    unit_aef = IN[r]['AEF'][i]
                    unit_mwh = G
                base_after_flow = (unit_aef * unit_mwh + flow_aef * flow_mwh) / max(unit_mwh + flow_mwh, eps)
                OUT[r]['FLOW_UNIT_FINAL_AEF'][i] = base_after_flow
                OUT[r]['Flow_UNIT_AEF_base'][i] = base_after_flow
                b = IN[r]['BESS'][i]
                if b < 0:
                    self.storage_manager.charge_bess(-b, base_after_flow)

        for r, U in (('north', UN), ('south', US), ('east', UE), ('central', UC)):
            if r in OUT:
                OUT[r]['UNIT_SRC_AEF'][i] = U
                OUT[r]['BESS_AEF'][i] = np.nan
        if 'central' in OUT:
            OUT['central']['PHS_AEF'][i] = phs_aef_to_grid if phs_aef_to_grid is not None else np.nan

        return phs_aef_to_grid, phs_deficit

    def _case_both_discharge_arr(self, i, bess_schedule, phs_schedule):
        """PHS first, then BESS — both draw from the shared pool."""
        IN, OUT = self._loop_in, self._loop_out
        eps = self.EPS

        wish_phs = max(phs_schedule, 0) if 'central' in IN else 0
        deliver_phs, phs_aef_to_grid, phs_deficit = self.storage_manager.discharge_phs(wish_phs)
        phs_aef_f = float(phs_aef_to_grid) if phs_aef_to_grid is not None else np.nan

        wish_bess = sum(max(v, 0) for v in bess_schedule.values())
        deliver_bess, bess_aef_to_grid, bess_deficit = self.storage_manager.discharge_bess(wish_bess)
        bess_aef_f = float(bess_aef_to_grid) if bess_aef_to_grid is not None else np.nan

        scale = (deliver_bess / wish_bess) if (deliver_bess > 0 and wish_bess > 0) else 0.0
        eff_b = {r: max(bess_schedule[r], 0) * scale for r in bess_schedule}

        def _src_full(r, is_central):
            G = IN[r]['Total_Gen (MWh)'][i]
            base = IN[r]['AEF'][i] * G
            bess_part = bess_aef_f * eff_b[r] if not np.isnan(bess_aef_f) else 0.0
            if is_central:
                phs_part = phs_aef_f * deliver_phs if not np.isnan(phs_aef_f) else 0.0
                num = base + bess_part + phs_part
                den = G + eff_b[r] + deliver_phs
            else:
                num = base + bess_part
                den = G + eff_b[r]
            return num / den if den > eps else IN[r]['AEF'][i]

        UN = _src_full('north', False) if 'north' in IN else 0.0
        UC = _src_full('central', True) if 'central' in IN else 0.0
        US = _src_full('south', False) if 'south' in IN else 0.0
        UE = _src_full('east', False) if 'east' in IN else 0.0

        if 'north' in OUT: OUT['north']['Flow_AEF'][i] = UC
        if 'south' in OUT: OUT['south']['Flow_AEF'][i] = UC
        if 'east' in OUT: OUT['east']['Flow_AEF'][i] = UC
        if 'central' in OUT:
            OUT['central']['Flow_N_AEF'][i] = UN
            OUT['central']['Flow_S_AEF'][i] = US
            OUT['central']['Flow_E_AEF'][i] = UE

        for r, U in (('north', UN), ('south', US), ('east', UE), ('central', UC)):
            if r in OUT:
                OUT[r]['UNIT_SRC_AEF'][i] = U
                OUT[r]['BESS_AEF'][i] = bess_aef_f
        if 'central' in OUT:
            OUT['central']['PHS_AEF'][i] = phs_aef_f

        # FLOW_UNIT_FINAL_AEF
        for r in ('north', 'south', 'east'):
            if r in IN:
                G = IN[r]['Total_Gen (MWh)'][i]
                if r == 'north':
                    fmwh = IN['north'].get('F_CN', np.zeros(0))[i] if 'F_CN' in IN['north'] else 0.0
                elif r == 'south':
                    fmwh = IN['south'].get('F_CS', np.zeros(0))[i] if 'F_CS' in IN['south'] else 0.0
                else:
                    fmwh = IN['east'].get('F_CE', np.zeros(0))[i] if 'F_CE' in IN['east'] else 0.0
                bess_part = bess_aef_f * eff_b[r] if not np.isnan(bess_aef_f) else 0.0
                num = IN[r]['AEF'][i] * G + UC * fmwh + bess_part
                den = G + fmwh + eff_b[r]
                OUT[r]['FLOW_UNIT_FINAL_AEF'][i] = num / den if den > eps else IN[r]['AEF'][i]

        if 'central' in IN:
            Gc = IN['central']['Total_Gen (MWh)'][i]
            f_nc = IN['central'].get('F_NC', np.zeros(0))[i] if 'F_NC' in IN['central'] else 0.0
            f_sc = IN['central'].get('F_SC', np.zeros(0))[i] if 'F_SC' in IN['central'] else 0.0
            f_ec = IN['central'].get('F_EC', np.zeros(0))[i] if 'F_EC' in IN['central'] else 0.0
            bess_central = bess_aef_f * eff_b['central'] if not np.isnan(bess_aef_f) else 0.0
            phs_central = phs_aef_f * deliver_phs if not np.isnan(phs_aef_f) else 0.0
            num = (IN['central']['AEF'][i] * Gc + bess_central + phs_central +
                   UN * f_nc + US * f_sc + UE * f_ec)
            den = Gc + eff_b['central'] + f_nc + f_sc + f_ec + deliver_phs
            OUT['central']['FLOW_UNIT_FINAL_AEF'][i] = num / den if den > eps else IN['central']['AEF'][i]

        return phs_aef_f, bess_aef_f, phs_deficit, bess_deficit


    def save_results(
        self,
        output_dir: Path,
        pool_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save AEF results to CSV files.

        Args:
            output_dir: Directory to save results
            pool_df: Storage pool DataFrame to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for region, df in self.region_data.items():
            file_path = output_dir / f"{region}.csv"
            df.to_csv(file_path)

        if pool_df is not None:
            pool_df.to_csv(output_dir / "storage_pools.csv")
