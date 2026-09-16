"""
Cross-regional power flow calculation for AEF.

This module provides functions for computing cross-regional power flows
and their impact on regional Average Emission Factors (AEF).
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class FlowCalculator:
    """
    Calculator for cross-regional power flows.

    Handles power flow between Taiwan's four main-island regions and any
    disconnected outlying-island load-serving zones.
    """

    EPS = 1e-12

    def __init__(self):
        """Initialize the flow calculator."""
        pass

    @staticmethod
    def safe_value(x) -> float:
        """
        Convert value to float, returning 0.0 for NaN/None.

        Args:
            x: Value to convert

        Returns:
            Float value or 0.0 if NaN/None
        """
        try:
            return 0.0 if pd.isna(x) else float(x)
        except Exception:
            return 0.0

    def compute_base_mix_aef(
        self,
        timestamp: pd.Timestamp,
        region_data: Dict[str, pd.DataFrame],
        flow_columns: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Compute base AEF without storage effects, considering cross-regional flows.

        Args:
            timestamp: Timestamp to compute for
            region_data: Dictionary mapping region names to DataFrames
                         Each DataFrame must have 'AEF' and flow columns
            flow_columns: Optional mapping of flow column names

        Returns:
            Dictionary mapping region codes to base AEF values
        """
        if flow_columns is None:
            flow_columns = {
                'F_CN': 'C_to_N',  # Central to North
                'F_NC': 'N_to_C',  # North to Central
                'F_CS': 'C_to_S',  # Central to South
                'F_SC': 'S_to_C',  # South to Central
                'F_CE': 'C_to_E',  # Central to East
                'F_EC': 'E_to_C',  # East to Central
            }

        all_regions = region_data
        mainland_region_names = {'north', 'central', 'south', 'east'}
        regions = {
            k: v for k, v in all_regions.items()
            if k in mainland_region_names
        }
        region_codes = {
            'north': 'N',
            'central': 'C',
            'south': 'S',
            'east': 'E',
            'island': 'I',
            'island_penghu': 'IP',
            'island_kinmen': 'IK',
            'island_lienchiang': 'IL',
        }

        # Get source AEF for each flow direction
        # Central acts as source for flows to N, S, E
        # N, S, E act as sources for flows to Central
        flow_aef = {}

        if 'central' in regions and timestamp in regions['central'].index:
            central_aef = self.safe_value(regions['central'].loc[timestamp, 'AEF'])
            flow_aef['C_to_N'] = central_aef
            flow_aef['C_to_S'] = central_aef
            flow_aef['C_to_E'] = central_aef

        for region in ['north', 'south', 'east']:
            if region in regions and timestamp in regions[region].index:
                aef = self.safe_value(regions[region].loc[timestamp, 'AEF'])
                flow_aef[f'{region_codes[region]}_to_C'] = aef

        result = {}

        # North: self-gen + flow from Central
        if 'north' in regions and timestamp in regions['north'].index:
            df = regions['north']
            gen = self.safe_value(df.loc[timestamp, 'Total_Gen (MWh)'])
            flow_in = self.safe_value(df.loc[timestamp, 'F_CN']) if 'F_CN' in df.columns else 0
            flow_aef_val = flow_aef.get('C_to_N', 0)

            numerator = self.safe_value(df.loc[timestamp, 'AEF']) * gen + flow_aef_val * flow_in
            denominator = gen + flow_in
            result['N'] = numerator / max(denominator, self.EPS)

        # Central: self-gen + flows from N, S, E
        if 'central' in regions and timestamp in regions['central'].index:
            df = regions['central']
            gen = self.safe_value(df.loc[timestamp, 'Total_Gen (MWh)'])
            flow_n = self.safe_value(df.loc[timestamp, 'F_NC']) if 'F_NC' in df.columns else 0
            flow_s = self.safe_value(df.loc[timestamp, 'F_SC']) if 'F_SC' in df.columns else 0
            flow_e = self.safe_value(df.loc[timestamp, 'F_EC']) if 'F_EC' in df.columns else 0

            numerator = (
                self.safe_value(df.loc[timestamp, 'AEF']) * gen +
                flow_aef.get('N_to_C', 0) * flow_n +
                flow_aef.get('S_to_C', 0) * flow_s +
                flow_aef.get('E_to_C', 0) * flow_e
            )
            denominator = gen + flow_n + flow_s + flow_e
            result['C'] = numerator / max(denominator, self.EPS)

        # South: self-gen + flow from Central
        if 'south' in regions and timestamp in regions['south'].index:
            df = regions['south']
            gen = self.safe_value(df.loc[timestamp, 'Total_Gen (MWh)'])
            flow_in = self.safe_value(df.loc[timestamp, 'F_CS']) if 'F_CS' in df.columns else 0
            flow_aef_val = flow_aef.get('C_to_S', 0)

            numerator = self.safe_value(df.loc[timestamp, 'AEF']) * gen + flow_aef_val * flow_in
            denominator = gen + flow_in
            result['S'] = numerator / max(denominator, self.EPS)

        # East: self-gen + flow from Central
        if 'east' in regions and timestamp in regions['east'].index:
            df = regions['east']
            gen = self.safe_value(df.loc[timestamp, 'Total_Gen (MWh)'])
            flow_in = self.safe_value(df.loc[timestamp, 'F_CE']) if 'F_CE' in df.columns else 0
            flow_aef_val = flow_aef.get('C_to_E', 0)

            numerator = self.safe_value(df.loc[timestamp, 'AEF']) * gen + flow_aef_val * flow_in
            denominator = gen + flow_in
            result['E'] = numerator / max(denominator, self.EPS)

        # Disconnected load-serving zones: no regional flows.
        for region, df in all_regions.items():
            if region in mainland_region_names or timestamp not in df.index:
                continue
            result[region_codes.get(region, region)] = self.safe_value(
                df.loc[timestamp, 'AEF']
            )

        return result

    def compute_final_unit_aef(
        self,
        region_df: pd.DataFrame,
        timestamp: pd.Timestamp,
        own_gen: float,
        flow_aef: float,
        flow_mwh: float,
        bess_aef_grid: Optional[float] = None,
        bess_mwh: float = 0.0,
        ess_aef_grid: Optional[float] = None,
        ess_mwh: float = 0.0
    ) -> float:
        """
        Compute final unit AEF considering generation, storage, and flows.

        Args:
            region_df: Region DataFrame with AEF data
            timestamp: Timestamp to compute for
            own_gen: Own generation (MWh)
            flow_aef: AEF of incoming flow (kg CO2e/kWh)
            flow_mwh: Incoming flow energy (MWh)
            bess_aef_grid: AEF of BESS discharge (kg CO2e/kWh)
            bess_mwh: BESS discharge to grid (MWh)
            ess_aef_grid: AEF of ESS discharge (kg CO2e/kWh)
            ess_mwh: ESS discharge to grid (MWh)

        Returns:
            Final unit AEF (kg CO2e/kWh)
        """
        base_aef = self.safe_value(region_df.loc[timestamp, 'AEF'])
        gen = self.safe_value(own_gen)

        bess_part = 0.0
        if bess_aef_grid is not None and not np.isnan(bess_aef_grid):
            bess_part = bess_aef_grid * self.safe_value(bess_mwh)

        ess_part = 0.0
        if ess_aef_grid is not None and not np.isnan(ess_aef_grid):
            ess_part = ess_aef_grid * self.safe_value(ess_mwh)

        flow_part = self.safe_value(flow_aef) * self.safe_value(flow_mwh)

        numerator = base_aef * gen + bess_part + ess_part + flow_part
        denominator = gen + self.safe_value(bess_mwh) + self.safe_value(ess_mwh) + self.safe_value(flow_mwh)

        if denominator > self.EPS:
            return numerator / denominator
        return base_aef

    def compute_regional_flows(
        self,
        region_data: Dict[str, pd.DataFrame],
        config: Optional[Dict] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute cross-regional flows for all regions.

        Args:
            region_data: Dictionary mapping region names to DataFrames
            config: Optional configuration dictionary

        Returns:
            Dictionary mapping region names to DataFrames with flow columns added
        """
        result = {}

        for region, df in region_data.items():
            df_copy = df.copy()

            # Set flow source AEF based on region
            if region == 'north':
                if 'central' in region_data:
                    df_copy['Flow_AEF_base'] = region_data['central']['AEF']
            elif region == 'south':
                if 'central' in region_data:
                    df_copy['Flow_AEF_base'] = region_data['central']['AEF']
            elif region == 'east':
                if 'central' in region_data:
                    df_copy['Flow_AEF_base'] = region_data['central']['AEF']
            elif region == 'central':
                if 'north' in region_data:
                    df_copy['Flow_N_AEF_base'] = region_data['north']['AEF']
                if 'south' in region_data:
                    df_copy['Flow_S_AEF_base'] = region_data['south']['AEF']
                if 'east' in region_data:
                    df_copy['Flow_E_AEF_base'] = region_data['east']['AEF']

            result[region] = df_copy

        return result
