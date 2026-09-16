"""
AEF (Average Emission Factor) computation module.

This module provides tools for computing average emission factors from
power generation data, considering cross-regional flows and storage systems.
"""

from streetlight.aef.calculator import AEFCalculator, compute_aef_row
from streetlight.aef.storage import StorageConfig, StorageState, StorageManager
from streetlight.aef.flow import FlowCalculator
from streetlight.aef.pipeline import AEFPipeline

__all__ = [
    'AEFCalculator',
    'compute_aef_row',
    'StorageConfig',
    'StorageState',
    'StorageManager',
    'FlowCalculator',
    'AEFPipeline',
]
