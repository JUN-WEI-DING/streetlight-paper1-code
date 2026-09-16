"""
Streetlight package for Taiwan streetlight and energy analysis.

Active modules cover configuration, PAR processing, AEF calculation, hardware
LCA support, streetlight simulation, and Pareto-style analysis support.
"""

__version__ = "0.1.0"

from streetlight import config, lca, paths

__all__ = [
    "__version__",
    "paths",
    "config",
    "lca",
]
