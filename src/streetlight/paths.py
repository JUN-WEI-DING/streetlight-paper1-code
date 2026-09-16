"""
Path resolution module for streetlight package.

Provides centralized path management that works across different machines
by resolving paths relative to the project root.

Environment Variable Overrides:
    STREETLIGHT_ROOT: Override project root directory
    STREETLIGHT_DATA_DIR: Override data directory
    STREETLIGHT_CONFIG_DIR: Override config directory
    STREETLIGHT_OUTPUTS_DIR: Override outputs directory
"""

import os
from pathlib import Path
from typing import Optional


def _find_repo_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the repository root by searching for .git directory.

    Args:
        start_path: Starting path for search (defaults to current file's location)

    Returns:
        Path to repository root
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    # Walk up the directory tree
    current = start_path
    while current != current.parent:  # Stop at filesystem root
        if (current / ".git").exists():
            return current
        current = current.parent

    # Fallback: assume src/ streetlight/ layout
    # If we're in src/streetlight/paths.py, go up two levels to src/, then one more to root
    path = Path(__file__).resolve()
    if path.parent.name == "streetlight" and path.parent.parent.name == "src":
        return path.parent.parent.parent

    # Last resort: use current working directory
    return Path.cwd()


def project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    env_override = os.getenv("STREETLIGHT_ROOT")
    if env_override:
        return Path(env_override)
    return _find_repo_root()


def src_dir() -> Path:
    """
    Get the src directory.

    Returns:
        Path to src directory
    """
    root = project_root()
    src = root / "src"
    return src if src.exists() else root


def data_dir() -> Path:
    """
    Get the data directory.

    Returns:
        Path to data directory
    """
    env_override = os.getenv("STREETLIGHT_DATA_DIR")
    if env_override:
        return Path(env_override)
    return project_root() / "data"


def config_dir() -> Path:
    """
    Get the config directory.

    Returns:
        Path to config directory
    """
    env_override = os.getenv("STREETLIGHT_CONFIG_DIR")
    if env_override:
        return Path(env_override)
    return project_root() / "config"


def outputs_dir() -> Path:
    """
    Get the outputs directory.

    Returns:
        Path to outputs directory
    """
    env_override = os.getenv("STREETLIGHT_OUTPUTS_DIR")
    if env_override:
        return Path(env_override)
    return project_root() / "outputs"


def scripts_dir() -> Path:
    """
    Get the scripts directory.

    Returns:
        Path to scripts directory
    """
    return project_root() / "scripts"


def logs_dir() -> Path:
    """
    Get the logs directory.

    Returns:
        Path to logs directory
    """
    return project_root() / "logs"


def power_data_dir() -> Path:
    """Get power data directory."""
    return data_dir() / "power"


def county_shapefile_path() -> Path:
    """
    Get path to Taiwan county shapefile.

    Returns:
        Path to the repo-local county boundary shapefile.
    """
    return data_dir() / "geo" / "county_boundaries.shp"


def streetlight_list_dir() -> Path:
    """Get streetlight list configuration directory."""
    return config_dir() / "streetlight_list"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The same path for convenience
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# Pipeline-specific paths


def aef_output_dir() -> Path:
    """Get AEF output directory."""
    return outputs_dir() / "AEF"


def aef_file(region: str) -> Path:
    """
    Get path to AEF output file for a specific region.

    Args:
        region: Region name (north, central, south, east, island)

    Returns:
        Path to region's AEF CSV file
    """
    return aef_output_dir() / f"{region.lower()}.csv"


def par_output_dir() -> Path:
    """Get PAR output directory."""
    return outputs_dir() / "par"


def par_long_format_path() -> Path:
    """Get path to canonical long-format PAR aggregation output."""
    return par_output_dir() / "par_aggregated.csv"


def par_wide_format_path() -> Path:
    """Get path to canonical wide-format PAR handoff artifact."""
    return par_output_dir() / "par_wide.csv"


def pareto_output_dir() -> Path:
    """Get Pareto analysis output directory."""
    return outputs_dir() / "pareto"


def city_results_path() -> Path:
    """Get path to city results parquet file."""
    return pareto_output_dir() / "city_results_df.parquet"


def plots_output_dir() -> Path:
    """Get plots output directory."""
    return outputs_dir() / "plots"


# Regional file paths


def region_generation_file(region: str) -> Path:
    """
    Get path to region's generation CSV file.

    Args:
        region: Region name (north, central, south, east, island)

    Returns:
        Path to region's generation CSV file
    """
    return power_data_dir() / f"{region.lower()}_unit_generation.csv"


def flow_file_path() -> Path:
    """Get path to inter-regional flow CSV file."""
    return power_data_dir() / "flow.csv"
