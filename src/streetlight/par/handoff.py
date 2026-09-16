"""
Canonical PAR handoff utilities for simulation and paper workflows.

This module converts county-level long-format PAR aggregation outputs into the
city/county wide table used by simulation, Pareto analysis, and the paper
baseline runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_par_aggregated(path: Path) -> pd.DataFrame:
    """Load long-format aggregated PAR output from CSV or parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def par_aggregated_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format county PAR aggregation into the canonical wide table.

    Expected minimum columns:
    - `time_utc`
    - `county`
    - one PAR value column, typically `mean_PAR_umol_m2_s`
    """
    required = {"time_utc", "county"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required PAR aggregation columns: {', '.join(missing)}")

    value_col = next(
        (
            c
            for c in (
                "mean_PAR_umol_m2_s",
                "mean_PAR",
                "PAR",
                "par",
            )
            if c in df.columns
        ),
        None,
    )
    if value_col is None:
        raise ValueError(
            "Cannot determine PAR value column. Expected one of: "
            "mean_PAR_umol_m2_s, mean_PAR, PAR, par"
        )

    wide = df[["time_utc", "county", value_col]].copy()
    wide["time_utc"] = pd.to_datetime(wide["time_utc"], errors="coerce")
    if wide["time_utc"].isna().any():
        raise ValueError("PAR aggregation contains non-parseable time_utc values")
    duplicate_mask = wide.duplicated(subset=["time_utc", "county"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            wide.loc[duplicate_mask, ["time_utc", "county"]]
            .sort_values(["time_utc", "county"])
            .drop_duplicates()
        )
        sample = ", ".join(
            f"({row.time_utc}, {row.county})"
            for row in duplicate_rows.head(5).itertuples(index=False)
        )
        raise ValueError(
            "PAR aggregation contains duplicate time_utc/county pairs; "
            "canonical handoff requires uniqueness. "
            f"Sample duplicates: {sample}"
        )

    wide = (
        wide.pivot(
            index="time_utc",
            columns="county",
            values=value_col,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    wide.columns.name = None
    return wide


def save_par_wide(df: pd.DataFrame, output_path: Path) -> None:
    """Save the canonical wide table to CSV or parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        df.to_parquet(output_path)
    else:
        df.to_csv(output_path, index=True, index_label="time_utc")


def interpolate_short_gaps(
    wide_df: pd.DataFrame,
    *,
    max_run: int = 6,
    freq: str = "10min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-interpolate short gaps in a wide PAR table and return imputation flags.

    Only internal NaN runs with length <= max_run are filled. Longer runs remain
    missing. Returned flags have the same shape as the value table and are True
    only where this function filled a previously missing value.
    """
    if not isinstance(wide_df.index, pd.DatetimeIndex):
        df = wide_df.copy()
        df.index = pd.to_datetime(df.index)
    else:
        df = wide_df.copy()

    df = df.sort_index()
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(expected)
    before_missing = df.isna()

    filled = df.interpolate(
        method="time",
        axis=0,
        limit=max_run,
        limit_direction="both",
        limit_area="inside",
    )

    # Revert runs longer than max_run because pandas' limit can partially fill
    # long runs depending on direction. This keeps the policy binary.
    for col in filled.columns:
        missing = before_missing[col]
        if not missing.any():
            continue
        group_id = (missing != missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(group_id).transform("sum")
        long_missing = missing & (run_lengths > max_run)
        filled.loc[long_missing, col] = pd.NA

    flags = before_missing & filled.notna()
    filled.index.name = wide_df.index.name or "time_utc"
    flags.index.name = filled.index.name
    return filled, flags


def prepare_par_for_simulation(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Build the canonical simulation handoff artifact from aggregated PAR output.

    Returns the wide DataFrame after saving it.
    """
    long_df = load_par_aggregated(input_path)
    wide_df = par_aggregated_to_wide(long_df)
    save_par_wide(wide_df, output_path)
    return wide_df
