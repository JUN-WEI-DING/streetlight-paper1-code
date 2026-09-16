from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from check_par_weekly_root import check_weekly_root
from run_paper1_final_data_pipeline import EXPECTED_END, EXPECTED_START
from streetlight.processing.power_adapter import (
    _normalize_to_10min_grid,
    _parse_regional_demand_timestamp,
    taipower_local_to_utc_naive,
)


DEMAND_REQUIRED_COLUMNS = {
    "date",
    "time",
    "source",
    "north_gen",
    "north_load",
    "central_gen",
    "central_load",
    "south_gen",
    "south_load",
    "east_gen",
    "east_load",
}
GENERATION_REQUIRED_COLUMNS = {
    "timestamp",
    "plant_name",
    "mapping_names",
    "energy_type",
    "used_mw",
    "capacity_mw",
    "status",
}
DEFAULT_CANONICAL_INPUT_BYTES = 700_000_000


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


def parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def check_path(path: Path, *, suffix: str, name: str) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(name=f"{name}.exists", ok=path.exists(), detail=str(path)),
        PreflightCheck(name=f"{name}.is_file", ok=path.is_file(), detail=str(path)),
        PreflightCheck(name=f"{name}.suffix", ok=path.suffix == suffix, detail=str(path)),
    ]
    return checks


def check_demand_parquet(path: Path) -> list[PreflightCheck]:
    checks = check_path(path, suffix=".parquet", name="demand")
    if not all(check.ok for check in checks):
        return checks

    try:
        columns = parquet_columns(path)
    except Exception as exc:
        return [*checks, PreflightCheck("demand.schema_readable", False, str(exc))]

    missing = sorted(DEMAND_REQUIRED_COLUMNS - columns)
    checks.append(
        PreflightCheck(
            "demand.required_columns",
            not missing,
            "missing=" + ",".join(missing) if missing else "all required columns present",
        )
    )
    if missing:
        return checks

    try:
        df = pd.read_parquet(path, columns=sorted(DEMAND_REQUIRED_COLUMNS))
        timestamps, _ = _normalize_to_10min_grid(_parse_regional_demand_timestamp(df))
        in_window = (
            (df["source"] == "power_demand")
            & (timestamps >= pd.Timestamp(EXPECTED_START))
            & (timestamps <= pd.Timestamp(EXPECTED_END))
        )
        checks.append(
            PreflightCheck(
                "demand.2024_rows",
                bool(in_window.any()),
                f"rows_in_window={int(in_window.sum())}",
            )
        )
        window = df.loc[in_window].copy()
        numeric_cols = sorted(DEMAND_REQUIRED_COLUMNS - {"date", "time", "source"})
        for col in numeric_cols:
            window[col] = pd.to_numeric(window[col], errors="coerce")
        invalid_numeric = window[numeric_cols].isna().any(axis=1)
        all_zero = window[numeric_cols].eq(0).all(axis=1)
        valid_timestamps = pd.DatetimeIndex(
            timestamps.loc[window.index[~(invalid_numeric | all_zero)]].dropna().unique()
        ).sort_values()
        expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="10min")
        valid_expected_timestamps = valid_timestamps.intersection(expected)
        missing_after_drop = expected.difference(valid_expected_timestamps)
        extra_after_drop = valid_timestamps.difference(expected)
        checks.append(
            PreflightCheck(
                "demand.flow_source_health",
                bool(len(valid_expected_timestamps)),
                (
                    f"valid_expected_timestamps={len(valid_expected_timestamps)} "
                    f"missing_timestamps_after_invalid_drop={len(missing_after_drop)} "
                    f"extra_timestamps_after_invalid_drop={len(extra_after_drop)} "
                    f"all_zero_regional_rows={int(all_zero.sum())} "
                    f"invalid_numeric_rows={int(invalid_numeric.sum())}"
                ),
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck("demand.2024_rows", False, str(exc)))
    return checks


def check_generation_parquet(path: Path) -> list[PreflightCheck]:
    checks = check_path(path, suffix=".parquet", name="generation")
    if not all(check.ok for check in checks):
        return checks

    try:
        columns = parquet_columns(path)
    except Exception as exc:
        return [*checks, PreflightCheck("generation.schema_readable", False, str(exc))]

    missing = sorted(GENERATION_REQUIRED_COLUMNS - columns)
    checks.append(
        PreflightCheck(
            "generation.required_columns",
            not missing,
            "missing=" + ",".join(missing) if missing else "all required columns present",
        )
    )
    if missing:
        return checks

    try:
        df = pd.read_parquet(path, columns=["timestamp"])
        timestamps = taipower_local_to_utc_naive(df["timestamp"])
        in_window = (timestamps >= pd.Timestamp(EXPECTED_START)) & (
            timestamps <= pd.Timestamp(EXPECTED_END)
        )
        checks.append(
            PreflightCheck(
                "generation.2024_rows",
                bool(in_window.any()),
                f"rows_in_window={int(in_window.sum())}",
            )
        )
    except Exception as exc:
        checks.append(PreflightCheck("generation.2024_rows", False, str(exc)))
    return checks


def check_par_root(root: Path, *, metadata_mode: str) -> list[PreflightCheck]:
    checks = check_weekly_root(root, metadata_mode=metadata_mode)
    missing_or_invalid = [check for check in checks if not check.ok]
    return [
        PreflightCheck("par_weekly_root.exists", root.exists(), str(root)),
        PreflightCheck(
            "par_weekly_root.expected_stores",
            not missing_or_invalid,
            f"expected={len(checks)} missing_or_invalid={len(missing_or_invalid)}",
        ),
    ]


def iter_files(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (child for child in path.rglob("*") if child.is_file())


def tree_size(path: Path) -> int:
    return sum(child.stat().st_size for child in iter_files(path))


def check_output_paths(staging_root: Path) -> list[PreflightCheck]:
    expected_dirs = [
        staging_root,
        staging_root / "par",
        staging_root / "power",
        staging_root / "aef",
        staging_root / "manifest",
    ]
    checks: list[PreflightCheck] = []
    for path in expected_dirs:
        checks.append(
            PreflightCheck(
                f"output_path.{path.name or 'root'}",
                not path.is_file(),
                f"{path} is not an existing file",
            )
        )
    return checks


def check_disk_space(staging_root: Path, *, min_free_multiplier: float) -> PreflightCheck:
    current_size = tree_size(staging_root) if staging_root.exists() else 0
    estimate = current_size if current_size else DEFAULT_CANONICAL_INPUT_BYTES
    required = int(math.ceil(estimate * min_free_multiplier))
    probe_path = staging_root if staging_root.exists() else staging_root.parent
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    usage = shutil.disk_usage(probe_path)
    return PreflightCheck(
        "disk.free_space",
        usage.free >= required,
        f"free_bytes={usage.free} required_bytes={required} estimate_bytes={estimate}",
    )


def run_checks(args: argparse.Namespace) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    checks.extend(check_demand_parquet(args.demand_path))
    checks.extend(check_generation_parquet(args.generation_path))
    checks.extend(check_par_root(args.par_weekly_root, metadata_mode=args.par_metadata))
    checks.extend(check_output_paths(args.staging_root))
    checks.append(
        check_disk_space(
            args.staging_root,
            min_free_multiplier=args.min_free_multiplier,
        )
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight the external inputs needed to rebuild Paper 1 canonical inputs."
    )
    parser.add_argument(
        "--demand-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--generation-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--par-weekly-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--par-metadata",
        choices=("none", "sample", "all"),
        default="sample",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("outputs/final_runs/paper1_canonical_inputs"),
    )
    parser.add_argument("--min-free-multiplier", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = run_checks(args)
    failures = [check for check in checks if not check.ok]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status}\t{check.name}\t{check.detail}")
    if failures:
        print(f"Paper 1 canonical input preflight: NOT READY ({len(failures)} failures)")
        return 1
    print("Paper 1 canonical input preflight: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
