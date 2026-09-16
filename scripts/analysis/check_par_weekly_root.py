from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


EXPECTED_YEAR = 2024
EXPECTED_NEXT_YEAR = 2025
EXPECTED_WEEK_COUNT = 52


@dataclass(frozen=True)
class StoreCheck:
    path: Path
    exists: bool
    metadata_checked: bool = False
    ok: bool = True
    message: str = ""


def expected_weekly_stores(root: Path) -> list[Path]:
    stores = [
        root / str(EXPECTED_YEAR) / f"{EXPECTED_YEAR}W{week:02d}.zarr"
        for week in range(1, EXPECTED_WEEK_COUNT + 1)
    ]
    stores.append(root / str(EXPECTED_NEXT_YEAR) / f"{EXPECTED_NEXT_YEAR}W01.zarr")
    return stores


def inspect_zarr_store(path: Path) -> StoreCheck:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - depends on optional local env
        return StoreCheck(
            path=path,
            exists=True,
            metadata_checked=True,
            ok=False,
            message=f"xarray import failed: {exc}",
        )

    try:
        ds = xr.open_zarr(path)
    except Exception as exc:
        return StoreCheck(
            path=path,
            exists=True,
            metadata_checked=True,
            ok=False,
            message=f"open_zarr failed: {exc}",
        )

    try:
        if "PAR" not in ds:
            return StoreCheck(
                path=path,
                exists=True,
                metadata_checked=True,
                ok=False,
                message="missing PAR variable",
            )
        missing_dims = [dim for dim in ("time", "latitude", "longitude") if dim not in ds.sizes]
        if missing_dims:
            return StoreCheck(
                path=path,
                exists=True,
                metadata_checked=True,
                ok=False,
                message=f"missing dimensions: {', '.join(missing_dims)}",
            )
        shape = tuple(int(ds.sizes[dim]) for dim in ("time", "latitude", "longitude"))
        return StoreCheck(
            path=path,
            exists=True,
            metadata_checked=True,
            ok=True,
            message=f"PAR shape time/latitude/longitude={shape}",
        )
    finally:
        ds.close()


def check_weekly_root(root: Path, *, metadata_mode: str) -> list[StoreCheck]:
    expected = expected_weekly_stores(root)
    checks: list[StoreCheck] = []

    metadata_paths: set[Path] = set()
    if metadata_mode == "sample":
        metadata_paths = {expected[0], expected[-1]}
    elif metadata_mode == "all":
        metadata_paths = set(expected)

    for path in expected:
        if not path.exists():
            checks.append(StoreCheck(path=path, exists=False, ok=False, message="missing"))
        elif path in metadata_paths:
            checks.append(inspect_zarr_store(path))
        else:
            checks.append(StoreCheck(path=path, exists=True, message="exists"))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a user-provided Himawari/P-Tree PAR weekly Zarr root "
            "matches the Paper 1 final data pipeline layout."
        )
    )
    parser.add_argument(
        "--weekly-root",
        type=Path,
        required=True,
        help="Root containing 2024/2024W*.zarr plus 2025/2025W01.zarr.",
    )
    parser.add_argument(
        "--metadata",
        choices=("none", "sample", "all"),
        default="sample",
        help="Open no stores, first/last stores, or all stores with xarray.open_zarr.",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=20,
        help="Maximum number of missing/invalid paths to print.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.weekly_root
    checks = check_weekly_root(root, metadata_mode=args.metadata)

    missing_or_invalid = [check for check in checks if not check.ok]
    metadata_failures = [check for check in missing_or_invalid if check.metadata_checked]
    existing = sum(1 for check in checks if check.exists)

    print(f"PAR weekly root: {root}")
    print(f"expected_stores={len(checks)} existing_stores={existing}")
    print(f"metadata_mode={args.metadata} metadata_failures={len(metadata_failures)}")

    if missing_or_invalid:
        print("missing_or_invalid:")
        for check in missing_or_invalid[: args.max_missing]:
            print(f"  {check.path}: {check.message}")
        remaining = len(missing_or_invalid) - args.max_missing
        if remaining > 0:
            print(f"  ... {remaining} more")
        return 1

    print("PAR weekly root check: READY")
    for check in checks:
        if check.metadata_checked:
            print(f"  {check.path}: {check.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
