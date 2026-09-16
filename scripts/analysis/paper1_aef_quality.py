#!/usr/bin/env python3
"""Audit the staged Paper 1 time sample without rebuilding private raw inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REGIONS = ("north", "central", "south", "east", "island_kinmen", "island_penghu", "island_lienchiang")


def build_aef_quality(canonical_inputs_dir: Path) -> dict[str, Any]:
    """Return deterministic CSV checks and separately identified preparation reports.

    Original absolute manifest paths are relocated using the original staging-root
    prefix. Every consumed original output is checked against its recorded hash.
    The manifests themselves have no self-hash and are identified separately.
    """
    root = Path(canonical_inputs_dir)
    consumed: set[str] = set()

    def csv(name: str, **kwargs: Any) -> pd.DataFrame:
        consumed.add(name)
        return pd.read_csv(root / name, **kwargs)

    def report(name: str) -> dict[str, Any]:
        consumed.add(name)
        return json.loads((root / name).read_text(encoding="utf-8"))

    def indexed(name: str, **kwargs: Any) -> pd.DataFrame:
        df = csv(name, index_col=0, **kwargs)
        df.index = pd.to_datetime(df.index)
        return df

    manifest = report("manifest/final_run_manifest.json")
    time_range = manifest["expected_time_range"]
    expected = pd.date_range(time_range["start"], time_range["end"], freq=time_range["freq"])

    def timestamps(index: pd.DatetimeIndex) -> dict[str, int]:
        return {
            "rows": len(index), "unique_timestamps": index.nunique(),
            "duplicate_rows": int(index.duplicated().sum()),
            "off_grid_rows": int((index != index.floor(time_range["freq"])).sum()),
            "outside_expected_rows": int((~index.isin(expected)).sum()),
            "missing_expected_timestamps": len(expected.difference(index)),
        }

    def flags(name: str) -> pd.DataFrame:
        df = indexed(name)
        if not df.dtypes.eq(bool).all():
            raise ValueError(f"Expected Boolean imputation flags: {name}")
        return df

    def gaps(name: str) -> tuple[pd.DatetimeIndex, dict[str, int]]:
        df = csv(name)
        parts = [pd.date_range(row.start, row.end, freq=time_range["freq"]) for row in df.itertuples()]
        index = pd.DatetimeIndex([stamp for part in parts for stamp in part])
        if len(index) != int(df.n_timestamps.sum()):
            raise ValueError(f"Gap report interval/count disagreement: {name}")
        return index, {"runs": len(df), "timestamps": len(index),
                       "longest_run": int(df.n_timestamps.max()) if len(df) else 0}

    par = indexed("par/par_wide.csv")
    par_flags = flags("par/par_imputed_flags.csv")
    generation = indexed("power/unit_generation.csv", usecols=[0]).index
    generation_flags = flags("power/unit_generation_imputed_flags.csv")
    imputed = generation_flags.index[generation_flags.any(axis=1)]
    before, before_stats = gaps("power/unit_generation_gap_report.csv")
    after, after_stats = gaps("power/unit_generation_remaining_gap_report.csv")
    restored = before.difference(after)
    flow = indexed("power/flow.csv")
    flow_gaps, flow_gap_stats = gaps("power/flow_gap_report.csv")
    common = par.index.intersection(generation).intersection(flow.index).sort_values()
    par_sample_flags = par_flags.reindex(common)
    generation_sample_flags = generation_flags.reindex(common)
    out: dict[str, Any] = {
        "expected_time_range": time_range, "expected_timestamps": len(expected),
        "par": {**timestamps(par.index), "cities": len(par.columns), "cell_denominator": par.size,
                "nonfinite_cells": int((~np.isfinite(par)).to_numpy().sum()),
                "imputed_cells": int(par_flags.to_numpy().sum()),
                "imputed_timestamps": int(par_flags.any(axis=1).sum())},
        "generation": {**timestamps(generation), "gap_before": before_stats, "gap_after": after_stats,
                       "restored_whole_gap_timestamps": len(restored),
                       "imputed_partial_row_timestamps": len(imputed.difference(before)),
                       "imputed_timestamps": len(imputed),
                       "imputed_cells": int(generation_flags.to_numpy().sum()),
                       "flag_cell_denominator": generation_flags.size,
                       "gap_after_matches_missing_output": after.equals(expected.difference(generation))},
        "flow": {**timestamps(flow.index), "columns": len(flow.columns),
                 "nonfinite_cells": int((~np.isfinite(flow)).to_numpy().sum()),
                 "gap": flow_gap_stats,
                 "gap_matches_missing_output": flow_gaps.equals(expected.difference(flow.index))},
        "intersection": {"timestamps": len(common), "excluded_expected_timestamps": len(expected.difference(common)),
                         "generation_missing": len(expected.difference(generation)),
                         "flow_missing": len(expected.difference(flow.index)),
                         "missing_both": len(expected.difference(generation).intersection(expected.difference(flow.index))),
                         "generation_only_removed": len(generation.difference(flow.index)),
                         "par_nonfinite_cells": int((~np.isfinite(par.reindex(common))).to_numpy().sum()),
                         "par_imputed_cells": int(par_sample_flags.to_numpy().sum()),
                         "par_cell_denominator": par_sample_flags.size,
                         "par_imputed_timestamps": int(par_sample_flags.any(axis=1).sum()),
                         "generation_imputed_cells": int(generation_sample_flags.to_numpy().sum()),
                         "generation_imputed_timestamps": len(common.intersection(imputed)),
                         "generation_restored_whole_gap_timestamps": len(common.intersection(restored))},
        "aef": {},
    }
    regional_generation = {}
    out["generation"]["regional_fuel_checks"] = {}
    for region in REGIONS:
        df = indexed(f"power/{region}_unit_generation.csv")
        regional_generation[region] = df
        fuel = df.loc[:, [column for column in df.columns
                          if column not in ("BESS", "PHS")
                          and not column.startswith(("F_", "Storage-"))]]
        # AEF clipping acts on fuel aggregates after unit columns are grouped.
        grouped = fuel.T.groupby([column.split("-", 1)[0] for column in fuel.columns]).sum(min_count=1).T
        negative = grouped < 0
        out["generation"]["regional_fuel_checks"][region] = {
            **timestamps(df.index), "nonstorage_unit_columns": len(fuel.columns),
            "negative_unit_cells": int((fuel < 0).to_numpy().sum()),
            "negative_fuel_aggregate_cells": int(negative.to_numpy().sum()),
            "negative_fuel_aggregate_timestamps": int(negative.any(axis=1).sum()),
            "negative_fuel_aggregate_timestamps_in_sample": int(negative.any(axis=1).reindex(common).sum()),
            "index_matches_common_sample": df.index.equals(common),
        }
    out["flow"]["by_link"] = {}
    for region, forward_name, reverse_name in (
        ("north", "F_CN", "F_NC"), ("east", "F_CE", "F_EC"), ("south", "F_CS", "F_SC"),
    ):
        signed = flow[forward_name]
        signed_common = signed.reindex(common)
        forward = regional_generation[region][forward_name].reindex(common)
        reverse = regional_generation["central"][reverse_name].reindex(common)
        out["flow"]["by_link"][forward_name] = {
            "positive_rows": int((signed > 0).sum()), "negative_rows": int((signed < 0).sum()),
            "zero_rows": int((signed == 0).sum()),
            "positive_rows_in_sample": int((signed_common > 0).sum()),
            "negative_rows_in_sample": int((signed_common < 0).sum()),
            "zero_rows_in_sample": int((signed_common == 0).sum()),
            "split_nonnegative": bool((forward >= 0).all() and (reverse >= 0).all()),
            "split_mutually_exclusive": bool(((forward * reverse) == 0).all()),
            "split_matches_signed_flow": bool(np.allclose(forward, signed_common.clip(lower=0))
                                               and np.allclose(reverse, (-signed_common).clip(lower=0))),
            "split_max_abs_error": float(max((forward - signed_common.clip(lower=0)).abs().max(),
                                               (reverse - (-signed_common).clip(lower=0)).abs().max())),
        }
    del regional_generation
    aef_common = par.index
    for region in REGIONS:
        df = indexed(f"aef/{region}.csv", usecols=["timestamp", "Total_Gen (MWh)", "Total_Emis", "FLOW_UNIT_FINAL_AEF"])
        final = df["FLOW_UNIT_FINAL_AEF"]
        finite = np.isfinite(final)
        entry = {**timestamps(df.index), "finite_final_aef": int(finite.sum()),
                 "nonfinite_final_aef": int((~finite).sum())}
        if region.startswith("island_"):
            # The calculator stores NaN totals when positive generation sums to
            # zero. Reconstruct its source ratio before the island median fill.
            ratio = df["Total_Emis"] / df["Total_Gen (MWh)"]
            source = ratio.mask(df["Total_Gen (MWh)"] <= 1e-12)
            fallback = source.isna()
            median = float(source.dropna().median()) if source.notna().any() else 0.0
            entry.update({"local_aef_fallback_rows": int(fallback.sum()),
                          "local_aef_fallback_rows_in_sample": int(fallback.reindex(common).sum()),
                          "local_aef_fallback_value": median,
                          "fallback_matches_local_median": bool(np.allclose(final[fallback], median))})
        out["aef"][region] = entry
        aef_common = aef_common.intersection(df.index[finite])
    out["intersection"].update({"finite_aef_all_regions": len(aef_common),
                               "finite_aef_index_matches_input_intersection": aef_common.equals(common)})
    generation_report = report("power/generation_quality_report.json")
    flow_report = report("power/flow_quality_report.json")
    invalid = indexed("power/generation_invalid_total_report.csv")
    out["generation"]["invalid_total_report_timestamps"] = len(invalid)
    out["generation"]["invalid_total_report_below_threshold"] = int(
        (invalid.total_generation_mw < invalid.min_valid_total_generation_mw).sum())
    out["generation"]["invalid_total_timestamps_restored"] = len(invalid.index.intersection(generation))
    out["generation"]["invalid_total_timestamps_in_sample"] = len(invalid.index.intersection(common))
    out["reported_preparation"] = {
        "evidence_scope": "Stored preparation reports; private raw inputs were not reprocessed.",
        "generation": {key: generation_report[key] for key in (
            "normalize_off_grid", "max_offset_minutes", "min_valid_total_generation_mw",
            "normalized_rows", "normalized_unique_timestamps", "impute_short_gaps", "max_impute_run", "combined")},
        "generation_invalid_total": {key: generation_report["invalid_total_generation"][key]
                                     for key in ("timestamps", "source_rows_dropped")},
        "flow": {key: flow_report[key] for key in (
            "unit_scale", "input_rows_after_filter", "duplicate_input_timestamps", "invalid_numeric_rows",
            "all_zero_regional_rows", "all_zero_regional_timestamps", "normalized_timestamps",
            "flow_nan_rows", "dropped_extra_timestamps")},
    }
    out["flow"]["normalization_report_rows"] = len(csv("power/flow_timestamp_normalization.csv"))
    out["generation"]["normalization_report_rows"] = len(csv("power/generation_timestamp_normalization.csv"))
    hashes = csv("manifest/final_output_hashes.csv")
    original_root = Path(manifest["staging_root"])
    recorded = {str(Path(row.path).relative_to(original_root)): row for row in hashes.itertuples()}
    checked = []
    for name in sorted(consumed):
        digest = hashlib.sha256()
        with (root / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        original = recorded.get(name)
        if original is not None and (actual != original.sha256 or (root / name).stat().st_size != original.size_bytes):
            raise ValueError(f"Canonical input differs from original output hash: {name}")
        if original is None and not name.startswith("manifest/"):
            raise ValueError(f"Canonical evidence absent from original hash manifest: {name}")
        checked.append({"path": name, "sha256": actual, "original_output_hash_verified": original is not None})
    out["provenance"] = {
        "original_run_id": manifest["run_id"], "original_output_hash_entries": len(recorded),
        "source_files": checked, "original_output_hashes_verified": sum(item["original_output_hash_verified"] for item in checked),
        "path_mapping": "Strip the recorded staging-root prefix and resolve within canonical inputs.",
        "scope": "Checks identify the staged snapshot; they do not revalidate raw downloads or source-record preparation.",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_inputs_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_aef_quality(args.canonical_inputs_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps({"expected_timestamps": result["expected_timestamps"],
                          "retained_timestamps": result["intersection"]["timestamps"],
                          "source_hashes_verified": result["provenance"]["original_output_hashes_verified"]}))


if __name__ == "__main__":
    main()
