"""SimaPro-calibrated hardware LCA calculator for Paper 1 designs.

Avoids the SimaPro re-import loop. Component-level GWP100 coefficients
are calibrated so that, at the calibration anchor (2.0x PV / 8.5x battery),
the per-stage outputs reproduce the SimaPro IPCC 2021 GWP100 V1.03
results from the controlled licensed rebuild record to within ~1%.

The calibration uses the closed-form decomposition:

    stage_total = SHARED_stage(fixed)
                + PV_modules    * G_module_per_unit
                + battery_units * G_battery_per_unit
                + transport_per_kg(stage) * (PV_mass + battery_mass)

For stages with multiple unknowns the under-determined slack is resolved
using literature ranges (footnoted by line). The SHARED contributions are
back-solved from the anchor-design SimaPro totals, so any future re-import of
SimaPro at a NEW knee can be reconciled by re-calibrating the per-unit
coefficients here against the new analysis output (run this script with
NEW totals -> new G_module / G_battery / transport coefficients).

Usage:
    python -m streetlight.lca.hardware
    python -m streetlight.lca.hardware --pv 1.85 --batt 8.45

API:
    from streetlight.lca import compute_lca_at_knee
    totals = compute_lca_at_knee(pv_factor=1.85, batt_factor=8.45)
    # -> {"SOLAR": {"A1_A3": ..., "A4": ..., "A5": ..., "B": ..., "C": ...},
    #     "TRAD":  {"A1_A3": ..., "A4": ..., "A5": ..., "B": ..., "C": ...}}
"""

from __future__ import annotations

import argparse
import json

# -----------------------------------------------------------------------------
# Knee-design physical units (must match build_dual_system.py)
# -----------------------------------------------------------------------------
# 1x design = 18 kW PV / 64 poles / 150 Wp = 1.875 modules per pole
# 1x design = 10 kWh / 64 poles / 1.4352 kWh per cell-unit = 0.10892 cells per pole
MODULES_PER_POLE_PER_FACTOR = 1.875
CELLS_PER_POLE_PER_FACTOR = 0.10892

PV_MODULE_MASS_KG = 7.59       # per 150 Wp module
BATT_SYSTEM_MASS_KG = 31.0     # per cell-unit incl. cells + BMS + enclosure
MPPT_MASS_KG = 0.145

# -----------------------------------------------------------------------------
# Calibration anchor: SimaPro IPCC 2021 GWP100 V1.03 totals at the anchor design
# (2.0x PV / 8.5x battery -> 3.75 modules / 0.9254 cells per pole).
# Source: controlled SimaPro analyse run on 2026-05-02, retained as licensed
# rebuild evidence rather than an active Python pipeline input.
# -----------------------------------------------------------------------------
ANCHOR_PV_FACTOR = 2.0
ANCHOR_BATT_FACTOR = 8.5
ANCHOR_MODULES = ANCHOR_PV_FACTOR * MODULES_PER_POLE_PER_FACTOR    # 3.75
ANCHOR_CELLS = ANCHOR_BATT_FACTOR * CELLS_PER_POLE_PER_FACTOR      # 0.92582 (~0.9254)

ANCHOR_SOLAR = {"A1_A3": 1400.7408, "A4": 15.715038, "A5": 170.70443,
                "B": 205.38569, "C": 14.117621}
ANCHOR_TRAD  = {"A1_A3":  543.35667, "A4": 13.364791, "A5": 169.69771,
                "B":   0.0,    "C": 10.369788}

# -----------------------------------------------------------------------------
# Component-level coefficients (kg CO2-eq per unit).
#
# Calibrated so that
#   compute_lca_at_knee(2.0, 8.5)["SOLAR"] == ANCHOR_SOLAR  (within ~1%)
#   compute_lca_at_knee(2.0, 8.5)["TRAD"]  == ANCHOR_TRAD   (within ~1%)
#
# The PV-module and battery-cell coefficients are constrained by the SOLAR
# A1-A3 and B totals respectively; transport coefficients by SOLAR vs TRAD
# A4 delta. SHARED_* are back-solved residuals (per stage).
# -----------------------------------------------------------------------------

# Battery system per cell-unit (cells + BMS bundled). Solved from anchor-design
# SOLAR B at ANCHOR_CELLS (replacement = 1x cells + 1x BMS + 1x MPPT + transport)
# minus per-piece MPPT contribution (~10 kg) and battery transport (~1.1 kg).
G_BATTERY_PER_UNIT = (
    ANCHOR_SOLAR["B"]
    - 10.0                                     # MPPT replacement, fixed
    - (ANCHOR_CELLS * BATT_SYSTEM_MASS_KG) * (1200/1000 * 0.011 + 200/1000 * 0.13)
) / ANCHOR_CELLS
# G_BATTERY_PER_UNIT ≈ 210 kg CO2-eq per cell-unit (LFP cell + BMS pack)

# Battery enclosure (sheet-steel cabinet, ~25 kg low-alloy steel)
G_BATTERY_ENCLOSURE = 50.0     # kg CO2-eq, fixed regardless of factor

# MPPT controller (small electronics, fixed)
G_MPPT = 10.0                  # kg CO2-eq

# Transport per kg of shipped material (sea 1200 km + lorry 200 km combined)
# sea: ~0.011 kg-CO2/tkm, lorry FAME 28t: ~0.13 kg-CO2/tkm
G_FREIGHT_SEA_PER_KG = 1200 / 1000.0 * 0.011    # kg-CO2 per kg shipped
G_FREIGHT_LORRY_PER_KG = 200 / 1000.0 * 0.13
G_FREIGHT_PER_KG = G_FREIGHT_SEA_PER_KG + G_FREIGHT_LORRY_PER_KG
# ≈ 0.0132 + 0.026 = 0.0392 kg-CO2 per kg shipped

# PV module -- solved from anchor SOLAR A1-A3 minus all other components.
# Anchor SOLAR A1-A3 = SHARED_A1A3 + 3.75*G_module + 0.9254*G_battery + G_encl + G_mppt
# Anchor TRAD A1-A3 = SHARED_A1A3 + (TRAD-only items: switch_box + switchgear + grounding)
# Estimate TRAD-only items in A1-A3 from masses (steel 18.5+5 kg @ 2 kg-CO2/kg + copper 1 kg @ 5):
TRAD_ONLY_A1A3 = (18.5 + 5.0) * 2.0 + 1.0 * 5.0    # = 52 kg-CO2/pole
SHARED_A1A3 = ANCHOR_TRAD["A1_A3"] - TRAD_ONLY_A1A3        # ≈ 491.36
SOLAR_ONLY_A1A3 = ANCHOR_SOLAR["A1_A3"] - SHARED_A1A3       # ≈ 909.38
# 3.75 * G_module = SOLAR_ONLY_A1A3 - 0.9254*G_battery - G_encl - G_mppt
G_PV_MODULE = (
    SOLAR_ONLY_A1A3
    - ANCHOR_CELLS * G_BATTERY_PER_UNIT
    - G_BATTERY_ENCLOSURE
    - G_MPPT
) / ANCHOR_MODULES
# G_PV_MODULE ≈ 175 kg-CO2 per 150 Wp single-Si module

# Stage A4 SHARED back-solve (TRAD A4 contains only TRAD masses transport):
TRAD_TRANSPORT_MASS_KG = (
    7.59 * 0  # no PV
    + 0       # no battery
    + 6.85    # luminaire (sea+lorry)
    + 56.0    # pole (lorry only domestic 200 km, but use combined for consistency)
    + 1.0     # accessories
    + 1454.72 # concrete (lorry 50 km local)
    + 25.6    # rebar
    + 0.528   # fasteners
    + 2.38554 # cable
    + 2.34    # conduit
)
# TRAD A4 ≈ 13.36; freight per kg differs by leg, so we treat shared transport
# as a calibrated constant rather than recomputing leg-by-leg.
SHARED_A4_KG_CO2 = ANCHOR_TRAD["A4"]    # back-solved; SOLAR A4 = SHARED + extras

# Stage A5 SHARED (civil work, mostly TRAD = SOLAR + ~1 kg extra):
SHARED_A5_KG_CO2 = ANCHOR_TRAD["A5"]    # ≈ 169.70

# Stage C SHARED (luminaire + pole + foundation EoL):
TRAD_EOL_MASS_KG = 6.85 + 56.0 + 1454.72 + 25.6 + 0.528 + 2.38554 + 2.34  # = 1548.4
SHARED_C_KG_CO2 = ANCHOR_TRAD["C"]      # ≈ 10.37 (TRAD has only shared EoL)

# Stage A5 minor extras when adding PV/battery (mounting per module):
G_A5_PER_PV_MODULE = (ANCHOR_SOLAR["A5"] - ANCHOR_TRAD["A5"]) / ANCHOR_MODULES   # ~0.27

# Stage C minor extras (transport + treatment of PV/battery materials):
SOLAR_EXTRA_C_AT_ANCHOR = ANCHOR_SOLAR["C"] - ANCHOR_TRAD["C"]                    # ~3.75 kg-CO2
SOLAR_EXTRA_MASS_AT_ANCHOR = (ANCHOR_MODULES * PV_MODULE_MASS_KG
                              + 2 * ANCHOR_CELLS * BATT_SYSTEM_MASS_KG)           # initial + 1 replacement batt
G_C_PER_KG_OF_EXTRA_MASS = SOLAR_EXTRA_C_AT_ANCHOR / SOLAR_EXTRA_MASS_AT_ANCHOR    # ~0.0437 kg-CO2/kg


def compute_lca_at_knee(
    pv_factor: float, batt_factor: float, *, battery_replacements: int = 1
) -> dict:
    """Return SOLAR and TRAD per-pole GWP100 dicts (kg CO2-eq) for one design.

    The TRAD totals are independent of (pv_factor, batt_factor) because
    the conventional grid-only LED design is fixed; they are returned for
    convenience so callers can compute the SOLAR-vs-TRAD delta directly.

    ``battery_replacements`` counts battery-cell/BMS replacements during the
    service horizon. The default preserves the reference year-12 replacement.
    One MPPT replacement remains fixed over 20 years. Stage C scales the same
    calibrated mass-based treatment proxy to the initial and replaced batteries;
    it is not a separate process-level battery recycling assessment.
    """
    if battery_replacements < 0 or int(battery_replacements) != battery_replacements:
        raise ValueError("battery_replacements must be a non-negative integer")
    pv_modules = pv_factor * MODULES_PER_POLE_PER_FACTOR
    batt_cells = batt_factor * CELLS_PER_POLE_PER_FACTOR

    pv_mass_kg = pv_modules * PV_MODULE_MASS_KG
    batt_mass_kg = batt_cells * BATT_SYSTEM_MASS_KG

    # ------- SOLAR -----------------------------------------------------------
    solar_a1a3 = (
        SHARED_A1A3
        + pv_modules * G_PV_MODULE
        + batt_cells * G_BATTERY_PER_UNIT
        + G_BATTERY_ENCLOSURE
        + G_MPPT
    )

    # SOLAR A4 = TRAD A4 (shared transport baseline) + extra freight for
    # PV modules + battery system + MPPT.
    solar_extra_freight = (pv_mass_kg + batt_mass_kg + MPPT_MASS_KG) * G_FREIGHT_PER_KG
    solar_a4 = SHARED_A4_KG_CO2 + solar_extra_freight

    # SOLAR A5 = TRAD A5 (shared install) + per-module mounting extra.
    solar_a5 = SHARED_A5_KG_CO2 + pv_modules * G_A5_PER_PV_MODULE

    # Battery-cell/BMS bundle and freight scale with replacement count; the
    # reference inventory has one MPPT replacement over the whole horizon.
    solar_b = (
        battery_replacements * batt_cells * G_BATTERY_PER_UNIT
        + G_MPPT
        + battery_replacements * batt_mass_kg * G_FREIGHT_PER_KG
    )

    # SOLAR C end-of-life = TRAD C (shared treatment) + extras scaling with
    # additional PV+battery mass, including every initial and replacement pack.
    extra_eol_mass = pv_mass_kg + (1 + battery_replacements) * batt_mass_kg
    solar_c = SHARED_C_KG_CO2 + extra_eol_mass * G_C_PER_KG_OF_EXTRA_MASS

    solar = {"A1_A3": solar_a1a3, "A4": solar_a4, "A5": solar_a5,
             "B": solar_b, "C": solar_c}

    # ------- TRAD ------------------------------------------------------------
    trad = dict(ANCHOR_TRAD)

    return {"SOLAR": solar, "TRAD": trad}


def report(pv_factor: float, batt_factor: float) -> None:
    out = compute_lca_at_knee(pv_factor, batt_factor)
    solar = out["SOLAR"]
    trad = out["TRAD"]
    fu = 64
    print(f"# LCA hardware totals (kg CO2-eq per pole) at PV={pv_factor}x, "
          f"BATT={batt_factor}x")
    print(f"# Anchored to SimaPro IPCC 2021 GWP100 V1.03 results at "
          f"{ANCHOR_PV_FACTOR}x/{ANCHOR_BATT_FACTOR}x.")
    print()
    print(f"{'Stage':<8} {'SOLAR':>10} {'TRAD':>10} {'Delta':>10}")
    delta_total = 0.0
    for k in ("A1_A3", "A4", "A5", "B", "C"):
        d = solar[k] - trad[k]
        delta_total += d
        print(f"{k:<8} {solar[k]:>10.2f} {trad[k]:>10.2f} {d:>+10.2f}")
    s_tot = sum(solar.values())
    t_tot = sum(trad.values())
    print(f"{'TOTAL':<8} {s_tot:>10.2f} {t_tot:>10.2f} {delta_total:>+10.2f}")
    print()
    print(f"Per-FU ({fu}-pole installation):")
    print(f"  SOLAR total = {s_tot * fu / 1000:.4f} t CO2-eq")
    print(f"  TRAD  total = {t_tot * fu / 1000:.4f} t CO2-eq")
    print(f"  Hardware burden delta = {delta_total * fu / 1000:.4f} t CO2-eq "
          f"({delta_total / 1000:.4f} t/pole)")


def calibration_check() -> None:
    """Confirm the anchor design reproduces ANCHOR totals."""
    out = compute_lca_at_knee(ANCHOR_PV_FACTOR, ANCHOR_BATT_FACTOR)
    print(f"# Calibration check at anchor design "
          f"({ANCHOR_PV_FACTOR}x / {ANCHOR_BATT_FACTOR}x)")
    print(f"{'Stage':<8} {'SOLAR_pred':>11} {'SOLAR_anchor':>13} {'err%':>7} "
          f"{'TRAD_pred':>10} {'TRAD_anchor':>12} {'err%':>7}")
    for k in ("A1_A3", "A4", "A5", "B", "C"):
        sp = out["SOLAR"][k]; sa = ANCHOR_SOLAR[k]
        tp = out["TRAD"][k];  ta = ANCHOR_TRAD[k]
        s_err = 100.0 * (sp - sa) / sa if sa != 0 else 0.0
        t_err = 100.0 * (tp - ta) / ta if ta != 0 else 0.0
        print(f"{k:<8} {sp:>11.2f} {sa:>13.2f} {s_err:>+7.2f} "
              f"{tp:>10.2f} {ta:>12.2f} {t_err:>+7.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pv", type=float, default=1.85, help="PV factor (default 1.85)")
    p.add_argument("--batt", type=float, default=8.45, help="Battery factor (default 8.45)")
    p.add_argument("--check", action="store_true", help="Run calibration check only")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    args = p.parse_args()
    if args.check:
        calibration_check()
        return
    if args.json:
        out = compute_lca_at_knee(args.pv, args.batt)
        print(json.dumps(out, indent=2))
    else:
        calibration_check()
        print()
        report(args.pv, args.batt)


if __name__ == "__main__":
    main()
