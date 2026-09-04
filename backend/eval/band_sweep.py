"""
Band sensitivity: what does widening or narrowing the error band actually buy?

The three-tier verdict has one free parameter. Too narrow and the system makes
confident calls it cannot support; too wide and it abstains on everything and
is useless (an early version of this build produced a +/-1.0 mm band against a
1 mm threshold and went BORDERLINE on 39% of cases).

This replays every validation measurement at a range of bands, using the
stored measurement and the known true height, and reports the trade-off
directly. The number to look at is `wrong_decisive` - a decisive verdict that
disagrees with the truth, i.e. a pack cleared that should have failed, or a
trader accused who was compliant.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = ("net_quantity", "mrp", "mfg_date")   # every field that can FAIL
NEAR_MM = 0.3


def load(path):
    recs = json.load(open(path))
    rows = []
    for r in recs:
        if "error" in r:
            continue
        for f in MEAS:
            rec = r["fields"].get(f, {})
            if rec.get("gt_mm") is None or rec.get("required_mm") is None:
                continue
            if rec.get("measured_mm") is None:
                continue
            # Only measurements the pipeline was willing to adjudicate: the
            # confidence and resolution gates have already had their say.
            if rec.get("verdict") not in ("PASS", "FAIL", "BORDERLINE"):
                continue
            rows.append((f, rec["measured_mm"], rec["gt_mm"],
                         rec["required_mm"]))
    return rows


def replay(rows, band):
    dec = wrong = near_n = near_dec = near_wrong = 0
    for f, m, gt, req in rows:
        truth = "PASS" if gt >= req else "FAIL"
        if m >= req + band:
            pred = "PASS"
        elif m <= req - band:
            pred = "FAIL"
        else:
            pred = "BORDERLINE"
        near = abs(gt - req) <= NEAR_MM
        if near:
            near_n += 1
        if pred != "BORDERLINE":
            dec += 1
            if near:
                near_dec += 1
            if pred != truth:
                wrong += 1
                if near:
                    near_wrong += 1
    return dec, wrong, near_n, near_dec, near_wrong


def main():
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(HERE, "..", "data", "eval_valid_records.json"))
    rows = load(path)
    if not rows:
        sys.exit("no adjudicated measurements in records")
    unc_path = os.path.join(HERE, "..", "data", "uncertainty.json")
    fitted = None
    if os.path.exists(unc_path):
        fitted = json.load(open(unc_path)).get("band_override_mm")

    print(f"{len(rows)} adjudicated measurements "
          f"({sum(1 for r in rows if abs(r[2]-r[3]) <= NEAR_MM)} within "
          f"{NEAR_MM} mm of their threshold)\n")
    print(f"{'band mm':>8s} {'decisive':>9s} {'wrong':>6s} "
          f"{'near decisive':>14s} {'near wrong':>11s}")
    print("-" * 54)
    bands = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20,
             0.30, 0.50, 1.00]
    if fitted:
        bands = sorted(set(bands + [round(fitted, 4)]))
    safest = None
    for b in bands:
        dec, wrong, nn, nd, nw = replay(rows, b)
        star = "  <- fitted" if fitted and abs(b - fitted) < 1e-9 else ""
        print(f"{b:8.3f} {dec/len(rows):9.3f} {wrong:6d} "
              f"{(nd/nn if nn else 0):14.3f} {nw:11d}{star}")
        if wrong == 0 and safest is None:
            safest = b
    print("-" * 54)
    if safest is not None:
        dec, _, nn, nd, _ = replay(rows, safest)
        print(f"smallest band with ZERO wrong decisive calls: {safest:.3f} mm "
              f"({dec/len(rows):.1%} of measurements still decisive)")
    else:
        print("every band tested produced at least one wrong decisive call")
    if fitted:
        dec, wrong, nn, nd, nw = replay(rows, fitted)
        print(f"fitted band {fitted:.4f} mm -> {dec/len(rows):.1%} decisive, "
              f"{wrong} wrong decisive ({nw} of them near the threshold)")
        if wrong:
            print("  ^ the fitted band is too tight for this population; "
                  "raise k or the percentile")


if __name__ == "__main__":
    main()
