"""
The three-way demo from the build plan, Section 8.

A single passing example proves nothing - a system that always says
"compliant" would pass it. This generates three packs that differ only in the
printed height of the net-quantity numerals and shows that the system
discriminates:

  A  comfortably compliant           -> PASS
  B  deliberately undersized print   -> FAIL
  C  printed AT the legal threshold  -> BORDERLINE, manual verification

C is the one that matters. A system that returns a confident verdict on C is
claiming a precision its own measurement error does not support.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.synth import Degradation, degrade, random_spec, render_master
from lmpc.calibration import MarkerSpec
from lmpc.pipeline import ScanConfig, scan_image
from lmpc.report import render_text
from lmpc.rules import PrintStyle, parse_net_quantity, required_height_mm


def build(out_dir: str, seed: int = 20260904):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    base = random_spec(rng)
    base.qty_value, base.qty_unit = 500, "g"      # -> Rule 7 Table I, 2 mm
    base.qty_key = "Net Qty."
    req = required_height_mm(
        "net_quantity", parse_net_quantity("500 g"), PrintStyle.NORMAL).required_mm

    cases = [
        ("A_compliant", req + 1.6, "comfortably above the Rule 7 minimum"),
        ("B_violation", req - 0.9, "deliberately undersized print"),
        ("C_threshold", req + 0.02, "printed at the legal threshold"),
    ]
    deg = Degradation(target_ppmm=16.0, tilt_x_deg=6.0, tilt_y_deg=-8.0,
                      roll_deg=2.0, ink_gain=0, blur_sigma_px=0.5,
                      light_strength=0.2, glare=0.12, noise_sigma=1.5,
                      jpeg_q=88)
    out = []
    for name, h, why in cases:
        spec = random_spec(random.Random(seed))
        spec.qty_value, spec.qty_unit, spec.qty_key = 500, "g", "Net Qty."
        spec.qty_height_mm = h
        spec.mrp_height_mm = 2.0
        master, gt = render_master(spec)
        photo = degrade(master, deg, random.Random(seed))
        path = os.path.join(out_dir, f"{name}.jpg")
        cv2.imwrite(path, photo, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        out.append((name, path, gt["fields"]["net_quantity"]["height_mm"],
                    req, why))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backend/data/demo")
    ap.add_argument("--full", action="store_true", help="print full reports")
    a = ap.parse_args()

    cases = build(a.out)
    cfg = ScanConfig(marker=MarkerSpec(marker_length_mm=40.0))
    print(f"\n{'case':14s} {'printed':>9s} {'required':>9s} {'measured':>9s} "
          f"{'err':>7s}  verdict")
    print("-" * 74)
    ok = True
    expected = {"A_compliant": "PASS", "B_violation": "FAIL",
                "C_threshold": "BORDERLINE"}
    for name, path, gt_mm, req, why in cases:
        rep = scan_image(cv2.imread(path), cfg)
        h = next(d["height"] for d in rep["declarations"]
                 if d["field"] == "net_quantity")
        m = h.get("measured_mm")
        v = h.get("verdict")
        err = f"{m - gt_mm:+.3f}" if m is not None else "  -  "
        print(f"{name:14s} {gt_mm:9.3f} {req:9.2f} "
              f"{(f'{m:.3f}' if m is not None else '  -  '):>9s} {err:>7s}  "
              f"{v}   ({why})")
        if v != expected[name]:
            ok = False
        if a.full:
            print(render_text(rep))
    print("-" * 74)
    print("discrimination check:",
          "PASS - the system separates all three" if ok else
          "FAILED - verdicts did not match A=PASS / B=FAIL / C=BORDERLINE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
