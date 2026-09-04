"""
Negative control for the coplanarity gate.

Renders the SAME label twice: once with the marker card in the label's plane
(the supported setup) and once with the card tilted onto a different plane,
as happens when the card is laid flat on the table beside a standing carton.
The second case is the dangerous one - the marker can look perfectly
square-on while every millimetre reading is wrong - so the gate must catch it.
"""
from __future__ import annotations

import math
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.synth import (MASTER_PPMM, Degradation, random_spec,
                        render_master)
from lmpc.calibration import MarkerSpec
from lmpc.pipeline import ScanConfig, scan_image


def _hinged_homography(deg, w_mm, h_mm, split_mm, phi_deg):
    """mm-plane -> image, for a plane hinged by phi about the line X=split_mm.

    phi = 0 reproduces the ordinary coplanar view exactly, so the same code
    path generates both the control and the treatment.
    """
    ax, ay, az = (math.radians(deg.tilt_x_deg), math.radians(deg.tilt_y_deg),
                  math.radians(deg.roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    d = 500.0
    f = deg.target_ppmm * d
    K = np.array([[f, 0, 0.0], [0, f, 0.0], [0, 0, 1.0]])
    p = math.radians(phi_deg)
    col_x = R @ np.array([math.cos(p), 0.0, math.sin(p)])
    col_y = R @ np.array([0.0, 1.0, 0.0])
    col_1 = R @ np.array([split_mm - w_mm / 2 - split_mm * math.cos(p),
                          -h_mm / 2,
                          -split_mm * math.sin(p)]) + np.array([0, 0, d])
    return K @ np.column_stack([col_x, col_y, col_1])


def build_split_plane(spec, panel_extra_tilt_deg: float, ppmm_out=16.0):
    """Compose a frame where the panel and the marker sit on DIFFERENT planes.

    The marker card keeps the base plane; the declaration panel is hinged away
    from it by `panel_extra_tilt_deg` about their shared edge - exactly the
    geometry of a card lying on the table next to a standing carton.
    """
    master, gt = render_master(spec)
    mh, mw = master.shape[:2]
    w_mm, h_mm = mw / MASTER_PPMM, mh / MASTER_PPMM
    split_mm = spec.panel_w_mm + 10.0
    split_px = int(split_mm * MASTER_PPMM)

    s = min(1.0, ppmm_out / MASTER_PPMM * 1.25)
    small = cv2.resize(master, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    eff = MASTER_PPMM * s
    px2mm = np.diag([1.0 / eff, 1.0 / eff, 1.0])

    deg = Degradation(target_ppmm=ppmm_out, tilt_x_deg=4.0, tilt_y_deg=-6.0)
    Hc = _hinged_homography(deg, w_mm, h_mm, split_mm, 0.0) @ px2mm
    Hp = _hinged_homography(deg, w_mm, h_mm, split_mm,
                            panel_extra_tilt_deg) @ px2mm

    ih, iw = small.shape[:2]
    corners = np.array([[0, 0, 1], [iw, 0, 1], [iw, ih, 1], [0, ih, 1]], float)
    pts = []
    for H in (Hc, Hp):
        q = corners @ H.T
        pts.append(q[:, :2] / q[:, 2:3])
    pts = np.vstack(pts)
    x0, y0 = pts[:, 0].min() - 40, pts[:, 1].min() - 40
    x1, y1 = pts[:, 0].max() + 40, pts[:, 1].max() + 40
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    W, H_ = int(min(5000, x1 - x0)), int(min(5000, y1 - y0))

    sx = int(split_px * s)
    mask_p = np.zeros((ih, iw), np.uint8); mask_p[:, :sx] = 255
    mask_c = np.zeros((ih, iw), np.uint8); mask_c[:, sx:] = 255

    def w_(img, Hm):
        return cv2.warpPerspective(img, T @ Hm, (W, H_),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    canvas = np.full((H_, W, 3), 235, np.uint8)
    for img_w, m_w in ((w_(small, Hp), w_(mask_p, Hp)),
                       (w_(small, Hc), w_(mask_c, Hc))):
        sel = m_w > 127
        canvas[sel] = img_w[sel]
    return canvas, gt


def main():
    rng = random.Random(4242)
    spec = random_spec(rng)
    spec.qty_value, spec.qty_unit, spec.qty_key = 500, "g", "Net Qty."
    spec.qty_height_mm = 3.0
    cfg = ScanConfig(marker=MarkerSpec(marker_length_mm=40.0))
    gt_mm = None
    print(f"{'panel tilt vs marker':>22s} {'measured':>9s} {'true':>7s} "
          f"{'err':>8s}  {'baseline':>9s}  gate")
    print("-" * 78)
    rows = []
    for extra in (0.0, 5.0, 10.0, 20.0, 30.0):
        img, gt = build_split_plane(spec, extra)
        gt_mm = gt["fields"]["net_quantity"]["height_mm"]
        rep = scan_image(img, cfg)
        h = next(d["height"] for d in rep["declarations"]
                 if d["field"] == "net_quantity")
        m = h.get("measured_mm")
        pl = rep["planarity"]
        gate = ("COPLANAR-OK" if pl["coplanar"] else "PANEL_NOT_COPLANAR")
        if pl["coplanar"] is None:
            gate = "untestable"
        err = f"{m-gt_mm:+.3f}" if m is not None else "   -   "
        print(f"{extra:19.0f} deg {(f'{m:.3f}' if m else '   -   '):>9s} "
              f"{gt_mm:7.3f} {err:>8s}  "
              f"{str(pl['median_slope_deg']):>9s}  {gate}")
        rows.append((extra, m, pl["coplanar"]))

    ok0 = rows[0][2] is not False
    caught = [r for r in rows[1:] if r[2] is False]
    print("-" * 78)
    print(f"coplanar control not falsely flagged : {ok0}")
    print(f"detected by the baseline statistic   : {len(caught)}/{len(rows)-1}"
          f" (at extra tilt {[r[0] for r in caught]})")
    print()
    print("READ THIS AS A MAGNITUDE TABLE, NOT A PASS/FAIL GATE.")
    print("The measured error column is the point: a plane mismatch between "
          "the marker")
    print("card and the declaration panel biases every reading, and the bias "
          "grows")
    print("with the mismatch. The baseline-slope statistic tracks it "
          "monotonically but")
    print("its null distribution on genuinely coplanar frames overlaps the "
          "signal, so it")
    print("is a weak indicator, not a reliable detector. Coplanarity is an "
          "OPERATOR")
    print("requirement: lay the card flat against the same face as the "
          "printing.")
    # The control must not false-fire. Detection is reported, not required.
    return 0 if ok0 else 1


if __name__ == "__main__":
    sys.exit(main())
