"""Command-line entry points: single image, batch folder, and live camera."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmpc import db
from lmpc.calibration import MarkerSpec, generate_marker
from lmpc.pipeline import ScanConfig, scan_image
from lmpc.report import render_text
from lmpc.rules import PrintStyle

COLOR = {"PASS": (60, 190, 60), "BORDERLINE": (0, 190, 240),
         "FAIL": (50, 50, 235), "NOT_ASSESSED": (150, 150, 150)}
OVERALL_COLOR = {"COMPLIANT": (60, 190, 60), "REVIEW_REQUIRED": (0, 190, 240),
                 "NON_COMPLIANT": (50, 50, 235)}


def _cfg(a) -> ScanConfig:
    cfg = ScanConfig(marker=MarkerSpec(a.dict, a.marker_mm, a.marker_id),
                     print_style=PrintStyle(a.style),
                     panel_area_cm2=a.panel_area_cm2, lang=a.lang)
    if getattr(a, "fast", False):
        cfg.psms = (6,)
    return cfg


def cmd_scan(a):
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    rep = scan_image(img, _cfg(a))
    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(render_text(rep))
    if a.save:
        db.save_scan(db.connect(), rep, a.image)
        print(f"[saved to scan history as {rep['scan_id']}]")


def cmd_batch(a):
    files = [os.path.join(a.folder, f) for f in sorted(os.listdir(a.folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
    con = db.connect() if a.save else None
    rows = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        rep = scan_image(img, _cfg(a))
        if con:
            db.save_scan(con, rep, f)
        rows.append((os.path.basename(f), rep["overall"],
                     len(rep["violations"]), len(rep["borderline"])))
        print(f"{os.path.basename(f):40s} {rep['overall']:16s} "
              f"viol={len(rep['violations'])} border={len(rep['borderline'])}")
    print(f"\n{len(rows)} images scanned.")


def _overlay(frame, rep, fps):
    h, w = frame.shape[:2]
    c = OVERALL_COLOR.get(rep["overall"], (150, 150, 150))
    cv2.rectangle(frame, (0, 0), (w, 46), (30, 30, 30), -1)
    cv2.putText(frame, rep["overall"], (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, c, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} fps", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    y = 76
    cal = rep["calibration"]
    txt = (f"scale {cal['px_per_mm_at_marker']:.1f} px/mm  tilt "
           f"{cal['tilt_deg']:.0f}deg" if cal["ok"] else "NO MARKER IN FRAME")
    cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 1, cv2.LINE_AA)
    y += 30
    for d in rep["declarations"]:
        hh = d.get("height")
        if hh and hh.get("measured_mm") is not None:
            col = COLOR.get(hh["verdict"], (150, 150, 150))
            line = (f"{d['field']}: {hh['measured_mm']:.2f}mm / "
                    f"{hh['required_mm']:.1f}mm  {hh['verdict']}")
        elif hh:
            col = COLOR["NOT_ASSESSED"]
            line = f"{d['field']}: not measurable"
        else:
            col = COLOR["PASS"] if d["present"] else COLOR["FAIL"]
            line = f"{d['field']}: {'found' if d['present'] else 'MISSING'}"
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    col, 2, cv2.LINE_AA)
        y += 27
    cv2.putText(frame, "[space] freeze+save   [q] quit", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    return frame


def cmd_camera(a):
    cap = cv2.VideoCapture(a.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        sys.exit("cannot open camera")
    cfg = _cfg(a)
    cfg.psms = (6,)                      # live path favours latency
    con = db.connect()
    last, rep = 0.0, None
    fps = 0.0
    print("live scanning - keep the marker in frame, coplanar with the label")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if now - last > a.interval:
            try:
                rep = scan_image(frame, cfg)
            except Exception as e:
                print("scan error:", e)
            fps = 1.0 / max(1e-6, now - last)
            last = now
        if rep:
            frame = _overlay(frame, rep, fps)
        cv2.imshow("LMPC Compliance Scanner", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and rep:
            p = os.path.join(os.path.dirname(__file__), "..", "data", "images",
                             f"{rep['scan_id']}.jpg")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, frame)
            db.save_scan(con, rep, p)
            print(render_text(rep))
            print(f"[frozen and saved as {rep['scan_id']}]")
    cap.release(); cv2.destroyAllWindows()


def cmd_calibrate_physical(a):
    """Fit the measurement uncertainty from REAL photographs of the printed
    known-height sheet.

    This is the calibration the build plan asks for: samples of ruler-checked
    known height, photographed with the actual camera and marker, with the
    band set from the deviation actually observed rather than assumed. Until
    it is run, the shipped band comes from synthetic captures and says so.
    """
    import json as _json
    import numpy as _np
    from lmpc.calibration import calibrate
    from lmpc.measure import measure_field_height
    from lmpc.ocr import content_crop, offset_result, run_ocr_multi

    here = os.path.dirname(os.path.abspath(__file__))
    truth_path = a.truth or os.path.join(here, "..", "assets",
                                         "height_calibration_truth.json")
    truth = _json.load(open(truth_path))["rows_top_to_bottom"]
    expected = [t["true_height_mm"] for t in truth]

    files = [os.path.join(a.folder, f) for f in sorted(os.listdir(a.folder))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".bmp"))]
    if not files:
        sys.exit(f"no images in {a.folder}")

    errs, rows, used, skipped = [], [], 0, []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            skipped.append((f, "unreadable")); continue
        cal = calibrate(img, MarkerSpec(a.dict, a.marker_mm, a.marker_id))
        if not cal.ok:
            skipped.append((f, cal.reason[:60])); continue
        crop, ox, oy = content_crop(cal.rectified)
        ocr = offset_result(run_ocr_multi(crop, cal.rect_px_per_mm), ox, oy)

        # keep the digit rows, top to bottom
        cand = []
        for line in ocr.lines:
            digits = sum(c.isdigit() for c in line.text)
            if digits < 6 or digits < 0.6 * len(line.text.replace(" ", "")):
                continue
            cand.append(line)
        cand.sort(key=lambda l: l.bbox[1])
        if len(cand) != len(expected):
            skipped.append((f, f"found {len(cand)} digit rows, expected "
                               f"{len(expected)}"))
            continue

        for line, exp in zip(cand, expected):
            x, y, w, h = line.bbox
            chars = [c for c in ocr.chars
                     if c[1] < x + w and c[1] + c[3] > x
                     and c[2] < y + h and c[2] + c[4] > y]
            src = cal.local_px_per_mm(x + w / 2, y + h / 2)
            m = measure_field_height(cal.rectified, line.bbox, chars,
                                     px_per_mm=cal.rect_px_per_mm,
                                     source_px_per_mm=src)
            if not m.ok:
                continue
            e = m.height_mm - exp
            errs.append(e)
            rows.append({"file": os.path.basename(f), "true_mm": exp,
                         "measured_mm": round(m.height_mm, 4),
                         "error_mm": round(e, 4), "src_px_per_mm": round(src, 2),
                         "n_glyphs": m.n_glyphs, "tilt_deg": round(cal.tilt_deg, 1)})
        used += 1

    if len(errs) < 10:
        print(f"only {len(errs)} usable measurements from {used} photo(s) - "
              f"not enough to fit a band. Shoot more, varying distance and "
              f"angle.")
        for f, why in skipped:
            print(f"   skipped {os.path.basename(f)}: {why}")
        sys.exit(1)

    a_ = _np.array(errs)
    bias, sigma = float(a_.mean()), float(a_.std(ddof=1))
    p95 = float(_np.percentile(_np.abs(a_), 95))
    band = max(abs(bias) + a.k * sigma, p95)
    out = {
        "bias_mm": round(bias, 5), "sigma_mm": round(sigma, 5), "k": a.k,
        "band_override_mm": round(band, 5),
        "min_px_per_mm": a.min_px_per_mm, "max_tilt_deg": a.max_tilt_deg,
        "max_baseline_slope_deg": a.max_slope_deg,
        "apply_bias_correction": bool(a.apply_bias_correction),
        "source": (f"PHYSICAL calibration: {len(errs)} numeral measurements "
                   f"from {used} photograph(s) of the printed known-height "
                   f"sheet, marker {a.marker_mm:g} mm, "
                   f"{time.strftime('%Y-%m-%d')}"),
        "measurements": rows,
    }
    dest = a.out or os.path.join(here, "..", "data", "uncertainty.json")
    with open(dest, "w") as fh:
        _json.dump(out, fh, indent=2)
    print(f"{len(errs)} measurements from {used} photo(s)")
    print(f"  bias  {bias:+.4f} mm")
    print(f"  sigma {sigma:.4f} mm")
    print(f"  p95|e| {p95:.4f} mm")
    print(f"  -> band +/-{band:.4f} mm  (written to {dest})")
    if skipped:
        print(f"  {len(skipped)} photo(s) skipped:")
        for f, why in skipped[:10]:
            print(f"    {os.path.basename(f)}: {why}")


def cmd_marker(a):
    img = generate_marker(MarkerSpec(a.dict), a.marker_id)
    cv2.imwrite(a.out, img)
    print(f"wrote {a.out}\nPrint at 100% scale, then MEASURE the printed black "
          f"square with a ruler or calipers and pass that value as "
          f"--marker-mm. Do not trust the nominal size.")


def cmd_report(a):
    r = db.get_scan(db.connect(), a.scan_id)
    if not r:
        sys.exit("no such scan")
    print(render_text(r))


def main():
    ap = argparse.ArgumentParser("lmpc", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--marker-mm", type=float, default=40.0,
                       help="TRUE measured side length of the printed marker")
        p.add_argument("--dict", default="DICT_4X4_50")
        p.add_argument("--marker-id", type=int, default=None)
        p.add_argument("--style", default="normal",
                       choices=["normal", "embossed"])
        p.add_argument("--panel-area-cm2", type=float, default=None)
        p.add_argument("--lang", default="eng")
        p.add_argument("--fast", action="store_true")

    p = sub.add_parser("scan"); p.add_argument("image")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", action="store_true"); common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("batch"); p.add_argument("folder")
    p.add_argument("--save", action="store_true"); common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("camera")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--interval", type=float, default=0.7)
    common(p); p.set_defaults(func=cmd_camera)

    p = sub.add_parser("marker")
    p.add_argument("--out", default="marker.png")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--dict", default="DICT_4X4_50")
    p.set_defaults(func=cmd_marker)

    p = sub.add_parser("calibrate-physical",
                       help="fit the error band from photos of the printed "
                            "known-height sheet")
    p.add_argument("folder")
    p.add_argument("--truth", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--min-px-per-mm", type=float, default=12.0)
    p.add_argument("--max-tilt-deg", type=float, default=35.0)
    p.add_argument("--max-slope-deg", type=float, default=0.35)
    p.add_argument("--apply-bias-correction", action="store_true")
    common(p); p.set_defaults(func=cmd_calibrate_physical)

    p = sub.add_parser("report"); p.add_argument("scan_id")
    p.set_defaults(func=cmd_report)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
