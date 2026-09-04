"""
Printable known-height sheet for PHYSICAL calibration.

The build plan (Section 3.3) is explicit that the borderline band must come
from photographing samples of ruler-measured known height with the actual
camera and marker, and computing how far the pipeline's estimate deviates -
"not an assumed constant". This sheet is that artefact.

Each row prints a digit string whose numeral height is set from the font's
own outline metrics, so the true height is known exactly and independently of
anything the scanner does. Rows are ordered by increasing height, which is
also how `cli.py calibrate-physical` pairs measurements with truth.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lmpc.calibration import MarkerSpec, generate_marker

DPI = 600                       # high, so the printed glyph is faithful
MM = DPI / 25.4
W, H = int(210 * MM), int(297 * MM)
MARKER_MM = 40.0
TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
DIGITS = "0123456789"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def fit(target_mm):
    lo, hi, best, berr = 4, 8000, None, 1e18
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(FONT, mid)
        bb = f.getbbox(DIGITS)
        h = bb[3] - bb[1]
        if abs(h - target_mm * MM) < berr:
            best, berr = f, abs(h - target_mm * MM)
        if h < target_mm * MM:
            lo = mid + 1
        else:
            hi = mid - 1
    bb = best.getbbox(DIGITS)
    return best, (bb[3] - bb[1]) / MM


def main():
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    h1 = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                            int(5.0 * MM))
    lab = ImageFont.truetype(FONT, int(2.4 * MM))

    y = int(14 * MM)
    d.text((int(15 * MM), y), "NUMERAL-HEIGHT CALIBRATION SHEET", font=h1, fill=0)
    y += int(9 * MM)
    d.text((int(15 * MM), y),
           "Print at 100%. Photograph with your camera at several distances "
           "and angles.", font=lab, fill=0)
    y += int(4.5 * MM)
    d.text((int(15 * MM), y),
           "Then: python backend/lmpc/cli.py calibrate-physical <folder> "
           "--marker-mm <measured>", font=lab, fill=0)
    y += int(9 * MM)

    mk_px = int(round(MARKER_MM * MM))
    mk = generate_marker(MarkerSpec(), 0, mk_px, 0)
    img.paste(Image.fromarray(mk), (int(140 * MM), int(14 * MM)))
    d.text((int(140 * MM), int(14 * MM) + mk_px + int(2 * MM)),
           f"marker nominal {MARKER_MM:g} mm - MEASURE IT", font=lab, fill=0)

    truth = []
    for t in TARGETS:
        f, actual = fit(t)
        bb = f.getbbox(DIGITS)
        d.text((int(15 * MM), y - bb[1]), DIGITS, font=f, fill=0)
        d.text((int(120 * MM), y), f"row target {t:.1f} mm", font=lab, fill=0)
        d.text((int(120 * MM), y + int(3.2 * MM)),
               f"TRUE numeral height {actual:.4f} mm", font=lab, fill=0)
        truth.append({"target_mm": t, "true_height_mm": round(actual, 5),
                      "font": os.path.basename(FONT), "digits": DIGITS})
        y += int((bb[3] - bb[1]) + 7 * MM)

    d.text((int(15 * MM), y + int(4 * MM)),
           "Rows are ordered by increasing height; the calibrator pairs "
           "measurements with truth in that order.", font=lab, fill=0)
    d.text((int(15 * MM), y + int(9 * MM)),
           "Verify a row with calipers before trusting the fitted band.",
           font=lab, fill=0)

    here = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(here, "height_calibration_sheet_A4_600dpi.png")
    img.save(png, dpi=(DPI, DPI))
    img.convert("RGB").save(png.replace(".png", ".pdf"), "PDF", resolution=DPI)
    with open(os.path.join(here, "height_calibration_truth.json"), "w") as fh:
        json.dump({"dpi": DPI, "marker_nominal_mm": MARKER_MM,
                   "rows_top_to_bottom": truth}, fh, indent=2)
    print("wrote", png)
    print("wrote", png.replace(".png", ".pdf"))
    print("wrote", os.path.join(here, "height_calibration_truth.json"))
    for t in truth:
        print(f"   target {t['target_mm']:.1f} mm -> true "
              f"{t['true_height_mm']:.4f} mm")


if __name__ == "__main__":
    main()
