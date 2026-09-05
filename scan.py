#!/usr/bin/env python3
"""
Drop photos in the inbox folder, run ./scan, get a plain-English result.

    ./scan                    scan every photo in the inbox folder
    ./scan photo.jpg          scan one file anywhere
    ./scan --marker-mm 39.4   tell it your printed card's size (once)

Scanned photos move to inbox/done/ with their result saved beside them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import cv2  # noqa: E402

from lmpc import db  # noqa: E402
from lmpc.calibration import MarkerSpec  # noqa: E402
from lmpc.friendly import render_friendly, render_one_liner  # noqa: E402
from lmpc.pipeline import ScanConfig, scan_image  # noqa: E402

CONFIG = os.path.join(ROOT, "scanner.json")
INBOX = os.path.join(ROOT, "inbox")
DONE = os.path.join(INBOX, "done")
IMAGES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")

SETUP = """
  ─────────────────────────────────────────────────────────────
   FIRST TIME ONLY

   To measure printing in millimetres I need something in the
   photo whose real size I know: the calibration card.

   1.  open backend/assets/calibration_card_A4_300dpi.pdf
       Print it at 100% scale — turn OFF "fit to page".

   2.  Measure the printed black square with a ruler.

   3.  Tell me that number once:

           ./scan --marker-mm 39.4

   I will remember it. After that, just drop photos in the
   inbox folder and run ./scan.

   I will not guess this number: every size I report is scaled
   by it, so a guess makes every answer wrong.
  ─────────────────────────────────────────────────────────────
"""


def read_image(path: str):
    """Load a photo. iPhone HEIC files go through macOS `sips` first."""
    if path.lower().endswith((".heic", ".heif")):
        out = os.path.join(tempfile.mkdtemp(), "c.jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", out],
                       capture_output=True)
        return cv2.imread(out) if os.path.exists(out) else None
    return cv2.imread(path)


def main() -> None:
    ap = argparse.ArgumentParser("scan")
    ap.add_argument("image", nargs="?", help="scan just this one file")
    ap.add_argument("--marker-mm", type=float,
                    help="measured size of your printed card; remembered")
    ap.add_argument("--details", action="store_true",
                    help="also print the full technical report")
    args = ap.parse_args()

    cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
    if args.marker_mm:
        cfg["marker_mm"] = args.marker_mm
        json.dump(cfg, open(CONFIG, "w"), indent=2)
        print(f"\n  Saved: your printed card is {args.marker_mm:g} mm across.")
    if not cfg.get("marker_mm"):
        print(SETUP)
        sys.exit(1)

    scfg = ScanConfig(marker=MarkerSpec(marker_length_mm=cfg["marker_mm"]))
    con = db.connect()

    if args.image:
        paths = [args.image]
        if not os.path.exists(args.image):
            sys.exit(f"  I cannot find {args.image}")
    else:
        os.makedirs(INBOX, exist_ok=True)
        paths = [os.path.join(INBOX, f) for f in sorted(os.listdir(INBOX))
                 if f.lower().endswith(IMAGES) and not f.startswith(".")]
        if not paths:
            print(f"\n  No photos to scan. Drop them in:\n    {INBOX}\n")
            return

    results = []
    for p in paths:
        name = os.path.basename(p)
        img = read_image(p)
        if img is None:
            print(f"\n  I could not open {name} as an image.\n")
            continue
        print(f"\n  reading {name} ...", flush=True)
        rep = scan_image(img, scfg)
        db.save_scan(con, rep, p)
        print(render_friendly(rep, filename=name, show_details=args.details))
        results.append((name, rep))

        if not args.image:          # inbox photos move out of the way
            os.makedirs(DONE, exist_ok=True)
            dest = os.path.join(DONE, name)
            stem, ext = os.path.splitext(name)
            n = 2
            while os.path.exists(dest):
                dest = os.path.join(DONE, f"{stem}_{n}{ext}")
                n += 1
            shutil.move(p, dest)
            with open(os.path.splitext(dest)[0] + "_result.txt", "w") as fh:
                fh.write(render_friendly(rep, filename=name, colour=False))

    if len(results) > 1:
        print("\n  Summary")
        for name, rep in results:
            print(render_one_liner(rep, name))
        print(f"\n  {len(results)} photo(s) scanned. They are now in "
              f"inbox/done/\n")


if __name__ == "__main__":
    main()
