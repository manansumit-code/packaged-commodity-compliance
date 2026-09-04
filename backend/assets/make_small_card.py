"""Small printable ArUco marker for tight labels (cosmetic tubs, tiny packs).

Same marker (DICT_4X4_50, id 0) as the full A4 card, but sized to sit in a
gap only ~2.5 cm wide next to the text, instead of covering it.

Print at 100% scale (no 'fit to page'), then MEASURE the printed black
square with a ruler/calipers and pass that measured value as --marker-mm.
Do not trust the nominal 20 mm below - printers scale slightly.
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lmpc.calibration import MarkerSpec, generate_marker

DPI = 300
MM = DPI / 25.4
MARKER_MM = 20.0          # nominal - measure the print, don't trust this
MARGIN_MM = 4.0           # white quiet-zone border kept around the marker
LABEL_MM = 10.0           # strip below for a tiny id caption

mk_px = int(round(MARKER_MM * MM))
margin_px = int(round(MARGIN_MM * MM))
label_px = int(round(LABEL_MM * MM))

W = mk_px + 2 * margin_px
H = mk_px + 2 * margin_px + label_px

img = Image.new("L", (W, H), 255)
mk = generate_marker(MarkerSpec(), 0, mk_px, 0)
img.paste(Image.fromarray(mk), (margin_px, margin_px))

d = ImageDraw.Draw(img)
F = "/System/Library/Fonts/Supplemental/Arial.ttf"
sm = ImageFont.truetype(F, int(2.6 * MM))
d.text((margin_px, mk_px + 2 * margin_px - int(1 * MM)),
       f"id0  nominal {MARKER_MM:g}mm - MEASURE ME", font=sm, fill=0)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "small_marker_id0_20mm.png")
img.save(out_png, dpi=(DPI, DPI))
img.convert("RGB").save(out_png.replace(".png", ".pdf"), "PDF", resolution=DPI)
print("wrote", out_png)
print("wrote", out_png.replace(".png", ".pdf"))
