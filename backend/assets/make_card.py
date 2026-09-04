"""Printable calibration card: ArUco marker of known size + verification ruler."""
import os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lmpc.calibration import MarkerSpec, generate_marker

DPI = 300
MM = DPI / 25.4
W, H = int(210 * MM), int(297 * MM)          # A4 portrait
MARKER_MM = 40.0

img = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(img)
F = "/System/Library/Fonts/Supplemental/Arial.ttf"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
h1 = ImageFont.truetype(FB, int(5.0 * MM))
h2 = ImageFont.truetype(FB, int(3.2 * MM))
bd = ImageFont.truetype(F, int(2.8 * MM))
sm = ImageFont.truetype(F, int(2.2 * MM))

y = int(15 * MM)
d.text((int(20 * MM), y), "COMPLIANCE SCANNER CALIBRATION CARD", font=h1, fill=0)
y += int(9 * MM)
d.text((int(20 * MM), y), "Legal Metrology (Packaged Commodities) Rules, 2011",
       font=bd, fill=0)
y += int(12 * MM)

# marker, drawn at exactly MARKER_MM
mk_px = int(round(MARKER_MM * MM))
mk = generate_marker(MarkerSpec(), 0, mk_px, 0)
img.paste(Image.fromarray(mk), (int(20 * MM), y))
d.rectangle([int(20 * MM) - 1, y - 1, int(20 * MM) + mk_px, y + mk_px],
            outline=0, width=1)
d.text((int(20 * MM) + mk_px + int(6 * MM), y),
       "ArUco DICT_4X4_50, id 0", font=h2, fill=0)
d.text((int(20 * MM) + mk_px + int(6 * MM), y + int(6 * MM)),
       f"Nominal side: {MARKER_MM:g} mm", font=bd, fill=0)
d.text((int(20 * MM) + mk_px + int(6 * MM), y + int(11 * MM)),
       "MEASURE IT AFTER PRINTING.", font=h2, fill=0)
d.text((int(20 * MM) + mk_px + int(6 * MM), y + int(16 * MM)),
       "Printers scale. Pass the measured", font=sm, fill=0)
d.text((int(20 * MM) + mk_px + int(6 * MM), y + int(20 * MM)),
       "value as --marker-mm, not 40.", font=sm, fill=0)
y += mk_px + int(14 * MM)

# verification ruler: 100 mm with 1 mm ticks
d.text((int(20 * MM), y), "Verification scale - check against a real ruler:",
       font=bd, fill=0)
y += int(7 * MM)
x0 = int(20 * MM)
for i in range(101):
    x = x0 + int(round(i * MM))
    ln = 8 if i % 10 == 0 else (5 if i % 5 == 0 else 3)
    d.line([x, y, x, y + int(ln * MM * 0.6)], fill=0, width=2 if i % 10 == 0 else 1)
    if i % 10 == 0:
        d.text((x - int(1.5 * MM), y + int(5.5 * MM)), str(i), font=sm, fill=0)
d.line([x0, y, x0 + int(100 * MM), y], fill=0, width=2)
y += int(14 * MM)

lines = [
    ("HOW TO USE", h2),
    ("1. Print this page at 100% scale (no 'fit to page', no scaling).", bd),
    ("2. Measure the printed black square edge-to-edge with a ruler or", bd),
    ("   calipers. Confirm the 100 mm scale above reads 100 mm.", bd),
    ("3. Lay the card FLAT AGAINST the label panel, in the SAME PLANE.", bd),
    ("   The card and the printed declarations must be coplanar - this is", bd),
    ("   what makes a photograph carry physical scale at all.", bd),
    ("4. Photograph both together, as square-on as practical.", bd),
    ("5. Scan, passing the measured side length:", bd),
    ("      python backend/lmpc/cli.py scan photo.jpg --marker-mm 40.0", sm),
    ("", sm),
    ("WHY A MARKER IS REQUIRED", h2),
    ("A photograph has no inherent physical scale: the same pixel height", bd),
    ("could be small print close up or large print far away. The marker is", bd),
    ("the only thing in frame with a known real-world size, so it is what", bd),
    ("converts pixels to millimetres. It is re-detected in EVERY frame, so", bd),
    ("moving or bumping the camera cannot silently invalidate the scale.", bd),
    ("", sm),
    ("SCOPE", h2),
    ("Flat / near-flat printed packaging only. Curved surfaces (bottles,", bd),
    ("jars, cans) break the coplanarity assumption. Embossed and debossed", bd),
    ("text is out of scope - it needs raking light to be visible at all.", bd),
]
for t, f in lines:
    d.text((int(20 * MM), y), t, font=f, fill=0)
    y += int((5.0 if f is h2 else 4.2) * MM)

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "calibration_card_A4_300dpi.png")
img.save(out_png, dpi=(DPI, DPI))
img.convert("RGB").save(out_png.replace(".png", ".pdf"), "PDF",
                        resolution=DPI)
print("wrote", out_png)
print("wrote", out_png.replace(".png", ".pdf"))
