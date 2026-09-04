"""
Synthetic label generator with INDEPENDENT ground truth.

The single thing that makes an accuracy number here meaningful is that ground
truth and measurement do not share a model:

  * Ground truth is the FreeType outline extent of the numerals, taken from
    the font metrics of the clean master render (`font.getbbox`), divided by
    the master's known px/mm. It is what a caliper on the printed page would
    read. The pipeline never sees the master, and ground truth is never
    produced by the pipeline's Otsu / connected-component / 50%-crossing code.

  * The photo the pipeline sees goes through an imaging chain the estimator
    does not model: ink gain, 3-D perspective, area-averaged downsampling to a
    realistic capture resolution, defocus blur, a non-uniform light field with
    a specular highlight, sensor noise, and JPEG.

Anything less than that produces a number like "99.8% accurate" that measures
PIL against OpenCV rather than measuring the architecture.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MASTER_PPMM = 40.0            # 40 px/mm ~= 1016 dpi master render

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]
FONTS = [f for f in FONTS if __import__("os").path.exists(f)]

DIGIT_SAMPLE = "0123456789"


def load_font(path: str, size_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size_px)


def digit_height_px(font: ImageFont.FreeTypeFont) -> float:
    """Numeral height straight from the font outline - the ground-truth ruler.

    Independent of any raster, threshold, or OpenCV operation. Verified
    against a 400 px high-resolution raster on four fonts: the outline extent
    and the physical ink extent agree to 0 px, so this is a real ruler and not
    a convention.
    """
    x0, y0, x1, y1 = font.getbbox(DIGIT_SAMPLE)
    return float(y1 - y0)


def printed_digit_heights(font: ImageFont.FreeTypeFont,
                          text: str) -> tuple[float, float]:
    """(median, max) outline height of the numerals ACTUALLY printed, in px.

    Ground truth must describe the same physical quantity the estimator
    reports. The estimator reports the median height of the numerals it can
    see, so the median over the printed digits is the matching truth. The max
    is carried alongside because round digits (0 3 6 8 9) overshoot the cap
    line by ~2%, and that definitional gap should be visible, not buried.
    """
    ds = [c for c in text if c in "0123456789"]
    if not ds:
        return digit_height_px(font), digit_height_px(font)
    hs = []
    for c in sorted(set(ds)):
        x0, y0, x1, y1 = font.getbbox(c)
        hs.extend([float(y1 - y0)] * ds.count(c))
    return float(np.median(hs)), float(max(hs))


def fit_font(path: str, target_mm: float,
             ppmm: float = MASTER_PPMM) -> tuple[ImageFont.FreeTypeFont, float]:
    """Smallest-error font size whose numeral height is `target_mm`."""
    target_px = target_mm * ppmm
    lo, hi = 4, 4000
    best, best_err = None, 1e18
    while lo <= hi:
        mid = (lo + hi) // 2
        f = load_font(path, mid)
        h = digit_height_px(f)
        err = abs(h - target_px)
        if err < best_err:
            best, best_err = f, err
        if h < target_px:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, digit_height_px(best) / ppmm


# --------------------------------------------------------------------------
BRANDS = ["SURYA FOODS", "GANGA AGRO", "NILGIRI MILLS", "AMRIT PACKAGED",
          "SHREE TRADERS", "KAVERI NATURALS", "DECCAN SPICES", "RATNA PURE"]
GENERIC = ["Refined Sunflower Oil", "Toor Dal", "Wheat Flour", "Basmati Rice",
           "Turmeric Powder", "Iodised Salt", "Roasted Peanuts", "Tea Leaves"]
CITIES = [("Pune", "411045"), ("Indore", "452010"), ("Guwahati", "781005"),
          ("Kochi", "682024"), ("Ludhiana", "141003"), ("Nashik", "422009")]

MRP_KEY_VARIANTS = ["MRP", "M.R.P.", "M.R.P", "Maximum Retail Price",
                    "Retail Sale Price", "MRP."]
QTY_KEY_VARIANTS = ["Net Qty.", "Net Quantity", "Net Wt.", "Net Weight",
                    "Net Content", "Net Qty"]
MFG_KEY_VARIANTS = ["Mfg. Date", "Date of Mfg.", "Mfd. on", "Packed on",
                    "Month & Year of Manufacture", "MFD"]
MAKER_KEY_VARIANTS = ["Manufactured by", "Mfd. by", "Packed by",
                      "Marketed by", "Manufactured & Packed by"]
CARE_KEY_VARIANTS = ["Customer Care", "Consumer Care", "For complaints contact",
                     "Customer Care Details", "Consumer complaints"]
QUANTITIES = [(50, "g"), (100, "g"), (200, "g"), (250, "g"), (500, "g"),
              (1, "kg"), (2, "kg"), (200, "ml"), (500, "ml"), (1, "L"),
              (750, "ml"), (150, "g"), (900, "g")]


@dataclass
class LabelSpec:
    seed: int = 0
    brand: str = ""
    generic: str = ""
    qty_value: float = 500
    qty_unit: str = "g"
    qty_key: str = "Net Qty."
    mrp: float = 199.0
    mrp_key: str = "MRP"
    mrp_tax_phrase: bool = True
    mfg_month: int = 3
    mfg_year: int = 2025
    mfg_key: str = "Mfg. Date"
    maker_key: str = "Manufactured by"
    care_key: str = "Customer Care"
    font_path: str = FONTS[0]
    body_font_path: str = FONTS[0]
    qty_height_mm: float = 2.4          # target numeral height, net quantity
    mrp_height_mm: float = 1.6
    mfg_height_mm: float = 1.6
    body_height_mm: float = 1.4
    omit: tuple[str, ...] = ()          # declarations deliberately removed
    distractors: bool = True            # rival prices/dates/batch numbers
    panel_w_mm: float = 100.0
    panel_h_mm: float = 118.0
    marker_mm: float = 40.0


def random_spec(rng: random.Random, qty_height_mm: Optional[float] = None,
                mrp_height_mm: Optional[float] = None,
                omit: tuple[str, ...] = ()) -> LabelSpec:
    qv, qu = rng.choice(QUANTITIES)
    city, pin = rng.choice(CITIES)
    fp = rng.choice(FONTS)
    return LabelSpec(
        seed=rng.randrange(10 ** 9),
        brand=rng.choice(BRANDS), generic=rng.choice(GENERIC),
        qty_value=qv, qty_unit=qu, qty_key=rng.choice(QTY_KEY_VARIANTS),
        mrp=round(rng.uniform(15, 999), rng.choice([0, 2])),
        mrp_key=rng.choice(MRP_KEY_VARIANTS),
        mrp_tax_phrase=rng.random() < 0.85,
        mfg_month=rng.randint(1, 12), mfg_year=rng.randint(2023, 2026),
        mfg_key=rng.choice(MFG_KEY_VARIANTS),
        maker_key=rng.choice(MAKER_KEY_VARIANTS),
        care_key=rng.choice(CARE_KEY_VARIANTS),
        font_path=fp, body_font_path=rng.choice(FONTS),
        qty_height_mm=qty_height_mm if qty_height_mm is not None
        else rng.uniform(0.8, 6.0),
        mrp_height_mm=mrp_height_mm if mrp_height_mm is not None
        else rng.uniform(1.0, 4.0),
        mfg_height_mm=rng.uniform(1.2, 2.5),
        body_height_mm=rng.uniform(1.2, 1.8),
        omit=omit,
        distractors=rng.random() < 0.75,
    )


def _city_for(seed: int):
    return CITIES[seed % len(CITIES)]


def render_master(spec: LabelSpec) -> tuple[np.ndarray, dict]:
    """Clean master render + ground truth. The pipeline must never see this."""
    ppmm = MASTER_PPMM
    W = int((spec.panel_w_mm + spec.marker_mm + 30) * ppmm)
    H = int((max(spec.panel_h_mm, spec.marker_mm + 20) + 12) * ppmm)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    # --- reference marker, coplanar with the panel by construction ---------
    import cv2 as _cv2
    dic = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_4X4_50)
    mk = _cv2.aruco.generateImageMarker(dic, 0, int(spec.marker_mm * ppmm))
    mx = int((spec.panel_w_mm + 16) * ppmm)
    my = int(20 * ppmm)
    img.paste(Image.fromarray(mk), (mx, my))

    d.rectangle([int(4 * ppmm), int(4 * ppmm),
                 int((spec.panel_w_mm + 4) * ppmm),
                 int((spec.panel_h_mm + 4) * ppmm)], outline=0, width=2)

    city, pin = _city_for(spec.seed)
    body_f, body_mm = fit_font(spec.body_font_path, spec.body_height_mm, ppmm)
    brand_f, _ = fit_font(spec.font_path, 5.5, ppmm)
    qty_f, qty_gt = fit_font(spec.font_path, spec.qty_height_mm, ppmm)
    mrp_f, mrp_gt = fit_font(spec.font_path, spec.mrp_height_mm, ppmm)
    mfg_f, mfg_gt = fit_font(spec.body_font_path, spec.mfg_height_mm, ppmm)

    qv = spec.qty_value
    qty_num = f"{qv:g}"
    qty_text = f"{spec.qty_key}: {qty_num} {spec.qty_unit}"
    mrp_num = f"{spec.mrp:.2f}" if spec.mrp != int(spec.mrp) else f"{int(spec.mrp)}"
    mrp_text = f"{spec.mrp_key}: Rs. {mrp_num}"
    if spec.mrp_tax_phrase:
        mrp_text += " (inclusive of all taxes)"
    mfg_text = f"{spec.mfg_key}: {spec.mfg_month:02d}/{spec.mfg_year}"

    x = 8 * ppmm
    y = 9 * ppmm

    def line(txt, f, gap_mm=2.2):
        nonlocal y
        d.text((x, y), txt, font=f, fill=0)
        bb = f.getbbox(txt)
        y += (bb[3] - bb[1]) + gap_mm * ppmm

    line(spec.brand, brand_f, 3.0)
    if "commodity_name" not in spec.omit:
        line(f"Common name: {spec.generic}", body_f)
    if "net_quantity" not in spec.omit:
        line(qty_text, qty_f, 3.0)
    if "mrp" not in spec.omit:
        line(mrp_text, mrp_f, 3.0)
    if "mfg_date" not in spec.omit:
        line(mfg_text, mfg_f)
    if "manufacturer" not in spec.omit:
        line(f"{spec.maker_key}: {spec.brand} Pvt. Ltd.", body_f, 0.8)
        line(f"Plot 14, MIDC Industrial Area, {city} - {pin}, India",
             body_f, 2.2)
    if "consumer_care" not in spec.omit:
        line(f"{spec.care_key}: 1800 266 {1000 + spec.seed % 8999}",
             body_f, 0.8)
        line(f"care@{spec.brand.split()[0].lower()}.co.in", body_f, 2.2)
    # Distractors. A real pack carries other numbers that look like the ones
    # being extracted: a best-before date, a batch/lot code, and frequently a
    # second price (a promotional or per-unit price). Without them the
    # value-anchored fallbacks in fields.py cannot be shown to have any
    # precision at all - there is nothing on the label for them to be wrong
    # about.
    if spec.distractors:
        bb_m = (spec.mfg_month % 12) + 1
        bb_y = spec.mfg_year + (1 if bb_m <= spec.mfg_month else 0)
        line(f"Best Before: {bb_m:02d}/{bb_y}   Use by end of that month",
             body_f, 0.8)
        line(f"Batch/Lot No.: B{spec.seed % 99999:05d}  "
             f"Expiry: {bb_m:02d}/{bb_y + 1}", body_f, 0.8)
        line(f"Special introductory price Rs. {max(5, spec.mrp * 0.8):.2f}  "
             f"| Rs. {max(1, spec.mrp / 5):.2f} per 100 g", body_f, 2.2)

    line("Ingredients: as declared on pack. Store in a cool dry place.",
         body_f)
    line("FSSAI Lic. No. 100" + f"{10000000 + spec.seed % 8999999}", body_f)

    # Ground truth: outline height of the numerals actually printed.
    qty_h = printed_digit_heights(qty_f, qty_num)
    mrp_h = printed_digit_heights(mrp_f, mrp_num)
    mfg_h = printed_digit_heights(mfg_f, f"{spec.mfg_month:02d}{spec.mfg_year}")

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)

    gt = {
        "seed": spec.seed,
        "master_px_per_mm": ppmm,
        "marker_mm": spec.marker_mm,
        "fields": {
            "net_quantity": {
                "present": "net_quantity" not in spec.omit,
                "text": f"{qty_num} {spec.qty_unit}",
                "value": qv, "unit": spec.qty_unit,
                "height_mm": qty_h[0] / ppmm, "height_max_mm": qty_h[1] / ppmm,
                "nominal_mm": qty_gt,
            },
            "mrp": {
                "present": "mrp" not in spec.omit,
                "text": mrp_num, "value": float(mrp_num),
                "height_mm": mrp_h[0] / ppmm, "height_max_mm": mrp_h[1] / ppmm,
                "nominal_mm": mrp_gt,
            },
            "mfg_date": {
                "present": "mfg_date" not in spec.omit,
                "text": f"{spec.mfg_month:02d}/{spec.mfg_year}",
                "month": spec.mfg_month, "year": spec.mfg_year,
                "height_mm": mfg_h[0] / ppmm, "height_max_mm": mfg_h[1] / ppmm,
                "nominal_mm": mfg_gt,
            },
            "manufacturer": {"present": "manufacturer" not in spec.omit},
            "consumer_care": {"present": "consumer_care" not in spec.omit},
            "commodity_name": {"present": "commodity_name" not in spec.omit,
                               "text": spec.generic},
        },
        "omitted": list(spec.omit),
        "distractors": spec.distractors,
        "mrp_tax_phrase": spec.mrp_tax_phrase,
        "font": spec.font_path.split("/")[-1],
    }
    return bgr, gt


# --------------------------------------------------------------------------
@dataclass
class Degradation:
    """One realistic capture. Every knob models a real imaging effect."""
    target_ppmm: float = 12.0      # capture resolution at the object plane
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    roll_deg: float = 0.0
    ink_gain: int = 0              # -1 erode, 0 none, +1 dilate (print/PSF)
    blur_sigma_px: float = 0.6     # defocus / lens MTF
    light_strength: float = 0.25   # non-uniform illumination
    glare: float = 0.0             # specular highlight strength
    noise_sigma: float = 2.0       # sensor read noise
    jpeg_q: int = 88

    @staticmethod
    def sample(rng: random.Random, tier: str = "realistic") -> "Degradation":
        if tier == "clean":
            return Degradation(target_ppmm=rng.uniform(18, 26),
                               tilt_x_deg=0, tilt_y_deg=0, roll_deg=0,
                               ink_gain=0, blur_sigma_px=0.0,
                               light_strength=0.0, glare=0.0,
                               noise_sigma=0.0, jpeg_q=97)
        if tier == "harsh":
            return Degradation(
                target_ppmm=rng.uniform(5, 11),
                tilt_x_deg=rng.uniform(-32, 32), tilt_y_deg=rng.uniform(-32, 32),
                roll_deg=rng.uniform(-12, 12),
                ink_gain=rng.choice([-1, 0, 1, 1]),
                blur_sigma_px=rng.uniform(0.6, 1.6),
                light_strength=rng.uniform(0.25, 0.55),
                glare=rng.uniform(0.0, 0.6),
                noise_sigma=rng.uniform(2.0, 7.0),
                jpeg_q=rng.randint(55, 80))
        return Degradation(                                   # "realistic"
            target_ppmm=rng.uniform(9, 20),
            tilt_x_deg=rng.uniform(-20, 20), tilt_y_deg=rng.uniform(-20, 20),
            roll_deg=rng.uniform(-8, 8),
            ink_gain=rng.choice([-1, 0, 0, 1]),
            blur_sigma_px=rng.uniform(0.3, 1.1),
            light_strength=rng.uniform(0.1, 0.4),
            glare=rng.uniform(0.0, 0.35),
            noise_sigma=rng.uniform(0.5, 4.0),
            jpeg_q=rng.randint(70, 92))


def _plane_homography(deg: Degradation, w_mm: float, h_mm: float):
    """Homography mm-plane -> photo pixels for a pinhole camera viewing the
    plane at the requested obliquity and resolution."""
    ax, ay, az = (math.radians(deg.tilt_x_deg), math.radians(deg.tilt_y_deg),
                  math.radians(deg.roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    d = 500.0                                   # mm from camera to plane
    f = deg.target_ppmm * d                     # focal length in px
    K = np.array([[f, 0, 0.0], [0, f, 0.0], [0, 0, 1.0]])
    # Centre the plane before rotating so tilt pivots about the label.
    C = np.array([[1, 0, -w_mm / 2], [0, 1, -h_mm / 2], [0, 0, 1]])
    T = np.column_stack([R[:, 0], R[:, 1], np.array([0, 0, d])])
    return K @ T @ C


def degrade(master_bgr: np.ndarray, deg: Degradation,
            rng: random.Random) -> np.ndarray:
    ppmm = MASTER_PPMM
    mh, mw = master_bgr.shape[:2]
    w_mm, h_mm = mw / ppmm, mh / ppmm

    img = master_bgr
    # 1. ink gain - print/PSF spread the estimator does not model
    if deg.ink_gain:
        k = np.ones((2, 2), np.uint8)
        img = (cv2.erode(img, k) if deg.ink_gain > 0 else cv2.dilate(img, k))

    # 2. area-averaged downsample to capture resolution (correct antialiasing)
    s = min(1.0, deg.target_ppmm / ppmm * 1.25)
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    eff_ppmm = ppmm * s

    # 3. perspective
    H = _plane_homography(deg, w_mm, h_mm)
    M = H @ np.diag([1.0 / eff_ppmm, 1.0 / eff_ppmm, 1.0])
    ih, iw = img.shape[:2]
    corners = np.array([[0, 0, 1], [iw, 0, 1], [iw, ih, 1], [0, ih, 1]], float)
    p = corners @ M.T
    p = p[:, :2] / p[:, 2:3]
    x0, y0 = p[:, 0].min() - 25, p[:, 1].min() - 25
    x1, y1 = p[:, 0].max() + 25, p[:, 1].max() + 25
    Tr = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    out_w, out_h = int(min(6000, x1 - x0)), int(min(6000, y1 - y0))
    img = cv2.warpPerspective(img, Tr @ M, (out_w, out_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(235, 235, 235))

    f = img.astype(np.float32)
    h, w = f.shape[:2]

    # 4. non-uniform illumination + specular highlight
    if deg.light_strength > 0 or deg.glare > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        light = 1.0 + deg.light_strength * (
            gx * (xx / w - 0.5) + gy * (yy / h - 0.5)) * 2.0
        f *= light[:, :, None]
        if deg.glare > 0:
            cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
            r = rng.uniform(0.12, 0.3) * min(h, w)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
            f += (deg.glare * 190.0 * blob)[:, :, None]

    # 5. defocus
    if deg.blur_sigma_px > 0:
        f = cv2.GaussianBlur(f, (0, 0), deg.blur_sigma_px)

    # 6. sensor noise
    if deg.noise_sigma > 0:
        f += np.random.normal(0, deg.noise_sigma, f.shape).astype(np.float32)

    img = np.clip(f, 0, 255).astype(np.uint8)

    # 7. JPEG
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), deg.jpeg_q])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def make_sample(rng: random.Random, tier: str = "realistic",
                spec: Optional[LabelSpec] = None,
                deg: Optional[Degradation] = None):
    spec = spec or random_spec(rng)
    master, gt = render_master(spec)
    deg = deg or Degradation.sample(rng, tier)
    photo = degrade(master, deg, rng)
    gt = dict(gt)
    gt["degradation"] = deg.__dict__.copy()
    gt["tier"] = tier
    return photo, gt, spec, deg
