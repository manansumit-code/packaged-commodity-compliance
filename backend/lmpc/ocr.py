"""
OCR front-end.

Runs entirely locally (Tesseract, offline) - no network call sits anywhere on
the scanning path, per build plan Section 7's "live demo network dependency".

Two things this module is careful about:

* It up-samples ONLY for recognition. Every coordinate it returns is mapped
  back to the rectified image's own pixel grid, because that grid is the one
  with the known px/mm. Measurement never sees the up-sampled image.
* It returns character boxes alongside word boxes. Character boxes are what
  let the measurement stage keep the digits and drop the currency symbol and
  the decimal point.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pytesseract

for _p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract",
           "/usr/bin/tesseract"):
    if os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        break
else:
    _found = shutil.which("tesseract")
    if _found:
        pytesseract.pytesseract.tesseract_cmd = _found


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    line_key: tuple


@dataclass
class Line:
    text: str
    words: list[Word]
    spans: list[tuple[int, int]]     # char span of each word within `text`

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x0 = min(w.x for w in self.words); y0 = min(w.y for w in self.words)
        x1 = max(w.x + w.w for w in self.words)
        y1 = max(w.y + w.h for w in self.words)
        return x0, y0, x1 - x0, y1 - y0

    def words_for_span(self, a: int, b: int) -> list[Word]:
        return [w for w, (s, e) in zip(self.words, self.spans)
                if s < b and e > a]


@dataclass
class OcrResult:
    lines: list[Line]
    chars: list[tuple[str, int, int, int, int]]   # (char, x, y, w, h)
    full_text: str
    scale: float
    engine: str = "tesseract"
    mean_conf: float = 0.0


def rebuild_lines(words: list[Word]) -> list[Line]:
    """Reconstruct printed lines geometrically from word boxes.

    Tesseract's own line grouping is unreliable on labels: in sparse-text mode
    it will happily split "Month & Year of Manufacture: 09/2023" into two
    separate "lines", which destroys the key-to-value association that field
    classification depends on. Because the image has already been rectified to
    a fronto-parallel plane, printed lines really are horizontal, so grouping
    by vertical centre is both simple and correct.
    """
    if not words:
        return []
    ws = sorted(words, key=lambda w: w.y + w.h / 2.0)
    groups: list[list[Word]] = []
    for w in ws:
        cy = w.y + w.h / 2.0
        if groups:
            g = groups[-1]
            g_cy = sum(x.y + x.h / 2.0 for x in g) / len(g)
            g_h = float(np.median([x.h for x in g]))
            if abs(cy - g_cy) < 0.6 * max(g_h, w.h):
                g.append(w)
                continue
        groups.append([w])

    lines: list[Line] = []
    for g in groups:
        g.sort(key=lambda w: w.x)
        # A very large horizontal gap means two unrelated columns that merely
        # happen to sit at the same height - split them.
        med_h = float(np.median([w.h for w in g]))
        runs, cur = [], [g[0]]
        for prev, w in zip(g, g[1:]):
            if w.x - (prev.x + prev.w) > 6.0 * med_h:
                runs.append(cur); cur = [w]
            else:
                cur.append(w)
        runs.append(cur)
        for run in runs:
            parts, spans, pos = [], [], 0
            for w in run:
                spans.append((pos, pos + len(w.text)))
                parts.append(w.text)
                pos += len(w.text) + 1
            lines.append(Line(" ".join(parts), run, spans))
    lines.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
    return lines


def _preprocess(bgr: np.ndarray, scale: float, variant: str = "clahe"
                ) -> np.ndarray:
    """One grayscale rendering of the panel, up-sampled for recognition.

    Every variant is produced at the SAME `scale`, which matters more than it
    looks: word coordinates are divided by the scale to get back to the
    rectified pixel grid, and that grid is the one with the known px/mm. If
    two variants were up-sampled differently, merging their word boxes would
    put an ROI in the wrong place and the glyph measurer would silently
    measure whatever happened to be there.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    if variant == "plain":
        # No enhancement at all. On a clean, evenly-lit label CLAHE can amplify
        # paper grain into speckle that breaks thin strokes, so the untouched
        # image is kept in the sweep as its own candidate.
        out = gray
    elif variant == "invert":
        # Reversed-out text - a white wordmark in a black box, white on a
        # coloured band - is a hole in the mask for every dark-on-light
        # assumption downstream. Tesseract itself expects dark on light, so
        # the only way to read those declarations is to hand it the negative.
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        out = cv2.bitwise_not(out)
    elif variant == "sharp":
        # Unsharp masking against the softening that phone denoise and JPEG
        # both apply. It recovers small type that is present but smeared -
        # which is the failure mode on a re-compressed messaging-app photo.
        base = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(base, (0, 0), 1.2)
        out = cv2.addWeighted(base, 1.7, blur, -0.7, 0)
    else:  # "clahe"
        # Flattens the local contrast loss caused by glare and by a
        # non-uniform light source across the panel.
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return out


OCR_VARIANTS = ("clahe", "plain", "sharp", "invert")


def content_crop(bgr: np.ndarray, pad: int = 24,
                 min_frac: float = 0.02) -> tuple[np.ndarray, int, int]:
    """Trim the rectified canvas to the region that actually contains ink.

    Rectifying into a metric plane produces a canvas sized by the visible
    extent of that plane, most of which is usually blank. OCR cost scales with
    canvas area, so trimming it is the difference between a per-frame budget
    that supports live video and one that does not.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape[:2]
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    _, m = cv2.threshold(cv2.GaussianBlur(small, (5, 5), 0), 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(m)
    if xs.size < small.size * 0.0005:
        return bgr, 0, 0
    x0 = max(0, int(xs.min() * 4) - pad); y0 = max(0, int(ys.min() * 4) - pad)
    x1 = min(w, int(xs.max() * 4) + pad); y1 = min(h, int(ys.max() * 4) + pad)
    if (x1 - x0) * (y1 - y0) < min_frac * w * h:
        return bgr, 0, 0
    return bgr[y0:y1, x0:x1], x0, y0


def offset_result(r: "OcrResult", ox: int, oy: int) -> "OcrResult":
    if ox == 0 and oy == 0:
        return r
    for line in r.lines:
        for w in line.words:
            w.x += ox; w.y += oy
    r.chars = [(c, x + ox, y + oy, w, h) for c, x, y, w, h in r.chars]
    return r


# Below this, up-sampling is not worth doing. Resampling the whole panel by a
# few percent costs a round of cubic interpolation - which softens strokes and
# shifts the noise pattern - and buys almost no extra detail, because the
# detail was never in the file. Worse, it makes the result unstable: a frame
# already near the target resolution would be resampled by 1.0075x or 1.0100x
# depending on a 0.25% difference in the declared marker size, and those two
# renderings are different enough that Tesseract ranks the preprocessing
# variants differently and reads different text off the same label.
SCALE_DEADBAND = 1.05


def choose_scale(rect_px_per_mm: float, target_px_per_mm: float = 28.0,
                 cap: float = 5.0) -> float:
    if rect_px_per_mm <= 0:
        return 1.0
    s = float(np.clip(target_px_per_mm / rect_px_per_mm, 1.0, cap))
    return 1.0 if s < SCALE_DEADBAND else s


def run_ocr(rect_bgr: np.ndarray, rect_px_per_mm: float = 0.0,
            lang: str = "eng", psm: int = 6,
            scale: Optional[float] = None,
            variant: str = "clahe") -> OcrResult:
    """OCR a rectified image; all output coordinates are in rectified pixels."""
    s = scale if scale is not None else choose_scale(rect_px_per_mm)
    img = _preprocess(rect_bgr, s, variant)
    # Tesseract collapses runs of spaces by default, which destroys the gap
    # between a key and its value on a label laid out in columns
    # ("Type of Ruling :        Single Line").
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"

    data = pytesseract.image_to_data(img, lang=lang, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    words: list[Word] = []
    confs: list[float] = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        confs.append(conf)
        words.append(Word(
            txt,
            int(round(data["left"][i] / s)), int(round(data["top"][i] / s)),
            int(round(data["width"][i] / s)), int(round(data["height"][i] / s)),
            conf,
            (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        ))

    lines = rebuild_lines(words)

    # Character boxes (Tesseract's box format is bottom-left origin).
    chars: list[tuple[str, int, int, int, int]] = []
    try:
        raw = pytesseract.image_to_boxes(img, lang=lang, config=cfg)
        ih = img.shape[0]
        for ln in raw.splitlines():
            parts = ln.split(" ")
            if len(parts) < 5:
                continue
            c, bx1, by1, bx2, by2 = parts[0], *map(int, parts[1:5])
            top = ih - by2
            chars.append((c, int(round(bx1 / s)), int(round(top / s)),
                          int(round((bx2 - bx1) / s)),
                          int(round((by2 - by1) / s))))
    except Exception:
        chars = []

    return OcrResult(lines, chars, "\n".join(l.text for l in lines), s,
                     mean_conf=float(np.mean(confs)) if confs else 0.0)


# A pass is judged only on text it was actually sure of, and only on text long
# enough to be a word. Both conditions are load-bearing.
#
# Scoring on raw word count makes an over-sharpened rendering "win":
# sharpening a photo of a textured table top manufactures hundreds of one- and
# two-character detections, which outnumber the real declarations. Raising the
# confidence floor alone does not fix it, because sharpening also RAISES the
# confidence of those fragments - it increases local contrast, which is most
# of what the confidence estimate responds to. Requiring three characters is
# what actually separates a word from a speck.
#
# Measured on two labels at two marginally different scales: at conf>=50 with
# no length rule, the winner flipped to the sharpened pass on a 0.25% input
# change and the declarations went from "302 Pages"/"170.00" to "1M"/"1". At
# conf>=70 with a 3-character minimum the same four passes rank stably.
SCORE_MIN_CONF = 70.0
SCORE_MIN_CHARS = 3
# Below this the fallback score is used: on a genuinely marginal frame no pass
# clears conf 70 at all, and every score would be zero.
FALLBACK_MIN_CONF = 50.0
# A challenger must beat the incumbent by this much to displace it. Passes
# often score within a point or two of each other, and without a margin the
# choice - and with it the character boxes the measurement stage uses - turns
# on noise.
SCORE_MARGIN = 1.08
# Words admitted from a NON-winning pass have to clear a higher bar still,
# because they are being added on the strength of a single reading with
# nothing to corroborate them.
MERGE_MIN_CONF = 60.0


def _score(r: "OcrResult", min_conf: float = SCORE_MIN_CONF) -> float:
    """How much real text a pass produced that it was confident about."""
    return float(sum(len(w.text) for l in r.lines for w in l.words
                     if w.conf >= min_conf and len(w.text) >= SCORE_MIN_CHARS))


def _iou(a: Word, b: Word) -> float:
    ix0 = max(a.x, b.x); iy0 = max(a.y, b.y)
    ix1 = min(a.x + a.w, b.x + b.w); iy1 = min(a.y + a.h, b.y + b.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _covered(w: Word, k: Word) -> float:
    """Fraction of w's area that lies inside k."""
    ix0 = max(w.x, k.x); iy0 = max(w.y, k.y)
    ix1 = min(w.x + w.w, k.x + k.w); iy1 = min(w.y + w.h, k.y + k.h)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / float(max(1, w.w * w.h))


def _merge_words(primary: list[Word], extra: list[list[Word]],
                 iou_thresh: float = 0.3,
                 cover_thresh: float = 0.6) -> list[Word]:
    """Add to the winning pass only what the others found that it missed.

    This is deliberately additive rather than a symmetric union. The winning
    pass is the one the rest of the pipeline was calibrated on, so it stays
    authoritative over every box it claimed; the other passes contribute only
    where they read confident text on ink the winner left blank - a wordmark
    reversed out of a black box, a declaration marooned beside a barcode that
    the block-layout mode ran together with its neighbour.

    A symmetric union is the wrong shape here for a specific reason: when two
    passes disagree about the same ink, "higher confidence wins" is a coin
    flip between two readings of one word, and taking it per-word shreds a
    line into alternating fragments from different passes. Overlap is tested
    by IoU and, because passes often box the same word at different tightness,
    also by simple containment.
    """
    kept = list(primary)
    for words in extra:
        for w in words:
            if w.conf < MERGE_MIN_CONF:
                continue
            if any(_iou(w, k) >= iou_thresh or _covered(w, k) >= cover_thresh
                   for k in kept):
                continue
            kept.append(w)
    return kept


def run_ocr_multi(rect_bgr: np.ndarray, rect_px_per_mm: float = 0.0,
                  lang: str = "eng", psms: tuple[int, ...] = (6, 11)
                  ) -> OcrResult:
    """Sweep preprocessing variants and page-segmentation modes, then MERGE.

    Two independent things defeat a single OCR pass on a real label:

      * Layout. A label mixes a dense address block with declarations
        scattered around a barcode and a logo. Tesseract's PSM 6 assumes one
        uniform block and runs unrelated columns together; PSM 11 finds the
        scattered text but gives up on the block. Neither is right for the
        whole panel.
      * Rendering. Glare wants CLAHE, clean print is hurt by it, a
        re-compressed photo wants sharpening, and a white-on-black wordmark
        is invisible to all three until the image is inverted.

    The old code ran a couple of PSMs and kept whichever scored highest,
    which threw away everything the losing pass had found - including, on
    this label, every reversed-out declaration. This version keeps the union.

    Cost is contained by sweeping variants at one PSM first, then spending
    the remaining PSMs only on the variant that won: 4 + 2 passes rather
    than 4 x 3. Every pass shares one up-sampling scale, so all coordinates
    land on the same rectified pixel grid and are safe to merge.
    """
    scale = choose_scale(rect_px_per_mm)
    base_psm = psms[0] if psms else 6

    results: list[tuple[str, OcrResult]] = []
    for variant in OCR_VARIANTS:
        try:
            results.append((variant, run_ocr(rect_bgr, rect_px_per_mm,
                                             lang=lang, psm=base_psm,
                                             scale=scale, variant=variant)))
        except Exception:
            continue
    if not results:
        return OcrResult([], [], "", scale)

    # Pick the primary pass deterministically: fixed order, and a challenger
    # must clear SCORE_MARGIN to displace the incumbent.
    scored = [(v, r, _score(r)) for v, r in results]
    if all(sc == 0 for _, _, sc in scored):
        scored = [(v, r, _score(r, FALLBACK_MIN_CONF)) for v, r in results]
    best_variant, best, best_score = scored[0]
    for v, r, sc in scored[1:]:
        if sc > best_score * SCORE_MARGIN:
            best_variant, best, best_score = v, r, sc

    extra = [r for _, r in results if r is not best]
    for psm in psms[1:]:
        try:
            extra.append(run_ocr(rect_bgr, rect_px_per_mm, lang=lang,
                                 psm=psm, scale=scale, variant=best_variant))
        except Exception:
            continue

    words = _merge_words(
        [w for l in best.lines for w in l.words],
        [[w for l in r.lines for w in l.words] for r in extra])
    lines = rebuild_lines(words)
    confs = [w.conf for w in words]

    # Character boxes come from the single best pass, NOT from the union.
    # The measurement stage cross-checks the number of glyphs it segments
    # against the number of digits the RECOGNISER read, and that check is
    # what separates a 0.037 mm sigma from a 0.241 mm one. Feeding it a
    # digit-column mask drawn from passes other than the one that produced
    # the text would break the correspondence the check depends on.
    return OcrResult(lines, best.chars, "\n".join(l.text for l in lines),
                     scale, mean_conf=float(np.mean(confs)) if confs else 0.0)
