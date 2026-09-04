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


def _preprocess(bgr: np.ndarray, scale: float) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    # CLAHE flattens the local contrast loss caused by glare and by a
    # non-uniform light source across the panel.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    return gray


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


def choose_scale(rect_px_per_mm: float, target_px_per_mm: float = 28.0,
                 cap: float = 5.0) -> float:
    if rect_px_per_mm <= 0:
        return 1.0
    return float(np.clip(target_px_per_mm / rect_px_per_mm, 1.0, cap))


def run_ocr(rect_bgr: np.ndarray, rect_px_per_mm: float = 0.0,
            lang: str = "eng", psm: int = 6,
            scale: Optional[float] = None) -> OcrResult:
    """OCR a rectified image; all output coordinates are in rectified pixels."""
    s = scale if scale is not None else choose_scale(rect_px_per_mm)
    img = _preprocess(rect_bgr, s)
    cfg = f"--oem 3 --psm {psm}"

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


def run_ocr_multi(rect_bgr: np.ndarray, rect_px_per_mm: float = 0.0,
                  lang: str = "eng", psms: tuple[int, ...] = (6, 11)) -> OcrResult:
    """Run several page-segmentation modes and keep the most productive one.

    Real labels mix a dense ingredient block with sparse scattered
    declarations; no single Tesseract PSM handles both well.
    """
    best: Optional[OcrResult] = None
    best_score = -1.0
    for psm in psms:
        try:
            r = run_ocr(rect_bgr, rect_px_per_mm, lang=lang, psm=psm)
        except Exception:
            continue
        score = sum(len(w.text) for l in r.lines for w in l.words) * (
            0.5 + r.mean_conf / 200.0)
        if score > best_score:
            best, best_score = r, score
    return best or OcrResult([], [], "", 1.0)
