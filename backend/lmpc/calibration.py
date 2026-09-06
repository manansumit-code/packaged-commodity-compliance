"""
Metric calibration: turn a photograph into a plane on which 1 px is a known
number of millimetres.

Method
------
A printed ArUco marker of known physical side length is kept in the same frame
as, and COPLANAR with, the label panel. Four marker corners in the image plus
four known corners in millimetre space give an exact planar homography - no
camera intrinsics, no focal length, no distance assumption. The homography is
recomputed on every frame, so bumping the camera cannot silently invalidate a
calibration (build plan Section 7).

The whole image is then rectified into that metric plane, so downstream OCR
and glyph measurement both operate on a fronto-parallel image with a constant,
known px/mm. Tilt is therefore corrected exactly rather than approximated.

What tilt still costs: a foreshortened region carries fewer real sensor pixels
per millimetre, so rectifying it interpolates rather than creates detail. That
is why `local_px_per_mm` reports the SOURCE resolution at the point being
measured, and why the pipeline refuses to adjudicate below a floor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_MARKER_MM = 40.0


@dataclass
class MarkerSpec:
    dictionary: str = DEFAULT_DICT
    marker_length_mm: float = DEFAULT_MARKER_MM
    marker_id: Optional[int] = None   # None = accept any detected id


@dataclass
class Calibration:
    """Result of locating the metric plane in one image."""
    ok: bool
    reason: str = ""
    marker_id: Optional[int] = None
    # homography mapping millimetre-plane (X, Y, 1) -> source image (u, v, 1)
    H_mm_to_img: Optional[np.ndarray] = None
    px_per_mm_at_marker: float = 0.0
    tilt_deg: float = 0.0
    taper: float = 1.0                # perspective foreshortening strength
    marker_corners_px: Optional[np.ndarray] = None
    # Which preprocessing variant the marker was finally detected on. "raw"
    # means it was clean; anything else is a signal that the printed card is
    # degraded and worth reprinting.
    detect_variant: str = "raw"
    # rectified metric image
    rectified: Optional[np.ndarray] = None
    rect_px_per_mm: float = 0.0
    rect_origin_mm: tuple[float, float] = (0.0, 0.0)

    def rect_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (self.rect_origin_mm[0] + x_px / self.rect_px_per_mm,
                self.rect_origin_mm[1] + y_px / self.rect_px_per_mm)

    def local_px_per_mm(self, x_rect_px: float, y_rect_px: float) -> float:
        """Real SOURCE-image resolution, in px/mm, at a rectified location."""
        if self.H_mm_to_img is None:
            return 0.0
        X, Y = self.rect_to_mm(x_rect_px, y_rect_px)
        return _jacobian_px_per_mm(self.H_mm_to_img, X, Y)


def _detect_variants(gray: np.ndarray):
    """Grayscale variants to try marker detection on, cheapest first.

    A marker printed on ordinary office paper and photographed under room
    light is not a clean binary square. Toner speckle, paper texture and JPEG
    ringing all put high-frequency noise inside the black cells, and ArUco's
    bit sampler reads that noise as bit errors, so the marker is found as a
    quad and then REJECTED at identification. The failure is silent: the
    caller only ever sees "no marker in frame".

    Each variant attacks one cause, and all of them stay at FULL resolution
    so that corner refinement keeps its sub-pixel accuracy - the px/mm ratio
    is derived from those corners, so resampling here would cost measurement
    precision to buy detection.
    """
    yield "raw", gray
    # Median filtering removes speckle without moving an edge, which is
    # exactly the trade this needs: it must not shift the corners.
    yield "median3", cv2.medianBlur(gray, 3)
    yield "median5", cv2.medianBlur(gray, 5)
    # A marker lit unevenly across its face (one side in shadow) defeats a
    # single global threshold; flattening local contrast first fixes it.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield "clahe", clahe
    yield "clahe+median3", cv2.medianBlur(clahe, 3)
    # Closing fills pinholes left by a printer that is low on toner.
    yield "median3+close", cv2.morphologyEx(
        cv2.medianBlur(gray, 3), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _get_detector(spec: MarkerSpec):
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement is what keeps the mm/px ratio honest.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 8
    # Ink bleed and JPEG ringing contaminate the outer rim of every bit cell.
    # Sampling each cell from a larger central margin reads the bit the
    # printer intended rather than the smear at its edge.
    params.perspectiveRemoveIgnoredMarginPerCell = 0.23
    # A printed-and-photographed marker carries a few genuinely wrong bits.
    # DICT_4X4_50 has a small Hamming distance, so this is raised only
    # slightly off the 0.6 default: enough to absorb toner speckle, not
    # enough to start inventing markers out of noise.
    params.errorCorrectionRate = 0.75
    return cv2.aruco.ArucoDetector(dic, params)


def generate_marker(spec: MarkerSpec, marker_id: int = 0,
                    px: int = 800, border_px: int = 80) -> np.ndarray:
    """Printable marker card. Print at 100% scale, then measure the printed
    black square with a ruler and pass the true value as marker_length_mm."""
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    img = cv2.aruco.generateImageMarker(dic, marker_id, px)
    return cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px,
                              cv2.BORDER_CONSTANT, value=255)


def _jacobian_px_per_mm(H: np.ndarray, X: float, Y: float,
                        d: float = 0.5) -> float:
    """sqrt(|det J|) of the mm->image map: isotropic source px per mm."""
    pts = np.array([[X, Y], [X + d, Y], [X, Y + d]], dtype=np.float64)
    pts = np.hstack([pts, np.ones((3, 1))]) @ H.T
    w = pts[:, 2:3]
    if np.any(np.abs(w) < 1e-9):
        return 0.0
    uv = pts[:, :2] / w
    dx = (uv[1] - uv[0]) / d
    dy = (uv[2] - uv[0]) / d
    return math.sqrt(abs(dx[0] * dy[1] - dx[1] * dy[0]))


def _quad_geometry(c: np.ndarray) -> tuple[float, float, float]:
    """(mean side px, tilt proxy in degrees, taper) for a square seen as `c`."""
    sides = [float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)]
    mean_side = float(np.mean(sides))
    lo, hi = min(sides), max(sides)
    tilt = math.degrees(math.acos(max(0.0, min(1.0, lo / hi)))) if hi > 0 else 90.0
    # Opposite sides are equal under an affine view; their ratio measures how
    # much true perspective (not just tilt) is present.
    taper = max(max(sides[0], sides[2]) / max(1e-6, min(sides[0], sides[2])),
                max(sides[1], sides[3]) / max(1e-6, min(sides[1], sides[3])))
    return mean_side, tilt, taper


def calibrate(bgr: np.ndarray, spec: MarkerSpec = MarkerSpec(),
              max_canvas_px: int = 5000,
              max_extent_mm: float = 400.0,
              rect_px_per_mm: Optional[float] = None) -> Calibration:
    """Locate the marker, build the metric homography, rectify the image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    detector = _get_detector(spec)
    corners, ids, variant = None, None, "raw"
    for vname, vimg in _detect_variants(gray):
        c, i, _ = detector.detectMarkers(vimg)
        if i is not None and len(i) > 0:
            corners, ids, variant = c, i, vname
            break
    if ids is None or len(ids) == 0:
        return Calibration(False, "No ArUco reference marker found in frame, "
                                  "on any of the six image variants tried. "
                                  "Physical measurement is impossible without "
                                  "a scale reference in the same plane.")

    ids = ids.flatten()
    pick = 0
    if spec.marker_id is not None:
        matches = np.where(ids == spec.marker_id)[0]
        if len(matches) == 0:
            return Calibration(False, f"Marker id {spec.marker_id} not present "
                                      f"(saw ids {sorted(set(ids.tolist()))}).")
        pick = int(matches[0])

    c = corners[pick].reshape(4, 2).astype(np.float64)
    mean_side, tilt, taper = _quad_geometry(c)
    L = spec.marker_length_mm
    px_per_mm = mean_side / L

    metric = np.array([[0, 0], [L, 0], [L, L], [0, L]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(metric, c.astype(np.float32)).astype(np.float64)

    scale = rect_px_per_mm or px_per_mm
    scale = float(np.clip(scale, 2.0, 60.0))

    # Metric-space bounding box of the visible image, clamped so that a steep
    # view cannot request a canvas the size of a building.
    h, w = gray.shape[:2]
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    Hinv = np.linalg.inv(H)
    p = np.hstack([img_corners, np.ones((4, 1))]) @ Hinv.T
    wcol = p[:, 2:3]
    valid = np.abs(wcol[:, 0]) > 1e-9
    mm_pts = p[valid, :2] / wcol[valid]
    if len(mm_pts) < 3:
        return Calibration(False, "Degenerate homography (view too oblique).")

    x0 = max(float(mm_pts[:, 0].min()), -max_extent_mm)
    y0 = max(float(mm_pts[:, 1].min()), -max_extent_mm)
    x1 = min(float(mm_pts[:, 0].max()), max_extent_mm)
    y1 = min(float(mm_pts[:, 1].max()), max_extent_mm)
    if x1 - x0 < L or y1 - y0 < L:
        return Calibration(False, "Metric plane extent collapsed; bad marker fit.")

    out_w = int(round((x1 - x0) * scale))
    out_h = int(round((y1 - y0) * scale))
    if max(out_w, out_h) > max_canvas_px:
        shrink = max_canvas_px / max(out_w, out_h)
        scale *= shrink
        out_w, out_h = int(out_w * shrink), int(out_h * shrink)

    S = np.array([[scale, 0, -x0 * scale],
                  [0, scale, -y0 * scale],
                  [0, 0, 1]], dtype=np.float64)
    M = S @ Hinv
    rect = cv2.warpPerspective(bgr, M, (out_w, out_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))

    return Calibration(
        ok=True, marker_id=int(ids[pick]), H_mm_to_img=H,
        px_per_mm_at_marker=px_per_mm, tilt_deg=tilt, taper=taper,
        marker_corners_px=c, rectified=rect, rect_px_per_mm=scale,
        rect_origin_mm=(x0, y0), detect_variant=variant,
    )


def calibrate_from_scale(bgr: np.ndarray, px_per_mm: float) -> Calibration:
    """Calibration for images of KNOWN scale (flatbed scan at a known DPI, or
    a synthetic render). Bypasses marker detection; no rectification needed."""
    H = np.array([[px_per_mm, 0, 0], [0, px_per_mm, 0], [0, 0, 1]], np.float64)
    return Calibration(ok=True, marker_id=None, H_mm_to_img=H,
                       px_per_mm_at_marker=px_per_mm, tilt_deg=0.0, taper=1.0,
                       rectified=bgr, rect_px_per_mm=px_per_mm,
                       rect_origin_mm=(0.0, 0.0))
