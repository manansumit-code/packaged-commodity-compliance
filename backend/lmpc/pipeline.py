"""
Scan pipeline: image in, compliance report out.

Order matters. Calibrate -> rectify -> OCR the rectified image -> classify
fields -> measure the numerals of the fields that carry a Rule 7 threshold ->
look the threshold up deterministically -> adjudicate with an error band.

The pipeline is allowed to say "I cannot tell". Three gates produce that
answer instead of a number:

  NO_REFERENCE_MARKER   nothing in frame establishes physical scale
  EXCESSIVE_TILT        the marker is so oblique that rectification is
                        interpolating detail rather than recovering it
  INSUFFICIENT_RESOLUTION  the numeral is rendered on too few real sensor
                        pixels for the measurement to mean anything
  PANEL_NOT_COPLANAR    the panel is on a grossly different plane from the
                        marker, so the marker's scale does not apply

A fourth, PANEL_TILT_PENALTY, does NOT withhold an answer: a mild plane
mismatch widens the error band instead of voiding the measurement.

Presence checking does NOT need scale, so a scan with no marker still returns
a full Rule 6 presence report - only the height half is withheld.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field as dc_field, replace
from typing import Any, Optional

import numpy as np

from . import calibration as calib
from .fields import FieldHit, extract_fields, union_box
from .measure import HeightMeasurement, measure_field_height
from .ocr import OcrResult, content_crop, offset_result, run_ocr_multi
from .rules import (MANDATORY_DECLARATIONS, PrintStyle, Uncertainty, Verdict,
                    declaration_rule, required_height_mm, verdict_for,
                    width_ratio_verdict)

# Declarations whose numerals carry a Rule 7 height threshold and that we can
# actually locate on the label. These are ADJUDICATED.
MEASURED_FIELDS = ("net_quantity", "mrp", "mfg_date")

# The remaining declarations are lettering, not numerals. Their height is
# measured and reported, but NOT adjudicated, for two reasons:
#
#   * the measurement is a median over whatever glyphs the line contains, so a
#     mixed-case line returns something between x-height and cap-height, which
#     is not the quantity Rule 7 names; and
#   * the error band in data/uncertainty.json was fitted on numerals, so it
#     does not describe this measurement.
#
# Reporting the number is useful - it tells you roughly how big the small
# print is. Turning it into a verdict would be inventing precision.
INFORMATIONAL_FIELDS = ("manufacturer", "commodity_name", "consumer_care")

UNCERTAINTY_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                                "uncertainty.json")


def load_uncertainty(path: str = UNCERTAINTY_PATH) -> Uncertainty:
    try:
        with open(path) as fh:
            d = json.load(fh)
        return Uncertainty(**{k: d[k] for k in
                              ("bias_mm", "sigma_mm", "k", "min_px_per_mm",
                               "max_tilt_deg", "max_baseline_slope_deg",
                               "max_baseline_spread_deg",
                               "hard_baseline_slope_deg",
                               "hard_baseline_spread_deg",
                               "coplanarity_mm_per_deg",
                               "source", "apply_bias_correction",
                               "band_override_mm") if k in d})
    except Exception:
        return Uncertainty()


@dataclass
class ScanConfig:
    marker: calib.MarkerSpec = dc_field(default_factory=calib.MarkerSpec)
    print_style: PrintStyle = PrintStyle.NORMAL
    panel_area_cm2: Optional[float] = None
    lang: str = "eng"
    known_px_per_mm: Optional[float] = None   # bypass marker (scanner input)
    psms: tuple[int, ...] = (6, 11)           # (6,) alone is ~2x faster
    uncertainty: Uncertainty = dc_field(default_factory=load_uncertainty)


def _to_jsonable(o: Any):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return _to_jsonable(asdict(o))
    if hasattr(o, "value") and hasattr(o, "name"):
        return o.value
    return o


def scan_image(bgr: np.ndarray, cfg: ScanConfig = None) -> dict:
    cfg = cfg or ScanConfig()
    t0 = time.time()
    unc = cfg.uncertainty
    gates: list[dict] = []

    # ---- 1. metric calibration -------------------------------------------
    if cfg.known_px_per_mm:
        cal = calib.calibrate_from_scale(bgr, cfg.known_px_per_mm)
        cal_note = "scale supplied by caller (no marker used)"
    else:
        cal = calib.calibrate(bgr, cfg.marker)
        cal_note = "ArUco reference marker"

    metric_ok = cal.ok
    if not cal.ok:
        # "No marker in frame" and "marker found but its geometry is
        # unusable" are different failures and deserve different advice;
        # reporting the second as the first tells the user to add a card
        # that is already in the picture.
        found = "No ArUco reference marker found" not in cal.reason
        gates.append({"gate": ("UNUSABLE_REFERENCE_MARKER" if found
                               else "NO_REFERENCE_MARKER"),
                      "blocking_height": True,
                      "detail": cal.reason})
        work = bgr
        rect_ppmm = 0.0
    else:
        if cal.tilt_deg > unc.max_tilt_deg or cal.taper > 1.35:
            metric_ok = False
            gates.append({
                "gate": "EXCESSIVE_TILT", "blocking_height": True,
                "detail": f"Marker is viewed at ~{cal.tilt_deg:.1f} deg "
                          f"(taper {cal.taper:.2f}); limit is "
                          f"{unc.max_tilt_deg:.0f} deg. Rectification would be "
                          f"interpolating, not recovering, detail. "
                          f"Re-shoot closer to square-on."})
        work = cal.rectified
        rect_ppmm = cal.rect_px_per_mm

    # ---- 2. OCR on the metric plane --------------------------------------
    crop, ox, oy = content_crop(work)
    ocr: OcrResult = offset_result(
        run_ocr_multi(crop, rect_ppmm, lang=cfg.lang, psms=cfg.psms), ox, oy)

    # ---- 3. field classification -----------------------------------------
    hits: dict[str, FieldHit] = extract_fields(ocr)

    # ---- 3b. legibility gate ---------------------------------------------
    # A declaration that OCR could not read is not the same thing as a
    # declaration that is not printed on the pack. Reporting the first as the
    # second is a false accusation against a trader, so when the imaging
    # conditions cannot support a "missing" claim, absence is reported as
    # NOT ASSESSED rather than as a violation.
    legible, legibility_reasons, label_ppmm, med_word_px = _legibility(
        ocr, cal, metric_ok, unc)
    if not legible:
        gates.append({"gate": "LOW_LEGIBILITY", "blocking_height": False,
                      "detail": "; ".join(legibility_reasons)})

    plan = _planarity(ocr, unc)
    band_penalty = 0.0
    if metric_ok and plan.get("severity") == "gross":
        metric_ok = False
        gates.append({"gate": "PANEL_NOT_COPLANAR", "blocking_height": True,
                      "detail": plan["detail"]})
    elif metric_ok and plan.get("severity") == "degraded":
        band_penalty = plan.get("band_penalty_mm", 0.0)
        gates.append({"gate": "PANEL_TILT_PENALTY", "blocking_height": False,
                      "detail": plan["detail"]})

    # ---- 4. measurement + adjudication -----------------------------------
    qty = hits["net_quantity"].value if hits["net_quantity"].present else None
    declarations = []
    for name, human in MANDATORY_DECLARATIONS:
        hit = hits[name]
        entry: dict[str, Any] = {
            "field": name,
            "requirement": human,
            "present": bool(hit.present),
            "assessable": bool(hit.assessable),
            "raw_text": hit.raw_text[:400],
            "value_text": hit.value_text,
            "value": _to_jsonable(hit.value),
            "match_confidence": round(hit.match_confidence, 2),
            "notes": ([hit.note] if hit.note else []) + (
                [] if (hit.present or not hit.assessable or legible)
                else ["Not found, but this frame is not legible enough to "
                      "claim the declaration is absent: "
                      + "; ".join(legibility_reasons)]),
            "presence_verdict": (
                Verdict.PASS.value if hit.present else
                Verdict.NOT_ASSESSED.value if not hit.assessable else
                Verdict.FAIL.value if legible else
                Verdict.NOT_ASSESSED.value),
            "height": None,
        }

        if hit.present and name in (MEASURED_FIELDS + INFORMATIONAL_FIELDS):
            entry["height"] = _assess_height(
                work, cal, metric_ok, ocr, hit, name, qty, cfg, gates,
                informational=name in INFORMATIONAL_FIELDS,
                band_penalty_mm=band_penalty)
        declarations.append(entry)

    # ---- 5. roll-up -------------------------------------------------------
    fails, borderline, unassessed = [], [], []
    for d in declarations:
        if d["presence_verdict"] == Verdict.FAIL.value:
            fails.append(f"{d['field']}: declaration not found")
        elif d["presence_verdict"] == Verdict.NOT_ASSESSED.value:
            unassessed.append(
                f"{d['field']}: presence not machine-assessable"
                + ("" if d["assessable"] else " by rule")
                + ("" if legible else " in a frame this illegible"))
        h = d.get("height")
        if h and not h.get("informational"):
            v = h.get("verdict")
            msg = (f"{d['field']}: numeral height "
                   f"{h.get('measured_mm')} mm vs required "
                   f"{h.get('required_mm')} mm")
            if v == Verdict.FAIL.value:
                fails.append(msg)
            elif v == Verdict.BORDERLINE.value:
                borderline.append(msg + " (within measurement error)")
            elif v == Verdict.NOT_ASSESSED.value:
                unassessed.append(f"{d['field']}: {h.get('reason', '')}")

            wv = h.get("width_verdict")
            wmsg = (f"{d['field']}: glyph width is "
                    f"{h.get('width_to_height_ratio')} of its height, below "
                    f"the one third Rule 7(3) requires")
            if wv == Verdict.FAIL.value:
                fails.append(wmsg)
            elif wv == Verdict.BORDERLINE.value:
                borderline.append(wmsg + " (within measurement error)")

    if fails:
        overall = "NON_COMPLIANT"
    elif borderline or unassessed:
        overall = "REVIEW_REQUIRED"
    else:
        overall = "COMPLIANT"

    return {
        "scan_id": uuid.uuid4().hex[:16],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
        "overall": overall,
        "violations": fails,
        "borderline": borderline,
        "not_assessed": unassessed,
        "declarations": declarations,
        "calibration": {
            "ok": bool(cal.ok), "usable_for_measurement": bool(metric_ok),
            "method": cal_note, "marker_id": cal.marker_id,
            "marker_length_mm": cfg.marker.marker_length_mm,
            "px_per_mm_at_marker": round(cal.px_per_mm_at_marker, 3),
            "rect_px_per_mm": round(rect_ppmm, 3),
            "tilt_deg": round(cal.tilt_deg, 2), "taper": round(cal.taper, 3),
            "reason": cal.reason,
        },
        "gates": gates,
        "planarity": plan,
        "legibility": {
            "presence_claims_reliable": bool(legible),
            "reasons": legibility_reasons,
            "label_source_px_per_mm": round(label_ppmm, 2) if label_ppmm else None,
            "median_word_height_px": round(med_word_px, 1),
        },
        "uncertainty": _to_jsonable(unc) | {"band_mm": round(unc.band_mm(), 4)},
        "ocr": {"engine": ocr.engine, "mean_conf": round(ocr.mean_conf, 1),
                "n_lines": len(ocr.lines), "ocr_scale": round(ocr.scale, 2),
                "text": ocr.full_text},
        "scope": SCOPE_STATEMENT,
    }


# Below this mean OCR confidence the frame cannot support an "this
# declaration is ABSENT" claim, and absence is reported as NOT ASSESSED
# instead. It does NOT block height measurement.
#
# Lowered from 65 to 60 on operator request, after real phone photographs of
# genuine retail packs came in at 60-64 and had every declaration withheld.
# The cost is real and runs the other way: at a lower bar the pipeline will
# start calling a declaration missing on frames the recogniser itself read
# poorly, and a false "missing" is an accusation against a trader. 60 is a
# judgement, not a measurement - there is no experiment in this repo that
# fixes the right value.
MIN_OCR_CONF = 60.0
MIN_WORD_PX = 11.0
# A field matched only by a bare value pattern - a currency amount with no
# "MRP" nearby, a date with no "Mfg." nearby - may well be a different
# declaration (a promotional price, a best-before date). Measuring it is fine;
# adjudicating it against a legal threshold is not, because the threshold
# belongs to a declaration we are not sure we are looking at.
MIN_MATCH_CONF_FOR_VERDICT = 0.5


def _planarity(ocr, unc):
    """Detect a label panel that is NOT coplanar with the reference marker.

    This is the last silent-wrong-answer mode in the design. The homography is
    fitted to the marker's plane; if the declarations sit on a different plane
    - a card lying flat on the table beside a standing carton - the marker can
    look perfectly square-on while every millimetre reading is wrong, and the
    tilt gate cannot see it.

    After rectification to the marker plane, a coplanar panel's printed lines
    are horizontal. A panel on a different plane keeps residual keystone, so
    its text baselines acquire a slope. Fitting a baseline through each line's
    word centres and looking at the slopes is therefore a direct test of the
    coplanarity assumption, using text the pipeline has already found.
    """
    slopes, ys = [], []
    for line in ocr.lines:
        ws = line.words
        # A baseline fitted through two or three closely-spaced words is
        # mostly OCR box jitter. Only long, well-spread lines carry a usable
        # slope, and admitting the rest is what makes the null distribution
        # too wide to separate from a real plane mismatch.
        if len(ws) < 4:
            continue
        x = np.array([w.x + w.w / 2.0 for w in ws], dtype=np.float64)
        y = np.array([w.y + w.h / 2.0 for w in ws], dtype=np.float64)
        mh = float(np.median([w.h for w in ws]))
        if x.max() - x.min() < 8.0 * mh:
            continue
        m = np.polyfit(x, y, 1)[0]
        slopes.append(m)
        ys.append(float(y.mean()))
    if len(slopes) < 3:
        return {"n_baselines": len(slopes), "median_slope_deg": None,
                "slope_spread_deg": None, "coplanar": None,
                "detail": "too few multi-word text lines to test coplanarity"}
    sl = np.degrees(np.arctan(np.array(slopes)))
    med = float(np.median(sl))
    spread = float(np.percentile(sl, 90) - np.percentile(sl, 10))
    # Slope and spread get independent thresholds. They measure different
    # things - a uniform lean versus disagreement between lines - and tying
    # the second to a multiple of the first made the spread test fire on
    # perfectly coplanar frames.
    # Three states, not two. The statistic is a WEAK indicator - eval/
    # test_coplanarity.py measured its null distribution overlapping the
    # signal - so treating "above the clean threshold" as proof of a plane
    # mismatch discards good captures. It is used instead to say how much
    # LESS the reading should be trusted, and only a gross excursion refuses
    # the scale outright.
    slope_excess = max(0.0, abs(med) - unc.max_baseline_slope_deg)
    spread_excess = max(0.0, spread - unc.max_baseline_spread_deg)
    gross = (abs(med) > unc.hard_baseline_slope_deg
             or spread > unc.hard_baseline_spread_deg)
    degraded = (not gross) and (slope_excess > 0 or spread_excess > 0)
    penalty = round(unc.coplanarity_mm_per_deg
                    * (slope_excess + 0.25 * spread_excess), 4) if degraded else 0.0

    severity = "gross" if gross else ("degraded" if degraded else "ok")
    if gross:
        detail = (f"Rectified text baselines slope {med:+.2f} deg (spread "
                  f"{spread:.2f} deg) instead of lying flat, past the "
                  f"{unc.hard_baseline_slope_deg:g} deg limit. The declaration "
                  f"panel is NOT coplanar with the reference marker, which "
                  f"makes the px/mm ratio wrong for the panel. Lay the marker "
                  f"card flat against the same face as the printing.")
    elif degraded:
        detail = (f"Rectified text baselines slope {med:+.2f} deg (spread "
                  f"{spread:.2f} deg), above the {unc.max_baseline_slope_deg:g} "
                  f"deg clean threshold but well inside the "
                  f"{unc.hard_baseline_slope_deg:g} deg limit. Heights are "
                  f"still measured; the error band is widened by "
                  f"{penalty:.3f} mm to cover the residual plane mismatch.")
    else:
        detail = ""
    return {
        "n_baselines": len(sl),
        "median_slope_deg": round(med, 3),
        "slope_spread_deg": round(spread, 3),
        "severity": severity,
        "band_penalty_mm": penalty,
        # kept for backwards compatibility: False now means GROSS only
        "coplanar": not gross,
        "detail": detail,
    }


def _legibility(ocr, cal, metric_ok, unc):
    """Can this frame support a claim that a declaration is ABSENT?"""
    reasons: list[str] = []
    words = [w for l in ocr.lines for w in l.words]
    med_word_px = float(np.median([w.h for w in words])) if words else 0.0

    label_ppmm = 0.0
    if metric_ok and words:
        samples = [cal.local_px_per_mm(w.x + w.w / 2, w.y + w.h / 2)
                   for w in words[:400]]
        samples = [s for s in samples if s > 0]
        if samples:
            label_ppmm = float(np.median(samples))

    if not words:
        reasons.append("no text was recognised anywhere in the frame")
    if ocr.mean_conf < MIN_OCR_CONF:
        reasons.append(f"mean OCR confidence {ocr.mean_conf:.0f} is below "
                       f"{MIN_OCR_CONF:.0f}")
    if med_word_px and med_word_px < MIN_WORD_PX:
        reasons.append(f"median recognised text is only {med_word_px:.0f} px "
                       f"tall (need {MIN_WORD_PX:.0f} px)")
    if label_ppmm and label_ppmm < unc.min_px_per_mm:
        reasons.append(f"label imaged at {label_ppmm:.1f} real px/mm, below "
                       f"the {unc.min_px_per_mm:.0f} px/mm floor")
    return (not reasons), reasons, label_ppmm, med_word_px


# Why a frame carries no usable millimetre scale. Each states the actual
# cause, because the only thing a user can act on is the real one.
_NO_SCALE_REASON = {
    "NO_REFERENCE_MARKER":
        "No reference marker in this frame, so there is no physical scale; "
        "height cannot be measured.",
    "UNUSABLE_REFERENCE_MARKER":
        "The reference marker was found but its geometry is unusable, so the "
        "physical scale is void; height cannot be measured.",
    "EXCESSIVE_TILT":
        "The reference marker was found, but the frame is too oblique for its "
        "scale to hold; height cannot be measured.",
    "PANEL_NOT_COPLANAR":
        "The reference marker was found, but the panel is not coplanar with "
        "it, so its scale does not apply here; height cannot be measured.",
}


def _assess_height(work, cal, metric_ok, ocr, hit, name, qty, cfg, gates,
                   informational: bool = False,
                   band_penalty_mm: float = 0.0) -> dict:
    unc = replace(cfg.uncertainty, extra_band_mm=band_penalty_mm)
    lk = required_height_mm(name, qty, cfg.print_style, cfg.panel_area_cm2)
    out: dict[str, Any] = {
        "informational": informational,
        "required_mm": lk.required_mm,
        "rule": f"Rule 7 Table {lk.table}" if lk.table in ("I", "II")
                else "Rule 7(3)",
        "rule_row": lk.band_label,
        "rule_basis": declaration_rule(name).source_note,
        "advisory_required_mm": lk.advisory_mm,
        "advisory_rule": (f"Rule 7 Table {lk.advisory_table}"
                          if lk.advisory_table else None),
        "print_style": cfg.print_style.value,
        "measured_mm": None,
        "verdict": Verdict.NOT_ASSESSED.value,
        "reason": "",
    }
    if not lk.ok:
        out["reason"] = lk.reason
        return out
    if not metric_ok:
        # Name the gate that actually voided the scale. A bare "no usable
        # scale" reads downstream as "no card in the photo", which is a lie
        # whenever the card was found and something else disqualified it.
        blocking = next((g["gate"] for g in gates
                         if g.get("blocking_height")), None)
        out["scale_gate"] = blocking
        out["reason"] = _NO_SCALE_REASON.get(
            blocking, "No usable physical scale for this frame; height "
                      "cannot be measured.")
        return out

    roi = union_box(hit.value_boxes)
    if roi is None:
        out["reason"] = "Field located but its numerals could not be boxed."
        return out

    x, y, w, h = roi
    chars = [c for c in ocr.chars
             if c[1] < x + w and c[1] + c[3] > x and
             c[2] < y + h and c[2] + c[4] > y]
    src_ppmm = cal.local_px_per_mm(x + w / 2, y + h / 2)
    out["source_px_per_mm"] = round(src_ppmm, 2)

    if src_ppmm > 0 and src_ppmm < unc.min_px_per_mm:
        out["reason"] = (
            f"Numeral is imaged at only {src_ppmm:.1f} real px/mm; the "
            f"measurable floor for this build is {unc.min_px_per_mm:.0f} "
            f"px/mm. Move the camera closer or raise capture resolution.")
        gates.append({"gate": "INSUFFICIENT_RESOLUTION", "field": name,
                      "blocking_height": True, "detail": out["reason"]})
        return out

    # A lettering field is measured over its whole line, so the recogniser's
    # digit count (a phone number, say) says nothing about how many glyphs
    # segmentation should find. Disable the cross-check rather than let it
    # fire spuriously.
    expected_digits = (0 if informational
                       else sum(c.isdigit() for c in (hit.value_text or "")))
    m: HeightMeasurement = measure_field_height(
        work, roi, chars, px_per_mm=cal.rect_px_per_mm,
        source_px_per_mm=src_ppmm, expected_digits=expected_digits)
    if not m.ok:
        out["reason"] = m.reason
        return out

    # Fewer numerals means the median has fewer votes. Once the glyph-count
    # consistency check is in place the calibration data no longer shows one-
    # and two-glyph readings to be worse (sigma 0.026 and 0.011 mm against
    # 0.039 for n>=3), so this is no longer a data-driven correction - it is
    # kept as a deliberately conservative one, because those samples are few
    # (30 and 13) and a legal verdict should not rest on a thin tail estimate.
    inflate = {1: 1.5, 2: 1.5}.get(m.n_glyphs, 1.0)
    eff = replace(unc, sigma_mm=unc.sigma_mm * inflate,
                  band_override_mm=None if unc.band_override_mm is None
                  else unc.band_override_mm * inflate)
    verdict, band = verdict_for(m.height_mm, lk.required_mm, eff)

    # Segmentation must agree with recognition about how many numerals are
    # there. When it does not, glyphs have merged or been dropped and the
    # median is taken over the wrong set - this is where every large error
    # lives. The height is still reported; it is simply not adjudicated.
    if not m.count_matches:
        verdict = Verdict.NOT_ASSESSED
        out["reason"] = (
            f"Measured {m.n_glyphs} numeral(s) but the recogniser read "
            f"{m.expected_digits} in '{hit.value_text}'. Segmentation and "
            f"recognition disagree, so the median height is taken over the "
            f"wrong set of glyphs. Height reported, not adjudicated.")
    elif hit.match_confidence < MIN_MATCH_CONF_FOR_VERDICT:
        verdict = Verdict.NOT_ASSESSED
        out["reason"] = (
            f"This declaration was identified only by a value pattern "
            f"(match confidence {hit.match_confidence:.2f}), not by a key "
            f"phrase, so it may be a different declaration. The height is "
            f"reported but not adjudicated.")

    out.update({
        "measured_mm": round(m.height_mm, 3),
        "measured_px": round(m.height_px, 2),
        "n_numerals_measured": m.n_glyphs,
        "glyph_spread_mm": round(m.spread_mm, 3),
        "median_width_mm": round(m.median_width_mm, 3),
        "error_band_mm": round(band, 3),
        "verdict": verdict.value,
        "low_glyph_count": m.low_glyph_count,
        "expected_digits": m.expected_digits,
        "glyph_count_matches_ocr": m.count_matches,
        "band_inflation": inflate,
    })
    if informational:
        out["informational"] = True
        out["verdict"] = Verdict.NOT_ASSESSED.value
        out["reason"] = (
            "Lettering, not numerals. The height shown is the median over the "
            "glyphs on this line, which for mixed-case text sits between "
            "x-height and cap-height, and the error band was fitted on "
            "numerals. Reported for information; not adjudicated.")
        out.pop("advisory_verdict", None)
        return out

    if lk.advisory_mm:
        av, _ = verdict_for(m.height_mm, lk.advisory_mm, eff)
        out["advisory_verdict"] = av.value
    if m.median_width_mm and m.height_mm:
        ratio = m.median_width_mm / m.height_mm
        out["width_to_height_ratio"] = round(ratio, 3)
        # Rule 7(3) is now adjudicated, not merely observed: after G.S.R.
        # 629(E) the width proviso is the entire operative content of that
        # sub-rule. A glyph can clear its height and still be unlawfully
        # condensed, and that is a defect nothing else in this report catches.
        wv, wnote = width_ratio_verdict(ratio, hit.value_text or "")
        out["width_verdict"] = wv
        out["width_rule"] = "Rule 7(3) (width >= 1/3 height)"
        if wnote:
            out["width_reason"] = wnote
    else:
        out["width_verdict"] = Verdict.NOT_ASSESSED.value
        out["width_reason"] = ("Glyph width was not measured, so the Rule 7(3) "
                               "width proviso could not be checked.")
    return out


SCOPE_STATEMENT = {
    "in_scope": [
        "Flat or near-flat printed packaging (cartons, boxes, pouches, "
        "printed labels on flat panels).",
        "Printed (ink) declarations.",
        "Latin-script declarations; Devanagari recognised only if the "
        "Hindi language pack is enabled.",
    ],
    "out_of_scope": [
        "Curved surfaces (bottles, jars, cans) - the label plane is not "
        "coplanar with the marker, so the metric homography is invalid.",
        "Embossed or debossed text - needs raking light to be visible at "
        "all; the embossed threshold column exists in the rule table but "
        "this build does not detect embossing.",
        "Determining the common/generic name of the commodity, which is "
        "not separable from brand text by rule.",
    ],
    "hard_requirement": "A printed ArUco marker of known side length must be "
                        "COPLANAR with, and visible in the same frame as, the "
                        "declaration panel. Without it a photograph carries no "
                        "physical scale and no height can be reported.",
}
