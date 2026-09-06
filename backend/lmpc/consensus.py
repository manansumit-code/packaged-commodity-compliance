"""
Multi-frame consensus: several photographs of one pack, one answer.

Why frames are combined at the RESULT level rather than by averaging the
images. Stacking pixels sounds like the stronger move and is the weaker one
here:

  * The quantity being measured IS the position of a stroke edge. Any
    misalignment between frames smears exactly that edge, so a stacking error
    corrupts the measurement instead of improving it.
  * Averaging cannot repair a field the recogniser mis-segmented. A vote can.
  * Independent frames give an EMPIRICAL error bar - how far apart real
    captures of one label actually land. Averaging them first throws that
    away, and it is the most defensible number the system can produce.

What consensus does and does not buy:

  RANDOM error falls - a glyph mis-segmented in one frame is outvoted, and a
  field no single frame could read is usually readable in some frame.

  SYSTEMATIC error does not. Every frame is scaled by the same measured
  marker length and judged against the same panel area, so an error in either
  shifts all frames together and they will agree beautifully while being
  wrong. Agreement across frames is therefore evidence about repeatability,
  never about the marker being right.

Because of that, the band is never NARROWED by having more frames. Three
samples do not license a tighter legal threshold. The band is widened when the
frames disagree more than the model predicts, and otherwise left alone.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Optional

import numpy as np

from .rules import Verdict, verdict_for

# A frame whose measurement the pipeline itself refused (glyph count
# disagreeing with the recogniser, say) is not evidence of the same weight as
# one it accepted. Refused readings are used only if nothing else exists.
_ADJUDICATED = {Verdict.PASS.value, Verdict.FAIL.value, Verdict.BORDERLINE.value}


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").split()).lower()


# A reading that only ONE frame produced needs at least this match confidence
# to be reported. The pipeline uses the same figure to decide whether a field
# is solid enough to adjudicate against a legal threshold.
MIN_LONE_FRAME_CONF = 0.5


def _vote_value(rows: list[tuple[str, float]],
                n_frames: int = 0) -> tuple[Optional[str], bool]:
    """Majority value across frames. Returns (value, frames_disagreed).

    An uncorroborated reading is treated with suspicion rather than accepted
    by default. Several frames of one pack will often each pick up a
    different scrap of noise where a declaration is genuinely hard to parse;
    taking the "winner" of that then reports a confident-looking value that
    exactly one frame ever saw. Observed on a jar whose MRP no frame could
    parse: one frame emitted a bare "7", nothing outvoted it, and the merged
    report showed the price as 7. Reporting nothing is the honest answer.
    """
    seen = [(v, c) for v, c in rows if _norm(v)]
    if not seen:
        return None, False
    counts = Counter(_norm(v) for v, _ in seen)
    top, n = counts.most_common(1)[0]
    disagreed = len(counts) > 1

    if n == 1:
        # Nothing corroborates any reading. Fall back to the most confident
        # one, and only if it is confident enough to stand alone.
        best = max(seen, key=lambda vc: vc[1])
        if n_frames >= 2 and best[1] < MIN_LONE_FRAME_CONF:
            return None, True
        return best[0], disagreed
    for v, _ in seen:
        if _norm(v) == top:
            return v, disagreed
    return seen[0][0], disagreed


def combine(reports: list[dict], names: list[str]) -> dict:
    """Merge per-frame reports into one, with agreement statistics."""
    if not reports:
        raise ValueError("no frames")
    if len(reports) == 1:
        out = copy.deepcopy(reports[0])
        out["frames"] = {"n": 1, "names": names, "note": "single frame; "
                         "no cross-frame agreement available"}
        return out

    # Base the merged report on the frame with the best imaging, so the
    # calibration/legibility sections describe a real capture rather than an
    # average of incompatible ones.
    def quality(r):
        return (r.get("legibility", {}).get("label_source_px_per_mm") or 0.0)
    base_i = max(range(len(reports)), key=lambda i: quality(reports[i]))
    out = copy.deepcopy(reports[base_i])
    out["frames"] = {"n": len(reports), "names": names,
                     "best_frame": names[base_i]}

    agreement: dict[str, Any] = {}
    merged_decls = []
    for idx, d0 in enumerate(out["declarations"]):
        field = d0["field"]
        d = copy.deepcopy(d0)
        rows = []
        for r in reports:
            dd = next((x for x in r["declarations"] if x["field"] == field), None)
            if dd:
                rows.append(dd)

        # ---- presence: a declaration found in ANY frame is present --------
        present_in = [i for i, dd in enumerate(rows) if dd.get("present")]
        d["present"] = bool(present_in)
        val, disagreed = _vote_value(
            [(dd.get("value_text"), dd.get("match_confidence") or 0.0)
             for dd in rows], n_frames=len(reports))
        d["value_text"] = val or ""
        if val is None and not any(dd.get("value_text") for dd in rows):
            d["present"] = bool(present_in)
        elif val is None:
            # every frame's reading was uncorroborated and weak
            d["present"] = False
            d["presence_verdict"] = Verdict.NOT_ASSESSED.value
        d["match_confidence"] = round(
            max((dd.get("match_confidence") or 0.0) for dd in rows), 2) if rows else 0.0
        if d["present"]:
            d["presence_verdict"] = Verdict.PASS.value

        # ---- height: median across frames that produced one ---------------
        h0 = d.get("height")
        if h0:
            hs = [dd.get("height") or {} for dd in rows]
            good = [h for h in hs
                    if h.get("measured_mm") and h.get("verdict") in _ADJUDICATED]
            weak = [h for h in hs if h.get("measured_mm")]
            used, trusted = (good, True) if good else (weak, False)
            info = any(h.get("informational") for h in hs if h)
            if used:
                vals = [float(h["measured_mm"]) for h in used]
                med = float(np.median(vals))
                spread = float(max(vals) - min(vals))
                req = next((h.get("required_mm") for h in used
                            if h.get("required_mm") is not None), None)
                h = copy.deepcopy(used[0])
                h["measured_mm"] = round(med, 3)
                h["frames_measured"] = len(vals)
                h["frame_values_mm"] = [round(v, 3) for v in vals]
                h["frame_spread_mm"] = round(spread, 3)
                h["measurement_trusted"] = trusted
                h["informational"] = info
                if not trusted:
                    h["reason"] = ("No frame produced a measurement the "
                                   "pipeline was willing to adjudicate; the "
                                   "median below is reported for information "
                                   "only. " + (h.get("reason") or ""))
                if req is not None and trusted and not info:
                    unc = _band_for(out, spread, len(vals))
                    v, band = verdict_for(med, req, unc)
                    h["verdict"] = v.value
                    h["band_mm"] = round(band, 4)
                    h["reason"] = ""
                elif info:
                    h["verdict"] = Verdict.NOT_ASSESSED.value
                d["height"] = h
            agreement[field] = {
                "frames_present": len(present_in),
                "frames_measured": len(used),
                "values_disagreed": disagreed,
                "spread_mm": (round(float(max(float(h["measured_mm"])
                                              for h in used)
                                         - min(float(h["measured_mm"])
                                               for h in used)), 3)
                              if len(used) > 1 else None),
            }
        merged_decls.append(d)

    out["declarations"] = merged_decls
    out["frame_agreement"] = agreement
    _rollup(out)
    return out


def _band_for(report: dict, spread_mm: float, n: int):
    """Model band, widened when the frames disagree more than it allows.

    Never narrowed. Averaging n readings would formally shrink the standard
    error, but the band here also carries systematic terms that repeat
    identically in every frame, and three samples are not grounds for
    claiming a tighter legal threshold.
    """
    from dataclasses import replace
    from .rules import Uncertainty
    u = report.get("uncertainty") or {}
    unc = Uncertainty(
        bias_mm=u.get("bias_mm", 0.0), sigma_mm=u.get("sigma_mm", 0.15),
        k=u.get("k", 2.0), band_override_mm=u.get("band_override_mm"),
        apply_bias_correction=bool(u.get("apply_bias_correction", False)))
    base = unc.band_mm()
    # Half the observed spread is the distance from the median to the extreme
    # frame: the smallest band that would have covered every frame.
    return replace(unc, extra_band_mm=max(0.0, (spread_mm / 2.0) - base))


def _rollup(out: dict) -> None:
    fails, borderline, unassessed = [], [], []
    for d in out["declarations"]:
        if d["presence_verdict"] == Verdict.FAIL.value:
            fails.append(f"{d['field']}: declaration not found")
        elif d["presence_verdict"] == Verdict.NOT_ASSESSED.value:
            unassessed.append(f"{d['field']}: presence not machine-assessable")
        h = d.get("height")
        if h and not h.get("informational"):
            v, msg = h.get("verdict"), (
                f"{d['field']}: numeral height {h.get('measured_mm')} mm vs "
                f"required {h.get('required_mm')} mm")
            if v == Verdict.FAIL.value:
                fails.append(msg)
            elif v == Verdict.BORDERLINE.value:
                borderline.append(msg + " (within measurement error)")
            elif v == Verdict.NOT_ASSESSED.value:
                unassessed.append(f"{d['field']}: {h.get('reason','')}")
            if h.get("width_verdict") == Verdict.FAIL.value:
                fails.append(f"{d['field']}: glyph width below the one third "
                             f"Rule 7(3) requires")
    out["violations"], out["borderline"], out["not_assessed"] = (
        fails, borderline, unassessed)
    out["overall"] = ("NON_COMPLIANT" if fails else
                      "REVIEW_REQUIRED" if (borderline or unassessed)
                      else "COMPLIANT")


def render_agreement(out: dict, colour: bool = True) -> str:
    """The cross-frame agreement table - the empirical error bar."""
    fr = out.get("frames") or {}
    ag = out.get("frame_agreement") or {}
    if fr.get("n", 1) < 2:
        return ""
    L = ["", "  AGREEMENT ACROSS FRAMES", ""]
    L.append(f"   {fr['n']} photo(s): " + ", ".join(fr.get("names", [])))
    L.append("")
    L.append(f"   {'declaration':22s} {'frames':>7s} {'median':>9s} "
             f"{'spread':>8s}   each frame")
    for d in out["declarations"]:
        h = d.get("height") or {}
        if not h.get("frame_values_mm"):
            continue
        vals = h["frame_values_mm"]
        # A field only one frame could measure has no spread - it has no
        # second opinion. Printing 0.00 there would read as perfect agreement,
        # which is the opposite of the truth.
        sp = (f"{h['frame_spread_mm']:>6.2f} mm" if len(vals) > 1
              else "     — " + " ")
        L.append(f"   {d['field']:22s} {len(vals):>4d}/{fr['n']:<2d} "
                 f"{h['measured_mm']:>7.2f} mm {sp}"
                 f"   {', '.join(f'{v:.2f}' for v in vals)}")
    dis = [f for f, a in ag.items() if a.get("values_disagreed")]
    if dis:
        L.append("")
        L.append("   frames read DIFFERENT text for: " + ", ".join(dis))
        L.append("   (the majority reading was used; check these by eye)")
    spreads = [(d.get("height") or {}).get("frame_spread_mm")
               for d in out["declarations"]
               if len((d.get("height") or {}).get("frame_values_mm") or []) > 1]
    spreads = [s for s in spreads if s is not None]
    single = [d["field"] for d in out["declarations"]
              if len((d.get("height") or {}).get("frame_values_mm") or []) == 1]
    if single:
        L.append("")
        L.append("   Only ONE photo could measure: " + ", ".join(single) + ".")
        L.append("   Those have no second opinion behind them. Re-shoot so")
        L.append("   more frames catch them.")
    if spreads:
        L.append("")
        L.append(f"   Widest disagreement between photos: {max(spreads):.3f} mm.")
        L.append("   This is a REAL error bar, measured on your own captures,")
        L.append("   not a figure fitted on simulated images.")
        L.append("   It does not cover errors shared by every frame - the")
        L.append("   marker size you typed in, and the panel area - because")
        L.append("   those shift all photos together.")
    return "\n".join(L)
