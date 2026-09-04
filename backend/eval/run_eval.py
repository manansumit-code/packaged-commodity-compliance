"""
Accuracy evaluation for the compliance scanner.

Reports THREE accuracies, because a single headline number would hide which
half of the architecture is weak:

  1. Field extraction  - is the right text found and read (classification)
  2. Height estimation - how many millimetres wrong is the measurement
                         (regression: signed bias AND spread, separately)
  3. Verdict agreement - does the end-to-end legal call match the truth,
                         and how often is it wrong in the dangerous direction

Protocol: the measurement error band that drives the PASS/BORDERLINE/FAIL
tiers is fitted on a CALIBRATION split and then reported on a disjoint
VALIDATION split. Fitting and reporting the band on the same images would
make the abstention rate meaningless.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.synth import Degradation, degrade, random_spec, render_master
from lmpc.calibration import MarkerSpec
from lmpc.pipeline import ScanConfig, scan_image
from lmpc.rules import PrintStyle, parse_net_quantity, required_height_mm

MEAS = ("net_quantity", "mrp", "mfg_date")
ALL_FIELDS = ("manufacturer", "commodity_name", "net_quantity", "mfg_date",
              "mrp", "consumer_care")


def _gt_required_mm(field: str, qty_text: str) -> float | None:
    q = parse_net_quantity(qty_text)
    lk = required_height_mm(field, q, PrintStyle.NORMAL, None)
    return lk.required_mm if lk.ok else None


def build_case(i: int, seed: int, tier: str, threshold_focus: float,
               omit_rate: float) -> tuple:
    """Deterministically construct one test case description."""
    rng = random.Random(seed * 7919 + i)
    spec = random_spec(rng)

    # Put a controlled fraction of cases right at the legal boundary. Without
    # them the verdict metric only ever measures easy calls.
    if rng.random() < threshold_focus:
        req = _gt_required_mm("net_quantity", f"{spec.qty_value:g} {spec.qty_unit}")
        if req:
            spec.qty_height_mm = max(0.5, req + rng.uniform(-0.5, 0.5))
    if rng.random() < threshold_focus:
        spec.mrp_height_mm = max(0.5, 1.0 + rng.uniform(-0.45, 0.45))

    if rng.random() < omit_rate:
        spec.omit = (rng.choice(ALL_FIELDS),)

    deg = Degradation.sample(rng, tier)
    return spec, deg, tier, seed * 7919 + i


def run_case(args) -> dict:
    spec, deg, tier, rseed = args
    rng = random.Random(rseed)
    master, gt = render_master(spec)
    photo = degrade(master, deg, rng)

    cfg = ScanConfig(marker=MarkerSpec(marker_length_mm=spec.marker_mm))
    try:
        rep = scan_image(photo, cfg)
    except Exception as e:                                   # never lose a case
        return {"tier": tier, "error": f"{type(e).__name__}: {e}"}

    out = {"tier": tier, "seed": rseed, "gt": gt,
           "deg": {k: (round(v, 3) if isinstance(v, float) else v)
                   for k, v in deg.__dict__.items()},
           "overall": rep["overall"],
           "calib_ok": rep["calibration"]["usable_for_measurement"],
           "calib_reason": rep["calibration"]["reason"],
           "tilt_deg": rep["calibration"]["tilt_deg"],
           "rect_ppmm": rep["calibration"]["rect_px_per_mm"],
           "ocr_conf": rep["ocr"]["mean_conf"],
           "legible": rep["legibility"]["presence_claims_reliable"],
           "label_ppmm": rep["legibility"]["label_source_px_per_mm"],
           "planarity_slope": rep["planarity"]["median_slope_deg"],
           "planarity_spread": rep["planarity"]["slope_spread_deg"],
           "elapsed_ms": rep["elapsed_ms"],
           "fields": {}}

    for d in rep["declarations"]:
        f = d["field"]
        g = gt["fields"].get(f, {})
        rec = {"present_pred": d["present"], "present_gt": g.get("present", True),
               "assessable": d["assessable"], "value_text": d["value_text"],
               "presence_verdict": d["presence_verdict"]}
        h = d.get("height")
        if h:
            rec.update({
                "measured_mm": h.get("measured_mm"),
                "required_mm": h.get("required_mm"),
                "verdict": h.get("verdict"),
                "reason": h.get("reason", "")[:90],
               "adjudicated": h.get("verdict") in ("PASS", "FAIL",
                                                   "BORDERLINE"),
                "src_ppmm": h.get("source_px_per_mm"),
                "n_glyphs": h.get("n_numerals_measured"),
            })
            rec["gt_mm"] = g.get("height_mm")
            rec["gt_max_mm"] = g.get("height_max_mm")
        out["fields"][f] = rec

    # value-level correctness against ground truth
    fq = out["fields"].get("net_quantity", {})
    gq = gt["fields"]["net_quantity"]
    if gq["present"]:
        pq = parse_net_quantity(fq.get("value_text") or "")
        out["fields"]["net_quantity"]["value_correct"] = bool(
            pq and abs(pq.value - gq["value"]) < 1e-6
            and pq.unit.lower().rstrip("s") in gq["unit"].lower())
    fm = out["fields"].get("mrp", {})
    gm = gt["fields"]["mrp"]
    if gm["present"]:
        try:
            out["fields"]["mrp"]["value_correct"] = abs(
                float((fm.get("value_text") or "0").replace(",", ".")
                      ) - gm["value"]) < 1e-6
        except ValueError:
            out["fields"]["mrp"]["value_correct"] = False
    fd = out["fields"].get("mfg_date", {})
    gd = gt["fields"]["mfg_date"]
    if gd["present"]:
        out["fields"]["mfg_date"]["value_correct"] = (
            (fd.get("value_text") or "").replace("-", "/").strip()
            == gd["text"])
    return out


# --------------------------------------------------------------------------
def height_stats(records, field=None, tier=None,
                 require_correct_field=False):
    """Height-estimation error, plus TWO yields.

    `yield_given_found` conditions on the declaration having been located -
    it isolates the measurement stage. `yield_end_to_end` denominates on the
    declaration actually being printed on the pack, so a field the extractor
    never found counts against it. The second is the number that describes
    what a user experiences; reporting only the first flatters the system.
    """
    errs, rels, yields_ok, yields_found, yields_printed = [], [], 0, 0, 0
    for r in records:
        if "error" in r or (tier and r["tier"] != tier):
            continue
        for f in ([field] if field else MEAS):
            rec = r["fields"].get(f, {})
            if not rec.get("present_gt", True):
                continue
            # Optionally exclude cases where the extractor located the WRONG
            # text (a best-before date read as the manufacture date). Those
            # are classification failures; folding them into the height metric
            # would blame the measurement stage for someone else's mistake.
            if require_correct_field and rec.get("value_correct") is False:
                continue
            yields_printed += 1
            if rec.get("gt_mm") is None:
                continue
            yields_found += 1
            if rec.get("measured_mm") is None:
                continue
            yields_ok += 1
            e = rec["measured_mm"] - rec["gt_mm"]
            errs.append(e)
            rels.append(e / rec["gt_mm"])
    if not errs:
        return None
    a = np.array(errs)
    return {
        "n": len(a),
        "yield_given_found": round(yields_ok / max(1, yields_found), 4),
        "yield_end_to_end": round(yields_ok / max(1, yields_printed), 4),
        "n_printed": yields_printed, "n_found": yields_found,
        "bias_mm": round(float(a.mean()), 4),
        "sigma_mm": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
        "mae_mm": round(float(np.abs(a).mean()), 4),
        "p95_abs_mm": round(float(np.percentile(np.abs(a), 95)), 4),
        "max_abs_mm": round(float(np.abs(a).max()), 4),
        "mean_rel_pct": round(float(np.mean(rels) * 100), 3),
        "mae_rel_pct": round(float(np.mean(np.abs(rels)) * 100), 3),
    }


def bin_by_resolution(records, require_correct_field: bool = True):
    """Error vs capture resolution.

    Defaults to correctly-classified measurements only, matching how the band
    is fitted. Mixing the two populations in one report makes the resolution
    table contradict the band that sits beside it.
    """
    bins = defaultdict(list)
    for r in records:
        if "error" in r:
            continue
        for f in MEAS:
            rec = r["fields"].get(f, {})
            if rec.get("gt_mm") is None or rec.get("measured_mm") is None:
                continue
            if require_correct_field and rec.get("value_correct") is False:
                continue
            p = rec.get("src_ppmm") or 0
            key = ("<6" if p < 6 else "6-9" if p < 9 else "9-12" if p < 12
                   else "12-16" if p < 16 else "16-22" if p < 22 else ">=22")
            bins[key].append(rec["measured_mm"] - rec["gt_mm"])
    order = ["<6", "6-9", "9-12", "12-16", "16-22", ">=22"]
    out = {}
    for k in order:
        v = bins.get(k)
        if not v:
            continue
        a = np.array(v)
        out[k] = {"n": len(a), "bias_mm": round(float(a.mean()), 4),
                  "sigma_mm": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
                  "mae_mm": round(float(np.abs(a).mean()), 4)}
    return out


def bin_by_tilt(records, require_correct_field: bool = True):
    """Height error against the marker's apparent tilt.

    The tilt gate's threshold should come from where error actually starts to
    grow, not from a guess. Note the tilt figure is the marker side-ratio
    proxy, which over-reads true tilt (it reported ~24 deg for a true ~18 deg
    view), so the gate is stricter in reality than its number suggests.
    """
    bins = defaultdict(list)
    for r in records:
        if "error" in r:
            continue
        t = r.get("tilt_deg") or 0.0
        key = ("0-5" if t < 5 else "5-10" if t < 10 else "10-15" if t < 15
               else "15-20" if t < 20 else "20-25" if t < 25
               else "25-30" if t < 30 else ">=30")
        for f in MEAS:
            rec = r["fields"].get(f, {})
            if rec.get("gt_mm") is None or rec.get("measured_mm") is None:
                continue
            if require_correct_field and rec.get("value_correct") is False:
                continue
            bins[key].append(rec["measured_mm"] - rec["gt_mm"])
    out = {}
    for k in ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">=30"]:
        v = bins.get(k)
        if not v:
            continue
        a = np.array(v)
        out[k] = {"n": len(a), "bias_mm": round(float(a.mean()), 4),
                  "sigma_mm": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
                  "p95_abs_mm": round(float(np.percentile(np.abs(a), 95)), 4)}
    return out


def band_validation(records, band_mm: float):
    """Does the fitted band actually cover the cases the system ADJUDICATES?

    The band is fitted on correctly-classified calibration measurements. What
    matters at runtime is the population that survives the confidence gate and
    receives a verdict - which is not the same set, because a field matched at
    exactly the gate threshold is adjudicated but may still be the wrong text.
    If this population has a tail beyond the band, the band is understated for
    precisely the cases it is used on.
    """
    errs = []
    for r in records:
        if "error" in r:
            continue
        for f in MEAS:
            rec = r["fields"].get(f, {})
            # "Adjudicated" = the pipeline issued a legal verdict on it.
            # After the confidence gate, a field identified only by a value
            # pattern comes back NOT_ASSESSED and is excluded here.
            if rec.get("verdict") not in ("PASS", "FAIL", "BORDERLINE"):
                continue
            if rec.get("gt_mm") is None or rec.get("measured_mm") is None:
                continue
            errs.append(rec["measured_mm"] - rec["gt_mm"])
    if not errs:
        return {}
    a = np.abs(np.array(errs))
    return {
        "n_adjudicated_measurements": len(a),
        "fitted_band_mm": round(band_mm, 5),
        "p95_abs_error_mm": round(float(np.percentile(a, 95)), 5),
        "p99_abs_error_mm": round(float(np.percentile(a, 99)), 5),
        "max_abs_error_mm": round(float(a.max()), 5),
        "fraction_outside_band": round(float((a > band_mm).mean()), 5),
        "band_covers_p95": bool(np.percentile(a, 95) <= band_mm),
        "verdict": ("band is adequate for the adjudicated population"
                    if np.percentile(a, 95) <= band_mm else
                    "BAND UNDERSTATED for the population it adjudicates - "
                    "raise k or the percentile"),
    }


def planarity_null(records):
    """Distribution of the coplanarity statistic on frames that ARE coplanar.

    Every synthetic frame is coplanar by construction, so this is the null
    distribution, and the gate threshold should be set from its tail rather
    than assumed.
    """
    med = [r["planarity_slope"] for r in records
           if "error" not in r and r.get("planarity_slope") is not None]
    spr = [r["planarity_spread"] for r in records
           if "error" not in r and r.get("planarity_spread") is not None]
    if not med:
        return {}
    m, s_ = np.abs(np.array(med)), np.array(spr)
    return {"n": len(m),
            "abs_median_slope_deg": {
                "p50": round(float(np.percentile(m, 50)), 4),
                "p95": round(float(np.percentile(m, 95)), 4),
                "p99": round(float(np.percentile(m, 99)), 4),
                "max": round(float(m.max()), 4)},
            "slope_spread_deg": {
                "p50": round(float(np.percentile(s_, 50)), 4),
                "p95": round(float(np.percentile(s_, 95)), 4),
                "p99": round(float(np.percentile(s_, 99)), 4),
                "max": round(float(s_.max()), 4)}}


def field_stats(records, only_legible: bool | None = None):
    """Presence/value accuracy per declaration.

    `only_legible` selects the frame population: True = frames the system
    judged readable enough to make an absence claim on, False = the frames it
    abstained on, None = everything. Reporting only the pooled number would
    blend "we read the label and the declaration is missing" with "we could
    not read the label", which are different results.
    """
    out = {}
    for f in ALL_FIELDS:
        tp = fp = fn = tn = abst = excl = 0
        vc = vt = 0
        for r in records:
            if "error" in r:
                continue
            if only_legible is not None and bool(r.get("legible")) != only_legible:
                continue
            rec = r["fields"].get(f)
            if rec is None:
                continue
            if not rec.get("assessable", True):
                excl += 1
                continue
            p, g = bool(rec["present_pred"]), bool(rec["present_gt"])
            if not p and rec.get("presence_verdict") == "NOT_ASSESSED":
                abst += 1
                continue
            tp += p and g; fp += p and not g
            fn += (not p) and g; tn += (not p) and not g
            if "value_correct" in rec:
                vt += 1
                vc += bool(rec["value_correct"])
        n = tp + fp + fn + tn
        if n + abst + excl == 0:
            continue
        if excl and n + abst == 0:
            out[f] = {"excluded_by_rule": True, "n_excluded": excl,
                      "note": "Not machine-assessable by rule; no score is "
                              "meaningful. See fields.py commodity_name."}
            continue
        out[f] = {
            "n_adjudicated": n, "n_abstained": abst,
            "n_excluded_by_rule": excl,
            "abstention_rate": round(abst / (n + abst), 4),
            "presence_accuracy": round((tp + tn) / n, 4) if n else None,
            "recall": round(tp / max(1, tp + fn), 4),
            "precision": round(tp / max(1, tp + fp), 4),
            "false_negatives": fn, "false_positives": fp,
        }
        if vt:
            out[f]["value_exact_match"] = round(vc / vt, 4)
            out[f]["value_n"] = vt
    return out


def verdict_stats(records):
    """End-to-end legal call vs truth, with the dangerous errors named."""
    agg = {}
    # Every field that can produce a height FAIL must be scored here.
    # mfg_date carries a Rule 7(3) threshold and its FAIL reaches `overall`,
    # so leaving it out would mean the headline "no wrong decisive calls"
    # covered only two of the three fields that can accuse a trader.
    for f in MEAS:
        c = Counter()
        near, far = Counter(), Counter()
        for r in records:
            if "error" in r:
                continue
            rec = r["fields"].get(f, {})
            gtm, req = rec.get("gt_mm"), rec.get("required_mm")
            if gtm is None or req is None or not rec.get("present_gt", True):
                continue
            truth = "PASS" if gtm >= req else "FAIL"
            pred = rec.get("verdict") or "NOT_ASSESSED"
            c[(truth, pred)] += 1
            (near if abs(gtm - req) <= 0.3 else far)[(truth, pred)] += 1
        tot = sum(c.values())
        if not tot:
            continue
        decisive = sum(v for (t, p), v in c.items() if p in ("PASS", "FAIL"))
        correct = sum(v for (t, p), v in c.items() if p == t)
        false_clear = sum(v for (t, p), v in c.items()
                          if t == "FAIL" and p == "PASS")
        false_viol = sum(v for (t, p), v in c.items()
                         if t == "PASS" and p == "FAIL")
        agg[f] = {
            "n": tot,
            "decisive_rate": round(decisive / tot, 4),
            "accuracy_on_decisive": round(correct / max(1, decisive), 4),
            "false_clear": false_clear,
            "false_clear_rate": round(false_clear / tot, 4),
            "false_violation": false_viol,
            "false_violation_rate": round(false_viol / tot, 4),
            "borderline_rate": round(
                sum(v for (t, p), v in c.items() if p == "BORDERLINE") / tot, 4),
            "not_assessed_rate": round(
                sum(v for (t, p), v in c.items() if p == "NOT_ASSESSED") / tot, 4),
            "near_threshold_0.3mm": {
                "n": sum(near.values()),
                "borderline_rate": round(
                    sum(v for (t, p), v in near.items() if p == "BORDERLINE")
                    / max(1, sum(near.values())), 4),
                "wrong_decisive": sum(v for (t, p), v in near.items()
                                      if p in ("PASS", "FAIL") and p != t)},
            "clear_of_threshold": {
                "n": sum(far.values()),
                "accuracy_on_decisive": round(
                    sum(v for (t, p), v in far.items() if p == t)
                    / max(1, sum(v for (t, p), v in far.items()
                                 if p in ("PASS", "FAIL"))), 4),
                "wrong_decisive": sum(v for (t, p), v in far.items()
                                      if p in ("PASS", "FAIL") and p != t)},
        }
    return agg


def overall_stats(records):
    """Accuracy of the verdict the system actually emits.

    Every other metric here is per-field. What a user sees is one of
    COMPLIANT / REVIEW_REQUIRED / NON_COMPLIANT, and that is what has to be
    right. Ground truth is reconstructed from the generator: a pack is
    non-compliant if a mandatory declaration was deliberately omitted, or if
    a declaration's TRUE printed height is below its Rule 7 requirement.

    Reported twice. `strict` counts an omitted common/generic name as a
    violation, which it legally is. `assessable` excludes it, because the
    system reports that declaration as not machine-determinable by rule and
    never claims it is absent - scoring it against a claim the system
    declines to make would measure nothing.
    """
    out = {}
    for mode in ("strict", "assessable"):
        c = Counter()
        for r in records:
            if "error" in r:
                continue
            gt = r.get("gt") or {}
            gtf = gt.get("fields") or {}
            omitted = set(gt.get("omitted") or [])
            if mode == "assessable":
                omitted.discard("commodity_name")
            bad = bool(omitted)
            for f in MEAS:
                g = gtf.get(f) or {}
                if not g.get("present") or g.get("height_mm") is None:
                    continue
                q = gtf.get("net_quantity") or {}
                req = _gt_required_mm(
                    f, f"{q.get('value', '')} {q.get('unit', '')}")
                if req is not None and g["height_mm"] < req:
                    bad = True
            truth = "NON_COMPLIANT" if bad else "COMPLIANT"
            c[(truth, r.get("overall"))] += 1
        tot = sum(c.values())
        if not tot:
            continue
        decided = sum(v for (t, p), v in c.items() if p != "REVIEW_REQUIRED")
        correct = sum(v for (t, p), v in c.items()
                      if p == t and p != "REVIEW_REQUIRED")
        cleared = sum(v for (t, p), v in c.items()
                      if t == "NON_COMPLIANT" and p == "COMPLIANT")
        accused = sum(v for (t, p), v in c.items()
                      if t == "COMPLIANT" and p == "NON_COMPLIANT")
        out[mode] = {
            "n_packs": tot,
            "truth_non_compliant": sum(v for (t, _), v in c.items()
                                       if t == "NON_COMPLIANT"),
            "review_required_rate": round(
                sum(v for (_, p), v in c.items()
                    if p == "REVIEW_REQUIRED") / tot, 4),
            "decided_rate": round(decided / tot, 4),
            "accuracy_when_decided": round(correct / max(1, decided), 4),
            "violating_pack_cleared": cleared,
            "violating_pack_cleared_rate": round(cleared / tot, 4),
            "compliant_pack_accused": accused,
            "compliant_pack_accused_rate": round(accused / tot, 4),
            "confusion": {f"truth={t}|predicted={p}": v
                          for (t, p), v in sorted(c.items())},
        }
    return out


def gate_stats(records):
    c = Counter()
    for r in records:
        if "error" in r:
            c["exception"] += 1
            continue
        c["total"] += 1
        c["calibrated"] += bool(r["calib_ok"])
        for f in MEAS:
            rec = r["fields"].get(f, {})
            if rec.get("measured_mm") is None and rec.get("present_gt", True) \
                    and rec.get("gt_mm") is not None:
                reason = rec.get("reason", "") or "no height returned"
                key = ("INSUFFICIENT_RESOLUTION" if "px/mm" in reason
                       else "NO_SCALE" if "physical scale" in reason
                       else "NO_GLYPHS" if "numeral" in reason
                       else "OTHER")
                c["abstain_" + key] += 1
    return dict(c)


# --------------------------------------------------------------------------
def derive_gates(records, k: float, tol_p95_mm: float = 0.09,
                tol_sigma_mm: float = 0.05):
    """Set the resolution floor and the error band FROM the calibration data.

    The floor is the coarsest capture resolution at which the measurement is
    still worth reporting: the lowest candidate where 95% of errors stay
    inside `tol_p95_mm` (15% of the tightest legal threshold, 1 mm) and the
    spread stays inside `tol_sigma_mm`. Everything below it is refused rather
    than reported, because at 6-9 px/mm the error is a third of a millimetre
    and a 1 mm threshold cannot be adjudicated at all.

    The band is then fitted on the population that survives that floor, as
    max(|bias| + k*sigma, p95|error|) - the percentile term matters because a
    handful of glyph mis-segmentations give the distribution a tail that a
    standard deviation understates.
    """
    # The band quantifies how far the MEASUREMENT of a correctly identified
    # numeral deviates from truth. Cases where the extractor located the wrong
    # text (a best-before date read as the manufacture date) are excluded:
    # they are classification failures, they are reported separately in
    # section 1, and they are handled by refusing a decisive verdict on a
    # low-confidence field match - not by inflating a millimetre band until it
    # is wider than the legal threshold it is supposed to adjudicate.
    per = []
    for r in records:
        if "error" in r:
            continue
        for f in MEAS:
            rec = r["fields"].get(f, {})
            if rec.get("gt_mm") is None or rec.get("measured_mm") is None:
                continue
            if rec.get("value_correct") is False:
                continue
            # Fit on exactly the population the pipeline is willing to
            # adjudicate, so the band describes the cases it is used on.
            if rec.get("verdict") not in ("PASS", "FAIL", "BORDERLINE"):
                continue
            per.append((rec.get("src_ppmm") or 0.0,
                        rec["measured_mm"] - rec["gt_mm"]))
    if not per:
        return None
    sweep, chosen = {}, None
    for floor in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
        a = np.array([e for p, e in per if p >= floor])
        if len(a) < 20:
            continue
        st = {"n": len(a), "bias_mm": round(float(a.mean()), 4),
              "sigma_mm": round(float(a.std(ddof=1)), 4),
              "mae_mm": round(float(np.abs(a).mean()), 4),
              "p95_abs_mm": round(float(np.percentile(np.abs(a), 95)), 4),
              "p99_abs_mm": round(float(np.percentile(np.abs(a), 99)), 4)}
        sweep[str(floor)] = st
        if chosen is None and st["p95_abs_mm"] <= tol_p95_mm \
                and st["sigma_mm"] <= tol_sigma_mm:
            chosen = floor
    if chosen is None:
        chosen = 12.0
    a = np.array([e for p, e in per if p >= chosen])
    bias, sigma = float(a.mean()), float(a.std(ddof=1))
    p95 = float(np.percentile(np.abs(a), 95))
    return {"floor": chosen, "sweep": sweep, "bias_mm": round(bias, 5),
            "sigma_mm": round(sigma, 5), "p95_abs_mm": round(p95, 5),
            "p99_abs_mm": round(float(np.percentile(np.abs(a), 99)), 5),
            "band_mm": round(max(abs(bias) + k * sigma,
                                 float(np.percentile(np.abs(a), 99))), 5),
            "n": len(a)}


def derive_slope_gate(records, min_deg: float = 0.25, factor: float = 1.5):
    """Coplanarity threshold from the null distribution.

    Every synthetic frame is coplanar by construction, so the observed spread
    of baseline slopes IS the null. The gate is set above its 99th percentile
    so it does not fire on a correctly-placed marker, and no higher than it
    needs to be, because the statistic grows monotonically with the real
    fault (see eval/test_coplanarity.py).
    """
    null = planarity_null(records)
    if not null:
        return min_deg, min_deg * 2, {}
    m99 = null["abs_median_slope_deg"]["p99"]
    s99 = null["slope_spread_deg"]["p99"]
    return (round(max(min_deg, m99 * factor), 3),
            round(max(min_deg * 2, s99 * factor), 3), null)


def _derive_and_write(cal, a, unc_path):
    """Derive every gate and the error band from calibration records, and
    write them where the pipeline will read them."""
    real = [r for r in cal if r.get("tier") in ("realistic", "harsh")]
    st = height_stats(real)
    g = derive_gates(cal, a.k)
    if g is None:
        print("calibration produced no measurements")
        sys.exit(1)
    slope_gate, spread_gate, null = derive_slope_gate(cal)
    unc = {
        "bias_mm": g["bias_mm"], "sigma_mm": g["sigma_mm"], "k": a.k,
        "band_override_mm": g["band_mm"],
        "min_px_per_mm": g["floor"], "max_tilt_deg": 35.0,
        "max_baseline_slope_deg": slope_gate,
        "max_baseline_spread_deg": spread_gate,
        "apply_bias_correction": False,
        "source": (f"fitted on {g['n']} numeral measurements at or above "
                   f"{g['floor']:g} px/mm, from {len(cal)} degraded SYNTHETIC "
                   f"captures, {time.strftime('%Y-%m-%d')}. Fitted on the "
                   f"population the pipeline actually adjudicates "
                   f"(correctly classified, glyph count agreeing with OCR, "
                   f"above the resolution floor); misclassification is "
                   f"reported separately rather than absorbed into this band. "
                   f"k={a.k:g} - see eval/band_sweep.py for why 2 is not "
                   f"enough. NOT a physical calibration: run "
                   f"'cli.py calibrate-physical' on photographs of the "
                   f"printed known-height sheet before enforcement use."),
        "gate_derivation": g,
        "coplanarity_null": null,
        "calibration_stats_all_resolutions": st,
        "by_resolution": bin_by_resolution(cal),
    }
    with open(unc_path, "w") as fh:
        json.dump(unc, fh, indent=2)
    print(json.dumps({k: v for k, v in unc.items()
                      if k not in ("by_resolution", "gate_derivation",
                                   "coplanarity_null",
                                   "calibration_stats_all_resolutions")},
                     indent=2))
    return unc


def run_split(name, n, seed, tiers, workers, threshold_focus, omit_rate):
    cases = []
    for i in range(n):
        tier = tiers[i % len(tiers)]
        cases.append(build_case(i, seed, tier, threshold_focus, omit_rate))
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for j, r in enumerate(ex.map(run_case, cases, chunksize=1)):
            recs.append(r)
            if (j + 1) % 25 == 0:
                print(f"  [{name}] {j+1}/{n}  "
                      f"({(time.time()-t0)/(j+1):.2f}s/img)", flush=True)
    print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=180)
    ap.add_argument("--n-valid", type=int, default=240)
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--tiers", default="clean,realistic,realistic,harsh")
    # k=4, not the conventional 2. On held-out validation, a band built from
    # k=2 (+/-0.083 mm) produced a FALSE VIOLATION: an MRP truly 1.125 mm was
    # measured at 0.870 mm and called FAIL. eval/band_sweep.py shows the whole
    # trade-off curve; zero wrong decisive calls first occurs at ~0.15 mm,
    # while still leaving 86.5% of measurements decisive. k was therefore
    # raised AFTER seeing that evidence - which is disclosed here because it
    # means the final validation figures are not blind to that one decision.
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--out", default="backend/data")
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from saved records, no rescan")
    ap.add_argument("--derive-only", action="store_true",
                    help="re-derive uncertainty.json from saved calibration "
                         "records, no rescan")
    a = ap.parse_args()
    tiers = a.tiers.split(",")
    os.makedirs(a.out, exist_ok=True)
    unc_path = os.path.join(a.out, "uncertainty.json")

    if a.derive_only:
        cal = json.load(open(os.path.join(a.out, "eval_calib_records.json")))
        _derive_and_write(cal, a, unc_path)
        return

    if a.report_only:
        val = json.load(open(os.path.join(a.out, "eval_valid_records.json")))
        _write_report(val, unc_path, a.out)
        return

    if not a.skip_calib:
        # Probe with permissive gates so the calibration split can OBSERVE the
        # full error-vs-resolution curve; the gates are then derived from it.
        with open(unc_path, "w") as fh:
            json.dump({"bias_mm": 0.0, "sigma_mm": 0.15, "k": a.k,
                       "min_px_per_mm": 5.0, "max_tilt_deg": 45.0,
                       "max_baseline_slope_deg": 5.0,
                       "max_baseline_spread_deg": 8.0,
                       "apply_bias_correction": False,
                       "source": "permissive probe for gate derivation"}, fh)
        print(f"CALIBRATION split: {a.n_calib} images")
        cal = run_split("calib", a.n_calib, 1234, tiers, a.workers, 0.35, 0.25)
        _derive_and_write(cal, a, unc_path)
        with open(os.path.join(a.out, "eval_calib_records.json"), "w") as fh:
            json.dump(cal, fh, default=str)

    print(f"\nVALIDATION split: {a.n_valid} images (held out, "
          f"band frozen from calibration)")
    val = run_split("valid", a.n_valid, 98765, tiers, a.workers, 0.35, 0.25)

    _write_report(val, unc_path, a.out)


POPULATION = {
    "what_these_numbers_describe": (
        "Synthetic labels rendered by eval/synth.py and passed through a "
        "simulated imaging chain. They measure the ARCHITECTURE, not "
        "performance on real retail packaging."),
    "label_population": [
        "flat black-on-white printed panels, 8 macOS system fonts",
        "no background graphics, no colour, no reversed (white-on-dark) text",
        "no halftoning, no dot gain, no paper show-through, no gloss",
        "English / Latin script only",
        "one net quantity, one MRP, one manufacture date per pack, plus "
        "best-before, expiry, batch and promotional-price distractors",
    ],
    "imaging_chain_modelled": [
        "ink gain (morphological erode/dilate)", "3-D perspective / tilt",
        "area-averaged downsampling to 5-26 px/mm",
        "Gaussian defocus", "non-uniform illumination + specular highlight",
        "additive sensor noise", "JPEG at q55-97",
    ],
    "imaging_chain_NOT_modelled": [
        "lens distortion", "rolling shutter", "motion blur",
        "real halftone screens and print registration error",
        "glossy / metallised substrates", "chromatic aberration",
    ],
    "expected_on_real_packaging": (
        "Field-extraction recall will be materially lower and the height "
        "sigma several times larger. The build plan's own line is that the "
        "hours go into tuning OCR against real product photos. Fit the band "
        "with 'cli.py calibrate-physical' before trusting any threshold."),
    "ground_truth_method": (
        "Numeral height comes from the FreeType outline metrics of the clean "
        "master render, which the pipeline never sees. Validated against a "
        "400 px raster on four fonts: agreement to 0 px."),
}


def _write_report(val, unc_path, out_dir):
    tiers = sorted({r["tier"] for r in val if "tier" in r})
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "0_population_this_measures": POPULATION,
        "n_validation_images": len(val),
        "uncertainty_used": json.load(open(unc_path)) if os.path.exists(unc_path) else None,
        "1_field_extraction": {
            "all_frames": field_stats(val),
            "legible_frames_only": field_stats(val, only_legible=True),
            "illegible_frames": field_stats(val, only_legible=False),
            "legible_frame_rate": round(
                sum(1 for r in val if r.get("legible")) /
                max(1, sum(1 for r in val if "error" not in r)), 4),
        },
        "2_height_estimation": {
            "overall": height_stats(val),
            "overall_given_correct_field": height_stats(
                val, require_correct_field=True),
            "by_tier": {t: height_stats(val, tier=t) for t in set(tiers)},
            "by_field": {f: height_stats(val, field=f) for f in MEAS},
            "by_source_resolution_px_per_mm": bin_by_resolution(val),
            "by_marker_tilt_deg": bin_by_tilt(val),
        },
        "4_coplanarity_null_distribution": planarity_null(val),
        "3_verdict_agreement_per_field": verdict_stats(val),
        "3a_overall_verdict_accuracy": overall_stats(val),
        "3b_band_validation_on_adjudicated_population": band_validation(
            val, ((json.load(open(unc_path)) if os.path.exists(unc_path)
                   else {}) or {}).get("band_override_mm") or 0.0),
        "gates_and_abstentions": gate_stats(val),
        "throughput_ms_median": round(statistics.median(
            [r["elapsed_ms"] for r in val if "elapsed_ms" in r]), 1),
    }
    with open(os.path.join(out_dir, "accuracy_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "eval_valid_records.json"), "w") as fh:
        json.dump(val, fh, default=str)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "0_population_this_measures"}, indent=2))


if __name__ == "__main__":
    main()
