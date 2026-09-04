"""Human-readable compliance report."""
from __future__ import annotations


def render_text(rep: dict) -> str:
    L: list[str] = []
    a = L.append
    a("=" * 74)
    a("  LEGAL METROLOGY (PACKAGED COMMODITIES) RULES, 2011")
    a("  COMPLIANCE SCAN REPORT")
    a("=" * 74)
    a(f"Scan ID   : {rep['scan_id']}")
    a(f"Timestamp : {rep['timestamp']}      ({rep['elapsed_ms']} ms)")
    a(f"VERDICT   : {rep['overall']}")
    a("")
    c = rep["calibration"]
    a("-- Physical calibration " + "-" * 50)
    if c["ok"]:
        a(f"   method {c['method']}, marker id {c['marker_id']} "
          f"({c['marker_length_mm']:g} mm)")
        a(f"   scale  {c['px_per_mm_at_marker']} px/mm at marker | "
          f"tilt {c['tilt_deg']} deg | taper {c['taper']}")
        if not c["usable_for_measurement"]:
            a("   NOT USABLE for measurement (see gates below)")
    else:
        a(f"   FAILED: {c['reason']}")
    for g in rep.get("gates", []):
        a(f"   [{g['gate']}] {g['detail']}")
    a("")
    a("-- Mandatory declarations (Rule 6) " + "-" * 39)
    for d in rep["declarations"]:
        mark = {"PASS": "OK  ", "FAIL": "MISS", "NOT_ASSESSED": "?   "}.get(
            d["presence_verdict"], "?   ")
        a(f"  [{mark}] {d['requirement']}")
        if d["value_text"]:
            a(f"          value: {d['value_text']}")
        for n in d["notes"]:
            a(f"          note : {n}")
    a("")
    a("-- Rule 7 letter/numeral height " + "-" * 42)
    any_h = False
    for d in rep["declarations"]:
        h = d.get("height")
        if not h:
            continue
        any_h = True
        a(f"  {d['field']}:")
        if h.get("measured_mm") is not None:
            a(f"     measured {h['measured_mm']:.3f} mm   "
              f"required {h['required_mm']:.2f} mm   "
              f"({h['rule']}, {h['rule_row']})")
            a(f"     verdict  {h['verdict']}   "
              f"error band +/-{h['error_band_mm']:.3f} mm "
              f"from {h['n_numerals_measured']} numeral(s) at "
              f"{h.get('source_px_per_mm')} px/mm")
            if h.get("advisory_required_mm"):
                a(f"     advisory under the stricter reading "
                  f"({h['advisory_rule']}): required "
                  f"{h['advisory_required_mm']:.2f} mm -> "
                  f"{h.get('advisory_verdict')}")
        else:
            a(f"     NOT ASSESSED - {h.get('reason')}")
    if not any_h:
        a("  (no declaration was measurable in this frame)")
    a("")
    if rep["violations"]:
        a("-- VIOLATIONS " + "-" * 60)
        for v in rep["violations"]:
            a(f"  * {v}")
        a("")
    if rep["borderline"]:
        a("-- BORDERLINE - manual verification recommended " + "-" * 26)
        for v in rep["borderline"]:
            a(f"  * {v}")
        a("")
    if rep["not_assessed"]:
        a("-- NOT ASSESSED " + "-" * 57)
        for v in rep["not_assessed"]:
            a(f"  * {v}")
        a("")
    u = rep["uncertainty"]
    a("-- Measurement uncertainty " + "-" * 47)
    a(f"   bias {u['bias_mm']:+.4f} mm, sigma {u['sigma_mm']:.4f} mm, "
      f"k={u['k']} -> band +/-{u['band_mm']:.3f} mm")
    a(f"   {u['source']}")
    a("")
    a("-- Scope of this assessment " + "-" * 46)
    for s in rep["scope"]["out_of_scope"]:
        a(f"   NOT covered: {s}")
    a(f"   Requires: {rep['scope']['hard_requirement']}")
    a("=" * 74)
    return "\n".join(L)
