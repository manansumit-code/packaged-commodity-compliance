"""Render accuracy_report.json as a readable summary."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "data", "accuracy_report.json")
r = json.load(open(PATH))
u = r.get("uncertainty_used") or {}
L = []
a = L.append

a("# Accuracy report\n")
a(f"Generated {r['generated']} on {r['n_validation_images']} held-out "
  f"validation images.\n")
a("**These numbers describe synthetic labels through a simulated imaging "
  "chain — they measure the architecture, not real retail packaging.** See "
  "`0_population_this_measures` in the JSON.\n")

a("\n## Gates, derived from the calibration split\n")
a(f"- resolution floor: **{u.get('min_px_per_mm')} px/mm** "
  f"(below this the height is refused)")
a(f"- error band: **±{u.get('band_override_mm')} mm** "
  f"(bias {u.get('bias_mm'):+}, sigma {u.get('sigma_mm')}, k={u.get('k')})")
a(f"- coplanarity: slope ≤ {u.get('max_baseline_slope_deg')}°, "
  f"spread ≤ {u.get('max_baseline_spread_deg')}°")
a(f"- max marker tilt: {u.get('max_tilt_deg')}°")
a(f"- bias subtracted from measurements: "
  f"**{u.get('apply_bias_correction')}**\n")
if u.get("gate_derivation", {}).get("sweep"):
    a("\n### Why that floor (error vs capture resolution)\n")
    a("| floor px/mm | n | bias mm | sigma mm | MAE mm | p95 abs mm |")
    a("|---|---|---|---|---|---|")
    for k, v in u["gate_derivation"]["sweep"].items():
        a(f"| ≥{k} | {v['n']} | {v['bias_mm']:+.4f} | {v['sigma_mm']:.4f} | "
          f"{v['mae_mm']:.4f} | {v['p95_abs_mm']:.4f} |")

a("\n## 1. Field extraction\n")
fe = r["1_field_extraction"]
a(f"Legible-frame rate: **{fe['legible_frame_rate']:.1%}** "
  f"(frames the system judged readable enough to claim a declaration absent)\n")
for scope in ("legible_frames_only", "all_frames"):
    a(f"\n### {scope.replace('_',' ')}\n")
    a("| declaration | recall | precision | adjudicated | abstained | "
      "value exact |")
    a("|---|---|---|---|---|---|")
    for f, v in fe[scope].items():
        if v.get("excluded_by_rule"):
            a(f"| {f} | — excluded by rule (n={v['n_excluded']}) | | | | |")
            continue
        vm = v.get("value_exact_match")
        a(f"| {f} | {v['recall']:.3f} | {v['precision']:.3f} | "
          f"{v['n_adjudicated']} | {v['n_abstained']} | "
          f"{(f'{vm:.3f}' if vm is not None else '—')} |")

a("\n## 2. Height estimation (mm)\n")
he = r["2_height_estimation"]
o = he["overall"]
a(f"Overall: bias **{o['bias_mm']:+.4f} mm**, sigma **{o['sigma_mm']:.4f} mm**, "
  f"MAE **{o['mae_mm']:.4f} mm**, p95 |e| **{o['p95_abs_mm']:.4f} mm** "
  f"(n={o['n']})")
a(f"Yield: **{o['yield_given_found']:.1%}** of located declarations produced a "
  f"height; **{o['yield_end_to_end']:.1%}** end-to-end of declarations "
  f"actually printed.")
c = he.get("overall_given_correct_field")
if c:
    a(f"\nConditioned on the extractor having read the RIGHT text "
      f"(classification failures excluded): bias **{c['bias_mm']:+.4f} mm**, "
      f"sigma **{c['sigma_mm']:.4f} mm**, MAE **{c['mae_mm']:.4f} mm**, "
      f"p95 |e| **{c['p95_abs_mm']:.4f} mm** (n={c['n']}). The gap between "
      f"this and the line above is misclassification, not mis-measurement.\n")
a("\n| split | n | bias mm | sigma mm | MAE mm | p95 abs mm | rel MAE % |")
a("|---|---|---|---|---|---|---|")
for label, v in [(f"tier: {k}", v) for k, v in he["by_tier"].items()] + \
                [(f"field: {k}", v) for k, v in he["by_field"].items()]:
    if not v:
        continue
    a(f"| {label} | {v['n']} | {v['bias_mm']:+.4f} | {v['sigma_mm']:.4f} | "
      f"{v['mae_mm']:.4f} | {v['p95_abs_mm']:.4f} | {v['mae_rel_pct']:.2f} |")

a("\n### by capture resolution\n")
a("| src px/mm | n | bias mm | sigma mm | MAE mm |")
a("|---|---|---|---|---|")
for k, v in he["by_source_resolution_px_per_mm"].items():
    a(f"| {k} | {v['n']} | {v['bias_mm']:+.4f} | {v['sigma_mm']:.4f} | "
      f"{v['mae_mm']:.4f} |")

a("\n### by marker tilt\n")
a("| tilt deg | n | bias mm | sigma mm | p95 abs mm |")
a("|---|---|---|---|---|")
for k, v in he.get("by_marker_tilt_deg", {}).items():
    a(f"| {k} | {v['n']} | {v['bias_mm']:+.4f} | {v['sigma_mm']:.4f} | "
      f"{v['p95_abs_mm']:.4f} |")

ov = r.get("3a_overall_verdict_accuracy") or {}
if ov:
    a("\n## 3. Overall verdict — the output the user actually sees\n")
    a("| ground truth | packs | non-compliant | REVIEW rate | decided | "
      "acc. when decided | violating pack CLEARED | compliant pack ACCUSED |")
    a("|---|---|---|---|---|---|---|---|")
    for m, v in ov.items():
        a(f"| {m} | {v['n_packs']} | {v['truth_non_compliant']} | "
          f"{v['review_required_rate']:.3f} | {v['decided_rate']:.3f} | "
          f"{v['accuracy_when_decided']:.3f} | "
          f"{v['violating_pack_cleared']} "
          f"({v['violating_pack_cleared_rate']:.3f}) | "
          f"{v['compliant_pack_accused']} "
          f"({v['compliant_pack_accused_rate']:.3f}) |")

a("\n## 3b. Verdict agreement per field\n")
a("| field | n | decisive | acc. on decisive | false CLEAR | false VIOLATION "
  "| borderline | not assessed |")
a("|---|---|---|---|---|---|---|---|")
for f, v in r["3_verdict_agreement_per_field"].items():
    a(f"| {f} | {v['n']} | {v['decisive_rate']:.3f} | "
      f"{v['accuracy_on_decisive']:.3f} | {v['false_clear']} "
      f"({v['false_clear_rate']:.3f}) | {v['false_violation']} "
      f"({v['false_violation_rate']:.3f}) | {v['borderline_rate']:.3f} | "
      f"{v['not_assessed_rate']:.3f} |")
a("")
for f, v in r["3_verdict_agreement_per_field"].items():
    n = v["near_threshold_0.3mm"]; c = v["clear_of_threshold"]
    a(f"- **{f}** — within 0.3 mm of the threshold (n={n['n']}): "
      f"{n['borderline_rate']:.1%} answered BORDERLINE, "
      f"{n['wrong_decisive']} wrong decisive calls. "
      f"Clear of the threshold (n={c['n']}): "
      f"accuracy {c['accuracy_on_decisive']:.3f}, "
      f"{c['wrong_decisive']} wrong decisive calls.")

bv = r.get("3b_band_validation_on_adjudicated_population") or {}
if bv:
    a("\n### Does the band cover the cases it adjudicates?\n")
    a(f"On the {bv['n_adjudicated_measurements']} validation measurements "
      f"that actually received a verdict: p95 |error| = "
      f"**{bv['p95_abs_error_mm']} mm**, p99 = {bv['p99_abs_error_mm']} mm, "
      f"max = {bv['max_abs_error_mm']} mm, against a fitted band of "
      f"**±{bv['fitted_band_mm']} mm**.")
    a(f"{bv['fraction_outside_band']:.2%} fall outside the band. "
      f"**{bv['verdict']}**\n")

cn = r.get("4_coplanarity_null_distribution") or {}
if cn:
    a("\n## 4. Coplanarity statistic, null distribution\n")
    a(f"On n={cn['n']} frames that ARE coplanar by construction: "
      f"|median baseline slope| p99 = "
      f"{cn['abs_median_slope_deg']['p99']}°, max = "
      f"{cn['abs_median_slope_deg']['max']}°; spread p99 = "
      f"{cn['slope_spread_deg']['p99']}°.")

a(f"\n## Throughput\n\nMedian **{r['throughput_ms_median']:.0f} ms** per scan "
  f"(2 page-segmentation modes; `fast=true` uses one).\n")

out = os.path.join(HERE, "..", "data", "ACCURACY.md")
open(out, "w").write("\n".join(L))
print("\n".join(L))
print(f"\n[written to {out}]", file=sys.stderr)
