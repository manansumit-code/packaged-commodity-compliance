"""Unit tests for the deterministic legal layer. Run: python backend/eval/test_rules.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmpc.fields import extract_fields
from lmpc.ocr import Line, OcrResult, Word
from lmpc.rules import (PrintStyle, Uncertainty, Verdict,
                        panel_area_cm2_cylindrical, panel_area_cm2_other,
                        panel_area_cm2_rectangular, parse_net_quantity,
                        required_height_mm, verdict_for, width_ratio_verdict)

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")


def approx(name, got, want, tol=1e-9):
    if got is None or abs(got - want) > tol:
        FAILED.append(f"{name}: got {got!r}, want ~{want!r}")


# ---- Rule 7 Table-I boundaries -------------------------------------------
# As substituted by G.S.R. 629(E) of 23.6.2017: ONE table, keyed on the area
# of the principal display panel in cm2, applying to every declaration.
# The Act prints the rows as "A < 50", "50 < A < 100", ...; we read each row
# as "up to and including" its upper bound.
for area, want in [(10.0, 1.0), (49.9, 1.0), (50.0, 1.0),
                   (50.1, 1.5), (100.0, 1.5),
                   (100.1, 2.5), (500.0, 2.5),
                   (500.1, 4.0), (2500.0, 4.0),
                   (2500.1, 6.0), (10000.0, 6.0)]:
    lk = required_height_mm("net_quantity", None, PrintStyle.NORMAL, area)
    approx(f"TableI[{area} cm2]", lk.required_mm, want)
    check(f"TableI[{area} cm2] names Table-I", lk.table, "I")

# the blown / formed / moulded column
for area, want in [(10.0, 1.5), (75.0, 3.0), (250.0, 4.0),
                   (1000.0, 6.0), (5000.0, 6.0)]:
    approx(f"TableI moulded[{area} cm2]",
           required_height_mm("net_quantity", None, PrintStyle.EMBOSSED,
                              area).required_mm, want)

# ---- the table is keyed on panel area, so it is refused without one ------
# This is the whole point of the 2017 amendment: net quantity no longer
# selects a row, so knowing "500 g" tells you nothing about the threshold.
lk = required_height_mm("net_quantity", parse_net_quantity("500 g"),
                        PrintStyle.NORMAL)
check("Table-I without a panel area is refused", lk.ok, False)
check("Table-I refusal names Table-I", lk.table, "I")
check("Table-I refusal explains why",
      "principal display panel" in lk.reason.lower(), True)
lk = required_height_mm("net_quantity", parse_net_quantity("12 N"),
                        PrintStyle.NORMAL)
check("a count declaration is refused the same way", lk.ok, False)

# ---- every declaration takes the same table -----------------------------
# Rule 7(2) carves out no declaration, and Rule 7(5) names net weight, retail
# sale price, expiry date and consumer care details expressly. The old build
# held MRP and the manufacture date to a flat 1 mm floor that G.S.R. 629(E)
# had already deleted; that under-enforced them by up to 5 mm.
for field in ("net_quantity", "mrp", "mfg_date", "consumer_care",
              "manufacturer"):
    lk = required_height_mm(field, parse_net_quantity("1 kg"),
                            PrintStyle.NORMAL, panel_area_cm2=623.7)
    approx(f"{field} takes Table-I", lk.required_mm, 4.0)
    check(f"{field} is not on the deleted floor", lk.table, "I")

# the superseded reading is still disclosed, but never binds
lk = required_height_mm("mrp", parse_net_quantity("1 kg"), PrintStyle.NORMAL,
                        panel_area_cm2=250.0)
approx("binding value is the amended table", lk.required_mm, 2.5)
approx("pre-2017 reading is reported alongside", lk.advisory_mm, 4.0)
check("pre-2017 reading is labelled as such",
      "pre-2017" in (lk.advisory_table or ""), True)

# ---- Rule 7(3): width not less than one third of height ------------------
check("clearly wide enough passes", width_ratio_verdict(0.55, "185")[0],
      Verdict.PASS.value)
check("clearly too narrow fails", width_ratio_verdict(0.20, "185")[0],
      Verdict.FAIL.value)
check("near the one-third line is borderline",
      width_ratio_verdict(0.33, "185")[0], Verdict.BORDERLINE.value)
check("an all-exempt declaration is not adjudicated",
      width_ratio_verdict(0.10, "111")[0], Verdict.NOT_ASSESSED.value)
check("an unmeasured width is not adjudicated",
      width_ratio_verdict(None, "185")[0], Verdict.NOT_ASSESSED.value)

# ---- Rule 7(4): how the panel area is computed ---------------------------
approx("7(4)(a) rectangular", panel_area_cm2_rectangular(29.7, 21.0), 623.7,
       tol=1e-6)
approx("7(4)(b) cylindrical is 40% of h x circumference",
       panel_area_cm2_cylindrical(12.0, circumference_cm=25.0), 120.0,
       tol=1e-6)
approx("7(4)(c) other shapes are 40% of total surface",
       panel_area_cm2_other(1000.0), 400.0, tol=1e-6)

# ---- quantity parsing ----------------------------------------------------
check("kg normalises", parse_net_quantity("Net Qty 1.5 kg").base_value, 1500.0)
check("multipack multiplies", parse_net_quantity("3 x 200 g").base_value, 600.0)
check("prefers weight over count",
      parse_net_quantity("10 N Net Wt 250 g").base_unit, "g")
check("no false unit from a price", parse_net_quantity("MRP Rs. 108"), None)
check("no false unit from a phone",
      parse_net_quantity("Customer Care 1800 266 2240"), None)
check("OCR unit repaired", parse_net_quantity("Net Qty 250 ¢").unit, "g")
check("repair is flagged", parse_net_quantity("Net Qty 250 ¢").ocr_recovered, True)

# ---- three-tier verdict --------------------------------------------------
u = Uncertainty(bias_mm=0.0, sigma_mm=0.05, k=2.0)      # band = 0.10
check("clear pass", verdict_for(2.50, 2.0, u)[0], Verdict.PASS)
check("clear fail", verdict_for(1.50, 2.0, u)[0], Verdict.FAIL)
check("just above threshold is borderline",
      verdict_for(2.05, 2.0, u)[0], Verdict.BORDERLINE)
check("just below threshold is borderline",
      verdict_for(1.95, 2.0, u)[0], Verdict.BORDERLINE)
check("exactly at threshold is borderline",
      verdict_for(2.00, 2.0, u)[0], Verdict.BORDERLINE)
check("bias is not silently subtracted",
      verdict_for(2.30, 2.0, Uncertainty(bias_mm=0.5, sigma_mm=0.01, k=2.0))[0],
      Verdict.BORDERLINE)
check("empirical band overrides k*sigma",
      verdict_for(2.30, 2.0, Uncertainty(sigma_mm=0.01, k=2.0,
                                         band_override_mm=0.5))[0],
      Verdict.BORDERLINE)


# ---- field classification against distractors ----------------------------
def mk(texts):
    lines = []
    for i, t in enumerate(texts):
        ws, x = [], 0
        for tok in t.split(" "):
            ws.append(Word(tok, x, i * 40, len(tok) * 10, 20, 90, (0, 0, 0)))
            x += len(tok) * 10 + 10
        sp, p = [], 0
        for w in ws:
            sp.append((p, p + len(w.text))); p += len(w.text) + 1
        lines.append(Line(t, ws, sp))
    return OcrResult(lines, [], "\n".join(texts), 1.0)


h = extract_fields(mk([
    "Net Qty.: 500 g", "MRP: Rs. 249.00 (inclusive of all taxes)",
    "Mfg. Date: 03/2025", "Best Before: 09/2025",
    "Batch/Lot No.: B04211 Expiry: 09/2026",
    "Special introductory price Rs. 199.00 | Rs. 49.80 per 100 g"]))
check("MRP not confused by promo price", h["mrp"].value_text, "249.00")
check("date not confused by best-before", h["mfg_date"].value_text, "03/2025")
check("net qty read", h["net_quantity"].value_text, "500 g")

h = extract_fields(mk(["Packed on: 07/2024   Best Before: 01/2025",
                       "Net Wt.: 200 g", "Rs. 88.50"]))
check("date after key, before rival key", h["mfg_date"].value_text, "07/2024")
check("bare price is a low-confidence MRP match",
      h["mrp"].match_confidence < 0.5, True)

h = extract_fields(mk(["Some Brand", "Net Content 250 g", "Best Before 05/2026"]))
check("best-before alone is not a mfg date", h["mfg_date"].present, False)

# wrapped declaration across lines
h = extract_fields(mk(["Month & Year of Manufacture:", "09/2023",
                       "Net Qty.: 100 g"]))
check("wrapped key/value recovered", h["mfg_date"].value_text, "09/2023")

# commodity name must not run on into the next declaration
h = extract_fields(mk(["Common name: Toor Dal", "Net Qty.: 500 g"]))
check("generic name stops at its line", h["commodity_name"].value_text, "Toor Dal")

# missing declarations are reported missing
h = extract_fields(mk(["Some Brand", "Ingredients: wheat, salt"]))
for f in ("mrp", "net_quantity", "mfg_date", "manufacturer", "consumer_care"):
    check(f"{f} absent", h[f].present, False)
check("generic name is excluded by rule, not failed",
      h["commodity_name"].assessable, False)

if FAILED:
    print(f"{len(FAILED)} FAILED:")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("all rule/field unit tests passed")
