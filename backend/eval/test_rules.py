"""Unit tests for the deterministic legal layer. Run: python backend/eval/test_rules.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmpc.fields import extract_fields
from lmpc.ocr import Line, OcrResult, Word
from lmpc.rules import (PrintStyle, Uncertainty, Verdict, parse_net_quantity,
                        required_height_mm, verdict_for)

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")


def approx(name, got, want, tol=1e-9):
    if got is None or abs(got - want) > tol:
        FAILED.append(f"{name}: got {got!r}, want ~{want!r}")


# ---- Rule 7 Table I boundaries (inclusive upper bounds) -------------------
for text, want in [("200 g", 1.0), ("199 g", 1.0), ("201 g", 2.0),
                   ("500 g", 2.0), ("501 g", 4.0), ("1 kg", 4.0),
                   ("100 ml", 1.0), ("500 ml", 2.0), ("1 L", 4.0),
                   ("2 x 100 g", 1.0), ("2 x 150 g", 2.0)]:
    q = parse_net_quantity(text)
    lk = required_height_mm("net_quantity", q, PrintStyle.NORMAL)
    approx(f"TableI[{text}]", lk.required_mm, want)

# embossed column
approx("TableI embossed 500g",
       required_height_mm("net_quantity", parse_net_quantity("500 g"),
                          PrintStyle.EMBOSSED).required_mm, 4.0)

# ---- Table II requires a panel area, and says so -------------------------
q = parse_net_quantity("12 N")
lk = required_height_mm("net_quantity", q, PrintStyle.NORMAL)
check("TableII without area is refused", lk.ok, False)
check("TableII names the table", lk.table, "II")
lk = required_height_mm("net_quantity", q, PrintStyle.NORMAL, panel_area_cm2=250)
approx("TableII 250cm2", lk.required_mm, 2.0)
lk = required_height_mm("net_quantity", q, PrintStyle.NORMAL, panel_area_cm2=3000)
approx("TableII 3000cm2", lk.required_mm, 6.0)

# ---- MRP: floor binds, table is advisory only ----------------------------
lk = required_height_mm("mrp", parse_net_quantity("1 kg"), PrintStyle.NORMAL)
approx("MRP binding = 7(3) floor", lk.required_mm, 1.0)
approx("MRP advisory = Table I", lk.advisory_mm, 4.0)

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
