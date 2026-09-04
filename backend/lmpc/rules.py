"""
Legal layer: Legal Metrology (Packaged Commodities) Rules, 2011.

Every number in this module is HARDCODED from the Rules text and is looked up
deterministically. No model, heuristic or learned component is allowed to
produce a legal threshold.

Primary source used for the tables: Rule 7 ("Size of letters and numerals") of
the Legal Metrology (Packaged Commodities) Rules, 2011, as reproduced in the
project's problem-statement build plan (SIH26034, Section 4).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------
# Rule 7 threshold tables
# --------------------------------------------------------------------------

# Table I - net quantity declared by WEIGHT or VOLUME.
# (upper_bound_inclusive_in_g_or_ml, normal_print_mm, embossed_mm)
# The final row uses `None` as "no upper bound".
RULE7_TABLE_I = [
    (200.0, 1.0, 2.0),
    (500.0, 2.0, 4.0),
    (None, 4.0, 6.0),
]

# Table II - net quantity declared by LENGTH, AREA or NUMBER.
# Keyed on the area of the principal display panel, in cm^2.
# (upper_bound_inclusive_cm2, normal_print_mm, embossed_mm)
RULE7_TABLE_II = [
    (100.0, 1.0, 2.0),
    (500.0, 2.0, 4.0),
    (2500.0, 4.0, 6.0),
    (None, 6.0, 6.0),
]

# Rule 7(3): general minimum letter height, independent of the tables above.
RULE7_GENERAL_FLOOR_MM = {"normal": 1.0, "embossed": 2.0}


class PrintStyle(str, Enum):
    NORMAL = "normal"
    EMBOSSED = "embossed"


class QuantityKind(str, Enum):
    WEIGHT_VOLUME = "weight_or_volume"   # -> Table I
    LENGTH_AREA_NUMBER = "length_area_number"  # -> Table II


# --------------------------------------------------------------------------
# Which threshold attaches to which declaration
# --------------------------------------------------------------------------
# This mapping is a LEGAL INTERPRETATION, not a measurement. It is isolated
# here, one row per declaration, with a source note, so that it can be
# corrected by a domain expert without touching the pipeline.
#
# The build plan (Section 3.3) explicitly flags this as "verify against the
# Rules text before finalising". Until that verification is done we take the
# conservative-for-the-accused reading: Rule 7's Table I/II is indexed BY the
# net quantity and is applied to the net-quantity numerals; every other
# mandatory declaration is held to the Rule 7(3) general 1 mm floor, and the
# stricter Table reading is reported alongside as ADVISORY ONLY.

class ThresholdBasis(str, Enum):
    TABLE = "rule7_table"            # Table I or II, chosen by quantity kind
    GENERAL_FLOOR = "rule7_3_floor"  # flat 1 mm / 2 mm


@dataclass(frozen=True)
class DeclarationRule:
    field_name: str
    basis: ThresholdBasis
    advisory_basis: Optional[ThresholdBasis]
    source_note: str


DECLARATION_HEIGHT_RULES: dict[str, DeclarationRule] = {
    "net_quantity": DeclarationRule(
        "net_quantity",
        ThresholdBasis.TABLE,
        None,
        "Rule 7(1) Table I/II. The table is indexed by the net quantity and is "
        "applied to the numerals of the net-quantity declaration. Settled.",
    ),
    "mrp": DeclarationRule(
        "mrp",
        ThresholdBasis.GENERAL_FLOOR,
        ThresholdBasis.TABLE,
        "Rule 7(3) general floor applied as the BINDING threshold. A stricter "
        "reading of Rule 7(1) applies the Table I/II height to every mandatory "
        "numeral including MRP; that value is reported as advisory only. "
        "VERIFY AGAINST THE BARE ACT BEFORE ENFORCEMENT USE.",
    ),
    "mfg_date": DeclarationRule(
        "mfg_date", ThresholdBasis.GENERAL_FLOOR, ThresholdBasis.TABLE,
        "Rule 7(3) general floor. Same open question as MRP.",
    ),
}

DEFAULT_DECLARATION_RULE = DeclarationRule(
    "*", ThresholdBasis.GENERAL_FLOOR, None,
    "Rule 7(3) general minimum letter height.",
)


def declaration_rule(field_name: str) -> DeclarationRule:
    return DECLARATION_HEIGHT_RULES.get(field_name, DEFAULT_DECLARATION_RULE)


# --------------------------------------------------------------------------
# Rule 6 - mandatory declarations on every pre-packaged commodity
# --------------------------------------------------------------------------
MANDATORY_DECLARATIONS = [
    ("manufacturer", "Name and address of manufacturer / packer / importer"),
    ("commodity_name", "Common or generic name of the commodity"),
    ("net_quantity", "Net quantity"),
    ("mfg_date", "Month and year of manufacture / pre-packing / import"),
    ("mrp", "Retail sale price (MRP) inclusive of all taxes"),
    ("consumer_care", "Consumer care name / address / phone / email"),
]


# --------------------------------------------------------------------------
# Net-quantity parsing
# --------------------------------------------------------------------------
_UNIT_TO_BASE = {
    # mass -> grams
    "mg": ("g", 0.001), "g": ("g", 1.0), "gm": ("g", 1.0), "gms": ("g", 1.0),
    "gram": ("g", 1.0), "grams": ("g", 1.0), "kg": ("g", 1000.0),
    "kgs": ("g", 1000.0), "kilogram": ("g", 1000.0),
    # volume -> millilitres
    "ml": ("ml", 1.0), "mls": ("ml", 1.0), "millilitre": ("ml", 1.0),
    "cl": ("ml", 10.0), "l": ("ml", 1000.0), "ltr": ("ml", 1000.0),
    "ltrs": ("ml", 1000.0), "litre": ("ml", 1000.0), "liter": ("ml", 1000.0),
    "lt": ("ml", 1000.0),
    # length / area / number -> Table II
    "mm": ("len", 1.0), "cm": ("len", 10.0), "m": ("len", 1000.0),
    "n": ("count", 1.0), "no": ("count", 1.0), "nos": ("count", 1.0),
    "pcs": ("count", 1.0), "piece": ("count", 1.0), "pieces": ("count", 1.0),
    "u": ("count", 1.0), "units": ("count", 1.0),
}

# "2 x 100 g", "500g", "1.5 L", "250 ml", "10 N"
_QTY_RE = re.compile(
    r"(?:(?P<mult>\d{1,3})\s*(?:x|X|×)\s*)?"
    r"(?P<value>\d{1,6}(?:[.,]\d{1,3})?)\s*"
    r"(?P<unit>mg|kgs|kg|kilogram|gms|gm|grams|gram|g|"
    r"mls|ml|millilitre|cl|ltrs|ltr|litre|liter|lt|l|"
    r"mm|cm|m|nos|no|pcs|pieces|piece|units|u|n)\b",
    re.IGNORECASE,
)


# Characters Tesseract most often substitutes for a unit on a real label.
# Applied ONLY when a strict parse has already failed, and the result is
# flagged so the report can say the reading was recovered, not read cleanly.
# Digit-only substitutions are deliberately absent: treating a bare "8" as a
# unit turns "Rs. 108" into "10 g". Only alphabetic and symbol confusions are
# corrected, and only when a "Net Qty." key phrase anchored the match.
_UNIT_CONFUSIONS = {
    "c": "g", "\u00a2": "g", "q": "g", "qm": "gm", "qms": "gms",
    "kq": "kg", "ka": "kg", "hg": "kg", "kgs": "kg",
    "mi": "ml", "rnl": "ml", "rni": "ml", "mt": "ml", "mj": "ml", "mis": "ml",
    "i": "l", "|": "l", "lir": "ltr", "itr": "ltr",
}


@dataclass
class NetQuantity:
    raw: str
    value: float             # as printed, e.g. 1.5
    unit: str                # as printed, e.g. "kg"
    multiplier: int          # 1, or N for "N x value unit"
    base_value: float        # total, in grams / ml / mm / count
    base_unit: str           # "g" | "ml" | "len" | "count"
    kind: QuantityKind
    ocr_recovered: bool = False


_LOOSE_QTY_RE = re.compile(
    r"(?:(?P<mult>\d{1,3})\s*(?:x|X|\u00d7)\s*)?"
    r"(?P<value>\d{1,6}(?:[.,]\d{1,3})?)\s*"
    r"(?P<unit>[A-Za-z\u00a2|]{1,4})(?![A-Za-z])")


def parse_net_quantity(text: str,
                       allow_ocr_confusions: bool = True) -> Optional[NetQuantity]:
    """Parse a printed net-quantity declaration into a normalised quantity."""
    if not text:
        return None
    best: Optional[NetQuantity] = None
    for m in _QTY_RE.finditer(text):
        unit_raw = m.group("unit").lower()
        base_unit, factor = _UNIT_TO_BASE[unit_raw]
        value = float(m.group("value").replace(",", "."))
        mult = int(m.group("mult")) if m.group("mult") else 1
        base_value = value * factor * mult
        kind = (QuantityKind.WEIGHT_VOLUME if base_unit in ("g", "ml")
                else QuantityKind.LENGTH_AREA_NUMBER)
        cand = NetQuantity(m.group(0).strip(), value, unit_raw, mult,
                           base_value, base_unit, kind)
        # Prefer weight/volume declarations; they are the ones Table I keys on.
        if best is None or (best.kind is not QuantityKind.WEIGHT_VOLUME
                            and cand.kind is QuantityKind.WEIGHT_VOLUME):
            best = cand
    if best is not None or not allow_ocr_confusions:
        return best

    # Second pass: the number read cleanly but the unit did not
    # ("250 g" recognised as "250 c"). Only single-token units are corrected,
    # and only against an explicit confusion table.
    for m in _LOOSE_QTY_RE.finditer(text):
        tok = m.group("unit").lower()
        fixed = _UNIT_CONFUSIONS.get(tok)
        if fixed is None or fixed not in _UNIT_TO_BASE:
            continue
        base_unit, factor = _UNIT_TO_BASE[fixed]
        value = float(m.group("value").replace(",", "."))
        mult = int(m.group("mult")) if m.group("mult") else 1
        kind = (QuantityKind.WEIGHT_VOLUME if base_unit in ("g", "ml")
                else QuantityKind.LENGTH_AREA_NUMBER)
        cand = NetQuantity(m.group(0).strip(), value, fixed, mult,
                           value * factor * mult, base_unit, kind, True)
        if best is None or (best.kind is not QuantityKind.WEIGHT_VOLUME
                            and cand.kind is QuantityKind.WEIGHT_VOLUME):
            best = cand
    return best


# --------------------------------------------------------------------------
# Threshold lookup
# --------------------------------------------------------------------------
@dataclass
class ThresholdLookup:
    required_mm: Optional[float]
    table: str                 # "I" | "II" | "7(3)"
    band_label: str
    ok: bool
    reason: str = ""
    advisory_mm: Optional[float] = None
    advisory_table: Optional[str] = None


def _pick(rows, key):
    for upper, normal, embossed in rows:
        if upper is None or key <= upper:
            return upper, normal, embossed
    return rows[-1]


def table_height_mm(qty: NetQuantity, style: PrintStyle,
                    panel_area_cm2: Optional[float]) -> ThresholdLookup:
    """Rule 7(1) Table I / Table II lookup."""
    if qty.kind is QuantityKind.WEIGHT_VOLUME:
        upper, normal, embossed = _pick(RULE7_TABLE_I, qty.base_value)
        label = (f"up to {upper:g} {qty.base_unit}" if upper
                 else f"above 500 {qty.base_unit}")
        return ThresholdLookup(
            normal if style is PrintStyle.NORMAL else embossed,
            "I", label, True)

    # Table II is keyed on the principal-display-panel area, which cannot be
    # recovered from a photograph of one face. It must be supplied.
    if panel_area_cm2 is None:
        return ThresholdLookup(
            None, "II", "", False,
            "Net quantity is declared by length/area/number, so Rule 7 Table II "
            "applies. Table II is keyed on the area of the principal display "
            "panel, which cannot be measured from a single photograph. Supply "
            "panel_area_cm2 to obtain a threshold.",
        )
    upper, normal, embossed = _pick(RULE7_TABLE_II, panel_area_cm2)
    label = (f"panel area up to {upper:g} cm2" if upper
             else "panel area above 2500 cm2")
    return ThresholdLookup(
        normal if style is PrintStyle.NORMAL else embossed, "II", label, True)


def required_height_mm(field_name: str, qty: Optional[NetQuantity],
                       style: PrintStyle = PrintStyle.NORMAL,
                       panel_area_cm2: Optional[float] = None) -> ThresholdLookup:
    """Binding minimum glyph height for one declaration, plus any advisory."""
    rule = declaration_rule(field_name)
    floor = RULE7_GENERAL_FLOOR_MM[style.value]

    table_lk = table_height_mm(qty, style, panel_area_cm2) if qty else None

    if rule.basis is ThresholdBasis.TABLE:
        if table_lk is None:
            return ThresholdLookup(
                None, "I/II", "", False,
                "Net quantity could not be parsed, so the Rule 7 table row "
                "cannot be selected.")
        return table_lk

    out = ThresholdLookup(floor, "7(3)", f"general floor ({style.value})", True)
    if rule.advisory_basis is ThresholdBasis.TABLE and table_lk and table_lk.ok:
        out.advisory_mm = table_lk.required_mm
        out.advisory_table = table_lk.table
    return out


# --------------------------------------------------------------------------
# Rule 7(2) - width advisory
# --------------------------------------------------------------------------
# Reported as an advisory observation only; it does not drive the verdict,
# because the exact sub-rule text has not been verified for this build.
RULE7_2_MIN_WIDTH_RATIO = 1.0 / 3.0
RULE7_2_EXEMPT_GLYPHS = {"1", "i", "l", "I"}


# --------------------------------------------------------------------------
# Three-tier verdict (build plan Section 3.3)
# --------------------------------------------------------------------------
class Verdict(str, Enum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass
class Uncertainty:
    """Empirically-measured error of the height estimator, in mm.

    `bias_mm` is the signed mean error (estimate - truth) and `sigma_mm` the
    standard deviation, both taken from the DEGRADED evaluation tier - not
    from clean renders. See eval/run_eval.py.
    """
    bias_mm: float = 0.0
    sigma_mm: float = 0.15
    k: float = 2.0
    min_px_per_mm: float = 12.0
    max_tilt_deg: float = 35.0
    max_baseline_slope_deg: float = 0.20
    max_baseline_spread_deg: float = 1.20
    source: str = "conservative default (no calibration run recorded)"
    # Subtracting the fitted bias from a measurement before adjudicating it is
    # a silent adjustment to a legal verdict. It stays OFF unless the bias was
    # fitted PHYSICALLY (ruler-measured samples through the real camera), and
    # even then the report states that it was applied. The bias is always
    # accounted for conservatively through the width of the band instead.
    apply_bias_correction: bool = False
    # Empirical band, used in preference to k*sigma when set. The error
    # distribution has a heavier tail than a Gaussian (a handful of glyph
    # mis-segmentations dominate), so a percentile of the observed absolute
    # error describes it better than a multiple of its standard deviation.
    band_override_mm: Optional[float] = None

    def band_mm(self) -> float:
        return (self.band_override_mm if self.band_override_mm is not None
                else abs(self.bias_mm) + self.k * self.sigma_mm)


def verdict_for(measured_mm: float, required_mm: float,
                unc: Uncertainty) -> tuple[Verdict, float]:
    band = unc.band_mm()
    value = measured_mm - unc.bias_mm if unc.apply_bias_correction else measured_mm
    if value >= required_mm + band:
        return Verdict.PASS, band
    if value <= required_mm - band:
        return Verdict.FAIL, band
    return Verdict.BORDERLINE, band
