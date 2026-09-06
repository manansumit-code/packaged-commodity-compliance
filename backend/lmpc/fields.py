"""
Field classification: which text block is which mandatory declaration.

Deterministic keyword/regex rules, on purpose (build plan Section 6). The
vocabulary of a legal declaration is bounded - "MRP", "M.R.P.", "Maximum
Retail Price", "Net Qty.", "Mfd. by" - so rules cover it, run in microseconds
per frame, work offline, and cannot invent a value that was not on the label.

Robustness to OCR noise comes from three places rather than from a model:
  * normalisation that erases the punctuation and spacing OCR gets wrong
    ("M.R.P." / "M R P" / "MRP" all collapse to MRP),
  * character classes for the classic confusions (0/O, 1/I/l, 5/S, 8/B),
  * a bounded fuzzy fallback on the keyword token only, never on the value.

The VALUE is always taken verbatim from the OCR text. Nothing in this module
can produce a number that is not literally present on the label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from difflib import SequenceMatcher
from typing import Optional

from .ocr import Line, OcrResult, Word
from .rules import parse_net_quantity


# --------------------------------------------------------------------------
# Normalisation with an index map back into the original string
# --------------------------------------------------------------------------
def normalize_with_map(s: str) -> tuple[str, list[int]]:
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch.upper())
            idx.append(i)
    return "".join(out), idx


def _orig_span(idx: list[int], a: int, b: int) -> tuple[int, int]:
    if not idx or a >= len(idx):
        return 0, 0
    b = min(b, len(idx))
    return idx[a], idx[b - 1] + 1


def _fuzzy_has(norm: str, token: str, thresh: float = 0.82) -> bool:
    """Sliding-window fuzzy containment - keyword tokens only."""
    n = len(token)
    if n < 3 or len(norm) < n:
        return False
    for i in range(0, len(norm) - n + 1):
        if SequenceMatcher(None, norm[i:i + n], token).ratio() >= thresh:
            return True
    return False


# --------------------------------------------------------------------------
# Patterns (applied to the NORMALISED, alnum-only, upper-cased line)
# --------------------------------------------------------------------------
MRP_KEYS = [r"MRP", r"MAXIMUMRETAILPRICE", r"MAXRETAILPRICE",
            r"RETAILSALEPRICE", r"MAXIMUMRETAILPRICERS", r"MAXIMUMPRICE"]
MRP_FUZZY = ["MAXIMUMRETAILPRICE", "RETAILSALEPRICE"]
MRP_CURRENCY = r"(?:RS|INR|R5|P5)?"
MRP_VALUE = r"(?P<val>\d{1,6}(?:\d{0,2})?)"

NETQTY_KEYS = [r"NETQTY", r"NETQUANTITY", r"NETWT", r"NETWEIGHT",
               r"NETCONTENT", r"NETCONTENTS", r"NETVOL", r"NETVOLUME"]
NETQTY_FUZZY = ["NETQUANTITY", "NETWEIGHT", "NETCONTENTS"]

# Normalisation strips "&" and ".", so every variant is listed in its
# post-normalisation form ("Month & Year of Manufacture" -> MONTHYEAROF...).
MFG_KEYS = [r"MFGDATE", r"DATEOFMFG", r"DATEOFMANUFACTURE", r"MFDON",
            r"MFGON", r"MANUFACTUREDON", r"PACKEDON", r"DATEOFPACKING",
            r"MONTHYEAROFMANUFACTURE", r"MONTHANDYEAROFMANUFACTURE",
            r"MONTHYEAROFPACKING", r"MONTHYEAROFIMPORT", r"DATEOFPKG",
            r"MFG", r"MFD", r"PKD", r"MFGDT", r"PKDON"]
MFG_FUZZY = ["DATEOFMANUFACTURE", "MANUFACTUREDON", "DATEOFPACKING"]

MAKER_KEYS = [r"MANUFACTUREDBY", r"MFDBY", r"MFGBY", r"PACKEDBY",
              r"MARKETEDBY", r"IMPORTEDBY", r"MANUFACTUREDANDPACKEDBY",
              r"MANUFACTURER"]
MAKER_FUZZY = ["MANUFACTUREDBY", "MARKETEDBY", "IMPORTEDBY", "PACKEDBY"]

CARE_KEYS = [r"CUSTOMERCARE", r"CONSUMERCARE", r"CUSTOMERCAREDETAILS",
             r"CONSUMERCAREDETAILS", r"FORCOMPLAINTS", r"FORQUERIES",
             r"CUSTOMERSERVICE", r"HELPLINE", r"TOLLFREE", r"GRIEVANCE"]
CARE_FUZZY = ["CUSTOMERCARE", "CONSUMERCARE", "CUSTOMERSERVICE"]

GENERIC_KEYS = [r"COMMONNAME", r"GENERICNAME", r"COMMONORGENERICNAME",
                r"NAMEOFCOMMODITY", r"PRODUCTNAME"]

TAX_PHRASE = [r"INCLUSIVEOFALLTAXES", r"INCLOFALLTAXES", r"INCLALLTAXES",
              r"INCOFALLTAXES"]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?:\b1800[\s\-]?\d{2,4}[\s\-]?\d{3,4}\b)|(?:\+?91[\s\-]?)?\b[6-9]\d{9}\b")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
DATE_PATTERNS = [
    re.compile(r"(?P<mon>0?[1-9]|1[0-2])[/\-.](?P<yr>20\d{2}|\d{2})"),
    re.compile(r"(?P<mname>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
               r"[A-Z]*[\s/\-.]*(?P<yr2>20\d{2}|\d{2})", re.IGNORECASE),
    re.compile(r"(?P<d>[0-3]?\d)[/\-.](?P<mon2>0?[1-9]|1[0-2])"
               r"[/\-.](?P<yr3>20\d{2}|\d{2})"),
]



# Negative keys. A pack carries other dates and other prices; the month and
# year of MANUFACTURE is a different declaration from a best-before or expiry
# date, and the maximum retail price is a different number from a promotional
# or per-unit price. Matching a date or an amount without excluding these is
# how a scanner reports a confident value for the wrong field.
MFG_NEGATIVE = [r"BESTBEFORE", r"BESTBEFOREEND", r"USEBY", r"USEBEFORE",
                r"EXPIRY", r"EXPDATE", r"EXPIRES", r"CONSUMEBEFORE",
                r"BESTBEF", r"EXP\b"]
MRP_NEGATIVE = [r"PER100G", r"PER100ML", r"PERKG", r"PERLITRE", r"PERUNIT",
                r"PERPIECE", r"INTRODUCTORY", r"SPECIALPRICE", r"OFFERPRICE",
                r"DISCOUNTEDPRICE", r"SAVE", r"WAS",
                # Legend-style labels (HUL and others) print the KEY in one
                # place and the VALUE somewhere else: "*MRP Rs (INCL. OF ALL
                # TAXES), USP, #MFD., B. NO. & @USE BEFORE: SEE BELOW." The
                # number nearest that key is not the price - it is whatever
                # digit the recogniser happened to find in the legend. On a
                # Pond's jar this read the MRP as "7".
                r"SEEBELOW", r"BELOW"]

# Lines that carry a number and a unit but are NOT a net-quantity
# declaration. "Size : 29.7 x 21 cm" is the dimensions of the article; read
# as a quantity it yields "21 cm", which is wrong twice over - it is not the
# net quantity, and it then anchors the Rule 7 height measurement onto the
# wrong numerals, so a height verdict gets attached to a field the pack never
# declared there.
NETQTY_NEGATIVE = [r"SIZE", r"DIMENSION", r"DIMENSIONS", r"LENGTH",
                   r"WIDTH", r"HEIGHT", r"THICKNESS", r"GSM"]


def _negative_cut(norm: str, idx: list[int], start_norm: int,
                  negatives) -> tuple[bool, Optional[int]]:
    """Where does the searchable region end because a rival key begins?

    Returns (blocked, cut_in_original_coords). `blocked` means a negative key
    sits BEFORE the value would - the whole line belongs to another
    declaration. Otherwise the search is truncated at the first negative key
    that follows, so "Mfg. 03/2025 Best Before 09/2025" still reads 03/2025.
    """
    first_after = None
    for pat in negatives:
        for m in re.finditer(pat, norm):
            if m.start() < start_norm:
                return True, None
            if first_after is None or m.start() < first_after:
                first_after = m.start()
    if first_after is None:
        return False, None
    return False, _orig_span(idx, first_after, first_after + 1)[0]


CURRENCY_RE = re.compile(r"(?:RS|INR|R5)\s*\.?\s*(?P<val>\d{1,6}(?:[.,]\d{1,2})?)",
                         re.IGNORECASE)

# The rupee sign and what Tesseract turns it into. Observed on a Pond's jar
# printed white-on-black: "AA* Rs 545/-" came back as "AA * = 545/" from one
# frame and "AA &545/-" from another - the glyph survives as "=", "&" or "%".
# These substitutes are far too weak to identify a price on their own, so
# they are only ever used together with the "/-" terminator below.
RUPEE_SIGNS = r"(?:RS|INR|R5|\u20b9|&|=|%|\?|\*)"

# A price written the Indian retail way: "545/-", "Rs. 545/-", "Rs 545/=".
# The trailing "/-" is what makes this safe to accept on a line carrying no
# MRP key at all: an ingredient list or an address does not contain it.
PRICE_SLASH_RE = re.compile(
    RUPEE_SIGNS + r"?\s*\.?\s*(?P<val>\d{1,6}(?:[.,]\d{1,2})?)\s*/\s*[-=<>~_]")

# A UNIT price - "Rs 2.73/ml", "49.80 per 100 g" - is not the retail sale
# price and must never be reported as one. Same jar prints both, side by side.
UNIT_PRICE_RE = re.compile(
    r"\d\s*(?:[.,]\d+)?\s*(?:/|per)\s*\d{0,3}\s*"
    r"(?:ml|mls|g|gm|gms|kg|kgs|l|ltr|litre|piece|pc|unit)\b", re.IGNORECASE)

# Legend-style labels declare the KEY in one place and the VALUE in another:
#   "*MRP Rs (INCL. OF ALL TAXES), USP, #MFD., B. NO. & @USE BEFORE: SEE BELOW."
# with the actual figures on a band underneath. HUL uses this across its
# range. Without following the pointer the pack reads as having no MRP at all.
MRP_LEGEND_MARKERS = [r"SEEBELOW", r"BELOW", r"OVERLEAF"]
MRP_LEGEND_LOOKAHEAD = 6
SHORT_MFG_TOKENS = ("MFD", "MFG", "PKD", "MFDBY")


# --------------------------------------------------------------------------
@dataclass
class FieldHit:
    name: str
    present: bool
    assessable: bool = True
    raw_text: str = ""
    value: Optional[object] = None
    value_text: str = ""
    value_boxes: list[tuple[int, int, int, int]] = dc_field(default_factory=list)
    line_bbox: Optional[tuple[int, int, int, int]] = None
    note: str = ""
    match_confidence: float = 0.0


def _boxes(words: list[Word]) -> list[tuple[int, int, int, int]]:
    return [(w.x, w.y, w.w, w.h) for w in words]


def union_box(boxes) -> Optional[tuple[int, int, int, int]]:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


# --------------------------------------------------------------------------
class Window:
    """A key-anchored search region spanning one line and its continuations.

    A declaration is frequently not confined to one OCR line - it wraps
    ("Manufactured by:" / "SURYA FOODS Pvt. Ltd." / street / city - PIN), and
    even when it does not wrap, OCR sometimes splits it. Searching only within
    a single line is the single largest cause of missed declarations, so the
    key is anchored on one line and the value is sought across that line and
    the next few geometrically-plausible ones.
    """

    def __init__(self, lines: list[Line], i: int, span: int = 2):
        self.lines = lines
        self.i = i
        parts: list[str] = []
        owner: list[Optional[tuple[int, int]]] = []
        idxs = [i]
        bx, by, bw, bh = lines[i].bbox
        prev_bottom, prev_h = by + bh, bh
        for j in range(i + 1, min(len(lines), i + 1 + span)):
            nx, ny, nw, nh = lines[j].bbox
            if ny - prev_bottom > 1.6 * max(prev_h, nh):
                break                       # too far below to be a wrap
            if nx > bx + bw or nx + nw < bx:
                break                       # no horizontal overlap
            idxs.append(j)
            prev_bottom, prev_h = ny + nh, nh
        for k, j in enumerate(idxs):
            if k:
                parts.append(" ")
                owner.append(None)
            t = lines[j].text
            parts.append(t)
            owner.extend((j, c) for c in range(len(t)))
        self.text = "".join(parts)
        self.owner = owner
        self.idxs = idxs

    def line_at(self, pos: int) -> Optional[int]:
        if 0 <= pos < len(self.owner) and self.owner[pos]:
            return self.owner[pos][0]
        return None

    def words_for_span(self, a: int, b: int) -> list[Word]:
        per: dict[int, list[int]] = {}
        for p in range(max(0, a), min(len(self.owner), b)):
            o = self.owner[p]
            if o:
                per.setdefault(o[0], []).append(o[1])
        out: list[Word] = []
        for j, cols in per.items():
            out.extend(self.lines[j].words_for_span(min(cols), max(cols) + 1))
        return out


def _windows(lines: list[Line], span: int = 2) -> list[Window]:
    return [Window(lines, i, span) for i in range(len(lines))]


def _key_in_first_line(win: Window, norm: str, idx: list[int],
                       keys, fuzzy) -> tuple[bool, float, int]:
    """Locate a key phrase, requiring it to start on the window's own line."""
    for k in keys:
        for m in re.finditer(k, norm):
            if win.line_at(idx[m.start()]) == win.i:
                return True, 1.0, m.end()
    first_len = len(win.lines[win.i].text)
    head = "".join(c for c in win.text[:first_len] if c.isalnum()).upper()
    for f in fuzzy:
        if _fuzzy_has(head, f):
            return True, 0.75, 0
    return False, 0.0, 0


def _neighbourhood(lines: list[Line], i: int, span: int = 2) -> str:
    lo, hi = max(0, i - span), min(len(lines), i + span + 1)
    return " ".join(l.text for l in lines[lo:hi])


def _parse_date(text: str):
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        g = m.groupdict()
        if g.get("mname"):
            mon, yr = _MONTHS[g["mname"][:3].upper()], g["yr2"]
        else:
            mon, yr = int(g.get("mon") or g.get("mon2")), (g.get("yr") or g.get("yr3"))
        yr = int(yr) if len(yr) == 4 else 2000 + int(yr)
        if not (1 <= mon <= 12 and 2000 <= yr <= 2099):
            continue
        return mon, yr, m.group(0), m.span()
    return None


# --------------------------------------------------------------------------
def extract_fields(ocr: OcrResult) -> dict[str, FieldHit]:
    lines = ocr.lines
    wins = _windows(lines)
    norms = [normalize_with_map(w.text) for w in wins]
    hits: dict[str, FieldHit] = {}

    def better(cur: Optional[FieldHit], new: FieldHit) -> FieldHit:
        return new if cur is None or new.match_confidence > cur.match_confidence else cur

    # ---------------- MRP ----------------
    best: Optional[FieldHit] = None
    for win, (norm, idx) in zip(wins, norms):
        ok, conf, kend = _key_in_first_line(win, norm, idx, MRP_KEYS, MRP_FUZZY)
        if not ok:
            continue
        blocked, cut = _negative_cut(norm, idx, kend, MRP_NEGATIVE)
        if blocked:
            continue
        region = norm[kend:]
        if cut is not None:
            cut_norm = next((k for k, o in enumerate(idx) if o >= cut),
                            len(norm))
            region = norm[kend:max(kend, cut_norm)]
        m = re.search(MRP_CURRENCY + MRP_VALUE, region)
        if not m:
            continue
        a, b = _orig_span(idx, kend + m.start("val"), kend + m.end("val"))
        raw_val = re.search(r"\d{1,6}(?:[.,]\d{1,2})?", win.text[a:a + 12])
        val_text = raw_val.group(0) if raw_val else m.group("val")
        try:
            val = float(val_text.replace(",", "."))
        except ValueError:
            continue
        ws = win.words_for_span(a, a + len(val_text))
        tax_ok = any(re.search(t, norm) for t in TAX_PHRASE) or any(
            re.search(t, normalize_with_map(
                _neighbourhood(lines, win.i))[0]) for t in TAX_PHRASE)
        best = better(best, FieldHit(
            "mrp", True, True, win.text, val, val_text, _boxes(ws),
            lines[win.i].bbox,
            "" if tax_ok else
            "Rule 6 requires the retail sale price to be stated as 'inclusive "
            "of all taxes'; that phrase was not found near it.", conf))

    if best is None:                       # value-anchored fallback
        for win in wins:
            ln_norm = normalize_with_map(lines[win.i].text)[0]
            if any(re.search(nk, ln_norm) for nk in MRP_NEGATIVE):
                continue
            m = CURRENCY_RE.search(win.text)
            if not m or win.line_at(m.start()) != win.i:
                continue
            try:
                val = float(m.group("val").replace(",", "."))
            except ValueError:
                continue
            a, b = m.span("val")
            best = better(best, FieldHit(
                "mrp", True, True, win.text, val, m.group("val"),
                _boxes(win.words_for_span(a, b)), lines[win.i].bbox,
                "Matched on a currency amount alone - no 'MRP' / 'Maximum "
                "Retail Price' key phrase was recognised. Verify manually "
                "that this is the retail sale price.", 0.45))
    # ---- MRP: Indian "545/-" price notation ------------------------------
    # Reached when no "MRP" key could be tied to a number. On a legend-style
    # label the key sentence and the figure are printed apart, and OCR splits
    # that sentence across lines besides, so the key is often not recoverable
    # at all - three frames of one Pond's jar produced "WAP", '") B. B.NO."
    # and one usable "MRP", never on the line carrying the price.
    #
    # The "/-" terminator carries the identification instead. It is specific
    # to a rupee amount: an ingredient list, an address, a batch code and a
    # phone number do not contain it, and a unit rate ends "/ml" rather than
    # "/-". That is enough to report the figure, though not enough to be sure
    # it is the RETAIL price rather than some other amount - hence a
    # confidence that adjudicates but carries a verify-by-eye note.
    if best is None:
        for win in wins:
            ln_norm = normalize_with_map(lines[win.i].text)[0]
            if any(re.search(nk, ln_norm) for nk in MRP_NEGATIVE):
                continue
            txt = lines[win.i].text
            m = PRICE_SLASH_RE.search(txt)
            if not m or UNIT_PRICE_RE.search(m.group(0)):
                continue
            try:
                val = float(m.group("val").replace(",", "."))
            except ValueError:
                continue
            a, b = m.span("val")
            best = better(best, FieldHit(
                "mrp", True, True, txt, val, m.group("val"),
                _boxes(lines[win.i].words_for_span(a, b)), lines[win.i].bbox,
                "Read from an amount written as '" + m.group("val") +
                "/-' with no 'MRP' key phrase attached to it - the label "
                "prints the key and the figure apart. Confirm by eye that "
                "this is the retail sale price.", 0.6))

    # ---- MRP: legend layout, key here / value below ----------------------
    if best is None or best.match_confidence < 0.5:
        legend = [i for i, ln in enumerate(lines)
                  for n in [normalize_with_map(ln.text)[0]]
                  if any(re.search(k, n) for k in MRP_KEYS)
                  and any(re.search(mk, n) for mk in MRP_LEGEND_MARKERS)]
        for li in legend:
            found = False
            for j in range(li + 1, min(li + 1 + MRP_LEGEND_LOOKAHEAD,
                                       len(lines))):
                txt = lines[j].text
                for m in PRICE_SLASH_RE.finditer(txt):
                    # The "/-" terminator is what already excludes a unit
                    # rate: "Rs 2.73/ml" ends in "/ml", not "/-", so it never
                    # reaches here. This check is a backstop, and is applied
                    # to the MATCH ONLY - widening it to the surrounding text
                    # made the unit price printed further along the same line
                    # ("545/-, Rs 2.73/ml") veto the real price.
                    if UNIT_PRICE_RE.search(m.group(0)):
                        continue
                    try:
                        val = float(m.group("val").replace(",", "."))
                    except ValueError:
                        continue
                    a, b = m.span("val")
                    best = better(best, FieldHit(
                        "mrp", True, True, txt, val, m.group("val"),
                        _boxes(lines[j].words_for_span(a, b)),
                        lines[j].bbox,
                        "Legend-style label: the 'MRP' key line points "
                        "elsewhere ('see below'), so the price was read from "
                        "the value band beneath it. Confirm by eye that this "
                        "is the retail sale price and not a unit rate.", 0.75))
                    found = True
                    break
                if found:
                    break
            if found:
                break

    hits["mrp"] = best or FieldHit("mrp", False, True,
                                   note="No MRP / maximum retail price "
                                        "declaration found in the OCR text.")

    # ---------------- Net quantity ----------------
    best = None
    for win, (norm, idx) in zip(wins, norms):
        ok, conf, kend = _key_in_first_line(win, norm, idx, NETQTY_KEYS,
                                            NETQTY_FUZZY)
        ln_norm = normalize_with_map(lines[win.i].text)[0]
        if not ok and any(re.search(nk, ln_norm) for nk in MRP_NEGATIVE):
            continue          # "Rs. 49.80 per 100 g" is a price, not a pack size
        if not ok and any(re.search(nk, ln_norm) for nk in NETQTY_NEGATIVE):
            continue          # "Size : 29.7 x 21 cm" is a dimension, not a qty
        search_text = win.text if ok else lines[win.i].text
        # OCR unit repair is only safe when a "Net Qty." key phrase anchored
        # the match; on unanchored text it would invent quantities.
        qty = parse_net_quantity(search_text, allow_ocr_confusions=ok)
        if qty is None:
            continue
        if not ok:
            conf = 0.5                     # bare "500 g" with no key phrase
        a = search_text.find(qty.raw)
        num = re.search(r"\d{1,6}(?:[.,]\d{1,3})?", qty.raw)
        na, nb = (a + num.start(), a + num.end()) if (a >= 0 and num) else (0, 0)
        ws = (win.words_for_span(na, nb) if ok
              else lines[win.i].words_for_span(na, nb))
        best = better(best, FieldHit(
            "net_quantity", True, True, search_text, qty, qty.raw, _boxes(ws),
            lines[win.i].bbox,
            ("Unit character recovered from a likely OCR substitution "
             f"('{qty.raw}' read as '{qty.value:g} {qty.unit}'); verify."
             if qty.ocr_recovered else
             "" if ok else "Quantity found without an explicit 'Net Qty. / "
                           "Net Wt.' key phrase."), conf))
    hits["net_quantity"] = best or FieldHit(
        "net_quantity", False, True,
        note="No net-quantity declaration found in the OCR text.")

    # ---------------- Month & year of manufacture ----------------
    best = None
    for win, (norm, idx) in zip(wins, norms):
        ok, conf, kend = _key_in_first_line(win, norm, idx, MFG_KEYS, MFG_FUZZY)
        if not ok:
            continue
        blocked, cut = _negative_cut(norm, idx, kend, MFG_NEGATIVE)
        if blocked:
            continue
        start = _orig_span(idx, kend, kend + 1)[0] if kend < len(idx) else 0
        p = _parse_date(win.text[start:cut] if cut else win.text[start:])
        if p is None:
            continue
        mon, yr, vtext, (va, vb) = p
        vspan = (start + va, start + vb)
        best = better(best, FieldHit(
            "mfg_date", True, True, win.text, {"month": mon, "year": yr},
            vtext, _boxes(win.words_for_span(*vspan)), lines[win.i].bbox,
            "", conf))

    if best is None:
        # Value-anchored fallback. A bare "MM/YYYY" is a strong signal on a
        # label, but it could also be a best-before date, so it is reported at
        # reduced confidence with the ambiguity stated. A short token next to
        # it that is one OCR error away from MFD/MFG/PKD raises confidence -
        # that is how "MED: 11/2026" is recovered.
        for win in wins:
            ln_norm = normalize_with_map(lines[win.i].text)[0]
            if any(re.search(nk, ln_norm) for nk in MFG_NEGATIVE):
                continue
            p = _parse_date(lines[win.i].text)
            if p is None:
                continue
            mon, yr, vtext, vspan = p
            head = "".join(c for c in lines[win.i].text[:vspan[0]]
                           if c.isalnum()).upper()
            near = any(SequenceMatcher(None, head[-len(t):], t).ratio() >= 0.6
                       for t in SHORT_MFG_TOKENS if len(head) >= len(t))
            best = better(best, FieldHit(
                "mfg_date", True, True, lines[win.i].text,
                {"month": mon, "year": yr}, vtext,
                _boxes(lines[win.i].words_for_span(*vspan)), lines[win.i].bbox,
                "Date matched without a clean 'Mfg./Mfd./Packed on' key "
                "phrase; it may be a best-before date. Verify manually."
                if not near else
                "Key phrase recovered through an OCR error tolerance "
                f"('{lines[win.i].text[:vspan[0]].strip()}').",
                0.6 if near else 0.35))
    hits["mfg_date"] = best or FieldHit(
        "mfg_date", False, True,
        note="No month-and-year of manufacture / packing found.")

    # ---------------- Manufacturer / packer / importer ----------------
    best = None
    for win, (norm, idx) in zip(wins, norms):
        ok, conf, kend = _key_in_first_line(win, norm, idx, MAKER_KEYS,
                                            MAKER_FUZZY)
        if not ok:
            continue
        addr = _neighbourhood(lines, win.i, 3)
        has_addr = bool(re.search(r"\b\d{6}\b", addr)) or len(addr) > 60
        best = better(best, FieldHit(
            "manufacturer", True, True, addr, addr.strip(), "",
            _boxes(lines[win.i].words), lines[win.i].bbox,
            "" if has_addr else
            "Declarant key phrase found but no accompanying address (no PIN "
            "code and little following text).", conf))
    hits["manufacturer"] = best or FieldHit(
        "manufacturer", False, True,
        note="No 'Manufactured/Packed/Marketed/Imported by' declaration found.")

    # ---------------- Consumer care ----------------
    best = None
    for win, (norm, idx) in zip(wins, norms):
        ok, conf, kend = _key_in_first_line(win, norm, idx, CARE_KEYS,
                                            CARE_FUZZY)
        ctx = _neighbourhood(lines, win.i, 3)
        email = EMAIL_RE.search(ctx)
        phone = PHONE_RE.search(ctx)
        if not ok and not (email or phone):
            continue
        if not ok:
            conf = 0.5
        parts = [p.group(0) for p in (email, phone) if p]
        best = better(best, FieldHit(
            "consumer_care", True, True, ctx,
            {"email": email.group(0) if email else None,
             "phone": phone.group(0) if phone else None},
            ", ".join(parts), _boxes(lines[win.i].words), lines[win.i].bbox,
            "" if (email or phone) else
            "Consumer-care heading found but no contact number or e-mail "
            "address was recognised next to it.", conf))
    hits["consumer_care"] = best or FieldHit(
        "consumer_care", False, True,
        note="No consumer-care name / phone / e-mail found.")

    # ---------------- Common or generic name ----------------
    best = None
    for win, (norm, idx) in zip(wins, norms):
        for k in GENERIC_KEYS:
            m = re.search(k, norm)
            if not m or win.line_at(idx[m.start()]) != win.i:
                continue
            a, _ = _orig_span(idx, m.end(), min(len(idx), m.end() + 1))
            # Stop at the end of the key's own line: the generic name is a
            # short phrase, and a window that runs on into the next
            # declaration would swallow the net quantity and the MRP with it.
            end = len(lines[win.i].text)
            txt = win.text[a:end].strip(" :-")
            best = FieldHit("commodity_name", True, True, win.text, txt, txt,
                            _boxes(lines[win.i].words), lines[win.i].bbox,
                            "", 1.0)
            break
        if best:
            break
    hits["commodity_name"] = best or FieldHit(
        "commodity_name", False, assessable=False,
        note="The common/generic name is not machine-determinable from a "
             "photograph unless the label states it with an explicit key "
             "phrase. Marketing brand text cannot be distinguished from the "
             "generic name by rule. Flagged for manual verification rather "
             "than reported as a violation.")

    return hits
