"""Plain-English rendering of a scan report, for people who are not engineers.

This module DERIVES NOTHING. Every verdict, number and threshold shown here is
read straight out of the report dict produced by pipeline.scan_image. If it
recomputed anything, this view and report.render_text would eventually
disagree, and the friendly one would be the one people trust.

The technical report stays available; this is a second view of the same facts,
not a replacement.
"""
from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------
# Colour, degraded away when the output is not a terminal
# --------------------------------------------------------------------------
def _use_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


class _C:
    def __init__(self, on: bool):
        self.on = on

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def green(self, s): return self._w("32", s)
    def red(self, s): return self._w("31", s)
    def yellow(self, s): return self._w("33", s)
    def blue(self, s): return self._w("36", s)
    def grey(self, s): return self._w("90", s)
    def bold(self, s): return self._w("1", s)
    def bold_green(self, s): return self._w("1;32", s)
    def bold_red(self, s): return self._w("1;31", s)
    def bold_yellow(self, s): return self._w("1;33", s)


# --------------------------------------------------------------------------
# Vocabulary: machine name -> what a person would call it
# --------------------------------------------------------------------------
FRIENDLY_NAME = {
    "manufacturer":   "Who made or packed it",
    "commodity_name": "What the product is",
    "net_quantity":   "How much is inside",
    "mfg_date":       "Month and year packed",
    "mrp":            "Price (MRP)",
    "consumer_care":  "Customer care contact",
}

# The same six, in the form that reads correctly inside a sentence
# ("The net quantity is printed 1.1 mm tall").
SENTENCE_NAME = {
    "manufacturer":   "maker's name and address",
    "commodity_name": "product name",
    "net_quantity":   "net quantity",
    "mfg_date":       "packing date",
    "mrp":            "price (MRP)",
    "consumer_care":  "customer care contact",
}

# One actionable sentence per gate, in the same words USAGE.md uses.
GATE_ADVICE = {
    "NO_REFERENCE_MARKER":
        "I could not find the printed calibration card in this photo, so I "
        "cannot measure how big the printing is. Lay the card flat next to "
        "the label, fully inside the picture, and shoot again.",
    "EXCESSIVE_TILT":
        "The photo was taken at too much of an angle. Hold the camera "
        "square-on to the pack — face it straight — and shoot again.",
    "INSUFFICIENT_RESOLUTION":
        "The printing came out too small in the photo to measure. Move the "
        "camera closer so the label and the card fill the frame, and shoot at "
        "your phone's full resolution.",
    "LOW_LEGIBILITY":
        "The text in this photo was not read confidently enough for me to "
        "say a declaration is MISSING — so I marked those as unchecked "
        "rather than accuse the pack. If you sent this picture through "
        "WhatsApp, send the original file instead: WhatsApp re-compresses "
        "photos and that alone is usually the cause. Otherwise reshoot "
        "sharper, with softer light and no flash glare.",
    "PANEL_NOT_COPLANAR":
        "The calibration card does not look like it is on the same flat "
        "surface as the printing. Lay the card flat against the same face as "
        "the label — not on the table beside a standing box — and shoot again.",
}

VERDICT_HEADLINE = {
    "COMPLIANT": ("[ OK ]", "Everything I could check looks fine",
                  "bold_green"),
    "NON_COMPLIANT": ("[ !! ]", "Found a problem", "bold_red"),
    "REVIEW_REQUIRED": ("[ ?? ]", "I could not check everything — "
                        "a person should look", "bold_yellow"),
}


def _short_reason(reason: str) -> str:
    """Turn one of the pipeline's technical refusals into everyday words."""
    r = (reason or "").lower()
    if "px/mm" in r and "floor" in r:
        return "the photo is not close enough to measure this"
    if "no usable physical scale" in r:
        return "no calibration card in the photo, so sizes cannot be measured"
    if "recogniser read" in r or "segmentation" in r:
        return "the digits ran together in the photo — shoot it sharper"
    if "value pattern" in r:
        return "I am not certain this is really that declaration, so I will "\
               "not judge it"
    if "lettering, not numerals" in r:
        return "letters, not numerals — measured for information only"
    if "could not be boxed" in r:
        return "I found it but could not get a clean measurement"
    if not reason:
        return "could not be measured"
    return "could not be measured reliably"


def _legibility_phrase(rep: dict) -> str:
    """The actual reason this frame cannot support an 'it is missing' claim."""
    for r in rep.get("legibility", {}).get("reasons", []):
        low = r.lower()
        if "ocr confidence" in low:
            return "the printing was too blurry to read confidently"
        if "median recognised text" in low:
            return "the lettering is too small in this photo"
        if "px/mm, below" in low:
            return "the photo was taken too far away"
        if "no text was recognised" in low:
            return "no readable text was found at all"
    return "photo not clear enough to say"


def _clip(s: str, n: int = 34) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _quality_lines(rep: dict, c: _C) -> list[str]:
    cal = rep["calibration"]
    out = []
    if not cal["ok"]:
        out.append("   " + c.red("Calibration card   not found in this photo"))
        out.append("   " + c.grey("                   without it I can check what is printed, "
                                  "but not how big it is"))
        return out

    ppmm = cal["px_per_mm_at_marker"]
    floor = rep["uncertainty"].get("min_px_per_mm", 6.0)
    if ppmm >= 16:
        sharp, col = "ideal", c.green
    elif ppmm >= 10:
        sharp, col = "fine", c.green
    elif ppmm >= floor:
        sharp, col = "workable, near the limit", c.yellow
    else:
        sharp, col = "too far away to measure", c.red
    out.append("   Closeness          " + col(sharp)
               + c.grey(f"   ({ppmm:.0f} pixels per mm; needs {floor:.0f}+, "
                        f"16 is ideal)"))

    tilt = cal["tilt_deg"]
    if tilt <= 15:
        ang, col = "good", c.green
    elif tilt <= 25:
        ang, col = "a little tilted", c.yellow
    else:
        ang, col = "too angled", c.red
    out.append("   Camera angle       " + col(ang)
               + c.grey(f"   ({tilt:.0f} degrees off square; keep under 25)"))
    return out


def render_friendly(rep: dict, filename: str = "", show_details: bool = False,
                    colour: bool | None = None) -> str:
    c = _C(_use_colour() if colour is None else colour)
    L: list[str] = []
    a = L.append
    bar = "─" * 66

    a("")
    a(c.blue(bar))
    a(c.bold(f"  {filename or 'photo'}"))
    a(c.blue(bar))
    a("")

    # ---- headline ---------------------------------------------------------
    badge, words, style = VERDICT_HEADLINE.get(
        rep["overall"], ("[ ?? ]", rep["overall"], "bold_yellow"))
    a("  " + getattr(c, style)(f"{badge}  {words}"))

    n_bad = sum(1 for d in rep["declarations"]
                if d["presence_verdict"] == "FAIL"
                or (d.get("height") or {}).get("verdict") == "FAIL")
    if n_bad:
        a("  " + c.grey(f"         {n_bad} item(s) below what the rules ask for"))
    a("")

    # ---- photo quality ----------------------------------------------------
    a(c.bold("  Photo quality"))
    L.extend(_quality_lines(rep, c))
    a("")

    # ---- presence ---------------------------------------------------------
    a(c.bold("  Is everything printed on the pack?"))
    for d in rep["declarations"]:
        name = FRIENDLY_NAME.get(d["field"], d["requirement"])
        pv = d["presence_verdict"]
        if pv == "PASS":
            mark, col, tail = "OK ", c.green, c.grey(_clip(d["value_text"] or ""))
        elif pv == "FAIL":
            mark, col, tail = "NO ", c.red, c.red("not printed on this pack")
        else:
            mark, col = " ? ", c.yellow
            tail = c.yellow(
                "a person must check this one" if not d["assessable"]
                else _legibility_phrase(rep))
        a(f"   {col('[' + mark + ']')} {name:<24} {tail}".rstrip())
    a("")

    # ---- text size --------------------------------------------------
    # This is the Rule 7 check and the number people actually came for, so
    # every declaration gets a row - including the ones with nothing to
    # measure, which say why rather than silently vanishing.
    a(c.bold("  TEXT SIZE — how tall the printing actually is"))
    a(" " * 35 + c.grey(f"{'printed':>9}{'minimum':>10}"))
    bands = []
    for d in rep["declarations"]:
        name = FRIENDLY_NAME.get(d["field"], d["field"])
        h = d.get("height")

        if not h:
            why = ("not found on the pack"
                   if d["presence_verdict"] == "FAIL" else
                   _legibility_phrase(rep)
                   if d["presence_verdict"] == "NOT_ASSESSED" else
                   "not measured — this tool measures numerals only")
            a(f"   {c.grey('[ - ]')} {name:<26}{'—':>9}{'—':>10}  "
              + c.grey(why))
            continue

        v, mm, need = h["verdict"], h.get("measured_mm"), h.get("required_mm")
        if h.get("error_band_mm") and not h.get("informational"):
            bands.append(h["error_band_mm"])
        meas = f"{mm:.2f} mm" if mm is not None else "—"
        req = f"{need:.1f} mm" if need is not None else "—"

        # Lettering: the number is real, the verdict would not be.
        if h.get("informational"):
            if mm is None:
                a(f"   {c.grey('[ - ]')} {name:<26}{'—':>9}{req:>10}  "
                  + c.grey(_short_reason(h.get("reason", ""))))
            else:
                a(f"   {c.blue('[ i ]')} {name:<26}{meas:>9}{req:>10}  "
                  + c.grey("letters, not numerals — measured, not judged"))
            continue

        if v == "PASS":
            mark, col, note = "[OK ]", c.green, ""
            adv = h.get("advisory_verdict")
            if adv in ("FAIL", "BORDERLINE"):
                note = c.yellow(f"ok, but a stricter reading of the rule "
                                f"wants {h['advisory_required_mm']:.1f} mm")
        elif v == "FAIL":
            mark, col, note = "[NO ]", c.red, c.red("TOO SMALL")
        elif v == "BORDERLINE":
            mark, col = "[ ~ ]", c.yellow
            note = c.yellow("too close to call — measure it by hand")
        else:
            mark, col = "[ ? ]", c.yellow
            note = c.yellow(("measured, but not judged: " if mm is not None
                             else "") + _short_reason(h.get("reason", "")))
        a(f"   {col(mark)} {name:<26}{meas:>9}{req:>10}  {note}".rstrip())

    if bands:
        a("")
        a(c.grey(f"   Digit heights are measured from the printed digits "
                 f"themselves and are"))
        a(c.grey(f"   accurate to about +/- {max(bands):.2f} mm. "
                 f"The minimum comes from Rule 7."))
        a(c.grey("   [ i ] rows are lettering: the size is real, but it is a "
                 "median over mixed"))
        a(c.grey("   upper and lower case, so it is not judged against the "
                 "rule."))
    a("")

    # ---- what to do -------------------------------------------------------
    todo: list[str] = []
    seen: set[str] = set()
    for g in rep.get("gates", []):
        adv = GATE_ADVICE.get(g["gate"])
        if adv and adv not in seen:
            seen.add(adv)
            todo.append(adv)
    for d in rep["declarations"]:
        name = SENTENCE_NAME.get(d["field"],
                                 d["field"].replace("_", " "))
        if d["presence_verdict"] == "FAIL":
            todo.append(f"\"{d['requirement']}\" is not printed on this pack. "
                        f"The rules require it. Add it to the label.")
        h = d.get("height") or {}
        if h.get("verdict") == "FAIL":
            todo.append(
                f"The {name} is printed {h['measured_mm']:.2f} mm tall. "
                f"The rules ask for at least {h['required_mm']:.1f} mm. "
                f"The label needs reprinting with bigger digits.")
        elif h.get("verdict") == "BORDERLINE":
            todo.append(
                f"The {name} measures {h['measured_mm']:.2f} mm against a "
                f"{h['required_mm']:.1f} mm minimum — too close for me to be "
                f"sure either way. Check it with a magnifier before deciding.")

    if todo:
        a(c.bold("  What to do next"))
        for t in todo:
            a("   * " + _wrap(t, 64))
        a("")
    else:
        a(c.bold("  What to do next"))
        a("   " + c.green("* Nothing. This pack passed every check I can make."))
        a("")

    a(c.grey("  This is a screening tool, not a legal ruling. It checks flat "
             "printed"))
    a(c.grey("  packs only, and cannot judge the product's generic name."))
    if show_details:
        from .report import render_text
        a("")
        a(c.grey(render_text(rep)))
    else:
        a(c.grey("  Full technical detail:  ./scan "
                 + (f"inbox/done/{filename} " if filename else "")
                 + "--details"))
    a(c.blue(bar))
    return "\n".join(L)


def _wrap(text: str, width: int) -> str:
    """Wrap, indenting continuation lines under the bullet."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ("\n" + " " * 5).join(lines)


def render_one_liner(rep: dict, filename: str, colour: bool | None = None) -> str:
    """Single line, for the batch summary table."""
    c = _C(_use_colour() if colour is None else colour)
    badge, words, style = VERDICT_HEADLINE.get(
        rep["overall"], ("[ ?? ]", rep["overall"], "bold_yellow"))
    return f"  {getattr(c, style)(badge)}  {_clip(filename, 30):<32}{words}"
