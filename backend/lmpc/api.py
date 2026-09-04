"""
FastAPI backend for the Packaged Commodity Compliance Scanner.

Everything on the scanning path is local: no cloud OCR, no model API, no
network dependency that can fail on venue wifi.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from . import db
from .calibration import MarkerSpec, generate_marker
from .pipeline import SCOPE_STATEMENT, ScanConfig, load_uncertainty, scan_image
from .report import render_text
from .rules import (DECLARATION_HEIGHT_RULES, MANDATORY_DECLARATIONS,
                    RULE7_GENERAL_FLOOR_MM, RULE7_TABLE_I, RULE7_TABLE_II,
                    PrintStyle)

app = FastAPI(title="Packaged Commodity Compliance Scanner",
              version="0.1.0-mvp",
              description="Legal Metrology (Packaged Commodities) Rules, 2011 "
                          "- automated label compliance scanning. SIH26034.")

CON = db.connect()
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
os.makedirs(IMG_DIR, exist_ok=True)


@app.get("/api/health")
def health():
    import pytesseract
    try:
        tv = str(pytesseract.get_tesseract_version())
    except Exception as e:
        tv = f"UNAVAILABLE: {e}"
    return {"ok": True, "opencv": cv2.__version__, "tesseract": tv,
            "uncertainty": load_uncertainty().__dict__}


@app.post("/api/scan")
async def scan(
    file: UploadFile = File(..., description="Label photo (jpg/png)"),
    marker_length_mm: float = Form(40.0),
    marker_dict: str = Form("DICT_4X4_50"),
    marker_id: Optional[int] = Form(None),
    print_style: str = Form("normal"),
    panel_area_cm2: Optional[float] = Form(None),
    known_px_per_mm: Optional[float] = Form(None),
    lang: str = Form("eng"),
    fast: bool = Form(False),
    save: bool = Form(True),
    officer: Optional[str] = Form(None),
):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")

    cfg = ScanConfig(
        marker=MarkerSpec(marker_dict, marker_length_mm, marker_id),
        print_style=PrintStyle(print_style),
        panel_area_cm2=panel_area_cm2,
        lang=lang, known_px_per_mm=known_px_per_mm)
    if fast:
        cfg.psms = (6,)

    rep = scan_image(img, cfg)

    path = None
    if save:
        path = os.path.join(IMG_DIR, f"{rep['scan_id']}.jpg")
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        db.save_scan(CON, rep, path, officer)
    rep["stored"] = bool(save)
    return JSONResponse(rep)


@app.get("/api/scans")
def scans(limit: int = Query(50, le=500), offset: int = 0,
          overall: Optional[str] = None, q: Optional[str] = None):
    return {"items": db.list_scans(CON, limit, offset, overall, q)}


@app.get("/api/scans/{scan_id}")
def scan_detail(scan_id: str):
    r = db.get_scan(CON, scan_id)
    if not r:
        raise HTTPException(404, "No such scan.")
    return r


@app.get("/api/scans/{scan_id}/report.txt", response_class=PlainTextResponse)
def scan_report(scan_id: str):
    r = db.get_scan(CON, scan_id)
    if not r:
        raise HTTPException(404, "No such scan.")
    return render_text(r)


@app.get("/api/stats")
def statistics():
    return db.stats(CON)


@app.get("/api/rules")
def rules():
    """Full disclosure of every legal threshold the system applies."""
    return {
        "source": "Legal Metrology (Packaged Commodities) Rules, 2011",
        "rule_7_table_I_weight_or_volume": [
            {"upper_bound_g_or_ml": u, "normal_mm": n, "embossed_mm": e}
            for u, n, e in RULE7_TABLE_I],
        "rule_7_table_II_length_area_number": [
            {"upper_bound_panel_area_cm2": u, "normal_mm": n, "embossed_mm": e}
            for u, n, e in RULE7_TABLE_II],
        "rule_7_3_general_floor_mm": RULE7_GENERAL_FLOOR_MM,
        "rule_6_mandatory_declarations": [
            {"field": f, "requirement": h} for f, h in MANDATORY_DECLARATIONS],
        "threshold_attachment": {
            k: {"basis": v.basis.value,
                "advisory_basis": v.advisory_basis.value if v.advisory_basis
                else None,
                "source_note": v.source_note}
            for k, v in DECLARATION_HEIGHT_RULES.items()},
        "scope": SCOPE_STATEMENT,
    }


@app.get("/api/uncertainty")
def uncertainty():
    u = load_uncertainty()
    return u.__dict__ | {"band_mm": u.band_mm()}


@app.get("/api/marker.png")
def marker(marker_id: int = 0, dictionary: str = "DICT_4X4_50"):
    """Printable reference marker. Print at 100%, measure the black square
    with a ruler, and pass that measurement as marker_length_mm when scanning.
    """
    img = generate_marker(MarkerSpec(dictionary), marker_id)
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png")
