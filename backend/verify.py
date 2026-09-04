"""
One-command end-to-end self-check.

    python backend/verify.py

Runs every part of the system and prints a pass/fail line for each: the
dependencies, the legal rule tables, the metric calibration, the scanning
pipeline, the three-way PASS/FAIL/BORDERLINE discrimination, the database,
the report renderer, and the HTTP API. Exits non-zero if anything fails.
"""
from __future__ import annotations

import json
import os
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

RESULTS: list[tuple[str, bool, str]] = []


def step(name):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                detail = fn() or ""
                RESULTS.append((name, True, f"{detail}  [{time.time()-t0:.1f}s]"))
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        run.__name__ = fn.__name__
        return run
    return deco


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# --------------------------------------------------------------------------
@step("dependencies")
def check_deps():
    import cv2
    import numpy
    import pytesseract
    v = pytesseract.get_tesseract_version()
    assert hasattr(cv2, "aruco"), "opencv-contrib is required for cv2.aruco"
    return f"opencv {cv2.__version__}, numpy {numpy.__version__}, tesseract {v}"


@step("legal rules + field classification unit tests")
def check_rules():
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval", "test_rules.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip().splitlines()[-1]


@step("marker generation + metric scale round-trip")
def check_calibration():
    import cv2
    from lmpc.calibration import MarkerSpec, calibrate, generate_marker
    card = os.path.join(HERE, "assets", "calibration_card_A4_300dpi.png")
    if not os.path.exists(card):
        subprocess.run([sys.executable, os.path.join(HERE, "assets", "make_card.py")],
                       capture_output=True, check=True)
    img = cv2.imread(card)
    cal = calibrate(img, MarkerSpec(marker_length_mm=40.0))
    assert cal.ok, cal.reason
    true_ppmm = 300 / 25.4
    err = abs(cal.px_per_mm_at_marker / true_ppmm - 1) * 100
    assert err < 1.0, f"scale error {err:.2f}% is too large"
    assert generate_marker(MarkerSpec()).shape[0] > 0
    return f"recovered {cal.px_per_mm_at_marker:.3f} px/mm vs true {true_ppmm:.3f} ({err:+.2f}%)"


@step("full scan pipeline on a generated label")
def check_pipeline():
    import cv2
    from eval.synth import make_sample
    from lmpc.calibration import MarkerSpec
    from lmpc.pipeline import ScanConfig, scan_image
    photo, gt, spec, deg = make_sample(random.Random(7), "realistic")
    rep = scan_image(photo, ScanConfig(marker=MarkerSpec(marker_length_mm=40.0)))
    assert rep["calibration"]["ok"], "calibration failed on a clean sample"
    found = [d["field"] for d in rep["declarations"] if d["present"]]
    assert len(found) >= 4, f"only found {found}"
    heights = [d["height"]["measured_mm"] for d in rep["declarations"]
               if d.get("height") and d["height"].get("measured_mm") is not None]
    assert heights, "no height measured"
    for key in ("overall", "declarations", "calibration", "gates", "legibility",
                "planarity", "uncertainty", "scope", "ocr"):
        assert key in rep, f"report is missing {key!r}"
    return f"{rep['overall']}, {len(found)}/6 declarations, {len(heights)} height(s)"


@step("three-way discrimination (PASS / FAIL / BORDERLINE)")
def check_demo():
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval", "demo_cases.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "separates all three" in r.stdout, r.stdout
    return "A=PASS, B=FAIL, C=BORDERLINE"


@step("coplanarity control (must not false-fire)")
def check_coplanarity():
    r = subprocess.run([sys.executable, os.path.join(HERE, "eval", "test_coplanarity.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return "coplanar control not falsely flagged"


@step("refusal gates fire when they should")
def check_gates():
    import numpy as np
    from lmpc.pipeline import ScanConfig, scan_image
    blank = np.full((600, 800, 3), 240, np.uint8)
    rep = scan_image(blank, ScanConfig())
    assert rep["calibration"]["ok"] is False
    assert any(g["gate"] == "NO_REFERENCE_MARKER" for g in rep["gates"])
    # with no marker, nothing may be reported as a height violation
    assert all(d.get("height") is None
               or d["height"]["verdict"] == "NOT_ASSESSED"
               for d in rep["declarations"])
    return "no-marker frame refused instead of guessed"


@step("database round-trip")
def check_db():
    import tempfile

    from lmpc import db
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    con = db.connect(path)
    rep = {"scan_id": "abc123", "timestamp": "2026-01-01T00:00:00",
           "overall": "NON_COMPLIANT", "violations": ["net_quantity: x"],
           "borderline": [], "declarations": [
               {"field": "net_quantity", "value_text": "500 g", "raw_text": ""}]}
    db.save_scan(con, rep)
    assert db.get_scan(con, "abc123")["overall"] == "NON_COMPLIANT"
    assert len(db.list_scans(con)) == 1
    assert db.stats(con)["total"] == 1
    return "insert / fetch / list / stats"


@step("human-readable report renderer")
def check_report():
    from eval.synth import make_sample
    from lmpc.calibration import MarkerSpec
    from lmpc.pipeline import ScanConfig, scan_image
    from lmpc.report import render_text
    photo, *_ = make_sample(random.Random(3), "clean")
    txt = render_text(scan_image(photo, ScanConfig(
        marker=MarkerSpec(marker_length_mm=40.0))))
    for must in ("COMPLIANCE SCAN REPORT", "Mandatory declarations",
                 "Rule 7 letter/numeral height", "Measurement uncertainty",
                 "Scope of this assessment"):
        assert must in txt, f"report missing section: {must}"
    return f"{len(txt.splitlines())} lines, all sections present"


@step("HTTP API")
def check_api():
    import cv2
    from eval.synth import make_sample
    port = _free_port()
    env = dict(os.environ, PYTHONPATH=HERE)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "lmpc.api:app", "--host", "127.0.0.1",
         "--port", str(port), "--app-dir", HERE, "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/api/health", timeout=1).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise AssertionError("API did not start")

        health = json.loads(urllib.request.urlopen(base + "/api/health").read())
        assert health["ok"]
        rules = json.loads(urllib.request.urlopen(base + "/api/rules").read())
        assert rules["rule_7_table_I_weight_or_volume"][0]["normal_mm"] == 1.0
        assert len(rules["rule_6_mandatory_declarations"]) == 6

        photo, *_ = make_sample(random.Random(11), "realistic")
        tmp = os.path.join(HERE, "data", "_verify.jpg")
        cv2.imwrite(tmp, photo)
        boundary = "----verify"
        with open(tmp, "rb") as fh:
            blob = fh.read()
        body = b""
        for k, v in (("marker_length_mm", "40.0"), ("save", "true")):
            body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{k}"\r\n\r\n{v}\r\n').encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="file"; filename="s.jpg"\r\n'
                 f"Content-Type: image/jpeg\r\n\r\n").encode() + blob + \
                f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            base + "/api/scan", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        rep = json.loads(urllib.request.urlopen(req, timeout=120).read())
        sid = rep["scan_id"]
        assert rep["overall"] in ("COMPLIANT", "REVIEW_REQUIRED", "NON_COMPLIANT")

        detail = json.loads(urllib.request.urlopen(
            f"{base}/api/scans/{sid}").read())
        assert detail["scan_id"] == sid
        txt = urllib.request.urlopen(
            f"{base}/api/scans/{sid}/report.txt").read().decode()
        assert "COMPLIANCE SCAN REPORT" in txt
        listing = json.loads(urllib.request.urlopen(base + "/api/scans").read())
        assert any(i["scan_id"] == sid for i in listing["items"])
        assert json.loads(urllib.request.urlopen(base + "/api/stats").read())["total"] >= 1
        assert json.loads(urllib.request.urlopen(base + "/api/uncertainty").read())["band_mm"] > 0
        png = urllib.request.urlopen(base + "/api/marker.png").read()
        assert png[:4] == b"\x89PNG"
        os.remove(tmp)
        return ("health, rules, scan, scans, detail, report.txt, stats, "
                "uncertainty, marker.png")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@step("scan latency")
def check_latency():
    import cv2
    from lmpc.calibration import MarkerSpec
    from lmpc.pipeline import ScanConfig, scan_image
    p = os.path.join(HERE, "data", "demo", "A_compliant.jpg")
    if not os.path.exists(p):
        subprocess.run([sys.executable, os.path.join(HERE, "eval", "demo_cases.py")],
                       capture_output=True, cwd=ROOT)
    img = cv2.imread(p)
    small = cv2.resize(img, (1920, int(1920 * img.shape[0] / img.shape[1])))
    out = []
    for label, frame, psms in (("full", img, (6, 11)), ("fast", img, (6,)),
                               ("fast@1080p", small, (6,))):
        cfg = ScanConfig(marker=MarkerSpec(marker_length_mm=40.0))
        cfg.psms = psms
        ts = [(lambda t0: (scan_image(frame, cfg), (time.time() - t0) * 1000)[1])(time.time())
              for _ in range(3)]
        out.append(f"{label} {statistics.median(ts):.0f}ms")
    return ", ".join(out)


def main():
    for fn in (check_deps, check_rules, check_calibration, check_pipeline,
               check_demo, check_coplanarity, check_gates, check_db,
               check_report, check_api, check_latency):
        fn()
    print()
    print("=" * 78)
    print("  PACKAGED COMMODITY COMPLIANCE SCANNER - SELF CHECK")
    print("=" * 78)
    for name, ok, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        print(f"          {detail}")
    bad = [r for r in RESULTS if not r[1]]
    print("=" * 78)
    print(f"  {len(RESULTS)-len(bad)}/{len(RESULTS)} checks passed")
    print("=" * 78)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
