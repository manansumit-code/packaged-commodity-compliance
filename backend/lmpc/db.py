"""SQLite scan history. One row per scan, full report retained verbatim."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Optional

# FastAPI runs sync endpoints in a thread pool, so one shared connection can
# be written from several threads at once. SQLite tolerates that only under a
# lock.
_LOCK = threading.Lock()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scans.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id       TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    overall       TEXT NOT NULL,
    n_violations  INTEGER NOT NULL,
    n_borderline  INTEGER NOT NULL,
    net_quantity  TEXT,
    mrp           TEXT,
    manufacturer  TEXT,
    image_path    TEXT,
    officer       TEXT,
    report_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_ts ON scans(ts DESC);
CREATE INDEX IF NOT EXISTS idx_scans_overall ON scans(overall);
"""


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _decl(rep: dict, name: str) -> Optional[str]:
    for d in rep.get("declarations", []):
        if d["field"] == name:
            return (d.get("value_text") or d.get("raw_text") or "")[:180]
    return None


def save_scan(con: sqlite3.Connection, rep: dict,
              image_path: Optional[str] = None,
              officer: Optional[str] = None) -> str:
    with _LOCK:
        con.execute(
            "INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rep["scan_id"], rep["timestamp"], rep["overall"],
             len(rep.get("violations", [])), len(rep.get("borderline", [])),
             _decl(rep, "net_quantity"), _decl(rep, "mrp"),
             _decl(rep, "manufacturer"), image_path, officer,
             json.dumps(rep)))
        con.commit()
    return rep["scan_id"]


def get_scan(con: sqlite3.Connection, scan_id: str) -> Optional[dict]:
    with _LOCK:
        r = con.execute("SELECT report_json FROM scans WHERE scan_id=?",
                        (scan_id,)).fetchone()
    return json.loads(r["report_json"]) if r else None


def list_scans(con: sqlite3.Connection, limit: int = 50, offset: int = 0,
               overall: Optional[str] = None,
               q: Optional[str] = None) -> list[dict[str, Any]]:
    sql = ("SELECT scan_id, ts, overall, n_violations, n_borderline, "
           "net_quantity, mrp, manufacturer, officer FROM scans WHERE 1=1")
    args: list[Any] = []
    if overall:
        sql += " AND overall=?"; args.append(overall)
    if q:
        sql += (" AND (manufacturer LIKE ? OR mrp LIKE ? OR net_quantity "
                "LIKE ? OR scan_id LIKE ?)")
        args += [f"%{q}%"] * 4
    sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with _LOCK:
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def stats(con: sqlite3.Connection) -> dict:
    with _LOCK:
        rows = con.execute(
            "SELECT overall, COUNT(*) c FROM scans GROUP BY overall").fetchall()
        viol = con.execute(
            "SELECT report_json FROM scans WHERE overall='NON_COMPLIANT' "
            "ORDER BY ts DESC LIMIT 500").fetchall()
    by = {r["overall"]: r["c"] for r in rows}
    counts: dict[str, int] = {}
    for r in viol:
        for v in json.loads(r["report_json"]).get("violations", []):
            counts[v.split(":")[0]] = counts.get(v.split(":")[0], 0) + 1
    return {"total": sum(by.values()), "by_verdict": by,
            "top_violated_declarations": dict(
                sorted(counts.items(), key=lambda kv: -kv[1])[:10])}
