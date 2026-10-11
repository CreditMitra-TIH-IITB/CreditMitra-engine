"""End-to-end smoke test against a running engine.

    python scripts/e2e_check.py statement1.pdf [statement2.pdf ...]
        [--engine http://127.0.0.1:8000]

Uploads each PDF through the real API, waits for the task, and prints what a
user would get: rows extracted, payee coverage, person/merchant split, how many
merchants resolved to a category, and the score. Exits non-zero if any task
fails or comes back without transactions. Prints aggregates only — never
narrations or payee names — so it is safe to run on real statements.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


def run_one(client: httpx.Client, pdf: Path, timeout_s: float) -> dict[str, Any]:
    started = time.time()
    with pdf.open("rb") as fh:
        resp = client.post(
            "/api/v1/statements/process", files={"pdf": (pdf.name, fh, "application/pdf")}
        )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        status = client.get(f"/api/v1/statements/status/{task_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        if time.time() - started > timeout_s:
            return {"file": pdf.name, "status": "timeout", "seconds": time.time() - started}
        time.sleep(2)

    result: dict[str, Any] = {
        "file": pdf.name,
        "status": status["status"],
        "seconds": round(time.time() - started, 1),
        "error": status.get("error"),
    }
    txns = status.get("transactions") or []
    merchants = [t for t in txns if t.get("payee_type") == "merchant"]
    resolved = [t for t in merchants if t.get("category") not in (None, "other")]
    result.update(
        rows=len(txns),
        empty_payee=sum(1 for t in txns if not t.get("payee")),
        types=dict(Counter(t.get("payee_type") for t in txns)),
        merchants_resolved=f"{len(resolved)}/{len(merchants)}",
        categories=dict(Counter(t["category"] for t in resolved).most_common(5)),
    )
    report = status.get("report")
    if report:
        result["score"] = f"{report['score']} {report['band']} / {report['archetype']}"
    else:
        result["score"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--engine", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    ok = True
    with httpx.Client(base_url=args.engine, timeout=60) as client:
        for pdf in args.pdfs:
            r = run_one(client, pdf, args.timeout)
            print(f"\n== {r['file']}  [{r['status']}, {r['seconds']}s]")
            for key in ("error", "rows", "empty_payee", "types", "merchants_resolved"):
                if r.get(key) not in (None, ""):
                    print(f"   {key:18s} {r[key]}")
            if r.get("categories"):
                print(f"   {'top categories':18s} {r['categories']}")
            print(f"   {'report':18s} {r.get('score') or 'none'}")
            if r["status"] != "completed" or not r.get("rows"):
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
