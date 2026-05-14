#!/usr/bin/env python3
"""
Annotation collection server for AURA human evaluation.

Serves the HTML evaluation form, collects annotations via POST, and provides
an admin dashboard and export endpoint.

Usage (from the aura conda env):
    cd /path/to/AURA_project
    python -m evaluation.human_eval_server

Or directly:
    python evaluation/human_eval_server.py

Endpoints:
    GET  /           - Serve the evaluation form (human_eval_forms.html)
    POST /submit     - Accept annotation submissions (JSON body)
    GET  /admin      - Admin dashboard showing annotator progress
    GET  /export     - Export all annotations as aggregated JSON
    GET  /health     - Health check
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_file, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ANNOTATIONS_DIR = RESULTS_DIR / "annotations"
HTML_FORM_PATH = RESULTS_DIR / "human_eval_forms.html"
JSON_FORM_PATH = RESULTS_DIR / "human_eval_forms.json"

HOST = "0.0.0.0"
PORT = 5050

# Ensure annotations directory exists
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotation_path(annotator_id: str) -> Path:
    """Return the JSON file path for a given annotator."""
    safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in annotator_id)
    return ANNOTATIONS_DIR / f"{safe_id}.json"


def _load_annotation(annotator_id: str) -> dict:
    """Load existing annotation data for an annotator."""
    path = _annotation_path(annotator_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_annotation(annotator_id: str, data: dict) -> None:
    """Save annotation data for an annotator."""
    path = _annotation_path(annotator_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_scenarios() -> list:
    """Load the scenario definitions (with hidden labels)."""
    if JSON_FORM_PATH.exists():
        with open(JSON_FORM_PATH) as f:
            return json.load(f)
    return []


def _count_completed_scenarios(ratings: dict, total_scenarios: int) -> int:
    """Count how many scenarios have all 8 ratings (4 dims x 2 responses)."""
    from evaluation.human_eval import EVAL_DIMENSIONS
    completed = 0
    for sid in range(total_scenarios):
        all_done = True
        for dim in EVAL_DIMENSIONS:
            key_a = f"s{sid}_a_{dim}"
            key_b = f"s{sid}_b_{dim}"
            if key_a not in ratings or key_b not in ratings:
                all_done = False
                break
        if all_done:
            completed += 1
    return completed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the HTML evaluation form."""
    if not HTML_FORM_PATH.exists():
        return Response(
            "<h1>Form not found</h1><p>Run the form generator first.</p>",
            status=404,
            content_type="text/html",
        )
    return send_file(HTML_FORM_PATH, mimetype="text/html")


@app.route("/health")
def health():
    """Health check."""
    return jsonify({"ok": True, "server": "human_eval_server", "port": PORT})


@app.route("/submit", methods=["POST"])
def submit():
    """
    Accept annotation submissions.

    Expected JSON body:
        {
            "annotator_id": "annotator_01",
            "ratings": {"s0_a_response_helpfulness": 4, ...},
            "partial": true/false
        }
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON body"}), 400

    annotator_id = body.get("annotator_id", "").strip()
    if not annotator_id:
        return jsonify({"ok": False, "error": "annotator_id is required"}), 400

    ratings = body.get("ratings", {})
    is_partial = body.get("partial", False)

    # Load existing data and merge (preserving older entries, overwriting with new)
    existing = _load_annotation(annotator_id)
    existing_ratings = existing.get("ratings", {})
    existing_ratings.update(ratings)

    # Load scenario count
    scenarios = _load_scenarios()
    total = len(scenarios)
    completed = _count_completed_scenarios(existing_ratings, total)

    record = {
        "annotator_id": annotator_id,
        "ratings": existing_ratings,
        "completed_scenarios": completed,
        "total_scenarios": total,
        "is_complete": not is_partial and completed == total,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "submissions": existing.get("submissions", 0) + 1,
    }
    if not is_partial:
        record["submitted_at"] = datetime.utcnow().isoformat() + "Z"

    _save_annotation(annotator_id, record)

    return jsonify({
        "ok": True,
        "annotator_id": annotator_id,
        "completed": completed,
        "total": total,
        "is_partial": is_partial,
    })


@app.route("/admin")
def admin():
    """Admin dashboard showing all annotators and their progress."""
    scenarios = _load_scenarios()
    total = len(scenarios)

    annotators = []
    for fpath in sorted(ANNOTATIONS_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            annotators.append({
                "id": data.get("annotator_id", fpath.stem),
                "completed": data.get("completed_scenarios", 0),
                "total": total,
                "is_complete": data.get("is_complete", False),
                "last_updated": data.get("last_updated", ""),
                "submissions": data.get("submissions", 0),
            })
        except Exception:
            continue

    n_annotators = len(annotators)
    n_complete = sum(1 for a in annotators if a["is_complete"])

    # Build HTML
    rows = ""
    for a in annotators:
        pct = round(a["completed"] / max(a["total"], 1) * 100)
        status = "COMPLETE" if a["is_complete"] else f"{pct}%"
        color = "#2e7d32" if a["is_complete"] else ("#ff9800" if pct > 0 else "#999")
        rows += f"""<tr>
  <td>{a['id']}</td>
  <td>{a['completed']}/{a['total']}</td>
  <td style="color:{color}; font-weight:600;">{status}</td>
  <td>{a['last_updated'][:19] if a['last_updated'] else '-'}</td>
  <td>{a['submissions']}</td>
</tr>
"""

    page = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>AURA Eval - Admin</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
th {{ background: #f0f0f0; }}
.summary {{ background: #e3f2fd; padding: 1em; border-radius: 8px; margin-bottom: 1em; }}
a {{ color: #1976d2; }}
</style></head><body>
<h1>AURA Human Evaluation - Admin</h1>
<div class="summary">
  <b>Annotators:</b> {n_annotators} registered | <b>Complete:</b> {n_complete}/{n_annotators} |
  <b>Target:</b> 20 annotators x {total} scenarios<br>
  <b>Status:</b> {"READY for analysis" if n_complete >= 20 else f"Need {max(20 - n_complete, 0)} more complete annotations"}
</div>
<p><a href="/">Evaluation Form</a> | <a href="/export">Export JSON</a></p>
<table>
<tr><th>Annotator</th><th>Progress</th><th>Status</th><th>Last Updated</th><th>Submissions</th></tr>
{rows if rows else '<tr><td colspan="5" style="text-align:center; color:#999;">No annotations yet.</td></tr>'}
</table>
</body></html>"""

    return Response(page, content_type="text/html")


@app.route("/export")
def export():
    """
    Export all annotations as aggregated JSON.

    Returns a JSON object with:
        - scenarios: the scenario definitions (with hidden labels)
        - annotations: list of per-annotator rating dicts
        - summary: counts and completion stats
    """
    scenarios = _load_scenarios()
    total = len(scenarios)

    all_annotations = []
    annotator_meta = []
    for fpath in sorted(ANNOTATIONS_DIR.glob("*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            all_annotations.append(data.get("ratings", {}))
            annotator_meta.append({
                "id": data.get("annotator_id", fpath.stem),
                "completed": data.get("completed_scenarios", 0),
                "is_complete": data.get("is_complete", False),
            })
        except Exception:
            continue

    result = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "scenarios": scenarios,
        "annotations": all_annotations,
        "annotator_meta": annotator_meta,
        "summary": {
            "total_scenarios": total,
            "total_annotators": len(all_annotations),
            "complete_annotators": sum(1 for a in annotator_meta if a["is_complete"]),
        },
    }
    return jsonify(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"AURA Human Evaluation Server")
    print(f"  Form:   {HTML_FORM_PATH}")
    print(f"  Data:   {ANNOTATIONS_DIR}")
    print(f"  Listen: http://{HOST}:{PORT}")
    print(f"  Admin:  http://{HOST}:{PORT}/admin")
    print(f"  Export: http://{HOST}:{PORT}/export")
    print()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
