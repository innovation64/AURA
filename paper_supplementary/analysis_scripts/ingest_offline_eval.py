"""Drop offline-eval JSON files into the annotations directory so they
participate in `analyze_human_eval.py` like any server-collected rater.

Each input file is the JSON the rater downloaded by clicking "Download
final ratings" in human_eval_offline.html. The file already has the
fields the server-side path produces; this script just normalises and
copies into evaluation/results/annotations/<id>.json.

Usage:
    python -m scripts.ingest_offline_eval path/to/file1.json path/to/file2.json ...

Or drop all received files into evaluation/inbox/ then:
    python -m scripts.ingest_offline_eval --inbox

Refuses to overwrite an existing annotation unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
INBOX     = ROOT / "evaluation" / "inbox"
ANN_DIR   = ROOT / "evaluation" / "results" / "annotations"


def ingest_one(src: Path, *, force: bool = False) -> str:
    with open(src) as f:
        data = json.load(f)

    aid = (data.get("annotator_id") or "").strip()
    if not aid:
        return f"SKIP {src.name}: no annotator_id"

    ratings = data.get("ratings") or {}
    if not isinstance(ratings, dict) or not ratings:
        return f"SKIP {src.name}: no ratings"

    safe_id = aid.replace(" ", "_").replace("/", "_")
    out = ANN_DIR / f"{safe_id}.json"

    completed = data.get("completed_scenarios")
    total     = data.get("total_scenarios", 50)
    if completed is None:
        # Recompute from ratings count
        completed = len({k.split("_")[0] for k in ratings.keys() if k.startswith("s")})

    record = {
        "annotator_id": aid,
        "ratings": ratings,
        "completed_scenarios": completed,
        "total_scenarios": total,
        "is_complete": bool(data.get("is_complete", completed == total)),
        "submitted_at": data.get("submitted_at") or data.get("saved_at")
                        or datetime.utcnow().isoformat() + "Z",
        "rater_kind": data.get("rater_kind", "human_offline"),
        "submissions": 1,
        "ingested_from": src.name,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }

    if out.exists() and not force:
        existing = json.loads(out.read_text())
        if existing.get("ratings") == ratings and existing.get("annotator_id") == aid:
            return f"NOOP {src.name}: identical record already at {out.name}"
        return (f"CONFLICT {src.name}: {out.name} exists and differs. "
                f"Use --force to overwrite (existing has "
                f"{len(existing.get('ratings', {}))} ratings, new has {len(ratings)}).")

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return (f"OK   {src.name} -> annotations/{out.name} "
            f"({len(ratings)} ratings, completed={completed}/{total}, "
            f"is_complete={record['is_complete']})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="JSON files to ingest")
    p.add_argument("--inbox", action="store_true",
                   help=f"Ingest all *.json in {INBOX}")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing annotation files of the same ID")
    args = p.parse_args()

    paths: list = []
    if args.inbox:
        if not INBOX.exists():
            INBOX.mkdir(parents=True)
            print(f"Created empty inbox at {INBOX}. Drop JSON files there.")
            return 0
        paths = sorted(INBOX.glob("*.json"))
    paths += [Path(f) for f in args.files]

    if not paths:
        print("No files. Pass paths or --inbox.", file=sys.stderr)
        return 1

    for src in paths:
        if not src.exists():
            print(f"MISS {src}: not found")
            continue
        try:
            print(ingest_one(src, force=args.force))
        except json.JSONDecodeError as e:
            print(f"ERR  {src.name}: invalid JSON ({e})")
        except Exception as e:
            print(f"ERR  {src.name}: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
