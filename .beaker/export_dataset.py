"""Export real Harvey LAB tasks into the Beaker JSONL dataset layout.

The labels Beaker optimizes against are the benchmark's own rubrics: every row
carries one LAB task's instructions as ``input`` and that task's ~60 binary
``criteria`` as ``expected``. Nothing is synthesized here — task ids come from
the frozen ``splits/{train,val}.txt`` lists and the rows are read straight out
of each task's ``task.json`` at the pinned ``HARVEY_LABS_COMMIT``.

Source documents stay out of the JSONL (they are binary and run to thousands of
files per task): the row records the task id + document filenames, and the spec
re-fetches that task folder from the pinned commit at rollout time, exactly like
``harvey-lab run`` does.

Usage (from the recipe root)::

    uv run python .beaker/export_dataset.py --train 8 --val 4

Rows land in ``.beaker/dataset/{train,val}.jsonl`` plus a ``manifest.json``
holding provenance (source repo, pinned commit, split counts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from harvey_lab.config import HARVEY_LABS_COMMIT
from harvey_lab.data.dataset import HarveyLabRecord, load_records, read_split
from harvey_lab.data.fetch import REPO, ensure_task_dirs


DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def _row(record: HarveyLabRecord) -> dict[str, Any]:
    """One standard JSONL case row for ``record``.

    ``input`` is everything the agent is handed (minus the documents, which the
    spec fetches by ``task_id``); ``expected`` is the rubric the judge grades
    against. ``group_key`` is the practice area so Beaker groups scores the way
    the benchmark is stratified.
    """
    return {
        "id": record.task_id,
        "input": {
            "task_id": record.task_id,
            "title": record.title,
            "work_type": record.work_type,
            "instructions": record.instructions,
            "deliverables": dict(record.deliverables),
            "documents": list(record.documents),
        },
        "expected": {
            "criteria": [
                {
                    "id": criterion.id,
                    "title": criterion.title,
                    "match_criteria": criterion.match_criteria,
                    "deliverables": list(criterion.deliverables),
                }
                for criterion in record.criteria
            ]
        },
        "metadata": {
            "practice_area": record.practice_area,
            "harvey_labs_commit": HARVEY_LABS_COMMIT,
            "task_fingerprint": record.task_fingerprint,
            "num_criteria": len(record.criteria),
            "num_documents": len(record.documents),
        },
        "group_key": record.practice_area,
    }


def _export_split(split: str, limit: int, cache_dir: Path | None) -> list[dict[str, Any]]:
    """Fetch the first ``limit`` tasks of ``split`` and turn them into rows.

    A task whose rubric carries no scoreable criterion is unscoreable (the
    recipe's evaluator excludes it from its averages), so it is dropped here
    rather than shipped as an unscoreable case.
    """
    task_ids = read_split(split)[:limit]
    tasks_root = ensure_task_dirs(task_ids, cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    for record in load_records(tasks_root, task_ids=task_ids):
        if not record.criteria:
            print(f"  skip {record.task_id}: no scoreable rubric criteria", file=sys.stderr)
            continue
        rows.append(_row(record))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=8, help="How many train tasks to export (default: 8).")
    parser.add_argument("--val", type=int, default=4, help="How many val tasks to export (default: 4).")
    parser.add_argument("--test", type=int, default=0, help="How many test tasks to export (default: 0, omitted).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Task-download cache (default: harvey_lab's).")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR, help=f"Output dir (default: {DATASET_DIR}).")
    args = parser.parse_args(argv)

    counts: dict[str, int] = {}
    for split, limit in (("train", args.train), ("val", args.val), ("test", args.test)):
        if limit <= 0:
            continue
        print(f"Exporting {limit} {split} task(s) ...", file=sys.stderr)
        rows = _export_split(split, limit, args.cache_dir)
        _write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    manifest = {
        "source": f"https://github.com/{REPO}",
        "commit": HARVEY_LABS_COMMIT,
        "splits": counts,
        "notes": (
            "Rows are Harvey LAB tasks from the frozen harvey_lab splits; `expected.criteria` is the task's own "
            "rubric. Source documents are not inlined — the spec fetches each task folder at `commit`."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {counts} to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
