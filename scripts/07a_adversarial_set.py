"""The adversarial question set for Phase 6 (Q4): validate it, and run the author's review.

  validate   re-check every record against the corpus (evidence alignment, absence checks)
             and, once a review has been applied, the approved set's composition
  review     write data/adversarial/review_candidates.md from the candidates, for the
             author to mark approve / reject
  apply      read the marked review file and set each record's status (and the author's
             comment) in data/adversarial/questions.jsonl, then validate

Usage:
    uv run python scripts/07a_adversarial_set.py validate
    uv run python scripts/07a_adversarial_set.py review
    uv run python scripts/07a_adversarial_set.py apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.adversarial import (
    parse_review,
    quota_problems,
    record_problems,
    render_review,
)
from src.evaluation.gold_spans import NormalisedText

QUESTIONS_PATH = Path("data/adversarial/questions.jsonl")
REVIEW_PATH = Path("data/adversarial/review_candidates.md")
PARSED_DIR = Path("data/processed/parsed")


def read_records() -> list[dict]:
    return [json.loads(x) for x in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()]


def write_records(records: list[dict]) -> None:
    QUESTIONS_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def validate(records: list[dict]) -> bool:
    raw = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))["full_text"]
        for p in sorted(PARSED_DIR.glob("*.json"))
    }
    docs = {d: NormalisedText(t) for d, t in raw.items()}
    ok = True
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        print("duplicate ids")
        ok = False
    for r in records:
        for p in record_problems(r, docs, raw):
            print(f"{r['id']}: {p}")
            ok = False
    if any(r["status"] != "candidate" for r in records):
        for p in quota_problems(records):
            print(f"quota: {p}")
            ok = False
    counts = {
        s: sum(r["status"] == s for r in records) for s in ("candidate", "approved", "rejected")
    }
    print(f"{len(records)} records {counts}: {'valid' if ok else 'INVALID'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "review", "apply"])
    args = parser.parse_args()
    records = read_records()
    if args.command == "validate":
        sys.exit(0 if validate(records) else 1)
    if args.command == "review":
        pending = [r for r in records if r["status"] == "candidate"]
        REVIEW_PATH.write_text(render_review(pending), encoding="utf-8")
        print(f"{len(pending)} candidates -> {REVIEW_PATH}")
        return
    decisions, comments, problems = parse_review(REVIEW_PATH.read_text(encoding="utf-8"))
    if problems:
        sys.exit("review not complete: " + "; ".join(problems))
    for r in records:
        if r["id"] in decisions:
            r["status"] = decisions[r["id"]]
        if r["id"] in comments:
            r["review_comment"] = comments[r["id"]]
    write_records(records)
    print(f"applied {len(decisions)} decisions, {len(comments)} comments")
    sys.exit(0 if validate(records) else 1)


if __name__ == "__main__":
    main()
