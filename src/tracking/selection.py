"""Selection of the pipeline registered in MLflow, by the rule fixed before any arm was
compared (DECISIONS.md, Phase 7):

  the `retrieved`-condition arm (generator x prompt) with the highest correct-and-grounded
  rate: the share of all the arm's questions whose answer is graded `correct` (strict) and
  `fully_grounded`. Ties: higher accuracy over all questions, then the earlier prompt version.

The denominator is every question in the arm, so declining cannot raise the rate, and a
correct answer that the judge finds unsupported by the context does not count. The winner's
lead over the runner-up is reported as a paired question-bootstrap interval, as Phase 3
reported its retrieval winner.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.evaluation.reliability import is_correct, mean_with_ci, question_ci

SELECTION_CONDITION = "retrieved"


def correct_and_grounded(row: dict) -> bool:
    return bool(is_correct(row["correctness"]["label"])) and bool(
        row["groundedness"]["fully_grounded"]
    )


def arm_key(row: dict) -> str:
    return f"{row['condition']}__{row['model_key']}__{row['prompt_id']}"


def arm_rates(rows: list[dict], n_resamples: int, seed: int) -> dict[str, dict]:
    """Correct-and-grounded rate and accuracy for every arm (both conditions), with
    question-bootstrap intervals."""
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[arm_key(r)].append(r)
    out = {}
    for key in sorted(by_arm):
        rs = sorted(by_arm[key], key=lambda r: r["financebench_id"])
        qids = [r["financebench_id"] for r in rs]
        if len(set(qids)) != len(qids):
            raise ValueError(f"arm {key} has more than one trace per question")
        cg = [float(correct_and_grounded(r)) for r in rs]
        acc = [float(r["correctness"]["label"] == "correct") for r in rs]
        out[key] = {
            "condition": rs[0]["condition"],
            "model_key": rs[0]["model_key"],
            "prompt_id": rs[0]["prompt_id"],
            "prompt_version": rs[0]["prompt_version"],
            "n": len(rs),
            "n_correct_and_grounded": int(sum(cg)),
            "correct_and_grounded": mean_with_ci(cg, qids, n_resamples, seed),
            "accuracy_all": float(np.mean(acc)),
        }
    return out


def rank(arms: dict[str, dict], condition: str = SELECTION_CONDITION) -> list[str]:
    candidates = [k for k, a in arms.items() if a["condition"] == condition]
    return sorted(
        candidates,
        key=lambda k: (
            -arms[k]["correct_and_grounded"]["mean"],
            -arms[k]["accuracy_all"],
            arms[k]["prompt_version"],
            arms[k]["model_key"],
        ),
    )


def paired_difference(
    rows: list[dict], a: str, b: str, n_resamples: int, seed: int
) -> dict[str, object]:
    """Correct-and-grounded rate of arm a minus arm b over the questions both answered, with
    a paired question-bootstrap interval."""
    va = {r["financebench_id"]: correct_and_grounded(r) for r in rows if arm_key(r) == a}
    vb = {r["financebench_id"]: correct_and_grounded(r) for r in rows if arm_key(r) == b}
    qids = sorted(set(va) & set(vb))
    d = np.asarray([float(va[q]) - float(vb[q]) for q in qids])
    return {
        "a": a,
        "b": b,
        "n_questions": len(qids),
        "diff": float(d.mean()),
        "ci95": question_ci(qids, lambda idx: float(d[idx].mean()), n_resamples, seed),
    }


def select(rows: list[dict], n_resamples: int, seed: int) -> dict:
    arms = arm_rates(rows, n_resamples, seed)
    ranking = rank(arms)
    if len(ranking) < 2:
        raise ValueError("selection needs at least two candidate arms")
    return {
        "arms": arms,
        "ranking": ranking,
        "winner": ranking[0],
        "winner_minus_runner_up": paired_difference(
            rows, ranking[0], ranking[1], n_resamples, seed
        ),
    }
