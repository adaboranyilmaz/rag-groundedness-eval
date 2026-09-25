"""The local HTML page used to hand-label the validation sample for groundedness.

The page shows, per item, only what the verification judge sees plus the answer: the
question, the five context excerpts (formatted exactly as the generator and the judge saw
them), the answer, and the decomposed claims. It deliberately hides the judge's verdicts,
the gold answer, the model's own citations and which model wrote the answer, and presents
items in a seeded shuffled order. Labels are kept in the browser's local storage while
working and exported as JSON (`groundedness_human.json`), tied to the sample by its hash.
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).with_name("labeling_page.html")


def render_labeling_page(page_items: list[dict], sample_sha256: str) -> str:
    """`page_items`: [{item_id, question, answer, excerpts: [{header, text}], claims:
    [{claim, kind}]}] in presentation order. `item_id` is opaque (a hash of the trace id):
    the trace id names the model, condition and prompt, and the page source must not.
    Embedded as JSON; `</` is escaped so filing text can never close the script element."""
    data = json.dumps({"sample_sha256": sample_sha256, "items": page_items}, ensure_ascii=False)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__DATA__", data.replace("</", r"<\/"))


REVIEW_TEMPLATE_PATH = Path(__file__).with_name("quadrant_review_page.html")


def render_review_page(page_items: list[dict], sample_sha256: str) -> str:
    """The Phase 6 review of the correct-but-ungrounded quadrant. Unlike the labelling page
    it shows the judge's claim verdicts and reasons, since the question is whether the judge
    is right; it still hides the model, the prompt and the gold answer. `page_items`:
    [{review_id, question, answer, excerpts: [{header, text}], claims: [{claim, kind,
    verdict, reason, supporting_excerpts}]}]; `review_id` is opaque."""
    data = json.dumps({"sample_sha256": sample_sha256, "items": page_items}, ensure_ascii=False)
    template = REVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__DATA__", data.replace("</", r"<\/"))


PREMISE_TEMPLATE_PATH = Path(__file__).with_name("premise_labeling_page.html")


def render_premise_page(page_items: list[dict], sample_sha256: str) -> str:
    """The author's blind labels for the adversarial false-premise answers (Phase 6, Q4):
    how each answer handles the premise, with the premise judge's definitions. Hides the
    model, the prompt and the judge's verdict. `page_items`: [{label_id, question,
    premise_note, answer}]; `label_id` is opaque."""
    data = json.dumps({"sample_sha256": sample_sha256, "items": page_items}, ensure_ascii=False)
    template = PREMISE_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__DATA__", data.replace("</", r"<\/"))
