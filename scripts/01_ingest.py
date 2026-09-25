"""Fetch the FinanceBench question set and its underlying 10-K/10-Q filings from EDGAR,
parse them to metadata-tagged text, and record corpus statistics.

Scope note (see DECISIONS.md): FinanceBench's `doc_name` also references some 8-K and
"Earnings" documents. PROJECT_SPEC.md §5 names only "10-K/10-Q PDFs or HTML" for the
EDGAR-fetch corpus, so those are parsed out of scope here and logged as skipped, not
silently dropped.

Usage: uv run python scripts/01_ingest.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import edgar
from src.ingestion.parser import ParsedDocument, merge_documents, parse_document

RAW_DIR = Path("data/raw")
# Overridable so the smoke evaluation (scripts/10_smoke_eval.py) can read the main cache.
EDGAR_CACHE_DIR = Path(os.environ.get("EDGAR_CACHE_DIR", RAW_DIR / "edgar_cache"))
FINANCEBENCH_DIR = RAW_DIR / "financebench"
PROCESSED_DIR = Path("data/processed/parsed")
RESULTS_DIR = Path("results/metrics")
OVERRIDES_PATH = Path("configs/edgar_company_overrides.json")

WORD_RE = re.compile(r"\S+")


def load_financebench_rows() -> list[dict]:
    """The local copy when there is one (it is DVC-versioned raw data); the Hub only for a
    first download. Checking the Hub on every run would let an upstream revision of the
    dataset silently replace the questions the reported numbers were computed on."""
    local_path = FINANCEBENCH_DIR / "financebench_merged.jsonl"
    if not local_path.exists():
        FINANCEBENCH_DIR.mkdir(parents=True, exist_ok=True)
        local_path = hf_hub_download(
            repo_id="PatronusAI/financebench",
            repo_type="dataset",
            filename="financebench_merged.jsonl",
            local_dir=FINANCEBENCH_DIR,
        )
    with open(local_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parsed_document_to_dict(doc: ParsedDocument) -> dict:
    return {
        "doc_id": doc.doc_id,
        "source_path": doc.source_path,
        "full_text": doc.full_text,
        "blocks": [asdict(b) for b in doc.blocks],
    }


def main() -> None:
    load_dotenv()
    user_agent = os.environ.get("EDGAR_USER_AGENT", "")
    if not user_agent and os.environ.get("EDGAR_OFFLINE") != "1":  # offline reads the cache only
        raise SystemExit("EDGAR_USER_AGENT is not set in .env — see .env.example.")

    EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FinanceBench question set from HuggingFace...")
    rows = load_financebench_rows()
    unique_docs = {r["doc_name"]: (r["company"], r["doc_type"], r["doc_period"]) for r in rows}
    print(f"  {len(rows)} questions over {len(unique_docs)} unique documents")

    overrides = {}
    if OVERRIDES_PATH.exists():
        overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))

    print("Loading SEC ticker map...")
    ticker_map = edgar.load_ticker_map(EDGAR_CACHE_DIR, user_agent)

    cik_cache: dict[str, edgar.CikResolution | None] = {}
    filings_cache: dict[str, list[dict]] = {}

    outcomes: dict[str, dict] = {}
    parsed_docs: dict[str, ParsedDocument] = {}

    for doc_name, (company, doc_type, _doc_period) in sorted(unique_docs.items()):
        parsed_name = edgar.parse_doc_name(doc_name)
        if parsed_name is None:
            outcomes[doc_name] = {
                "status": "skipped_out_of_scope",
                "reason": f"doc_type={doc_type!r} is not 10-K/10-Q; see PROJECT_SPEC.md §5",
            }
            continue

        if company not in cik_cache:
            cik_cache[company] = edgar.resolve_cik(
                company, ticker_map, overrides, EDGAR_CACHE_DIR, user_agent
            )
        resolution = cik_cache[company]
        if resolution is None:
            outcomes[doc_name] = {"status": "unresolved_company", "company": company}
            continue

        if resolution.cik not in filings_cache:
            filings_cache[resolution.cik] = edgar.get_filings(
                resolution.cik, EDGAR_CACHE_DIR, user_agent
            )
        filing = edgar.find_filing(parsed_name, resolution.cik, filings_cache[resolution.cik])
        if filing is None:
            outcomes[doc_name] = {
                "status": "unresolved_filing",
                "company": company,
                "cik": resolution.cik,
                "cik_method": resolution.method,
            }
            continue

        # Parse the accession-keyed cache file itself. An earlier version parsed a
        # doc_name-keyed copy written only if absent, so when a fetcher fix changed which
        # filing a name resolved to, the stale copy was kept and parsed while this
        # record showed the new accession (see DECISIONS.md, Phase 3).
        try:
            doc_path = edgar.fetch_document(filing, EDGAR_CACHE_DIR, user_agent)
        except edgar.EdgarOffline:
            raise  # a rebuild missing raw data must stop, not record a fetch error
        except Exception as exc:  # noqa: BLE001 - record and continue the batch
            outcomes[doc_name] = {"status": "fetch_error", "error": str(exc)}
            continue

        # A 10-K that incorporates its financial statements by reference from the annual
        # report to shareholders files that report as Exhibit 13 (SEC Reg. S-K Item 601);
        # without it the corpus lacks the statements themselves (CVS's 2018 10-K did).
        # If the filing index itself can't be read (EDGAR serves a persistent 503 for some
        # index pages while the filing's documents download fine), keep the verified
        # primary document and record the failed lookup rather than dropping the filing.
        exhibits: list[str] = []
        exhibit_paths: list[Path] = []
        exhibit_lookup_error: str | None = None
        if filing.form == "10-K":
            try:
                exhibits = edgar.find_exhibits(filing, "EX-13", EDGAR_CACHE_DIR, user_agent)
            except edgar.EdgarOffline:
                raise
            except Exception as exc:  # noqa: BLE001
                exhibit_lookup_error = str(exc)
                print(f"  [warn] {doc_name}: Exhibit 13 lookup failed ({exc})")
        try:
            exhibit_paths = [
                edgar.fetch_exhibit(filing, ex, EDGAR_CACHE_DIR, user_agent) for ex in exhibits
            ]
        except edgar.EdgarOffline:
            raise
        except Exception as exc:  # noqa: BLE001 - an EX-13 that exists but won't download
            outcomes[doc_name] = {"status": "fetch_error", "error": f"exhibit: {exc}"}
            continue

        try:
            parsed = parse_document(doc_path, doc_name)
            if exhibit_paths:
                parsed = merge_documents(
                    doc_name, [parsed, *(parse_document(p, doc_name) for p in exhibit_paths)]
                )
        except Exception as exc:  # noqa: BLE001
            outcomes[doc_name] = {"status": "parse_error", "error": str(exc)}
            continue

        # Independent check that the parsed document is the period EDGAR says it is:
        # compare the cover page's stated period end with the filing's reportDate.
        cover_date = edgar.cover_period_end(parsed.full_text)
        period_check = (
            "cover_not_found"
            if cover_date is None
            else ("match" if cover_date == filing.report_date else "mismatch")
        )

        parsed_docs[doc_name] = parsed
        (PROCESSED_DIR / f"{doc_name}.json").write_text(
            json.dumps(parsed_document_to_dict(parsed)), encoding="utf-8"
        )
        outcomes[doc_name] = {
            "status": "ok",
            "company": company,
            "cik": resolution.cik,
            "cik_method": resolution.method,
            "accession_number": filing.accession_number,
            "form": filing.form,
            "report_date": filing.report_date,
            "filing_date": filing.filing_date,
            "exhibits_appended": exhibits,
            "exhibit_lookup_error": exhibit_lookup_error,
            "cover_period_end": cover_date,
            "period_check": period_check,
            "n_blocks": len(parsed.blocks),
            "n_chars": len(parsed.full_text),
        }
        print(f"  [{outcomes[doc_name]['status']}] {doc_name}")

    status_counts = Counter(o["status"] for o in outcomes.values())
    print("\nIngestion outcomes:", dict(status_counts))
    period_checks = Counter(o["period_check"] for o in outcomes.values() if o["status"] == "ok")
    lookup_failed = sorted(n for n, o in outcomes.items() if o.get("exhibit_lookup_error"))
    print("Exhibit 13 lookups failed (primary document kept):", lookup_failed or "none")
    print("Cover-page period checks:", dict(period_checks))
    for name, o in sorted(outcomes.items()):
        if o.get("period_check") == "mismatch":
            print(
                f"  PERIOD MISMATCH {name}: cover says {o['cover_period_end']}, "
                f"EDGAR reportDate {o['report_date']}"
            )

    n_pages = 0
    n_words = 0
    n_table_blocks = 0
    n_total_blocks = 0
    for doc in parsed_docs.values():
        pages = {b.page for b in doc.blocks if b.page is not None}
        n_pages += len(pages)
        n_words += len(WORD_RE.findall(doc.full_text))
        n_table_blocks += sum(1 for b in doc.blocks if b.is_table)
        n_total_blocks += len(doc.blocks)

    corpus_stats = {
        "financebench_questions": len(rows),
        "financebench_unique_docs": len(unique_docs),
        "documents_ingested": len(parsed_docs),
        "page_count": n_pages,
        "word_count": n_words,
        "word_count_method": (
            "whitespace-split token approximation, not a specific model tokenizer"
        ),
        "table_block_density": (n_table_blocks / n_total_blocks) if n_total_blocks else 0.0,
        "ingestion_outcomes": outcomes,
        "status_counts": dict(status_counts),
        "period_check_counts": dict(period_checks),
        "exhibit_lookup_failed": lookup_failed,
    }
    out_path = RESULTS_DIR / "corpus_stats.json"
    out_path.write_text(json.dumps(corpus_stats, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
