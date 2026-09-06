"""EDGAR filing fetcher: company -> CIK resolution, filing lookup, cached download.

Every network call is disk-cached and rate-limited to stay well under SEC's fair-access
policy (https://www.sec.gov/os/webmaster-faq#developers), and every request carries the
descriptive User-Agent EDGAR requires.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

MIN_REQUEST_INTERVAL_S = 0.11  # SEC allows 10 req/s; stay comfortably under that
_last_request_at = 0.0

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
BROWSE_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&company={company}&type={form}&dateb=&owner=include&count=10&output=atom"
)
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{document}"

_CORP_SUFFIXES = [
    " INCORPORATED",
    " CORPORATION",
    " COMPANY",
    " HOLDINGS",
    " GROUP",
    " LIMITED",
    " INC",
    " CORP",
    " LTD",
    " PLC",
    " LLC",
    " CO",
]

# Below this length, a substring match is too likely to be coincidental (e.g. "AES" is
# a substring of "BAE SYSTEMS", "KE" of "FOOTLOCKER" post-stripping) — require an exact
# normalized match instead, and let genuinely unresolvable names fall through to the
# EDGAR browse-prefix search.
MIN_SUBSTRING_MATCH_LEN = 5


class EdgarUserAgentMissing(RuntimeError):
    """Raised when EDGAR_USER_AGENT is not configured; EDGAR blocks anonymous scrapers."""


def _get(url: str, user_agent: str, cache_path: Path | None = None) -> bytes:
    """Rate-limited, disk-cached, EDGAR-compliant GET."""
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()

    if not user_agent:
        raise EdgarUserAgentMissing(
            "EDGAR_USER_AGENT is empty. Set it in .env — EDGAR blocks anonymous scrapers."
        )

    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    _last_request_at = time.monotonic()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data


def normalize_company_name(name: str) -> str:
    """Uppercase, strip punctuation and common corporate suffixes for matching."""
    normalized = name.upper()
    normalized = re.sub(r"/[A-Z]{2,4}$", "", normalized).strip()  # trailing /NY, /NEW, /DE tags
    normalized = re.sub(r"[.,]", "", normalized)
    changed = True
    while changed:
        changed = False
        for suffix in _CORP_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                changed = True
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return normalized


def load_ticker_map(cache_dir: Path, user_agent: str) -> list[dict]:
    """Fetch (and cache) SEC's ticker -> CIK -> title mapping."""
    cache_path = cache_dir / "company_tickers.json"
    raw = _get(TICKERS_URL, user_agent, cache_path=cache_path)
    return list(json.loads(raw).values())


@dataclass(frozen=True)
class CikResolution:
    company_query: str
    cik: str  # zero-padded to 10 digits
    matched_name: str
    method: str  # ticker_exact | title_normalized | edgar_browse_prefix | manual_override


def resolve_cik(
    company: str,
    ticker_map: list[dict],
    overrides: dict[str, str],
    cache_dir: Path,
    user_agent: str,
) -> CikResolution | None:
    """Resolve a FinanceBench company name to a CIK, trying cheapest methods first."""
    if company in overrides:
        cik = str(overrides[company]).zfill(10)
        return CikResolution(company, cik, company, "manual_override")

    query_norm = normalize_company_name(company)

    for entry in ticker_map:
        if entry["ticker"].upper() == company.upper():
            return CikResolution(
                company, str(entry["cik_str"]).zfill(10), entry["title"], "ticker_exact"
            )

    for entry in ticker_map:
        title_norm = normalize_company_name(entry["title"])
        if title_norm == query_norm:
            return CikResolution(
                company, str(entry["cik_str"]).zfill(10), entry["title"], "title_exact"
            )

    if len(query_norm) >= MIN_SUBSTRING_MATCH_LEN:
        for entry in ticker_map:
            title_norm = normalize_company_name(entry["title"])
            if len(title_norm) >= MIN_SUBSTRING_MATCH_LEN and (
                query_norm in title_norm or title_norm in query_norm
            ):
                return CikResolution(
                    company, str(entry["cik_str"]).zfill(10), entry["title"], "title_substring"
                )

    # Falls back to EDGAR's own company-name search, which also covers filers that
    # have since delisted (and so are absent from company_tickers.json).
    for form in ("10-K", "10-Q"):
        url = BROWSE_URL.format(company=company.replace(" ", "+"), form=form)
        cache_path = cache_dir / "browse" / f"{query_norm}_{form}.atom"
        try:
            raw = _get(url, user_agent, cache_path=cache_path)
        except urllib.error.URLError:
            continue
        cik = _first_cik_from_atom(raw)
        if cik:
            return CikResolution(company, cik, company, "edgar_browse_prefix")

    return None


def _first_cik_from_atom(raw: bytes) -> str | None:
    # <cik> lives inside a non-namespaced <company-info> embedded as the entry's
    # text/xml content, so a plain tag-name scan is simpler than namespaced XPath.
    root = ElementTree.fromstring(raw)
    for el in root.iter():
        if el.tag.endswith("cik") and el.text:
            return el.text.strip().zfill(10)
    return None


@dataclass(frozen=True)
class FilingRef:
    cik: str
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return ARCHIVES_URL.format(
            cik_int=int(self.cik),
            accession_nodash=self.accession_nodash,
            document=self.primary_document,
        )


@dataclass(frozen=True)
class CompanyFilings:
    fiscal_year_end: str  # "MMDD", e.g. "1231" or Amcor's "0630"
    entries: list[dict]


def get_filings(cik: str, cache_dir: Path, user_agent: str) -> CompanyFilings:
    """Return every filing entry for a CIK, merging the recent window with older pages."""
    cik10 = cik.zfill(10)
    cache_path = cache_dir / "submissions" / f"CIK{cik10}.json"
    raw = _get(SUBMISSIONS_URL.format(cik10=cik10), user_agent, cache_path=cache_path)
    data = json.loads(raw)

    entries = _columns_to_rows(data["filings"]["recent"])
    for older in data["filings"].get("files", []):
        older_cache = cache_dir / "submissions" / older["name"]
        older_url = f"https://data.sec.gov/submissions/{older['name']}"
        older_raw = _get(older_url, user_agent, cache_path=older_cache)
        entries.extend(_columns_to_rows(json.loads(older_raw)))
    return CompanyFilings(data.get("fiscalYearEnd") or "1231", entries)


def _columns_to_rows(columnar: dict) -> list[dict]:
    keys = list(columnar.keys())
    n = len(columnar[keys[0]])
    return [{k: columnar[k][i] for k in keys} for i in range(n)]


@dataclass(frozen=True)
class ParsedDocName:
    company_slug: str
    form: str  # "10-K" | "10-Q"
    year: int
    quarter: int | None


_DOC_NAME_RE = re.compile(
    r"^(?P<company>.+)_(?P<year>\d{4})(?:Q(?P<quarter>[1-4]))?_(?P<form>10K|10Q)$"
)


def parse_doc_name(doc_name: str) -> ParsedDocName | None:
    """Parse a FinanceBench doc_name like 'AMD_2015_10K' or '3M_2023Q2_10Q'.

    Returns None for forms this project doesn't fetch from EDGAR (8-K, Earnings) —
    see DECISIONS.md: scope is limited to what the spec names, 10-K/10-Q.
    """
    match = _DOC_NAME_RE.match(doc_name)
    if not match:
        return None
    form = "10-K" if match["form"] == "10K" else "10-Q"
    quarter = int(match["quarter"]) if match["quarter"] else None
    return ParsedDocName(match["company"], form, int(match["year"]), quarter)


def find_filing(
    parsed: ParsedDocName, cik: str, company_filings: CompanyFilings
) -> FilingRef | None:
    """Pick the filing entry matching the target form + fiscal period.

    FinanceBench's `year` label is the fiscal year the report falls in, per that
    company's own calendar — not necessarily the calendar year of the reportDate. For a
    10-K this is a non-issue (a fiscal year's annual report always carries a reportDate
    in the calendar year the spec's `year` names, whatever the fiscal year end is). For
    a 10-Q we have to derive the expected quarter-end month/year from the company's
    actual `fiscalYearEnd` (e.g. Amcor's is June 30, so its "FY2023 Q2" quarter end is
    December 2022, not a calendar Q2) — assuming calendar-aligned quarters silently
    misses every non-calendar-fiscal-year filer.
    """
    filings = company_filings.entries
    candidates = [f for f in filings if f["form"] == parsed.form and f["reportDate"]]

    if parsed.form == "10-K":
        matches = [f for f in candidates if f["reportDate"].startswith(str(parsed.year))]
    else:
        fye_month = int(company_filings.fiscal_year_end[:2])
        raw_month = fye_month - 3 * (4 - parsed.quarter)
        if raw_month <= 0:
            expected_month, expected_year = raw_month + 12, parsed.year - 1
        else:
            expected_month, expected_year = raw_month, parsed.year
        expected_idx = expected_year * 12 + expected_month

        matches = []
        for f in candidates:
            report_year, report_month = int(f["reportDate"][:4]), int(f["reportDate"][5:7])
            if abs((report_year * 12 + report_month) - expected_idx) <= 1:
                matches.append(f)

    if not matches:
        return None
    matches.sort(key=lambda f: f["filingDate"])
    best = matches[0]
    return FilingRef(
        cik=cik,
        accession_number=best["accessionNumber"],
        form=best["form"],
        filing_date=best["filingDate"],
        report_date=best["reportDate"],
        primary_document=best["primaryDocument"],
    )


def fetch_document(filing: FilingRef, cache_dir: Path, user_agent: str) -> bytes:
    """Download (or read from cache) the primary document for a filing."""
    suffix = Path(filing.primary_document).suffix or ".bin"
    cache_path = cache_dir / "documents" / f"{filing.cik}_{filing.accession_nodash}{suffix}"
    return _get(filing.document_url, user_agent, cache_path=cache_path)
