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

import lxml.html

MIN_REQUEST_INTERVAL_S = 0.11  # SEC allows 10 req/s; stay comfortably under that
MAX_ATTEMPTS = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_BASE_DELAY_S = 2.0  # 2, 4, 8, 16 s between attempts
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
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(MAX_ATTEMPTS):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_S:
            time.sleep(MIN_REQUEST_INTERVAL_S - elapsed)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            _last_request_at = time.monotonic()
            break
        except urllib.error.HTTPError as exc:
            _last_request_at = time.monotonic()
            # EDGAR answers bursts with 429/503; back off and retry. Other statuses
            # (404 etc.) are real answers and are raised immediately.
            if exc.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_BASE_DELAY_S * 2**attempt)

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


# A 52/53-week fiscal year ending on the Saturday/Sunday nearest December 31 can end in
# the first days of January (Johnson & Johnson's fiscal 2022 ended January 1, 2023).
# Such a year is named for the calendar year it almost entirely covers, so a period end
# in this window belongs to the previous year. Filers whose year ends in late January or
# February (Walmart, Best Buy, Ulta) name the year by its end date and are unaffected.
_EARLY_JANUARY_LAST_DAY = 7


def _is_early_january(month: int, day: int) -> bool:
    return month == 1 and day <= _EARLY_JANUARY_LAST_DAY


def fiscal_year_label(report_date: str) -> int:
    """The fiscal-year label (FinanceBench's `year`) for a 10-K with this `reportDate`."""
    year, month, day = (int(p) for p in report_date.split("-"))
    return year - 1 if _is_early_january(month, day) else year


def effective_year_end_month(fiscal_year_end: str) -> int:
    """Month in which the fiscal year effectively ends, from EDGAR's "MMDD"
    `fiscalYearEnd`; an early-January year end counts as December."""
    month, day = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
    return 12 if _is_early_january(month, day) else month


def find_filing(
    parsed: ParsedDocName, cik: str, company_filings: CompanyFilings
) -> FilingRef | None:
    """Pick the filing entry matching the target form + fiscal period.

    FinanceBench's `year` label is the fiscal year the report falls in, per that
    company's own calendar — not necessarily the calendar year of the reportDate. For a
    10-K the label is the reportDate's calendar year, except for a 52/53-week year that
    ends in the first days of January, which carries the previous year's label (see
    `fiscal_year_label`; matching on the raw reportDate year once fetched J&J's fiscal
    2021 10-K for "JOHNSON_JOHNSON_2022_10K"). For a 10-Q we have to derive the expected
    quarter-end month/year from the company's actual `fiscalYearEnd` (e.g. Amcor's is
    June 30, so its "FY2023 Q2" quarter end is December 2022, not a calendar Q2) —
    assuming calendar-aligned quarters silently misses every non-calendar-fiscal-year
    filer.
    """
    filings = company_filings.entries
    candidates = [f for f in filings if f["form"] == parsed.form and f["reportDate"]]

    if parsed.form == "10-K":
        matches = [f for f in candidates if fiscal_year_label(f["reportDate"]) == parsed.year]
    else:
        fye_month = effective_year_end_month(company_filings.fiscal_year_end)
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


def document_cache_path(filing: FilingRef, cache_dir: Path) -> Path:
    """Where a filing's primary document is cached. Keyed by CIK + accession number — the
    filing's identity — never by FinanceBench `doc_name`: a name-keyed copy survives a
    change in which filing the name resolves to, and silently keeps the old document."""
    suffix = Path(filing.primary_document).suffix or ".bin"
    return cache_dir / "documents" / f"{filing.cik}_{filing.accession_nodash}{suffix}"


def fetch_document(filing: FilingRef, cache_dir: Path, user_agent: str) -> Path:
    """Download (or find in cache) the primary document for a filing; return its path."""
    cache_path = document_cache_path(filing, cache_dir)
    _get(filing.document_url, user_agent, cache_path=cache_path)
    return cache_path


INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.htm"
)


def parse_filing_index(index_html: bytes) -> list[tuple[str, str]]:
    """(type, document filename) for every document row of an EDGAR filing index page.
    The filename comes from the row's link, not the cell text, which for inline-XBRL
    documents carries a trailing " iXBRL" marker."""
    rows = []
    for tr in lxml.html.fromstring(index_html).iter("tr"):
        cells = [c.text_content().strip() for c in tr.iter("td")]
        links = [a.get("href") for a in tr.iter("a") if a.get("href")]
        if len(cells) >= 4 and cells[3] and links:
            rows.append((cells[3], links[0].rsplit("/", 1)[-1]))
    return rows


def find_exhibits(
    filing: FilingRef, exhibit_type: str, cache_dir: Path, user_agent: str
) -> list[str]:
    """Filenames of a filing's exhibits of `exhibit_type` (e.g. "EX-13" matches "EX-13"
    and "EX-13.1", not "EX-10.13"), in index order."""
    url = INDEX_URL.format(
        cik_int=int(filing.cik),
        accession_nodash=filing.accession_nodash,
        accession=filing.accession_number,
    )
    cache_path = cache_dir / "index" / f"{filing.cik}_{filing.accession_nodash}-index.htm"
    return select_exhibits(
        parse_filing_index(_get(url, user_agent, cache_path=cache_path)), exhibit_type
    )


def select_exhibits(rows: list[tuple[str, str]], exhibit_type: str) -> list[str]:
    """Filenames whose type is `exhibit_type` or a numbered variant of it ("EX-13.1")."""
    return [doc for typ, doc in rows if typ == exhibit_type or typ.startswith(exhibit_type + ".")]


def fetch_exhibit(filing: FilingRef, document: str, cache_dir: Path, user_agent: str) -> Path:
    """Download (or find in cache) one exhibit of a filing; return its path. Keyed by
    CIK + accession + exhibit filename, so it never collides with the primary document."""
    cache_path = cache_dir / "documents" / f"{filing.cik}_{filing.accession_nodash}__{document}"
    url = ARCHIVES_URL.format(
        cik_int=int(filing.cik), accession_nodash=filing.accession_nodash, document=document
    )
    _get(url, user_agent, cache_path=cache_path)
    return cache_path


_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
_COVER_PERIOD_RE = re.compile(
    r"(?:fiscal\s+year|quarterly\s+period)\s+ended\s*:?\s*"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})\s*,\s*(?P<year>\d{4})",
    re.IGNORECASE,
)


def cover_period_end(full_text: str) -> str | None:
    """The period-end date stated on a 10-K/10-Q cover page ("For the fiscal year ended
    January 1, 2023", "For the quarterly period ended July 29, 2023") as ISO
    "YYYY-MM-DD", or None if the cover wording isn't found. Only the first ~5,000
    characters are searched: the cover page, not later narrative mentioning other
    periods."""
    head = full_text[:5000].replace("\xa0", " ")
    match = _COVER_PERIOD_RE.search(head)
    if not match or match["month"].lower() not in _MONTHS:
        return None
    month = _MONTHS[match["month"].lower()]
    return f"{int(match['year']):04d}-{month:02d}-{int(match['day']):02d}"
