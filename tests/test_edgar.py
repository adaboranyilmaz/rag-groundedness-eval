"""Unit tests for src/ingestion/edgar.py's pure logic: name normalization, CIK
resolution tiers, doc_name parsing, and fiscal-period filing matching. Network calls
(`_get`, the EDGAR browse-prefix fallback, live submissions fetches) aren't exercised
here — those were validated against the real EDGAR API during development."""

import pytest

from src.ingestion import edgar


class TestNormalizeCompanyName:
    def test_strips_common_corporate_suffixes(self):
        assert edgar.normalize_company_name("3M Co") == "3M"
        assert edgar.normalize_company_name("Apple Inc.") == "APPLE"

    def test_strips_trailing_state_tag(self):
        assert edgar.normalize_company_name("CORNING INC /NY") == "CORNING"

    def test_removes_punctuation_and_spaces(self):
        assert edgar.normalize_company_name("Johnson & Johnson") == "JOHNSONJOHNSON"


class TestResolveCik:
    TICKER_MAP = [
        {"cik_str": 66740, "ticker": "MMM", "title": "3M CO"},
        {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
        {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
        {"cik_str": 874761, "ticker": "AES", "title": "AES CORP"},
    ]

    def test_resolves_via_ticker_exact(self, tmp_path):
        res = edgar.resolve_cik("AMD", self.TICKER_MAP, {}, tmp_path, "test-agent")
        assert res.cik == "0000002488"
        assert res.method == "ticker_exact"

    def test_resolves_via_title_exact(self, tmp_path):
        res = edgar.resolve_cik("3M", self.TICKER_MAP, {}, tmp_path, "test-agent")
        assert res.cik == "0000066740"
        assert res.method == "title_exact"

    def test_corporate_suffix_stripping_enables_exact_match(self, tmp_path):
        # "AES Corporation" only matches "AES CORP" after both sides drop their
        # corporate-suffix words — this was previously a substring false positive
        # against an unrelated company ("BAE SYSTEMS") before suffix stripping and a
        # minimum substring length were added.
        res = edgar.resolve_cik("AES Corporation", self.TICKER_MAP, {}, tmp_path, "test-agent")
        assert res.cik == "0000874761"
        assert res.method == "title_exact"

    def test_resolves_via_title_substring_when_long_enough(self, tmp_path):
        res = edgar.resolve_cik("Amazon", self.TICKER_MAP, {}, tmp_path, "test-agent")
        assert res.cik == "0001018724"
        assert res.method == "title_substring"

    def test_manual_override_takes_priority_over_everything(self, tmp_path):
        res = edgar.resolve_cik("3M", self.TICKER_MAP, {"3M": "9999999999"}, tmp_path, "test-agent")
        assert res.cik == "9999999999"
        assert res.method == "manual_override"


class TestParseDocName:
    def test_parses_10k(self):
        parsed = edgar.parse_doc_name("AMD_2015_10K")
        assert parsed.form == "10-K"
        assert parsed.year == 2015
        assert parsed.quarter is None

    def test_parses_10q_with_quarter(self):
        parsed = edgar.parse_doc_name("3M_2023Q2_10Q")
        assert parsed.form == "10-Q"
        assert parsed.year == 2023
        assert parsed.quarter == 2

    def test_company_slug_can_contain_underscores(self):
        # Regression test: an early version anchored on "no underscores in company
        # name" and silently misclassified this real 10-K as out-of-scope.
        parsed = edgar.parse_doc_name("JOHNSON_JOHNSON_2022_10K")
        assert parsed.company_slug == "JOHNSON_JOHNSON"
        assert parsed.form == "10-K"

    def test_returns_none_for_out_of_scope_forms(self):
        assert edgar.parse_doc_name("AMCOR_2022_8K_dated-2022-07-01") is None
        assert edgar.parse_doc_name("PEPSICO_2023Q1_EARNINGS") is None


class TestFindFiling:
    def test_10k_matches_by_report_year(self):
        filings = edgar.CompanyFilings(
            fiscal_year_end="1231",
            entries=[
                {
                    "form": "10-K",
                    "reportDate": "2018-12-31",
                    "filingDate": "2019-02-07",
                    "accessionNumber": "acc-2018",
                    "primaryDocument": "a.htm",
                },
                {
                    "form": "10-K",
                    "reportDate": "2019-12-31",
                    "filingDate": "2020-02-07",
                    "accessionNumber": "acc-2019",
                    "primaryDocument": "b.htm",
                },
            ],
        )
        parsed = edgar.parse_doc_name("X_2018_10K")
        result = edgar.find_filing(parsed, "1", filings)
        assert result.accession_number == "acc-2018"

    def test_10q_uses_actual_fiscal_year_end_not_calendar_quarters(self):
        # Amcor's real fiscal year ends June 30, so its "FY2023 Q2" 10-Q has a
        # reportDate in calendar December 2022, not a calendar Q2 month. Assuming
        # calendar-aligned quarters here previously caused this exact filing to go
        # unresolved.
        filings = edgar.CompanyFilings(
            fiscal_year_end="0630",
            entries=[
                {
                    "form": "10-Q",
                    "reportDate": "2022-09-30",
                    "filingDate": "2022-11-01",
                    "accessionNumber": "acc-q1",
                    "primaryDocument": "q1.htm",
                },
                {
                    "form": "10-Q",
                    "reportDate": "2022-12-31",
                    "filingDate": "2023-02-01",
                    "accessionNumber": "acc-q2",
                    "primaryDocument": "q2.htm",
                },
                {
                    "form": "10-Q",
                    "reportDate": "2023-03-31",
                    "filingDate": "2023-05-01",
                    "accessionNumber": "acc-q3",
                    "primaryDocument": "q3.htm",
                },
            ],
        )
        parsed = edgar.parse_doc_name("AMCOR_2023Q2_10Q")
        result = edgar.find_filing(parsed, "1", filings)
        assert result.accession_number == "acc-q2"

    @staticmethod
    def _filings(fiscal_year_end: str, form: str, report_dates: list[str]):
        return edgar.CompanyFilings(
            fiscal_year_end=fiscal_year_end,
            entries=[
                {
                    "form": form,
                    "reportDate": d,
                    "filingDate": d,  # only ordering matters
                    "accessionNumber": f"acc-{d}",
                    "primaryDocument": f"{d}.htm",
                }
                for d in report_dates
            ],
        )

    def test_10k_early_january_year_end_takes_previous_years_label(self):
        # Johnson & Johnson's fiscal 2022 ended January 1, 2023; its fiscal 2021 ended
        # January 2, 2022. Matching on the reportDate's calendar year once picked the
        # fiscal 2021 10-K for "JOHNSON_JOHNSON_2022_10K".
        filings = self._filings("0101", "10-K", ["2022-01-02", "2023-01-01"])
        result = edgar.find_filing(edgar.parse_doc_name("JOHNSON_JOHNSON_2022_10K"), "1", filings)
        assert result.report_date == "2023-01-01"

    def test_10k_late_january_year_end_keeps_its_own_label(self):
        # Walmart / Best Buy name the fiscal year by its (late Jan / early Feb) end date.
        filings = self._filings("0131", "10-K", ["2018-01-31", "2019-01-31"])
        result = edgar.find_filing(edgar.parse_doc_name("WALMART_2019_10K"), "1", filings)
        assert result.report_date == "2019-01-31"
        filings = self._filings("0130", "10-K", ["2018-02-03", "2019-02-02"])
        result = edgar.find_filing(edgar.parse_doc_name("BESTBUY_2019_10K"), "1", filings)
        assert result.report_date == "2019-02-02"

    def test_10q_early_january_year_end_uses_calendar_quarters(self):
        filings = self._filings("0101", "10-Q", ["2022-07-03", "2023-04-02", "2023-07-02"])
        result = edgar.find_filing(edgar.parse_doc_name("JOHNSON_JOHNSON_2023Q2_10Q"), "1", filings)
        assert result.report_date == "2023-07-02"

    def test_10q_late_january_year_end_best_buy(self):
        # Best Buy fiscal 2024 Q2 is the quarter ended July 29, 2023. The corpus once held
        # the quarter ended May 4, 2024 under this name (a stale name-keyed cache copy).
        filings = self._filings("0130", "10-Q", ["2023-07-29", "2023-10-28", "2024-05-04"])
        result = edgar.find_filing(edgar.parse_doc_name("BESTBUY_2024Q2_10Q"), "1", filings)
        assert result.report_date == "2023-07-29"

    def test_returns_none_when_no_candidate_matches(self):
        filings = edgar.CompanyFilings(fiscal_year_end="1231", entries=[])
        parsed = edgar.parse_doc_name("X_2018_10K")
        assert edgar.find_filing(parsed, "1", filings) is None


class TestFiscalYearLabel:
    def test_calendar_year_end(self):
        assert edgar.fiscal_year_label("2022-12-31") == 2022

    def test_early_january_end_belongs_to_previous_year(self):
        assert edgar.fiscal_year_label("2023-01-01") == 2022
        assert edgar.fiscal_year_label("2021-01-03") == 2020

    def test_late_january_and_february_ends_keep_their_year(self):
        assert edgar.fiscal_year_label("2019-01-31") == 2019
        assert edgar.fiscal_year_label("2023-01-28") == 2023
        assert edgar.fiscal_year_label("2019-02-02") == 2019

    def test_effective_year_end_month(self):
        assert edgar.effective_year_end_month("0103") == 12
        assert edgar.effective_year_end_month("0130") == 1
        assert edgar.effective_year_end_month("0630") == 6


class TestCoverPeriodEnd:
    def test_annual_report_cover(self):
        text = "FORM 10-K\n\nFor the fiscal year ended\xa0January\xa01, 2023\n\nor"
        assert edgar.cover_period_end(text) == "2023-01-01"

    def test_quarterly_report_cover(self):
        text = "FORM 10-Q\nFor the quarterly period ended July 29 , 2023\nOR"
        assert edgar.cover_period_end(text) == "2023-07-29"

    def test_missing_cover_wording(self):
        assert edgar.cover_period_end("Annual report for 2022") is None

    def test_only_the_cover_region_is_searched(self):
        text = "x" * 6000 + "For the fiscal year ended December 31, 2019"
        assert edgar.cover_period_end(text) is None


def test_document_cache_path_is_keyed_by_filing_not_doc_name(tmp_path):
    ref = edgar.FilingRef(
        cik="0000764478",
        accession_number="0000764478-23-000042",
        form="10-Q",
        filing_date="2023-09-01",
        report_date="2023-07-29",
        primary_document="bby-20230729.htm",
    )
    path = edgar.document_cache_path(ref, tmp_path)
    assert path == tmp_path / "documents" / "0000764478_000076447823000042.htm"


def _index_row(seq: str, href: str, label: str, typ: str) -> str:
    return (
        f"<tr><td>{seq}</td><td>desc</td><td><a href='{href}'>{label}</a></td>"
        f"<td>{typ}</td><td>9</td></tr>"
    )


INDEX_HTML = (
    "<html><body><table class='tableFile'>"
    "<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>"
    + _index_row("1", "/ix?doc=/Archives/edgar/data/1/2/x-10k.htm", "x-10k.htm iXBRL", "10-K")
    + _index_row("2", "/Archives/edgar/data/1/2/ex1013.htm", "ex1013.htm", "EX-10.13")
    + _index_row("3", "/Archives/edgar/data/1/2/ex131.htm", "ex131.htm", "EX-13.1")
    + _index_row("4", "/Archives/edgar/data/1/2/ex13.htm", "ex13.htm", "EX-13")
    + _index_row("", "/Archives/edgar/data/1/2/0-1.txt", "0-1.txt", "")
    + "</table></body></html>"
).encode()


class TestFilingIndex:
    def test_parses_type_and_filename_from_link(self):
        rows = edgar.parse_filing_index(INDEX_HTML)
        assert rows[0] == ("10-K", "x-10k.htm")  # not "x-10k.htm iXBRL"
        assert ("EX-13.1", "ex131.htm") in rows
        assert all(typ for typ, _ in rows)  # the untyped submission-text row is skipped

    def test_selects_exhibit_13_and_numbered_variants_only(self):
        rows = edgar.parse_filing_index(INDEX_HTML)
        assert edgar.select_exhibits(rows, "EX-13") == ["ex131.htm", "ex13.htm"]


class TestGetRetries:
    @staticmethod
    def _stub(monkeypatch, codes):
        """urlopen stub raising HTTPError for each code in `codes`, then succeeding."""
        import io
        import urllib.error

        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if len(calls) <= len(codes):
                raise urllib.error.HTTPError(request.full_url, codes[len(calls) - 1], "x", {}, None)
            return io.BytesIO(b"ok")

        monkeypatch.setattr(edgar.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
        return calls

    def test_retries_throttling_then_succeeds(self, monkeypatch):
        calls = self._stub(monkeypatch, [503, 429])
        assert edgar._get("https://example.test/a", "ua") == b"ok"
        assert len(calls) == 3

    def test_does_not_retry_a_real_404(self, monkeypatch):
        import urllib.error

        calls = self._stub(monkeypatch, [404])
        with pytest.raises(urllib.error.HTTPError):
            edgar._get("https://example.test/a", "ua")
        assert len(calls) == 1

    def test_gives_up_after_max_attempts(self, monkeypatch):
        import urllib.error

        calls = self._stub(monkeypatch, [503] * 10)
        with pytest.raises(urllib.error.HTTPError):
            edgar._get("https://example.test/a", "ua")
        assert len(calls) == edgar.MAX_ATTEMPTS
