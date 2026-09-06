"""Unit tests for src/ingestion/edgar.py's pure logic: name normalization, CIK
resolution tiers, doc_name parsing, and fiscal-period filing matching. Network calls
(`_get`, the EDGAR browse-prefix fallback, live submissions fetches) aren't exercised
here — those were validated against the real EDGAR API during development."""

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

    def test_returns_none_when_no_candidate_matches(self):
        filings = edgar.CompanyFilings(fiscal_year_end="1231", entries=[])
        parsed = edgar.parse_doc_name("X_2018_10K")
        assert edgar.find_filing(parsed, "1", filings) is None
