import json

import pytest

from wqb.local.fetch import (
    FetchError,
    RateLimit,
    parse_company_facts,
    parse_sec_tickers,
    parse_stooq_txt,
)

BULK = (
    "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
    "AAPL.US,D,20240102,000000,185.0,186.5,184.2,185.6,42000000,0\n"
    "AAPL.US,D,20240103,000000,184.0,185.0,183.0,183.5,38000000,0\n"
)
PER_SYMBOL = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,185.0,186.5,184.2,185.6,42000000\n"
)


# ------------------------------------------------------------------- stooq
def test_parses_the_bulk_archive_format():
    rows = parse_stooq_txt(BULK)
    assert len(rows) == 2
    assert rows[0]["Date"] == "2024-01-02"      # YYYYMMDD normalised to ISO
    assert rows[0]["Close"] == "185.6"


def test_parses_the_per_symbol_format():
    rows = parse_stooq_txt(PER_SYMBOL)
    assert rows[0]["Date"] == "2024-01-02"
    assert rows[0]["Volume"] == "42000000.0"


def test_both_formats_produce_the_same_columns():
    assert set(parse_stooq_txt(BULK)[0]) == set(parse_stooq_txt(PER_SYMBOL)[0])


def test_empty_and_headerless_input_is_tolerated():
    assert parse_stooq_txt("") == []
    assert parse_stooq_txt("garbage,data\n1,2") == []


def test_unparseable_rows_are_skipped():
    text = BULK + "AAPL.US,D,20240104,000000,,,,,,\n"
    assert len(parse_stooq_txt(text)) == 2


def test_missing_volume_column_defaults_to_zero():
    text = "<DATE>,<OPEN>,<HIGH>,<LOW>,<CLOSE>\n20240102,1,2,0.5,1.5\n"
    assert parse_stooq_txt(text)[0]["Volume"] == "0.0"


# ---------------------------------------------------------------- sec edgar
def test_parses_ticker_to_cik_map():
    payload = json.dumps({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
    }).encode()
    assert parse_sec_tickers(payload) == {"AAPL": 320193, "MSFT": 789019}


def test_ticker_map_rejects_unusable_payloads():
    with pytest.raises(FetchError, match="not valid JSON"):
        parse_sec_tickers(b"<html>")
    with pytest.raises(FetchError, match="no usable rows"):
        parse_sec_tickers(b"{}")


def test_company_facts_uses_the_filing_date_not_the_period_end():
    """The period end is weeks before the number is public; keying off it would
    leak information nobody could have traded on."""
    payload = json.dumps({"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"end": "2023-09-30", "filed": "2023-11-03", "val": 350.0},
        {"end": "2023-12-31", "filed": "2024-02-02", "val": 352.0},
    ]}}}}}).encode()
    series = parse_company_facts(payload, ["Assets"])["Assets"]
    assert [d for d, _ in series] == ["2023-11-03", "2024-02-02"]


def test_company_facts_sorts_and_dedupes_by_date():
    payload = json.dumps({"facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"filed": "2024-02-02", "val": 1.0},
        {"filed": "2023-11-03", "val": 2.0},
        {"filed": "2024-02-02", "val": 3.0},
    ]}}}}}).encode()
    series = parse_company_facts(payload, ["Assets"])["Assets"]
    assert [d for d, _ in series] == ["2023-11-03", "2024-02-02"]
    assert series[-1][1] == 3.0


def test_company_facts_ignores_requested_tags_that_are_absent():
    payload = json.dumps({"facts": {"us-gaap": {}}}).encode()
    assert parse_company_facts(payload, ["Assets"]) == {}


def test_company_facts_rejects_non_json():
    with pytest.raises(FetchError, match="not valid JSON"):
        parse_company_facts(b"not json")


# ---------------------------------------------------------------- rate limit
def test_rate_limit_spaces_requests(monkeypatch):
    """SEC asks for no more than 10 requests a second."""
    slept = []
    monkeypatch.setattr("wqb.local.fetch.time.sleep", slept.append)
    clock = iter([0.0, 0.0, 0.05, 0.05])
    monkeypatch.setattr("wqb.local.fetch.time.monotonic", lambda: next(clock))

    limiter = RateLimit(interval=0.15)
    limiter.wait()
    limiter.wait()
    assert slept and slept[0] > 0
