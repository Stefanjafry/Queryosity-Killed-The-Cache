"""Tests for the pure truncation classifier."""

from __future__ import annotations

from src.bayesopt.check_truncation import classify_counts

TEST_CACHES = [102400, 262144, 524288]


def test_clean_when_no_cap_pileup() -> None:
    # genuinely small, naturally varying (JOB-like)
    counts = [32352, 31431, 31126, 30000, 25000, 10000, 500, 72]
    res = classify_counts(counts, TEST_CACHES)
    assert res.verdict == "CLEAN"
    assert res.effective_cap is None


def test_corrupt_in_range_at_2gb_cap() -> None:
    # cluster clamped just under 262144, at/below the largest test cache
    counts = [262092, 262092, 262091, 262089, 261985, 200000, 50000]
    res = classify_counts(counts, TEST_CACHES)
    assert res.verdict == "CORRUPT_IN_RANGE"
    assert res.effective_cap == 262144
    assert res.n_at_cap >= 2


def test_ok_above_range_at_6gb_cap() -> None:
    # cluster clamped just under 786432, ABOVE the largest test cache (524288)
    counts = [786400, 786380, 786000, 600000, 300000, 50000]
    res = classify_counts(counts, TEST_CACHES)
    assert res.verdict == "OK_ABOVE_RANGE"
    assert res.effective_cap == 786432


def test_single_large_query_is_not_truncation() -> None:
    # one big query, rest much smaller, near no cap -> not flagged as a cap
    counts = [400000, 120000, 90000, 50000, 10000]
    res = classify_counts(counts, TEST_CACHES)
    assert res.effective_cap is None
    assert res.verdict == "CLEAN"


def test_clamped_top_flag_distinguishes_identical_from_varying() -> None:
    identical = [262100, 262099, 262098, 262097, 262096]   # spread < 0.2%
    varying = [261782, 260387, 258446, 250000, 240000]      # spread > 0.2%
    assert classify_counts(identical, TEST_CACHES).clamped_top is True
    assert classify_counts(varying, TEST_CACHES).clamped_top is False
