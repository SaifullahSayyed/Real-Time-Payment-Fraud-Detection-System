"""
tests/test_slice3_scorer.py
────────────────────────────
Slice 3 tests: scoring engine (deterministic rule-based scorer).

Coverage:
  ✓ Clean transaction scores low
  ✓ Cold-start adds to score
  ✓ Micro-amount adds to score
  ✓ Critical amount adds to score
  ✓ High hourly velocity adds to score
  ✓ Critical hourly velocity adds more than high
  ✓ Amount outlier (z-score > 3σ) adds to score
  ✓ New merchant adds to score for established user
  ✓ High-risk hour adds to score
  ✓ High-risk currency adds to score
  ✓ Foreign currency (non-INR) adds small amount
  ✓ Established user discount reduces score
  ✓ Multiple risk factors combine additively
  ✓ Score always in [0, 1]
  ✓ RiskLevel consistent with score (via _score_to_risk)
  ✓ Determinism: identical fv → identical result
  ✓ _squash is monotonically increasing
  ✓ _squash(0) > 0 and _squash(2) < 1
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from fraud_detector.features.extractor import (
    FeatureVector,
    AMOUNT_MICRO,
    AMOUNT_HIGH,
    AMOUNT_CRITICAL,
    HOURLY_TX_HIGH,
    HOURLY_TX_CRITICAL,
    ZSCORE_HIGH,
    ZSCORE_CRITICAL,
    MIN_SAMPLES_FOR_ZSCORE,
    HIGH_RISK_CURRENCIES,
)
from fraud_detector.models.transaction import RiskLevel, _score_to_risk
from fraud_detector.scoring.scorer import ScoringResult, _squash, score_transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_NaN = float("nan")


def _fv(**overrides) -> FeatureVector:
    """Build a minimal 'clean' FeatureVector with any field overridden.

    Defaults are deliberately set so the established-user discount does NOT
    apply (total_tx_count=5, known_merchant_count=2), keeping baseline raw
    weight at 0 and making individual rule effects clearly visible.
    """
    defaults = dict(
        amount_paise=50_000,
        is_micro_amount=False,
        is_high_amount=False,
        is_critical_amount=False,
        hourly_tx_count=2,
        hourly_amount_paise=100_000,
        daily_tx_count=4,
        daily_amount_paise=200_000,
        is_high_hourly_velocity=False,
        is_critical_hourly_velocity=False,
        is_high_daily_tx=False,
        is_high_daily_amount=False,
        zscore_amount=_NaN,
        is_amount_outlier=False,
        is_amount_extreme_outlier=False,
        is_new_merchant=False,
        known_merchant_count=2,    # < 5 → no established-user discount
        hour_of_day=12,
        day_of_week=1,
        is_high_risk_hour=False,
        is_cold_start=False,
        total_tx_count=5,          # < 20 → no established-user discount
        is_domestic=True,
        is_high_risk_currency=False,
    )
    defaults.update(overrides)
    return FeatureVector(**defaults)


def _clean_fv() -> FeatureVector:
    """A 'clean' transaction: established user, normal amount, normal velocity."""
    return _fv()


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


class TestBasicScoring:
    def test_clean_transaction_low_risk(self):
        result = score_transaction(_clean_fv())
        assert result.score < 0.30, f"Clean tx should be LOW risk, got {result.score:.4f}"
        assert result.risk_level == RiskLevel.LOW

    def test_score_always_in_range(self):
        for _ in range(10):
            result = score_transaction(_clean_fv())
            assert 0.0 <= result.score <= 1.0

    def test_risk_level_consistent_with_score(self):
        result = score_transaction(_clean_fv())
        assert result.risk_level == _score_to_risk(result.score)

    def test_flags_is_list(self):
        result = score_transaction(_clean_fv())
        assert isinstance(result.flags, list)

    def test_raw_weight_field_present(self):
        result = score_transaction(_clean_fv())
        assert isinstance(result.raw_weight, float)


# ---------------------------------------------------------------------------
# Individual rule effects
# ---------------------------------------------------------------------------


class TestRuleEffects:
    def _compare(self, baseline: FeatureVector, risky: FeatureVector) -> None:
        """Assert that risky scores higher than baseline."""
        base_result = score_transaction(baseline)
        risky_result = score_transaction(risky)
        assert risky_result.score > base_result.score, (
            f"Risky ({risky_result.score:.4f}) should exceed baseline ({base_result.score:.4f})"
        )

    def test_cold_start_increases_score(self):
        self._compare(_fv(is_cold_start=False), _fv(is_cold_start=True, total_tx_count=0))

    def test_micro_amount_increases_score(self):
        self._compare(_fv(), _fv(is_micro_amount=True))

    def test_critical_amount_increases_score(self):
        self._compare(_fv(), _fv(is_critical_amount=True))

    def test_high_amount_increases_score(self):
        self._compare(_fv(), _fv(is_high_amount=True))

    def test_critical_amount_worse_than_high(self):
        high_result = score_transaction(_fv(is_high_amount=True))
        crit_result = score_transaction(_fv(is_critical_amount=True))
        assert crit_result.score > high_result.score

    def test_high_velocity_increases_score(self):
        self._compare(_fv(), _fv(is_high_hourly_velocity=True))

    def test_critical_velocity_worse_than_high(self):
        high = score_transaction(_fv(is_high_hourly_velocity=True))
        crit = score_transaction(_fv(is_critical_hourly_velocity=True, is_high_hourly_velocity=True))
        assert crit.score > high.score

    def test_amount_outlier_increases_score(self):
        self._compare(_fv(), _fv(is_amount_outlier=True, zscore_amount=4.0))

    def test_amount_extreme_outlier_worse_than_outlier(self):
        outlier = score_transaction(_fv(is_amount_outlier=True, zscore_amount=4.0))
        extreme = score_transaction(_fv(is_amount_outlier=True, is_amount_extreme_outlier=True, zscore_amount=6.0))
        assert extreme.score > outlier.score

    def test_new_merchant_increases_score_for_established_user(self):
        self._compare(
            _fv(is_new_merchant=False, known_merchant_count=5),
            _fv(is_new_merchant=True, known_merchant_count=5),
        )

    def test_high_risk_hour_increases_score(self):
        self._compare(_fv(is_high_risk_hour=False), _fv(is_high_risk_hour=True))

    def test_high_risk_currency_increases_score(self):
        self._compare(_fv(), _fv(is_high_risk_currency=True))

    def test_foreign_currency_increases_score(self):
        self._compare(_fv(is_domestic=True), _fv(is_domestic=False))

    def test_established_user_discount(self):
        """25 txns, 10 merchants → discount should lower score vs. new user."""
        established = _fv(total_tx_count=25, known_merchant_count=10)
        newer = _fv(total_tx_count=5, known_merchant_count=2)
        est_result = score_transaction(established)
        new_result = score_transaction(newer)
        assert est_result.score <= new_result.score


# ---------------------------------------------------------------------------
# Multi-risk stacking
# ---------------------------------------------------------------------------


class TestMultiRiskStacking:
    def test_multiple_flags_score_higher_than_single(self):
        single = score_transaction(_fv(is_cold_start=True, total_tx_count=0))
        multi = score_transaction(_fv(
            is_cold_start=True,
            total_tx_count=0,
            is_micro_amount=True,
            is_high_risk_hour=True,
        ))
        assert multi.score > single.score

    def test_all_risk_factors_critical(self):
        """Pile on all risk factors — should reach CRITICAL."""
        result = score_transaction(_fv(
            is_cold_start=True,
            total_tx_count=0,
            is_micro_amount=True,
            is_critical_amount=True,
            is_critical_hourly_velocity=True,
            is_high_hourly_velocity=True,
            is_high_daily_tx=True,
            is_high_daily_amount=True,
            is_amount_extreme_outlier=True,
            is_amount_outlier=True,
            zscore_amount=7.0,
            is_high_risk_hour=True,
            is_high_risk_currency=True,
            is_domestic=False,
        ))
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.score >= 0.85


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestScorerDeterminism:
    def test_identical_fv_identical_result(self):
        fv = _fv(is_cold_start=True, total_tx_count=0)
        r1 = score_transaction(fv)
        r2 = score_transaction(fv)
        assert r1.score == r2.score
        assert r1.flags == r2.flags
        assert r1.risk_level == r2.risk_level

    def test_score_independent_of_call_order(self):
        fv_a = _fv(is_cold_start=True)
        fv_b = _fv(is_high_risk_hour=True)
        # Score A then B
        a1 = score_transaction(fv_a)
        b1 = score_transaction(fv_b)
        # Score B then A (reversed)
        b2 = score_transaction(fv_b)
        a2 = score_transaction(fv_a)
        assert a1.score == a2.score
        assert b1.score == b2.score


# ---------------------------------------------------------------------------
# _squash properties
# ---------------------------------------------------------------------------


class TestSquash:
    def test_output_in_range(self):
        for raw in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            s = _squash(raw)
            assert 0.0 < s < 1.0, f"_squash({raw}) = {s} out of (0,1)"

    def test_monotonically_increasing(self):
        vals = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        squashed = [_squash(v) for v in vals]
        for i in range(len(squashed) - 1):
            assert squashed[i] < squashed[i + 1], (
                f"Not monotone at index {i}: {squashed[i]} >= {squashed[i+1]}"
            )

    def test_squash_zero_is_positive(self):
        assert _squash(0.0) > 0.0

    def test_squash_two_is_below_one(self):
        assert _squash(2.0) < 1.0
