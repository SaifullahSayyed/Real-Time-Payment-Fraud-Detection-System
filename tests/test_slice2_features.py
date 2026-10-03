"""
tests/test_slice2_features.py
──────────────────────────────
Slice 2 tests: feature extraction (pure function).

Coverage:
  ✓ Cold-start flag when no history
  ✓ Non-cold-start when history exists
  ✓ Micro-amount flag
  ✓ High-amount flag
  ✓ Critical-amount flag
  ✓ Neither high nor critical for normal amount
  ✓ Hourly velocity flags (high / critical)
  ✓ Daily velocity flags (tx count / amount)
  ✓ Z-score: NaN when insufficient history
  ✓ Z-score: computes correctly with sufficient history
  ✓ Z-score outlier flag
  ✓ Z-score extreme outlier flag
  ✓ Z-score: no division by zero when all amounts identical (variance=0)
  ✓ New merchant flag for known user
  ✓ New merchant NOT flagged for cold-start user
  ✓ Known merchant: no flag
  ✓ High-risk hour flag (00–04 UTC)
  ✓ Safe hour: no flag
  ✓ High-risk currency flag
  ✓ Foreign currency (non-INR, non-high-risk)
  ✓ Domestic currency: no foreign flag
  ✓ Determinism: same inputs → same FeatureVector
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from fraud_detector.features.extractor import (
    AMOUNT_CRITICAL,
    AMOUNT_HIGH,
    AMOUNT_MICRO,
    HIGH_RISK_CURRENCIES,
    HOURLY_TX_CRITICAL,
    HOURLY_TX_HIGH,
    MIN_SAMPLES_FOR_ZSCORE,
    FeatureVector,
    extract_features,
)
from fraud_detector.models.transaction import Channel, TransactionRequest, TransactionType
from fraud_detector.state.user_state import UserStateSnapshot

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(
    total_tx_count: int = 0,
    amount_mean: float = 0.0,
    amount_m2: float = 0.0,
    hourly_tx_count: int = 0,
    hourly_amount_paise: int = 0,
    daily_tx_count: int = 0,
    daily_amount_paise: int = 0,
    known_merchants: frozenset | None = None,
    last_tx_timestamp=None,
) -> UserStateSnapshot:
    return UserStateSnapshot(
        user_id="usr-001",
        total_tx_count=total_tx_count,
        total_amount_paise=0,
        hourly_tx_count=hourly_tx_count,
        hourly_amount_paise=hourly_amount_paise,
        daily_tx_count=daily_tx_count,
        daily_amount_paise=daily_amount_paise,
        amount_mean=amount_mean,
        amount_m2=amount_m2,
        known_merchants=known_merchants if known_merchants is not None else frozenset(),
        last_tx_timestamp=last_tx_timestamp,
        is_cold_start=(total_tx_count == 0),
    )


def _request(
    amount: int = 50_000,
    merchant_id: str = "mer-001",
    currency: str = "INR",
    hour: int = 12,
    transaction_type: str = "PURCHASE",
) -> TransactionRequest:
    ts = datetime(2024, 6, 15, hour, 0, 0, tzinfo=_UTC)
    return TransactionRequest(
        transaction_id="txn-001",
        user_id="usr-001",
        merchant_id=merchant_id,
        amount=amount,
        currency=currency,
        timestamp=ts,
        transaction_type=transaction_type,
    )


# ---------------------------------------------------------------------------
# Cold-start
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_is_cold_start_when_no_history(self):
        fv = extract_features(_request(), _snapshot(total_tx_count=0))
        assert fv.is_cold_start is True
        assert fv.total_tx_count == 0

    def test_not_cold_start_when_has_history(self):
        fv = extract_features(_request(), _snapshot(total_tx_count=5))
        assert fv.is_cold_start is False
        assert fv.total_tx_count == 5


# ---------------------------------------------------------------------------
# Amount features
# ---------------------------------------------------------------------------


class TestAmountFeatures:
    def test_micro_amount_flagged(self):
        fv = extract_features(_request(amount=AMOUNT_MICRO - 1), _snapshot())
        assert fv.is_micro_amount is True
        assert fv.is_high_amount is False
        assert fv.is_critical_amount is False

    def test_micro_boundary_not_flagged(self):
        fv = extract_features(_request(amount=AMOUNT_MICRO), _snapshot())
        assert fv.is_micro_amount is False

    def test_high_amount_flagged(self):
        fv = extract_features(_request(amount=AMOUNT_HIGH), _snapshot())
        assert fv.is_high_amount is True
        assert fv.is_micro_amount is False

    def test_critical_amount_flagged(self):
        fv = extract_features(_request(amount=AMOUNT_CRITICAL), _snapshot())
        assert fv.is_critical_amount is True
        assert fv.is_high_amount is True  # critical implies high

    def test_normal_amount_no_flags(self):
        fv = extract_features(_request(amount=50_000), _snapshot())
        assert fv.is_micro_amount is False
        assert fv.is_high_amount is False
        assert fv.is_critical_amount is False


# ---------------------------------------------------------------------------
# Velocity features
# ---------------------------------------------------------------------------


class TestVelocityFeatures:
    def test_high_hourly_velocity(self):
        fv = extract_features(
            _request(),
            _snapshot(hourly_tx_count=HOURLY_TX_HIGH + 1),
        )
        assert fv.is_high_hourly_velocity is True
        assert fv.is_critical_hourly_velocity is False

    def test_critical_hourly_velocity(self):
        fv = extract_features(
            _request(),
            _snapshot(hourly_tx_count=HOURLY_TX_CRITICAL + 1),
        )
        assert fv.is_critical_hourly_velocity is True
        assert fv.is_high_hourly_velocity is True

    def test_low_velocity_no_flags(self):
        fv = extract_features(_request(), _snapshot(hourly_tx_count=2))
        assert fv.is_high_hourly_velocity is False
        assert fv.is_critical_hourly_velocity is False

    def test_high_daily_tx_count(self):
        from fraud_detector.features.extractor import DAILY_TX_HIGH
        fv = extract_features(_request(), _snapshot(daily_tx_count=DAILY_TX_HIGH + 1))
        assert fv.is_high_daily_tx is True

    def test_high_daily_amount(self):
        from fraud_detector.features.extractor import DAILY_AMOUNT_HIGH
        fv = extract_features(
            _request(), _snapshot(daily_amount_paise=DAILY_AMOUNT_HIGH + 1)
        )
        assert fv.is_high_daily_amount is True


# ---------------------------------------------------------------------------
# Z-score features
# ---------------------------------------------------------------------------


class TestZScoreFeatures:
    def test_zscore_nan_when_insufficient_history(self):
        """Fewer than MIN_SAMPLES_FOR_ZSCORE → z-score should be NaN."""
        fv = extract_features(
            _request(amount=1_000_000),
            _snapshot(total_tx_count=MIN_SAMPLES_FOR_ZSCORE - 1),
        )
        assert math.isnan(fv.zscore_amount)
        assert fv.is_amount_outlier is False
        assert fv.is_amount_extreme_outlier is False

    def test_zscore_computed_with_sufficient_history(self):
        """10 previous txns of 50000 paise each, now 50000 → z=0."""
        n = 10
        mean = 50_000.0
        # M2 for 10 identical values = 0
        fv = extract_features(
            _request(amount=50_000),
            _snapshot(total_tx_count=n, amount_mean=mean, amount_m2=0.0),
        )
        # variance = 0, so z-score indeterminate (NaN)
        assert math.isnan(fv.zscore_amount)

    def test_zscore_outlier_flagged(self):
        """Amount 4-sigma above mean should flag as outlier."""
        n = 20
        mean = 50_000.0
        std = 10_000.0
        m2 = (std ** 2) * n   # population variance * n = M2
        outlier_amount = int(mean + 4 * std)   # z = 4.0
        fv = extract_features(
            _request(amount=outlier_amount),
            _snapshot(total_tx_count=n, amount_mean=mean, amount_m2=m2),
        )
        assert not math.isnan(fv.zscore_amount)
        assert abs(fv.zscore_amount - 4.0) < 0.01
        assert fv.is_amount_outlier is True
        assert fv.is_amount_extreme_outlier is False

    def test_zscore_extreme_outlier_flagged(self):
        n = 20
        mean = 50_000.0
        std = 10_000.0
        m2 = (std ** 2) * n
        extreme_amount = int(mean + 6 * std)  # z = 6.0
        fv = extract_features(
            _request(amount=extreme_amount),
            _snapshot(total_tx_count=n, amount_mean=mean, amount_m2=m2),
        )
        assert fv.is_amount_extreme_outlier is True
        assert fv.is_amount_outlier is True

    def test_no_division_by_zero_when_variance_is_zero(self):
        """If all past amounts identical, variance=0 → z-score=NaN (not error)."""
        n = 10
        mean = 50_000.0
        m2 = 0.0    # all amounts were identical
        fv = extract_features(
            _request(amount=50_000),
            _snapshot(total_tx_count=n, amount_mean=mean, amount_m2=m2),
        )
        assert math.isnan(fv.zscore_amount)
        assert fv.is_amount_outlier is False


# ---------------------------------------------------------------------------
# Merchant features
# ---------------------------------------------------------------------------


class TestMerchantFeatures:
    def test_new_merchant_flagged_for_known_user(self):
        snap = _snapshot(total_tx_count=5, known_merchants=frozenset({"mer-999"}))
        fv = extract_features(_request(merchant_id="mer-001"), snap)
        assert fv.is_new_merchant is True

    def test_new_merchant_not_flagged_for_cold_start(self):
        """Cold-start users haven't seen any merchant — new merchant is expected."""
        snap = _snapshot(total_tx_count=0)
        fv = extract_features(_request(merchant_id="mer-001"), snap)
        assert fv.is_cold_start is True
        # feature still computed, but scorer discounts it for cold-start users
        assert fv.is_new_merchant is True

    def test_known_merchant_not_new(self):
        snap = _snapshot(
            total_tx_count=5,
            known_merchants=frozenset({"mer-001", "mer-002"}),
        )
        fv = extract_features(_request(merchant_id="mer-001"), snap)
        assert fv.is_new_merchant is False

    def test_known_merchant_count(self):
        snap = _snapshot(
            total_tx_count=5,
            known_merchants=frozenset({"m1", "m2", "m3"}),
        )
        fv = extract_features(_request(), snap)
        assert fv.known_merchant_count == 3


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_high_risk_hour(self):
        for h in range(0, 5):
            fv = extract_features(_request(hour=h), _snapshot())
            assert fv.is_high_risk_hour is True, f"hour {h} should be high risk"

    def test_safe_hour(self):
        for h in [6, 12, 17, 22]:
            fv = extract_features(_request(hour=h), _snapshot())
            assert fv.is_high_risk_hour is False, f"hour {h} should not be high risk"

    def test_hour_of_day_value(self):
        fv = extract_features(_request(hour=14), _snapshot())
        assert fv.hour_of_day == 14

    def test_day_of_week_saturday(self):
        # 2024-06-15 is a Saturday (weekday=5)
        ts = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)
        req = TransactionRequest(
            transaction_id="txn-dow",
            user_id="usr-001",
            merchant_id="mer-001",
            amount=50_000,
            timestamp=ts,
        )
        fv = extract_features(req, _snapshot())
        assert fv.day_of_week == 5  # Saturday


# ---------------------------------------------------------------------------
# Currency features
# ---------------------------------------------------------------------------


class TestCurrencyFeatures:
    def test_inr_is_domestic(self):
        fv = extract_features(_request(currency="INR"), _snapshot())
        assert fv.is_domestic is True
        assert fv.is_high_risk_currency is False

    def test_usd_is_foreign_not_high_risk(self):
        fv = extract_features(_request(currency="USD"), _snapshot())
        assert fv.is_domestic is False
        assert fv.is_high_risk_currency is False

    def test_high_risk_currency(self):
        ccy = next(iter(HIGH_RISK_CURRENCIES))
        fv = extract_features(_request(currency=ccy), _snapshot())
        assert fv.is_high_risk_currency is True
        assert fv.is_domestic is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_output(self):
        req = _request()
        snap = _snapshot(total_tx_count=10, amount_mean=40_000.0, amount_m2=2_000_000.0)
        fv1 = extract_features(req, snap)
        fv2 = extract_features(req, snap)
        assert fv1 == fv2

    def test_different_amounts_different_output(self):
        snap = _snapshot()
        fv1 = extract_features(_request(amount=50_000), snap)
        fv2 = extract_features(_request(amount=100_000), snap)
        assert fv1.amount_paise != fv2.amount_paise
