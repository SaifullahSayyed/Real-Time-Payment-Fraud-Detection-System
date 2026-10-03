"""
tests/test_slice1_models.py
───────────────────────────
Slice 1 tests: Pydantic input validation and output contracts.

Coverage targets:
  ✓ Happy-path: minimal valid transaction
  ✓ Happy-path: all optional fields populated
  ✓ amount = float   → rejected (strict int)
  ✓ amount = 0       → rejected (gt=0)
  ✓ amount < 0       → rejected
  ✓ amount = max+1   → rejected
  ✓ amount = max     → accepted
  ✓ Naive datetime   → rejected (not tz-aware)
  ✓ Non-UTC tz       → rejected (offset != 0)
  ✓ UTC+0 offset     → accepted
  ✓ Invalid currency → rejected (wrong format)
  ✓ Lowercase currency → rejected
  ✓ Unknown currency (valid format) → accepted (flagged downstream)
  ✓ transaction_id empty string → rejected
  ✓ transaction_id too long → rejected
  ✓ transaction_id bad chars → rejected
  ✓ Extra unknown field → rejected (extra="forbid")
  ✓ Missing required fields → rejected
  ✓ Invalid enum value for channel → rejected
  ✓ Invalid enum value for transaction_type → rejected
  ✓ FraudScore risk/score consistency enforced
  ✓ FraudScore is_duplicate field round-trips
  ✓ _score_to_risk boundary values
  ✓ utc_now() returns UTC-aware datetime
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from fraud_detector.models import (
    Channel,
    FraudScore,
    RiskLevel,
    TransactionRequest,
    TransactionType,
    _score_to_risk,
    utc_now,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_NOW_UTC = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)
_MAX_PAISE = 10_000_000_000


def _base() -> dict:
    """Minimal valid transaction payload."""
    return {
        "transaction_id": "txn-001",
        "user_id": "usr-001",
        "merchant_id": "mer-001",
        "amount": 50000,            # 500 INR in paise
        "timestamp": _NOW_UTC,
    }


def _make(**overrides) -> dict:
    d = _base()
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_minimal_valid_transaction(self):
        tx = TransactionRequest(**_base())
        assert tx.amount == 50000
        assert tx.currency == "INR"
        assert tx.channel == Channel.ONLINE
        assert tx.transaction_type == TransactionType.PURCHASE

    def test_all_optional_fields(self):
        tx = TransactionRequest(
            **_make(
                merchant_name="Flipkart",
                ip_address="192.168.1.1",
                device_id="dev-abc-123",
                channel="UPI",
                transaction_type="TRANSFER",
                currency="USD",
            )
        )
        assert tx.merchant_name == "Flipkart"
        assert tx.ip_address == "192.168.1.1"
        assert tx.channel == Channel.UPI

    def test_amount_at_max_boundary(self):
        tx = TransactionRequest(**_make(amount=_MAX_PAISE))
        assert tx.amount == _MAX_PAISE

    def test_amount_of_one_paise(self):
        tx = TransactionRequest(**_make(amount=1))
        assert tx.amount == 1

    def test_utc_plus_zero_zoneinfo(self):
        """datetime with UTC+00:00 from timedelta must be accepted."""
        ts = datetime(2024, 1, 1, tzinfo=timezone(timedelta(0)))
        tx = TransactionRequest(**_make(timestamp=ts))
        assert tx.timestamp.utcoffset().total_seconds() == 0

    def test_unknown_currency_code_is_accepted(self):
        """Unknown but syntactically valid currency codes pass validation."""
        tx = TransactionRequest(**_make(currency="XYZ"))
        assert tx.currency == "XYZ"

    def test_immutability(self):
        """Frozen model must raise on attribute assignment."""
        tx = TransactionRequest(**_base())
        with pytest.raises((TypeError, ValidationError)):
            tx.amount = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# amount validation
# ---------------------------------------------------------------------------


class TestAmountValidation:
    def test_float_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**_make(amount=500.0))
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("amount",) for e in errors)

    def test_zero_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount=0))

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount=-1))

    def test_very_negative_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount=-999_999_999))

    def test_above_max_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount=_MAX_PAISE + 1))

    def test_string_amount_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount="5000"))

    def test_none_amount_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(amount=None))


# ---------------------------------------------------------------------------
# timestamp validation
# ---------------------------------------------------------------------------


class TestTimestampValidation:
    def test_naive_datetime_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**_make(timestamp=datetime(2024, 1, 1, 12, 0, 0)))
        assert any(e["loc"] == ("timestamp",) for e in exc_info.value.errors())

    def test_non_utc_positive_offset_rejected(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=ist)))

    def test_non_utc_negative_offset_rejected(self):
        pst = timezone(timedelta(hours=-8))
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=pst)))

    def test_string_iso_utc_accepted(self):
        """Pydantic should parse an ISO-8601 UTC string."""
        tx = TransactionRequest(**_make(timestamp="2024-06-15T10:00:00Z"))
        assert tx.timestamp.tzinfo is not None

    def test_string_non_utc_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(timestamp="2024-06-15T10:00:00+05:30"))

    def test_none_timestamp_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(timestamp=None))


# ---------------------------------------------------------------------------
# currency validation
# ---------------------------------------------------------------------------


class TestCurrencyValidation:
    def test_two_letters_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency="IN"))

    def test_four_letters_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency="INRX"))

    def test_lowercase_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency="inr"))

    def test_digits_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency="1NR"))

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency=""))

    def test_spaces_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(currency="I R"))


# ---------------------------------------------------------------------------
# ID field validation
# ---------------------------------------------------------------------------


class TestIdValidation:
    def test_empty_transaction_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(transaction_id=""))

    def test_too_long_transaction_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(transaction_id="a" * 65))

    def test_max_length_id_accepted(self):
        tx = TransactionRequest(**_make(transaction_id="a" * 64))
        assert len(tx.transaction_id) == 64

    def test_bad_chars_in_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(transaction_id="txn@001"))

    def test_spaces_in_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(transaction_id="txn 001"))

    def test_empty_user_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(user_id=""))

    def test_empty_merchant_id_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(merchant_id=""))


# ---------------------------------------------------------------------------
# Extra / missing field handling
# ---------------------------------------------------------------------------


class TestSchemaEnforcement:
    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(secret_flag=True))

    def test_missing_transaction_id_rejected(self):
        d = _base()
        del d["transaction_id"]
        with pytest.raises(ValidationError):
            TransactionRequest(**d)

    def test_missing_user_id_rejected(self):
        d = _base()
        del d["user_id"]
        with pytest.raises(ValidationError):
            TransactionRequest(**d)

    def test_missing_amount_rejected(self):
        d = _base()
        del d["amount"]
        with pytest.raises(ValidationError):
            TransactionRequest(**d)

    def test_missing_timestamp_rejected(self):
        d = _base()
        del d["timestamp"]
        with pytest.raises(ValidationError):
            TransactionRequest(**d)


# ---------------------------------------------------------------------------
# Enum validation
# ---------------------------------------------------------------------------


class TestEnumValidation:
    def test_invalid_channel_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(channel="CARRIER_PIGEON"))

    def test_invalid_transaction_type_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(**_make(transaction_type="BRIBE"))

    def test_valid_channels_accepted(self):
        for ch in Channel:
            tx = TransactionRequest(**_make(channel=ch.value))
            assert tx.channel == ch

    def test_valid_transaction_types_accepted(self):
        for tt in TransactionType:
            tx = TransactionRequest(**_make(transaction_type=tt.value))
            assert tx.transaction_type == tt


# ---------------------------------------------------------------------------
# FraudScore validation
# ---------------------------------------------------------------------------


class TestFraudScore:
    def _valid_score(self, score: float, risk: RiskLevel) -> FraudScore:
        return FraudScore(
            transaction_id="txn-001",
            user_id="usr-001",
            score=score,
            risk_level=risk,
            flags=["test"],
            scored_at=_NOW_UTC,
        )

    def test_low_score_low_risk(self):
        fs = self._valid_score(0.1, RiskLevel.LOW)
        assert fs.risk_level == RiskLevel.LOW

    def test_medium_score_medium_risk(self):
        fs = self._valid_score(0.45, RiskLevel.MEDIUM)
        assert fs.risk_level == RiskLevel.MEDIUM

    def test_high_score_high_risk(self):
        fs = self._valid_score(0.7, RiskLevel.HIGH)
        assert fs.risk_level == RiskLevel.HIGH

    def test_critical_score_critical_risk(self):
        fs = self._valid_score(0.9, RiskLevel.CRITICAL)
        assert fs.risk_level == RiskLevel.CRITICAL

    def test_inconsistent_risk_raises(self):
        with pytest.raises(ValidationError):
            self._valid_score(0.1, RiskLevel.CRITICAL)  # 0.1 is LOW, not CRITICAL

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_score(-0.01, RiskLevel.LOW)

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._valid_score(1.001, RiskLevel.CRITICAL)

    def test_is_duplicate_defaults_false(self):
        fs = self._valid_score(0.1, RiskLevel.LOW)
        assert fs.is_duplicate is False

    def test_is_duplicate_true_round_trips(self):
        fs = FraudScore(
            transaction_id="txn-dup",
            user_id="usr-001",
            score=0.2,
            risk_level=RiskLevel.LOW,
            is_duplicate=True,
            scored_at=_NOW_UTC,
        )
        assert fs.is_duplicate is True


# ---------------------------------------------------------------------------
# _score_to_risk boundary values
# ---------------------------------------------------------------------------


class TestScoreToRisk:
    def test_boundary_0_00(self):
        assert _score_to_risk(0.0) == RiskLevel.LOW

    def test_boundary_0_299(self):
        assert _score_to_risk(0.299) == RiskLevel.LOW

    def test_boundary_0_30(self):
        assert _score_to_risk(0.30) == RiskLevel.MEDIUM

    def test_boundary_0_599(self):
        assert _score_to_risk(0.599) == RiskLevel.MEDIUM

    def test_boundary_0_60(self):
        assert _score_to_risk(0.60) == RiskLevel.HIGH

    def test_boundary_0_849(self):
        assert _score_to_risk(0.849) == RiskLevel.HIGH

    def test_boundary_0_85(self):
        assert _score_to_risk(0.85) == RiskLevel.CRITICAL

    def test_boundary_1_00(self):
        assert _score_to_risk(1.0) == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# utc_now helper
# ---------------------------------------------------------------------------


class TestUtcNow:
    def test_returns_aware_datetime(self):
        dt = utc_now()
        assert dt.tzinfo is not None

    def test_returns_utc(self):
        dt = utc_now()
        assert dt.utcoffset().total_seconds() == 0

    def test_is_recent(self):
        import time
        before = datetime.now(tz=_UTC)
        time.sleep(0.001)
        result = utc_now()
        time.sleep(0.001)
        after = datetime.now(tz=_UTC)
        assert before <= result <= after
