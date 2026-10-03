"""
fraud_detector/models/transaction.py
─────────────────────────────────────
Pydantic v2 input/output contracts for the fraud-detection system.

Design invariants enforced here (so nothing else has to):
  • amount is ALWAYS a strict int (paise / minor units) and must be > 0.
  • timestamp must be timezone-aware UTC; naive datetimes are rejected.
  • transaction_id, user_id, merchant_id are non-empty strings capped at 64 chars.
  • currency is an ISO-4217 three-letter uppercase code (unknown codes pass through
    but are flagged downstream as HIGH risk).
  • channel / transaction_type use closed enums so typos surface at the boundary.
  • ip_address and device_id are optional; if present they are length-capped strings.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AMOUNT_MAX_PAISE = 10_000_000_000  # 10 Cr – beyond this amount is suspicious
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_MIN_TIMESTAMP = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_MAX_TIMESTAMP = datetime(2035, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TransactionType(str, Enum):
    PURCHASE = "PURCHASE"
    REFUND = "REFUND"
    TRANSFER = "TRANSFER"
    WITHDRAWAL = "WITHDRAWAL"
    TOPUP = "TOPUP"


class Channel(str, Enum):
    ONLINE = "ONLINE"
    POS = "POS"
    ATM = "ATM"
    UPI = "UPI"
    NEFT = "NEFT"
    IMPS = "IMPS"
    RTGS = "RTGS"
    APP = "APP"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Annotated helper types
# ---------------------------------------------------------------------------

# Strict int that must fit in int64; floats are refused.
StrictInt64 = Annotated[int, Field(strict=True, ge=-(2**63), le=2**63 - 1)]

# Non-empty string, max 64 chars, alphanumeric + hyphens/underscores only.
EntityId = Annotated[str, Field(min_length=1, max_length=64)]

# Optional free-text fields (names, notes) – no alphanumeric restriction.
ShortText = Annotated[str, Field(min_length=1, max_length=256)]

# ---------------------------------------------------------------------------
# Core domain model: TransactionRequest
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    """Input to the fraud scorer.  Every field is validated strictly.

    Pydantic forbids extra fields so typo-keys surface as errors rather than
    silently disappearing.
    """

    model_config = ConfigDict(
        frozen=True,          # immutable after construction
        extra="forbid",       # reject unknown keys
        populate_by_name=True,
    )

    transaction_id: EntityId = Field(
        description="Globally unique identifier for this transaction."
    )
    user_id: EntityId = Field(description="Unique identifier for the payer.")
    merchant_id: EntityId = Field(description="Unique identifier for the merchant.")

    # Money: strict int (paise).  Floats are refused by strict=True.
    amount: Annotated[int, Field(strict=True, gt=0, le=_AMOUNT_MAX_PAISE)] = Field(
        description=(
            "Transaction amount in paise (integer minor units). "
            "Must be > 0 and ≤ 10,00,00,00,000."
        )
    )

    currency: Annotated[str, Field(min_length=3, max_length=3)] = Field(
        default="INR",
        description="ISO-4217 three-letter currency code (upper-case).",
    )

    timestamp: AwareDatetime = Field(
        description="UTC timestamp of the transaction (must carry tz info)."
    )

    transaction_type: TransactionType = Field(
        default=TransactionType.PURCHASE,
        description="Nature of the transaction.",
    )

    channel: Channel = Field(
        default=Channel.ONLINE,
        description="Channel through which the transaction was made.",
    )

    merchant_name: Optional[ShortText] = Field(
        default=None,
        description="Human-readable merchant name (optional).",
    )

    ip_address: Optional[Annotated[str, Field(max_length=45)]] = Field(
        default=None,
        description="IPv4 or IPv6 address of the client (optional).",
    )

    device_id: Optional[Annotated[str, Field(max_length=128)]] = Field(
        default=None,
        description="Opaque device fingerprint (optional).",
    )

    latitude: Optional[Annotated[float, Field(ge=-90.0, le=90.0)]] = Field(
        default=None,
        description="GPS latitude in [-90.0, +90.0] (optional).",
    )

    longitude: Optional[Annotated[float, Field(ge=-180.0, le=180.0)]] = Field(
        default=None,
        description="GPS longitude in [-180.0, +180.0] (optional).",
    )

    # -----------------------------------------------------------------------
    # Field-level validators
    # -----------------------------------------------------------------------

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, v: str) -> str:
        if not _CURRENCY_RE.match(v):
            raise ValueError(
                f"Currency must be exactly 3 uppercase ASCII letters, got {v!r}"
            )
        return v

    @field_validator("transaction_id", "user_id", "merchant_id")
    @classmethod
    def _validate_id_chars(cls, v: str) -> str:
        if not _ID_RE.match(v):
            raise ValueError(
                f"ID must be 1-64 alphanumeric/hyphen/underscore characters, got {v!r}"
            )
        return v

    # -----------------------------------------------------------------------
    # Model-level validator
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_timestamp_is_utc(self) -> TransactionRequest:
        """Ensure the timestamp is in UTC and within permitted operational range.

        AwareDatetime guarantees tzinfo is present; we additionally enforce UTC
        so the entire system operates on a single timezone.
        """
        ts = self.timestamp
        # Normalise to UTC offset object for comparison
        utc_offset = ts.utcoffset()
        if utc_offset is None or utc_offset.total_seconds() != 0:
            raise ValueError(
                f"timestamp must be UTC (offset 0), got offset {utc_offset}"
            )
        if ts < _MIN_TIMESTAMP or ts > _MAX_TIMESTAMP:
            raise ValueError(
                f"timestamp {ts.isoformat()} outside permitted operational range "
                f"[{_MIN_TIMESTAMP.isoformat()}, {_MAX_TIMESTAMP.isoformat()}]"
            )
        return self

    @model_validator(mode="after")
    def _validate_geo_coordinates(self) -> TransactionRequest:
        """Latitude and longitude must either both be provided or both be None."""
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError(
                "latitude and longitude must either both be provided or both be omitted."
            )
        return self


# ---------------------------------------------------------------------------
# Output model: FraudScore
# ---------------------------------------------------------------------------


class FraudScore(BaseModel):
    """Response from the fraud scorer.  Read-only once created."""

    model_config = ConfigDict(frozen=True)

    transaction_id: EntityId
    user_id: EntityId
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        description="Fraud probability in [0, 1]."
    )
    risk_level: RiskLevel
    flags: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons that contributed to the score.",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True if this transaction_id was seen before (cached result).",
    )
    scored_at: AwareDatetime = Field(
        description="UTC wall-clock time when scoring completed.",
    )

    @model_validator(mode="after")
    def _risk_matches_score(self) -> FraudScore:
        """Risk level must be consistent with the numeric score."""
        s = self.score
        expected = _score_to_risk(s)
        if self.risk_level != expected:
            raise ValueError(
                f"risk_level {self.risk_level} inconsistent with score {s:.4f} "
                f"(expected {expected})"
            )
        return self


def _score_to_risk(score: float) -> RiskLevel:
    """Canonical mapping from numeric score to risk level.

    Thresholds (inclusive upper bound):
      [0.00, 0.30) → LOW
      [0.30, 0.60) → MEDIUM
      [0.60, 0.85) → HIGH
      [0.85, 1.00] → CRITICAL
    """
    if score < 0.30:
        return RiskLevel.LOW
    if score < 0.60:
        return RiskLevel.MEDIUM
    if score < 0.85:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=timezone.utc)
