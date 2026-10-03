"""
fraud_detector/features/extractor.py
──────────────────────────────────────
Stateless feature extraction: given a TransactionRequest + UserStateSnapshot,
produce a FeatureVector used by the scorer.

All features are deterministic: same inputs → same features, always.
No randomness, no I/O, no global mutable state touched here.

Features:
  amount_*        – absolute amount analysis
  velocity_*      – transaction rate in time windows
  zscore_amount   – how many σ this amount is from the user's mean
  merchant_*      – is this a new merchant? how many unique merchants seen?
  time_*          – hour-of-day, day-of-week risk
  cold_start      – boolean flag for new users
  currency_*      – is currency INR (domestic)?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from fraud_detector.models.transaction import TransactionRequest
from fraud_detector.state.user_state import UserStateSnapshot

# ---------------------------------------------------------------------------
# Thresholds (named constants, never magic numbers)
# ---------------------------------------------------------------------------

# Amount thresholds in paise
AMOUNT_MICRO = 100           # < 1 INR — suspicious micro-transaction
AMOUNT_HIGH = 10_000_000    # ≥ 1 lakh INR — high-value
AMOUNT_CRITICAL = 100_000_000  # ≥ 10 lakh INR — critical-value

# Velocity thresholds
HOURLY_TX_HIGH = 5           # > 5 tx/hr is suspicious
HOURLY_TX_CRITICAL = 10      # > 10 tx/hr is very suspicious
DAILY_TX_HIGH = 15           # > 15 tx/day
DAILY_AMOUNT_HIGH = 50_000_000  # > 5 lakh/day

# Amount z-score threshold
ZSCORE_HIGH = 3.0            # 3-sigma is an outlier
ZSCORE_CRITICAL = 5.0        # 5-sigma is very unusual

# Minimum samples for z-score to be meaningful
MIN_SAMPLES_FOR_ZSCORE = 5

# Hours considered high-risk (late night / very early morning)
HIGH_RISK_HOURS = frozenset(range(0, 5))   # 00:00–04:59 UTC

# Known OFAC/high-risk currencies (synthetic list — real system would use a live feed)
HIGH_RISK_CURRENCIES = frozenset({"XXX", "ZWD"})


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureVector:
    """Fully derived, immutable feature vector for a single transaction."""

    # ── Amount features ──────────────────────────────────────────────────
    amount_paise: int
    is_micro_amount: bool         # amount < AMOUNT_MICRO
    is_high_amount: bool          # amount ≥ AMOUNT_HIGH
    is_critical_amount: bool      # amount ≥ AMOUNT_CRITICAL

    # ── Velocity features ────────────────────────────────────────────────
    hourly_tx_count: int
    hourly_amount_paise: int
    daily_tx_count: int
    daily_amount_paise: int
    is_high_hourly_velocity: bool
    is_critical_hourly_velocity: bool
    is_high_daily_tx: bool
    is_high_daily_amount: bool

    # ── Amount z-score (versus user's own history) ───────────────────────
    zscore_amount: float          # NaN when insufficient history
    is_amount_outlier: bool       # |z| > ZSCORE_HIGH
    is_amount_extreme_outlier: bool  # |z| > ZSCORE_CRITICAL

    # ── Merchant features ─────────────────────────────────────────────────
    is_new_merchant: bool         # merchant never seen before for this user
    known_merchant_count: int     # number of distinct merchants seen so far

    # ── Temporal features ─────────────────────────────────────────────────
    hour_of_day: int              # 0–23 UTC
    day_of_week: int              # 0=Monday … 6=Sunday
    is_high_risk_hour: bool       # 00:00–04:59 UTC

    # ── User state features ───────────────────────────────────────────────
    is_cold_start: bool           # user has no prior transaction history
    total_tx_count: int           # lifetime tx count (before this tx)

    # ── Currency features ─────────────────────────────────────────────────
    is_domestic: bool             # currency == "INR"
    is_high_risk_currency: bool   # currency in HIGH_RISK_CURRENCIES

    # ── Device & Geo features ─────────────────────────────────────────────
    is_new_device: bool = False
    device_count_1h: int = 0
    is_high_device_velocity: bool = False
    travel_speed_kmh: float = 0.0
    is_impossible_travel: bool = False


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def extract_features(
    request: TransactionRequest,
    snapshot: UserStateSnapshot,
) -> FeatureVector:
    """Compute a FeatureVector from a validated request and user snapshot.

    This function is pure: no I/O, no side effects, completely deterministic.
    """
    ts: datetime = request.timestamp
    amount = request.amount

    # ── Amount ─────────────────────────────────────────────────────────────
    is_micro = amount < AMOUNT_MICRO
    is_high = amount >= AMOUNT_HIGH
    is_critical = amount >= AMOUNT_CRITICAL

    # ── Velocity ───────────────────────────────────────────────────────────
    h_count = snapshot.hourly_tx_count
    h_amount = snapshot.hourly_amount_paise
    d_count = snapshot.daily_tx_count
    d_amount = snapshot.daily_amount_paise

    # ── Z-score ────────────────────────────────────────────────────────────
    n = snapshot.total_tx_count   # count BEFORE this transaction
    zscore = float("nan")
    if n >= MIN_SAMPLES_FOR_ZSCORE:
        variance = snapshot.amount_m2 / n  # population variance
        if variance > 0:
            std = math.sqrt(variance)
            zscore = (amount - snapshot.amount_mean) / std
        # If variance == 0 (all previous amounts identical):
        # zscore stays NaN (indeterminate, we don't divide by zero)

    is_outlier = not math.isnan(zscore) and abs(zscore) > ZSCORE_HIGH
    is_extreme = not math.isnan(zscore) and abs(zscore) > ZSCORE_CRITICAL

    # ── Merchant ───────────────────────────────────────────────────────────
    is_new_merchant = request.merchant_id not in snapshot.known_merchants

    # ── Temporal ───────────────────────────────────────────────────────────
    hour = ts.hour
    dow = ts.weekday()

    # ── Currency ───────────────────────────────────────────────────────────
    currency = request.currency
    is_domestic = currency == "INR"
    is_high_risk_ccy = currency in HIGH_RISK_CURRENCIES

    # ── Device & Geo ───────────────────────────────────────────────────────
    is_new_device = bool(
        request.device_id is not None
        and not snapshot.is_cold_start
        and request.device_id not in snapshot.known_devices
    )
    dev_count = snapshot.device_count_1h
    is_high_dev_vel = dev_count > 2

    travel_speed = 0.0
    is_impossible = False
    if (
        request.latitude is not None
        and request.longitude is not None
        and snapshot.last_latitude is not None
        and snapshot.last_longitude is not None
        and snapshot.last_tx_timestamp is not None
    ):
        dist_km = haversine_distance_km(
            snapshot.last_latitude,
            snapshot.last_longitude,
            request.latitude,
            request.longitude,
        )
        dt_hours = abs((ts - snapshot.last_tx_timestamp).total_seconds()) / 3600.0
        if dt_hours > 0:
            travel_speed = dist_km / dt_hours
        elif dist_km > 5.0:
            travel_speed = 99999.0
        else:
            travel_speed = 0.0

        if travel_speed > 900.0:
            is_impossible = True

    return FeatureVector(
        # Amount
        amount_paise=amount,
        is_micro_amount=is_micro,
        is_high_amount=is_high,
        is_critical_amount=is_critical,
        # Velocity
        hourly_tx_count=h_count,
        hourly_amount_paise=h_amount,
        daily_tx_count=d_count,
        daily_amount_paise=d_amount,
        is_high_hourly_velocity=(h_count > HOURLY_TX_HIGH),
        is_critical_hourly_velocity=(h_count > HOURLY_TX_CRITICAL),
        is_high_daily_tx=(d_count > DAILY_TX_HIGH),
        is_high_daily_amount=(d_amount > DAILY_AMOUNT_HIGH),
        # Z-score
        zscore_amount=zscore,
        is_amount_outlier=is_outlier,
        is_amount_extreme_outlier=is_extreme,
        # Merchant
        is_new_merchant=is_new_merchant,
        known_merchant_count=len(snapshot.known_merchants),
        # Temporal
        hour_of_day=hour,
        day_of_week=dow,
        is_high_risk_hour=(hour in HIGH_RISK_HOURS),
        # User state
        is_cold_start=snapshot.is_cold_start,
        total_tx_count=n,
        # Currency
        is_domestic=is_domestic,
        is_high_risk_currency=is_high_risk_ccy,
        # Device & Geo
        is_new_device=is_new_device,
        device_count_1h=dev_count,
        is_high_device_velocity=is_high_dev_vel,
        travel_speed_kmh=travel_speed,
        is_impossible_travel=is_impossible,
    )
