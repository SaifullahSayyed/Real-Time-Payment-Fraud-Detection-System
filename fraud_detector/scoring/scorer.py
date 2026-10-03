"""
fraud_detector/scoring/scorer.py
──────────────────────────────────
Deterministic, idempotent fraud scorer.

Design:
  • Each rule returns a (weight, flag_message) pair.
  • Weights are summed and clamped to [0, 1] with a logistic squash.
  • Same FeatureVector → same score, always (no random seeds, no clocks inside).
  • Cold-start: new users get a baseline score bump but are not auto-blocked.
  • Idempotency is enforced by the StateStore layer, not here; the scorer
    is stateless and can be called with any snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from fraud_detector.features.extractor import FeatureVector
from fraud_detector.models.transaction import RiskLevel, _score_to_risk

# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------


class _RuleResult(NamedTuple):
    weight: float       # additive risk weight (can be negative = reward)
    flag: str           # human-readable reason (empty string = not triggered)


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def _rule_cold_start(fv: FeatureVector) -> _RuleResult:
    if fv.is_cold_start:
        return _RuleResult(0.20, "COLD_START: no transaction history for user")
    return _RuleResult(0.0, "")


def _rule_micro_amount(fv: FeatureVector) -> _RuleResult:
    """Micro-transactions are often used to probe stolen cards."""
    if fv.is_micro_amount:
        return _RuleResult(0.25, f"MICRO_AMOUNT: {fv.amount_paise} paise")
    return _RuleResult(0.0, "")


def _rule_critical_amount(fv: FeatureVector) -> _RuleResult:
    if fv.is_critical_amount:
        return _RuleResult(0.30, f"CRITICAL_AMOUNT: {fv.amount_paise} paise ≥ ₹10L")
    elif fv.is_high_amount:
        return _RuleResult(0.15, f"HIGH_AMOUNT: {fv.amount_paise} paise ≥ ₹1L")
    return _RuleResult(0.0, "")


def _rule_hourly_velocity(fv: FeatureVector) -> _RuleResult:
    if fv.is_critical_hourly_velocity:
        return _RuleResult(
            0.35,
            f"CRITICAL_VELOCITY: {fv.hourly_tx_count} tx in last hour",
        )
    if fv.is_high_hourly_velocity:
        return _RuleResult(
            0.20,
            f"HIGH_VELOCITY: {fv.hourly_tx_count} tx in last hour",
        )
    return _RuleResult(0.0, "")


def _rule_daily_velocity(fv: FeatureVector) -> _RuleResult:
    flags = []
    weight = 0.0
    if fv.is_high_daily_tx:
        weight += 0.15
        flags.append(f"HIGH_DAILY_TX: {fv.daily_tx_count} tx in 24h")
    if fv.is_high_daily_amount:
        weight += 0.15
        flags.append(f"HIGH_DAILY_AMOUNT: {fv.daily_amount_paise} paise in 24h")
    if flags:
        return _RuleResult(weight, "; ".join(flags))
    return _RuleResult(0.0, "")


def _rule_amount_zscore(fv: FeatureVector) -> _RuleResult:
    if fv.is_amount_extreme_outlier:
        return _RuleResult(
            0.30,
            f"EXTREME_AMOUNT_OUTLIER: z={fv.zscore_amount:.2f}",
        )
    if fv.is_amount_outlier:
        return _RuleResult(
            0.20,
            f"AMOUNT_OUTLIER: z={fv.zscore_amount:.2f}",
        )
    return _RuleResult(0.0, "")


def _rule_new_merchant(fv: FeatureVector) -> _RuleResult:
    """New merchant is mildly suspicious; offset by how many we've seen."""
    if fv.is_new_merchant and not fv.is_cold_start:
        # Less suspicious as the user's merchant diversity grows
        penalty = max(0.05, 0.15 - fv.known_merchant_count * 0.01)
        return _RuleResult(
            penalty,
            f"NEW_MERCHANT: first time at this merchant (seen {fv.known_merchant_count} before)",
        )
    return _RuleResult(0.0, "")


def _rule_high_risk_hour(fv: FeatureVector) -> _RuleResult:
    if fv.is_high_risk_hour:
        return _RuleResult(
            0.10,
            f"HIGH_RISK_HOUR: transaction at {fv.hour_of_day:02d}:xx UTC",
        )
    return _RuleResult(0.0, "")


def _rule_high_risk_currency(fv: FeatureVector) -> _RuleResult:
    if fv.is_high_risk_currency:
        return _RuleResult(0.30, f"HIGH_RISK_CURRENCY: currency is flagged")
    return _RuleResult(0.0, "")


def _rule_foreign_currency(fv: FeatureVector) -> _RuleResult:
    """Foreign (non-INR) transaction is mildly elevated risk."""
    if not fv.is_domestic and not fv.is_high_risk_currency:
        return _RuleResult(0.05, "FOREIGN_CURRENCY: non-INR transaction")
    return _RuleResult(0.0, "")


# Reward for established users with many known merchants
def _rule_established_user_discount(fv: FeatureVector) -> _RuleResult:
    if fv.total_tx_count >= 20 and fv.known_merchant_count >= 5:
        return _RuleResult(-0.10, "")
    return _RuleResult(0.0, "")


def _rule_impossible_travel(fv: FeatureVector) -> _RuleResult:
    if fv.is_impossible_travel:
        return _RuleResult(
            0.40,
            f"IMPOSSIBLE_TRAVEL: {fv.travel_speed_kmh:.0f} km/h between locations exceeds physical limit",
        )
    return _RuleResult(0.0, "")


def _rule_new_device(fv: FeatureVector) -> _RuleResult:
    if fv.is_new_device:
        return _RuleResult(0.10, "NEW_DEVICE: unseen device fingerprint")
    return _RuleResult(0.0, "")


def _rule_device_velocity(fv: FeatureVector) -> _RuleResult:
    if fv.is_high_device_velocity:
        return _RuleResult(
            0.25,
            f"DEVICE_VELOCITY: {fv.device_count_1h} distinct devices used in 1h",
        )
    return _RuleResult(0.0, "")


# ---------------------------------------------------------------------------
# Rule registry — ORDER MATTERS for flag clarity but NOT for final score
# (scores are additive, not cascading)
# ---------------------------------------------------------------------------

_RULES = [
    _rule_cold_start,
    _rule_micro_amount,
    _rule_critical_amount,
    _rule_hourly_velocity,
    _rule_daily_velocity,
    _rule_amount_zscore,
    _rule_new_merchant,
    _rule_high_risk_hour,
    _rule_high_risk_currency,
    _rule_foreign_currency,
    _rule_established_user_discount,
    _rule_impossible_travel,
    _rule_new_device,
    _rule_device_velocity,
]


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringResult:
    score: float            # [0, 1]
    risk_level: RiskLevel
    flags: list[str]
    raw_weight: float       # pre-squash sum (useful for debugging)


# ---------------------------------------------------------------------------
# Logistic squash
# ---------------------------------------------------------------------------

def _squash(raw: float) -> float:
    """Map raw weight sum → [0, 1] via a calibrated logistic function.

    Calibration intent:
      raw=0.0  → score≈0.05  (baseline for any transaction)
      raw=0.5  → score≈0.50
      raw=1.0  → score≈0.88
    Achieved by: σ(k*(raw - 0.5)) with k=5, shifted to start at 0.05.

    We keep the minimum at ~0.05 so even clean transactions have a non-zero
    score (fraud systems should always communicate uncertainty).
    """
    k = 5.0
    shifted = k * (raw - 0.5)
    return 1.0 / (1.0 + math.exp(-shifted))


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------


def score_transaction(fv: FeatureVector) -> ScoringResult:
    """Score a transaction given its feature vector.

    Deterministic: same fv → same result, guaranteed.
    No I/O, no mutable state, no randomness.
    """
    raw_weight = 0.0
    flags: list[str] = []

    for rule in _RULES:
        result = rule(fv)
        raw_weight += result.weight
        if result.flag:
            flags.append(result.flag)

    # Clamp raw before squash to avoid extreme logistic saturation
    clamped = max(0.0, min(raw_weight, 2.0))
    score = _squash(clamped)

    # Clamp final score to [0, 1] (logistic already does this, but be explicit)
    score = max(0.0, min(1.0, score))

    risk = _score_to_risk(score)

    return ScoringResult(
        score=score,
        risk_level=risk,
        flags=flags,
        raw_weight=raw_weight,
    )
