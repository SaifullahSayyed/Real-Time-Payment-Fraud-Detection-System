"""
fraud_detector/decision/cost_matrix.py
──────────────────────────────────────
Cost-optimal Decision Engine.

Decisions:
  • APPROVE  – Allow payment without customer interruption.
  • STEP_UP  – Challenge with 2FA / OTP step-up authentication.
  • REVIEW   – Route to manual human analyst review queue.
  • BLOCK    – Hard decline immediately.

Cost Matrix Model:
  Total Cost = (FN * C_missed_fraud) + (FP_block * C_false_alarm)
             + (Reviews * C_review) + (StepUps * C_stepup)

Thresholds are dynamically adapted as ground-truth review labels are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class Decision(str, Enum):
    APPROVE = "APPROVE"
    STEP_UP = "STEP_UP"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class CostMatrix:
    """Operational unit costs in INR paise."""
    c_missed_fraud: int = 500_000   # ₹5,000 average fraud loss
    c_false_block: int = 50_000     # ₹500 customer churn / dispute cost
    c_manual_review: int = 5_000    # ₹50 human review labour cost
    c_step_up: int = 500            # ₹5 SMS/OTP infrastructure & minor friction


@dataclass
class DecisionResult:
    decision: Decision
    score: float
    expected_cost_paise: int
    thresholds: dict[str, float]
    action_note: str


class DecisionEngine:
    """Maps continuous fraud scores into business decisions minimizing expected cost."""

    def __init__(
        self,
        cost_matrix: CostMatrix | None = None,
        thresh_stepup: float = 0.30,
        thresh_review: float = 0.60,
        thresh_block: float = 0.85,
        max_step_per_label: float = 0.02,
        max_drift_per_hour: float = 0.10,
    ) -> None:
        self.costs = cost_matrix or CostMatrix()
        self.thresh_stepup = thresh_stepup
        self.thresh_review = thresh_review
        self.thresh_block = thresh_block
        self.max_step_per_label = max_step_per_label
        self.max_drift_per_hour = max_drift_per_hour
        self._drift_history: list[tuple[float, float]] = []

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "step_up": round(self.thresh_stepup, 4),
            "review": round(self.thresh_review, 4),
            "block": round(self.thresh_block, 4),
        }

    def decide(self, score: float, amount_paise: int) -> DecisionResult:
        """Compute the cost-optimal decision for a given score and amount."""
        # Scale missed-fraud cost with transaction value if larger than default
        c_fn = max(self.costs.c_missed_fraud, amount_paise)

        if score >= self.thresh_block:
            decision = Decision.BLOCK
            # Expected cost = probability of false alarm * c_false_block
            expected_cost = int((1.0 - score) * self.costs.c_false_block)
            note = "High probability fraud: transaction declined."
        elif score >= self.thresh_review:
            decision = Decision.REVIEW
            expected_cost = self.costs.c_manual_review + int((1.0 - score) * 1000)
            note = "Suspicious activity: routed to analyst review queue."
        elif score >= self.thresh_stepup:
            decision = Decision.STEP_UP
            expected_cost = self.costs.c_step_up
            note = "Moderate risk: step-up 2FA/OTP challenge required."
        else:
            decision = Decision.APPROVE
            # Expected cost = probability of missed fraud * c_fn
            expected_cost = int(score * c_fn)
            note = "Clean transaction: automatically approved."

        return DecisionResult(
            decision=decision,
            score=score,
            expected_cost_paise=expected_cost,
            thresholds=self.thresholds,
            action_note=note,
        )

    def update_from_review_feedback(
        self, tp: int, fp: int, tn: int, fn: int, now: float | None = None
    ) -> dict[str, float]:
        """Recalibrate thresholds dynamically based on empirical review outcomes.

        Guarantees:
          • Step per label is bounded to at most `max_step_per_label` (default 0.02).
          • Total drift across all reviews in any 1-hour rolling window is bounded
            to at most `max_drift_per_hour` (default 0.10).
        """
        total_eval = tp + fp + tn + fn
        if total_eval < 5:
            return self.thresholds

        import time
        t_now = now if now is not None else time.monotonic()

        # Prune drift records older than 1 hour (3600 seconds)
        cutoff = t_now - 3600.0
        self._drift_history = [(t, d) for t, d in self._drift_history if t > cutoff]
        hourly_drift = sum(abs(d) for _, d in self._drift_history)

        step = self.max_step_per_label
        if hourly_drift + step > self.max_drift_per_hour:
            # Hourly drift budget exhausted — cannot drift any further this hour
            return self.thresholds

        # If false positives are high, shift block and review thresholds upward
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        if fp_rate > 0.08:
            # Too many false alarms: relax block and review thresholds slightly
            self.thresh_block = min(0.92, self.thresh_block + step)
            self.thresh_review = min(0.70, self.thresh_review + step)
            self._drift_history.append((t_now, step))
        elif fn_rate > 0.05:
            # Too many missed frauds: tighten block and review thresholds
            self.thresh_block = max(0.75, self.thresh_block - step)
            self.thresh_review = max(0.50, self.thresh_review - step)
            self._drift_history.append((t_now, -step))

        return self.thresholds
