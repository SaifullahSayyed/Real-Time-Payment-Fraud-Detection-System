"""
fraud_detector/scoring/isolation_forest.py
──────────────────────────────────────────
Adaptive anomaly detection using scikit-learn Isolation Forest.

Key design:
  • Trains strictly on a rolling window of verified legitimate transactions.
  • Transforms FeatureVector instances into standardized numeric vectors.
  • Converts decision_function scores into a calibrated [0, 1] anomaly score.
  • Combines rule-based score + forest anomaly score deterministically.
  • Re-fitting on schedule allows model to adapt to new baseline patterns.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from sklearn.ensemble import IsolationForest

from fraud_detector.features.extractor import FeatureVector


class IsolationForestScorer:
    """Adaptive anomaly detector for unseen fraud patterns."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        contamination: float = 0.05,
        forest_weight: float = 0.35,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.forest_weight = forest_weight
        self._model: IsolationForest | None = None
        self._is_fitted: bool = False

    @staticmethod
    def _vectorize(fv: FeatureVector) -> list[float]:
        """Convert FeatureVector into a numeric vector suitable for trees."""
        z = fv.zscore_amount
        if math.isnan(z) or math.isinf(z):
            z = 0.0

        return [
            math.log1p(float(fv.amount_paise)),
            float(fv.hourly_tx_count),
            float(fv.daily_tx_count),
            float(fv.hour_of_day),
            float(fv.day_of_week),
            float(fv.known_merchant_count),
            float(z),
            float(fv.device_count_1h),
            float(fv.travel_speed_kmh),
            1.0 if fv.is_new_merchant else 0.0,
            1.0 if fv.is_new_device else 0.0,
            1.0 if fv.is_domestic else 0.0,
        ]

    def fit(self, normal_features: Sequence[FeatureVector]) -> None:
        """Fit the isolation forest on a batch of legitimate feature vectors."""
        if not normal_features:
            return

        X = np.array([self._vectorize(fv) for fv in normal_features], dtype=np.float64)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,  # Single-threaded for reproducibility
        )
        model.fit(X)
        self._model = model
        self._is_fitted = True

    def score(self, fv: FeatureVector) -> float:
        """Score an individual feature vector.

        Returns anomaly score in [0.0, 1.0], where 1.0 is extremely anomalous.
        If the model is not fitted yet, returns neutral 0.15.
        """
        if not self._is_fitted or self._model is None:
            return 0.15

        x = np.array([self._vectorize(fv)], dtype=np.float64)
        # decision_function: lower means more anomalous (e.g. -0.2 to +0.2)
        raw_score = float(self._model.decision_function(x)[0])
        # Logistic squashing centered at 0 with steep slope
        # raw_score < 0 is an anomaly -> maps towards 1.0
        # raw_score > 0 is normal -> maps towards 0.0
        anomaly_prob = 1.0 / (1.0 + math.exp(raw_score * 12.0))
        return max(0.0, min(1.0, anomaly_prob))

    def combine(self, rule_score: float, fv: FeatureVector) -> tuple[float, float]:
        """Combine rule score and forest anomaly score.

        Returns (combined_score, forest_score).
        """
        if not self._is_fitted:
            return rule_score, 0.0

        f_score = self.score(fv)
        w = self.forest_weight
        # If forest detects severe anomaly (> 0.70) while rules missed it,
        # ensure strong boost
        boosted = max(rule_score, f_score) if f_score > 0.70 else (1 - w) * rule_score + w * f_score
        combined = max(0.0, min(1.0, boosted))
        return combined, f_score

    def schedule_refit(self, recent_legit_features: Sequence[FeatureVector]) -> None:
        """Re-fit the model on an updated rolling window of legit transactions."""
        self.fit(recent_legit_features)
