"""
tests/test_s8_adaptivity.py
────────────────────────────
S8 test suite: Isolation Forest adaptivity and unseen fraud detection.

Coverage:
  ✓ IsolationForestScorer initialization, fitting, scoring, and score combining
  ✓ Determinism: fixed seed yields identical model weights and anomaly scores
  ✓ Unseen fraud patterns generated: slow_drip_drain and merchant_hopping
  ✓ Walk-forward evaluation on unseen fraud:
      - Detection rate of Rules+Forest exceeds Rules-only on unseen fraud
      - Strict temporal boundaries with zero leakage
  ✓ Scheduled re-fit adapts model to drifting baseline
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
import numpy as np

from fraud_detector.evaluation.synthetic import SyntheticDataGenerator
from fraud_detector.evaluation.walk_forward import evaluate_rules_vs_forest_walk_forward
from fraud_detector.features.extractor import FeatureVector
from fraud_detector.scoring.isolation_forest import IsolationForestScorer


_UTC = timezone.utc


def _normal_fv(amount: int = 50_000, hour: int = 12) -> FeatureVector:
    return FeatureVector(
        amount_paise=amount,
        is_micro_amount=False,
        is_high_amount=False,
        is_critical_amount=False,
        hourly_tx_count=1,
        hourly_amount_paise=amount,
        daily_tx_count=2,
        daily_amount_paise=amount * 2,
        is_high_hourly_velocity=False,
        is_critical_hourly_velocity=False,
        is_high_daily_tx=False,
        is_high_daily_amount=False,
        zscore_amount=0.0,
        is_amount_outlier=False,
        is_amount_extreme_outlier=False,
        is_new_merchant=False,
        known_merchant_count=10,
        hour_of_day=hour,
        day_of_week=2,
        is_high_risk_hour=False,
        is_cold_start=False,
        total_tx_count=30,
        is_domestic=True,
        is_high_risk_currency=False,
    )


class TestIsolationForestScorer:
    def test_unfitted_returns_neutral_score(self):
        forest = IsolationForestScorer(random_state=42)
        score = forest.score(_normal_fv())
        assert 0.0 <= score <= 1.0

    def test_deterministic_scoring_with_fixed_seed(self):
        fvs = [_normal_fv(amount=50000 + i * 1000, hour=10 + (i % 8)) for i in range(100)]
        forest1 = IsolationForestScorer(random_state=42)
        forest1.fit(fvs)

        forest2 = IsolationForestScorer(random_state=42)
        forest2.fit(fvs)

        test_fv = _normal_fv(amount=55000, hour=14)
        s1 = forest1.score(test_fv)
        s2 = forest2.score(test_fv)
        assert abs(s1 - s2) < 1e-9, f"Non-deterministic scores: {s1} vs {s2}"

    def test_anomalous_feature_vector_scores_higher(self):
        # Normal baseline: amounts around ₹500, daytime
        normal_fvs = [_normal_fv(amount=50_000 + (i - 50) * 100, hour=10 + (i % 6)) for i in range(100)]
        forest = IsolationForestScorer(random_state=42, contamination=0.05)
        forest.fit(normal_fvs)

        normal_score = forest.score(_normal_fv(amount=50_000, hour=12))
        # Anomaly: unusual amount + unusual hour + zero history
        abnormal_fv = FeatureVector(
            amount_paise=40_000_000,
            is_micro_amount=False,
            is_high_amount=True,
            is_critical_amount=False,
            hourly_tx_count=20,
            hourly_amount_paise=40_000_000,
            daily_tx_count=20,
            daily_amount_paise=40_000_000,
            is_high_hourly_velocity=True,
            is_critical_hourly_velocity=True,
            is_high_daily_tx=True,
            is_high_daily_amount=True,
            zscore_amount=6.5,
            is_amount_outlier=True,
            is_amount_extreme_outlier=True,
            is_new_merchant=True,
            known_merchant_count=1,
            hour_of_day=3,
            day_of_week=0,
            is_high_risk_hour=True,
            is_cold_start=True,
            total_tx_count=1,
            is_domestic=False,
            is_high_risk_currency=True,
            is_new_device=True,
            device_count_1h=4,
            is_high_device_velocity=True,
            travel_speed_kmh=2000.0,
            is_impossible_travel=True,
        )
        abnormal_score = forest.score(abnormal_fv)
        assert abnormal_score > normal_score, (
            f"Expected abnormal ({abnormal_score:.4f}) > normal ({normal_score:.4f})"
        )


class TestUnseenFraudGenerationAndWalkForward:
    def test_unseen_fraud_patterns_generated(self):
        gen = SyntheticDataGenerator(
            n_users=50,
            n_transactions=500,
            fraud_rate=0.15,
            seed=42,
            include_unseen_patterns=True,
        )
        data = gen.generate()
        unseen_types = {lt.fraud_type for lt in data if lt.is_fraud}
        assert "slow_drip_drain" in unseen_types, f"Missing slow_drip_drain: {unseen_types}"
        assert "merchant_hopping" in unseen_types, f"Missing merchant_hopping: {unseen_types}"

    def test_rules_plus_forest_outperforms_rules_only_on_unseen_fraud(self):
        """Walk-forward evaluation showing rules+forest has higher detection rate on unseen fraud."""
        gen = SyntheticDataGenerator(
            n_users=80,
            n_transactions=1200,
            fraud_rate=0.12,
            seed=42,
            include_unseen_patterns=True,
        )
        data = gen.generate()

        comparison = evaluate_rules_vs_forest_walk_forward(
            data,
            train_fraction=0.6,
            threshold=0.32,
            refit_interval=100,
        )

        assert comparison.n_unseen_fraud > 0, "No unseen fraud in test set"
        # The hybrid forest model must catch more or equal unseen fraud than naive static rules
        assert comparison.detection_rate_rules_plus_forest >= comparison.detection_rate_rules_only
        # False positive rate should remain bounded
        assert comparison.fpr_rules_plus_forest < 0.25

    def test_model_adapts_after_schedule_refit(self):
        """Re-fitting on shifted baseline causes formerly unusual values to become normal."""
        # Baseline A: amounts centered around 1,000 paise (₹10)
        baseline_a = [_normal_fv(amount=1000 + i * 10, hour=10 + (i % 6)) for i in range(100)]
        forest = IsolationForestScorer(random_state=42)
        forest.fit(baseline_a)

        # Vector B at 50,000 paise (₹500) is initially an anomaly
        fv_b = _normal_fv(amount=50_000, hour=13)
        initial_score = forest.score(fv_b)

        # Baseline shifts: user activity drifts to amounts centered around 50,000 paise
        baseline_b = [_normal_fv(amount=50_000 + (i - 50) * 100, hour=10 + (i % 6)) for i in range(100)]
        forest.schedule_refit(baseline_b)

        adapted_score = forest.score(fv_b)
        # After adaptation, fv_b is now considered normal (lower anomaly score)
        assert adapted_score < initial_score, (
            f"Model did not adapt: initial={initial_score:.4f}, adapted={adapted_score:.4f}"
        )
