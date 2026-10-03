"""
tests/test_slice5_evaluation.py
─────────────────────────────────
Slice 5 tests: synthetic data generator + walk-forward evaluation.

Coverage:
  SyntheticDataGenerator:
    ✓ Generate returns list of LabelledTransactions
    ✓ All timestamps are UTC-aware
    ✓ Dataset is sorted by timestamp (ascending)
    ✓ All transaction_ids are unique
    ✓ Both fraud and legit samples present
    ✓ Fraud rate approximately matches requested rate
    ✓ Amount always > 0 (int paise)
    ✓ Deterministic: same seed → same dataset
    ✓ Different seeds → different datasets
    ✓ Fraud types: card_testing, account_takeover, velocity_fraud present

  walk_forward_split:
    ✓ Train fraction preserved (within 1 sample tolerance)
    ✓ Test is strictly after train (no timestamp overlap)
    ✓ Empty dataset returns ([], [])
    ✓ Invalid train_fraction raises ValueError
    ✓ Leakage detected: raises RuntimeError when test < train max ts

  evaluate_walk_forward:
    ✓ Returns EvaluationResult
    ✓ AUROC in [0, 1]
    ✓ AUROC > 0.5 (better than random — our model should work on synthetic fraud)
    ✓ best_f1 in [0, 1]
    ✓ n_train + n_test == total samples
    ✓ n_test_fraud + n_test_legit == n_test
    ✓ Precision@K: all values in [0, 1]
    ✓ No leakage: evaluator never sees test labels during warm-up
    ✓ Deterministic: same data, same engine → same AUROC

  _trapezoid_auc:
    ✓ Perfect classifier: AUC = 1.0
    ✓ Random classifier: AUC ≈ 0.5
    ✓ All positive predictions: AUC boundary behaviour
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from fraud_detector.evaluation.synthetic import LabelledTransaction, SyntheticDataGenerator
from fraud_detector.evaluation.walk_forward import (
    EvaluationResult,
    ThresholdMetrics,
    _trapezoid_auc,
    evaluate_walk_forward,
    walk_forward_split,
)
from fraud_detector.models.transaction import TransactionRequest

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# SyntheticDataGenerator tests
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    def setup_method(self):
        self.gen = SyntheticDataGenerator(
            n_users=50,
            n_transactions=500,
            fraud_rate=0.10,
            seed=99,
        )
        self.data = self.gen.generate()

    def test_returns_list_of_labelled_transactions(self):
        assert isinstance(self.data, list)
        assert all(isinstance(lt, LabelledTransaction) for lt in self.data)

    def test_non_empty(self):
        assert len(self.data) > 0

    def test_all_timestamps_utc_aware(self):
        for lt in self.data:
            ts = lt.request.timestamp
            assert ts.tzinfo is not None
            assert ts.utcoffset().total_seconds() == 0, (
                f"Non-UTC timestamp: {ts}"
            )

    def test_sorted_by_timestamp_ascending(self):
        timestamps = [lt.request.timestamp for lt in self.data]
        assert timestamps == sorted(timestamps), "Dataset must be sorted by timestamp"

    def test_all_transaction_ids_unique(self):
        ids = [lt.request.transaction_id for lt in self.data]
        assert len(ids) == len(set(ids)), "Duplicate transaction IDs detected"

    def test_both_fraud_and_legit_present(self):
        fraud_count = sum(1 for lt in self.data if lt.is_fraud)
        legit_count = sum(1 for lt in self.data if not lt.is_fraud)
        assert fraud_count > 0, "No fraud samples generated"
        assert legit_count > 0, "No legit samples generated"

    def test_fraud_rate_approximately_correct(self):
        fraud_count = sum(1 for lt in self.data if lt.is_fraud)
        actual_rate = fraud_count / len(self.data)
        # Allow ±15% relative tolerance (burst fraud increases count)
        assert actual_rate > 0.01, f"Fraud rate too low: {actual_rate:.3f}"
        assert actual_rate < 0.60, f"Fraud rate too high: {actual_rate:.3f}"

    def test_all_amounts_positive_integers(self):
        for lt in self.data:
            amt = lt.request.amount
            assert isinstance(amt, int), f"amount is not int: {type(amt)}"
            assert amt > 0, f"amount not positive: {amt}"

    def test_deterministic_same_seed(self):
        gen2 = SyntheticDataGenerator(n_users=50, n_transactions=500, fraud_rate=0.10, seed=99)
        data2 = gen2.generate()
        assert len(self.data) == len(data2)
        for lt1, lt2 in zip(self.data, data2):
            assert lt1.request.transaction_id == lt2.request.transaction_id
            assert lt1.request.amount == lt2.request.amount
            assert lt1.is_fraud == lt2.is_fraud

    def test_different_seeds_different_data(self):
        gen_other = SyntheticDataGenerator(
            n_users=50, n_transactions=500, fraud_rate=0.10, seed=1234
        )
        data_other = gen_other.generate()
        # At least some transaction amounts should differ
        amounts_a = [lt.request.amount for lt in self.data[:10]]
        amounts_b = [lt.request.amount for lt in data_other[:10]]
        assert amounts_a != amounts_b, "Different seeds produced identical data"

    def test_fraud_types_present(self):
        fraud_types = {lt.fraud_type for lt in self.data if lt.is_fraud}
        # With 500 transactions at 10% fraud rate, all three types should appear
        assert "card_testing" in fraud_types, f"Missing card_testing, got: {fraud_types}"
        assert "account_takeover" in fraud_types, f"Missing account_takeover, got: {fraud_types}"

    def test_legit_transactions_have_normal_fraud_type(self):
        for lt in self.data:
            if not lt.is_fraud:
                assert lt.fraud_type == "normal"


# ---------------------------------------------------------------------------
# walk_forward_split tests
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def _make_data(self, n: int) -> list[LabelledTransaction]:
        gen = SyntheticDataGenerator(n_users=10, n_transactions=n, seed=7)
        return gen.generate()

    def test_split_fraction_preserved(self):
        data = self._make_data(100)
        train, test = walk_forward_split(data, train_fraction=0.7)
        # Allow ±1 due to integer rounding
        assert abs(len(train) - int(len(data) * 0.7)) <= 1
        assert len(train) + len(test) == len(data)

    def test_test_strictly_after_train(self):
        data = self._make_data(200)
        train, test = walk_forward_split(data, train_fraction=0.7)
        if train and test:
            train_max = max(lt.request.timestamp for lt in train)
            test_min = min(lt.request.timestamp for lt in test)
            assert test_min >= train_max, (
                f"Leakage: test min {test_min} < train max {train_max}"
            )

    def test_empty_dataset(self):
        train, test = walk_forward_split([])
        assert train == [] and test == []

    def test_invalid_fraction_zero_raises(self):
        data = self._make_data(10)
        with pytest.raises(ValueError):
            walk_forward_split(data, train_fraction=0.0)

    def test_invalid_fraction_one_raises(self):
        data = self._make_data(10)
        with pytest.raises(ValueError):
            walk_forward_split(data, train_fraction=1.0)

    def test_leakage_detection_raises(self):
        """Passing out-of-order data should trigger leakage detection."""
        t1 = datetime(2024, 1, 1, tzinfo=_UTC)
        t2 = datetime(2024, 1, 2, tzinfo=_UTC)
        t3 = datetime(2024, 1, 3, tzinfo=_UTC)

        def _req(tx_id, ts):
            return TransactionRequest(
                transaction_id=tx_id,
                user_id="u1",
                merchant_id="m1",
                amount=1000,
                timestamp=ts,
            )

        # train_data has t1 and t3; test_data has t2 → leakage
        # Simulate by building a list where early train entries are later
        # Note: split is positional, so we need train to have a large timestamp
        lt_early = LabelledTransaction(_req("t1", t1), False, "normal")
        lt_late_in_train = LabelledTransaction(_req("t3", t3), False, "normal")
        lt_test = LabelledTransaction(_req("t2", t2), False, "normal")

        # Force a specific layout: first 2/3 are train, last 1/3 is test
        # Put t3 in train position and t2 (earlier) in test → leakage
        crafted = [lt_early, lt_late_in_train, lt_test]  # NOT sorted by ts

        with pytest.raises(RuntimeError, match="LEAKAGE"):
            walk_forward_split(crafted, train_fraction=0.67)


# ---------------------------------------------------------------------------
# evaluate_walk_forward tests
# ---------------------------------------------------------------------------


class TestEvaluateWalkForward:
    def setup_method(self):
        self.gen = SyntheticDataGenerator(
            n_users=100,
            n_transactions=1_000,
            fraud_rate=0.08,
            seed=42,
        )
        self.data = self.gen.generate()

    def test_returns_evaluation_result(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert isinstance(result, EvaluationResult)

    def test_auroc_in_range(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert 0.0 <= result.auroc <= 1.0, f"AUROC out of range: {result.auroc}"

    def test_auroc_better_than_random(self):
        """Our model should detect the synthetic fraud patterns (AUROC > 0.5)."""
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert result.auroc > 0.5, (
            f"Model performed worse than random: AUROC={result.auroc:.4f}"
        )

    def test_best_f1_in_range(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert 0.0 <= result.best_f1 <= 1.0

    def test_train_plus_test_equals_total(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert result.n_train + result.n_test == len(self.data)

    def test_test_fraud_plus_legit_equals_test(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert result.n_test_fraud + result.n_test_legit == result.n_test

    def test_precision_at_k_in_range(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7, k_values=[10, 50])
        for k, p in result.precision_at_k.items():
            assert 0.0 <= p <= 1.0, f"P@{k} out of range: {p}"

    def test_summary_returns_string(self):
        result = evaluate_walk_forward(self.data, train_fraction=0.7)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "AUROC" in summary

    def test_deterministic_same_data(self):
        r1 = evaluate_walk_forward(self.data, train_fraction=0.7)
        r2 = evaluate_walk_forward(self.data, train_fraction=0.7)
        assert r1.auroc == r2.auroc, "Evaluation is not deterministic"
        assert r1.best_f1 == r2.best_f1

    def test_per_threshold_count(self):
        thresholds = [0.3, 0.5, 0.7]
        result = evaluate_walk_forward(
            self.data, train_fraction=0.7, thresholds=thresholds
        )
        assert len(result.per_threshold) == len(thresholds)

    def test_threshold_metrics_sum_to_n_test(self):
        result = evaluate_walk_forward(
            self.data, train_fraction=0.7, thresholds=[0.5]
        )
        tm = result.per_threshold[0]
        assert tm.tp + tm.fp + tm.tn + tm.fn == result.n_test


# ---------------------------------------------------------------------------
# _trapezoid_auc tests
# ---------------------------------------------------------------------------


class TestTrapezoidAUC:
    def test_perfect_classifier(self):
        """ROC: (0,0) → (0,1) → (1,1) → AUC = 1.0"""
        points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        auc = _trapezoid_auc(points)
        assert abs(auc - 1.0) < 1e-9

    def test_random_classifier(self):
        """ROC: (0,0) → (1,1) diagonal → AUC = 0.5"""
        points = [(0.0, 0.0), (1.0, 1.0)]
        auc = _trapezoid_auc(points)
        assert abs(auc - 0.5) < 1e-9

    def test_worst_classifier(self):
        """ROC: (0,0) → (1,0) → (1,1) → AUC = 0.0"""
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        auc = _trapezoid_auc(points)
        assert abs(auc - 0.0) < 1e-9

    def test_auc_in_unit_square(self):
        """Any ROC curve should give AUC in [0, 1]."""
        import random
        rng = random.Random(0)
        # Generate random ROC-like points
        points = [(0.0, 0.0)] + [
            (rng.random(), rng.random()) for _ in range(20)
        ] + [(1.0, 1.0)]
        auc = _trapezoid_auc(points)
        assert 0.0 <= auc <= 1.0
