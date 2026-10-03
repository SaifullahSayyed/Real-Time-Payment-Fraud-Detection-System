"""
fraud_detector/evaluation/walk_forward.py
───────────────────────────────────────────
Walk-forward (time-based) evaluation for the fraud detection engine.

Rules to prevent leakage:
  1. Data is sorted by timestamp BEFORE splitting.
  2. The train window contains all transactions up to the split point.
  3. The test window starts strictly AFTER the split point — never overlaps.
  4. The engine is evaluated on the test window in chronological order.
  5. After each test transaction, state IS updated (online learning setting).
  6. No random shuffling at any point.

Metrics:
  • Precision, Recall, F1 at each configurable score threshold.
  • AUROC (Area Under ROC) computed with trapezoidal integration — stdlib only.
  • Precision@K (top-K transactions by score).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from fraud_detector.engine import FraudEngine
from fraud_detector.evaluation.synthetic import LabelledTransaction
from fraud_detector.state.user_state import UserStateStore


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False positive rate = FP / (FP + TN)."""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0


@dataclass
class EvaluationResult:
    n_train: int
    n_test: int
    n_test_fraud: int
    n_test_legit: int

    # Per-threshold metrics (one entry per threshold in the sweep)
    per_threshold: list[ThresholdMetrics] = field(default_factory=list)

    # AUROC computed from the threshold sweep
    auroc: float = 0.0

    # Precision@K
    precision_at_k: dict[int, float] = field(default_factory=dict)

    # Best threshold by F1
    best_threshold: float = 0.5
    best_f1: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Walk-forward Evaluation Summary",
            f"  Train size : {self.n_train:,}",
            f"  Test  size : {self.n_test:,}  "
            f"(fraud={self.n_test_fraud}, legit={self.n_test_legit})",
            f"  AUROC      : {self.auroc:.4f}",
            f"  Best F1    : {self.best_f1:.4f}  @ threshold={self.best_threshold:.2f}",
        ]
        for k, p in sorted(self.precision_at_k.items()):
            lines.append(f"  P@{k:<4}     : {p:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------


def walk_forward_split(
    data: list[LabelledTransaction],
    train_fraction: float = 0.7,
) -> tuple[list[LabelledTransaction], list[LabelledTransaction]]:
    """Split a time-sorted dataset into train and test windows.

    The split is purely positional (time-based).  No random shuffling.

    Returns (train, test) where:
      - train = first `train_fraction` of rows (by time)
      - test  = remaining rows (strictly later timestamps)
    """
    if not data:
        return [], []
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    split_idx = int(len(data) * train_fraction)
    train = data[:split_idx]
    test = data[split_idx:]

    # Verify no timestamp leakage
    if train and test:
        train_max_ts = max(lt.request.timestamp for lt in train)
        test_min_ts = min(lt.request.timestamp for lt in test)
        if test_min_ts < train_max_ts:
            raise RuntimeError(
                f"LEAKAGE DETECTED: test set contains timestamps "
                f"({test_min_ts}) before train max ({train_max_ts}). "
                f"Ensure data is sorted by timestamp before splitting."
            )

    return train, test


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS = [i / 100 for i in range(1, 100)]   # 0.01 … 0.99
_DEFAULT_K_VALUES = [10, 50, 100]


def evaluate_walk_forward(
    data: list[LabelledTransaction],
    train_fraction: float = 0.7,
    thresholds: Sequence[float] | None = None,
    k_values: Sequence[int] | None = None,
) -> EvaluationResult:
    """
    Walk-forward evaluation of the FraudEngine on a labelled dataset.

    Process:
      1. Sort by timestamp.
      2. Split: train / test (time-based, no leakage).
      3. Warm-up: run the engine over the train set (builds per-user state).
         Ground-truth labels are NOT fed back to the engine — it only sees
         transactions (as in production).
      4. Evaluate: run the engine over the test set in chronological order.
         After each test transaction, state IS committed (online setting).
      5. Compute metrics: AUROC, precision/recall/F1 at each threshold,
         precision@K.
    """
    if thresholds is None:
        thresholds = _DEFAULT_THRESHOLDS
    if k_values is None:
        k_values = _DEFAULT_K_VALUES

    # Step 1: sort by timestamp
    data_sorted = sorted(data, key=lambda lt: lt.request.timestamp)

    # Step 2: split
    train_data, test_data = walk_forward_split(data_sorted, train_fraction)

    # Step 3: warm-up (train set)
    store = UserStateStore()
    engine = FraudEngine(store=store)

    for lt in train_data:
        engine.evaluate(lt.request)  # commits state; score ignored during warm-up

    # Step 4: evaluate on test set
    scored: list[tuple[float, bool]] = []  # (model_score, is_fraud)

    for lt in test_data:
        result = engine.evaluate(lt.request)
        scored.append((result.score, lt.is_fraud))

    # Step 5: metrics
    n_fraud = sum(1 for _, label in scored if label)
    n_legit = len(scored) - n_fraud

    # Per-threshold metrics
    threshold_metrics = []
    for thresh in thresholds:
        tm = ThresholdMetrics(threshold=thresh)
        for score, label in scored:
            predicted_fraud = score >= thresh
            if predicted_fraud and label:
                tm.tp += 1
            elif predicted_fraud and not label:
                tm.fp += 1
            elif not predicted_fraud and label:
                tm.fn += 1
            else:
                tm.tn += 1
        threshold_metrics.append(tm)

    # AUROC via trapezoidal rule on (FPR, TPR) pairs from threshold sweep
    # Sort by FPR ascending
    roc_points = sorted(
        [(tm.fpr, tm.recall) for tm in threshold_metrics],
        key=lambda p: p[0],
    )
    # Add (0, 0) and (1, 1) anchor points
    roc_points = [(0.0, 0.0)] + roc_points + [(1.0, 1.0)]
    auroc = _trapezoid_auc(roc_points)

    # Best F1
    best_f1 = -1.0
    best_thresh = 0.5
    for tm in threshold_metrics:
        if tm.f1 > best_f1:
            best_f1 = tm.f1
            best_thresh = tm.threshold

    # Precision@K
    scored_desc = sorted(scored, key=lambda x: x[0], reverse=True)
    prec_at_k = {}
    for k in k_values:
        top_k = scored_desc[:k]
        if top_k:
            prec_at_k[k] = sum(1 for _, label in top_k if label) / len(top_k)
        else:
            prec_at_k[k] = 0.0

    return EvaluationResult(
        n_train=len(train_data),
        n_test=len(test_data),
        n_test_fraud=n_fraud,
        n_test_legit=n_legit,
        per_threshold=threshold_metrics,
        auroc=auroc,
        precision_at_k=prec_at_k,
        best_threshold=best_thresh,
        best_f1=best_f1,
    )


def _trapezoid_auc(points: list[tuple[float, float]]) -> float:
    """Compute AUC using the trapezoidal rule. Points need not be sorted."""
    pts = sorted(points, key=lambda p: p[0])
    area = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


@dataclass
class UnseenEvaluationComparison:
    n_train: int
    n_test: int
    n_unseen_fraud: int
    detection_rate_rules_only: float
    detection_rate_rules_plus_forest: float
    fpr_rules_only: float
    fpr_rules_plus_forest: float
    threshold: float


def evaluate_rules_vs_forest_walk_forward(
    data: list[LabelledTransaction],
    train_fraction: float = 0.7,
    threshold: float = 0.35,
    refit_interval: int = 150,
) -> UnseenEvaluationComparison:
    """Compare Rules-only vs Rules+Forest detection and FPR on unseen fraud patterns."""
    from fraud_detector.features.extractor import extract_features
    from fraud_detector.scoring.isolation_forest import IsolationForestScorer
    from fraud_detector.scoring.scorer import score_transaction

    data_sorted = sorted(data, key=lambda lt: lt.request.timestamp)
    train_data, test_data = walk_forward_split(data_sorted, train_fraction)

    store = UserStateStore()
    forest = IsolationForestScorer(random_state=42)

    # Warm-up train set
    legit_train_fvs = []
    for lt in train_data:
        snap = store.get_snapshot(lt.request.user_id, lt.request.timestamp)
        fv = extract_features(lt.request, snap)
        if not lt.is_fraud:
            legit_train_fvs.append(fv)
        store.commit(
            user_id=lt.request.user_id,
            transaction_id=lt.request.transaction_id,
            amount_paise=lt.request.amount,
            timestamp=lt.request.timestamp,
            merchant_id=lt.request.merchant_id,
            score=0.1,
            flags=[],
            device_id=lt.request.device_id,
            latitude=lt.request.latitude,
            longitude=lt.request.longitude,
        )

    # Initial fit of Isolation Forest on legitimate historical baseline
    forest.fit(legit_train_fvs)

    # Online test evaluation
    unseen_types = {"slow_drip_drain", "merchant_hopping"}
    unseen_total = 0
    unseen_detected_rules = 0
    unseen_detected_forest = 0

    legit_total = 0
    legit_fp_rules = 0
    legit_fp_forest = 0

    rolling_legit = list(legit_train_fvs)
    legit_counter = 0

    for lt in test_data:
        snap = store.get_snapshot(lt.request.user_id, lt.request.timestamp)
        fv = extract_features(lt.request, snap)
        rule_res = score_transaction(fv)
        combined_score, _ = forest.combine(rule_res.score, fv)

        store.commit(
            user_id=lt.request.user_id,
            transaction_id=lt.request.transaction_id,
            amount_paise=lt.request.amount,
            timestamp=lt.request.timestamp,
            merchant_id=lt.request.merchant_id,
            score=combined_score,
            flags=rule_res.flags,
            device_id=lt.request.device_id,
            latitude=lt.request.latitude,
            longitude=lt.request.longitude,
        )

        is_unseen = lt.is_fraud and lt.fraud_type in unseen_types
        if is_unseen:
            unseen_total += 1
            if rule_res.score >= threshold:
                unseen_detected_rules += 1
            if combined_score >= threshold:
                unseen_detected_forest += 1
        elif not lt.is_fraud:
            legit_total += 1
            if rule_res.score >= threshold:
                legit_fp_rules += 1
            if combined_score >= threshold:
                legit_fp_forest += 1

            # Online model updating on verified clean events
            rolling_legit.append(fv)
            legit_counter += 1
            if legit_counter % refit_interval == 0:
                forest.schedule_refit(rolling_legit[-400:])

    dr_rules = (unseen_detected_rules / unseen_total) if unseen_total > 0 else 0.0
    dr_forest = (unseen_detected_forest / unseen_total) if unseen_total > 0 else 0.0
    fpr_rules = (legit_fp_rules / legit_total) if legit_total > 0 else 0.0
    fpr_forest = (legit_fp_forest / legit_total) if legit_total > 0 else 0.0

    return UnseenEvaluationComparison(
        n_train=len(train_data),
        n_test=len(test_data),
        n_unseen_fraud=unseen_total,
        detection_rate_rules_only=dr_rules,
        detection_rate_rules_plus_forest=dr_forest,
        fpr_rules_only=fpr_rules,
        fpr_rules_plus_forest=fpr_forest,
        threshold=threshold,
    )
