"""
fraud_detector/engine.py
──────────────────────────
FraudEngine: the single entry point for the entire pipeline.

Flow for each request:
  1. Validate input (already done by Pydantic at API layer, but engine re-checks).
  2. Check idempotency: if transaction_id seen before, return cached result.
  3. Fetch immutable state snapshot (under per-user lock).
  4. Extract features (pure, stateless).
  5. Score (pure, stateless, deterministic).
  6. Commit state update (atomic, idempotent guard inside).
  7. Build and return FraudScore.

Thread safety: steps 2-3 and step 6 each acquire the per-user lock independently.
There is a race window between step 3 (read) and step 6 (write): a concurrent
request may commit between our snapshot and our commit.  This is acceptable
because:
  a) Idempotency guard in commit() prevents double-counting.
  b) The snapshot is already conservative (slightly stale is safer than blocking
     all reads behind a write lock for latency).
  c) Step 2 prevents a duplicate from re-scoring at all.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fraud_detector.features.extractor import extract_features
from fraud_detector.models.transaction import FraudScore, TransactionRequest, utc_now
from fraud_detector.scoring.scorer import score_transaction
from fraud_detector.state.user_state import UserStateStore


class FraudEngine:
    """Top-level orchestrator.  One instance per application."""

    def __init__(self, store: UserStateStore | None = None) -> None:
        self._store = store or UserStateStore()

    @property
    def store(self) -> UserStateStore:
        return self._store

    def evaluate(self, request: TransactionRequest) -> FraudScore:
        """Score a transaction request.

        Idempotent: calling with the same transaction_id returns the cached
        FraudScore without modifying state.
        """
        now = utc_now()

        # ── Step 2: Idempotency check ────────────────────────────────────
        from fraud_detector.state.user_state import compute_payload_hash
        payload_hash = compute_payload_hash(
            request.user_id, request.amount, request.merchant_id, request.currency
        )
        cached = self._store.get_cached_result(
            request.user_id, request.transaction_id, payload_hash=payload_hash
        )
        if cached is not None:
            cached_score, cached_flags = cached
            from fraud_detector.models.transaction import _score_to_risk
            return FraudScore(
                transaction_id=request.transaction_id,
                user_id=request.user_id,
                score=cached_score,
                risk_level=_score_to_risk(cached_score),
                flags=cached_flags,
                is_duplicate=True,
                scored_at=now,
            )

        # ── Step 3: Snapshot (read, under lock) ──────────────────────────
        snapshot = self._store.get_snapshot(request.user_id, request.timestamp)

        # ── Step 4: Feature extraction (pure) ────────────────────────────
        fv = extract_features(request, snapshot)

        # ── Step 5: Score (pure, deterministic) ──────────────────────────
        result = score_transaction(fv)

        # ── Step 6: Commit state (atomic, idempotent) ────────────────────
        self._store.commit(
            user_id=request.user_id,
            transaction_id=request.transaction_id,
            amount_paise=request.amount,
            timestamp=request.timestamp,
            merchant_id=request.merchant_id,
            score=result.score,
            flags=result.flags,
            payload_hash=payload_hash,
            device_id=request.device_id,
            latitude=request.latitude,
            longitude=request.longitude,
        )

        # ── Step 7: Build response ────────────────────────────────────────
        return FraudScore(
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            score=result.score,
            risk_level=result.risk_level,
            flags=result.flags,
            is_duplicate=False,
            scored_at=now,
        )
