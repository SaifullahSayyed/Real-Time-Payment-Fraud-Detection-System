"""
tests/test_slice4_state_engine.py
───────────────────────────────────
Slice 4 tests: UserStateStore + FraudEngine (integration).

Coverage:
  UserStateStore:
    ✓ Fresh user: cold-start snapshot
    ✓ Commit updates tx count
    ✓ Commit updates amount statistics
    ✓ Commit registers merchant
    ✓ Commit records seen_ids
    ✓ Duplicate commit is a no-op (idempotent)
    ✓ get_cached_result returns stored values
    ✓ is_duplicate returns True after commit, False before
    ✓ Velocity: only counts transactions inside the window
    ✓ Out-of-order events: last_tx_timestamp tracks most recent (not insertion order)
    ✓ Idempotency set eviction when > MAX_SEEN_IDS (oldest evicted)
    ✓ Two users are independent (no cross-contamination)
    ✓ Concurrent commits: final count correct under threading (no lost updates)
    ✓ reset_user clears state

  FraudEngine:
    ✓ First call: is_duplicate=False
    ✓ Second call same tx_id: is_duplicate=True
    ✓ Idempotent: score same on duplicate
    ✓ Score stored in state after first call
    ✓ Cold-start user scores higher than established user (all else equal)
    ✓ Engine uses per-user snapshot (users independent)
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from fraud_detector.engine import FraudEngine
from fraud_detector.models.transaction import TransactionRequest
from fraud_detector.state.user_state import (
    MAX_SEEN_IDS,
    UserStateStore,
)

_UTC = timezone.utc
_T0 = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)


def _ts(offset_seconds: int = 0) -> datetime:
    return _T0 + timedelta(seconds=offset_seconds)


def _req(
    tx_id: str = "txn-001",
    user_id: str = "usr-001",
    merchant_id: str = "mer-001",
    amount: int = 50_000,
    timestamp: datetime | None = None,
) -> TransactionRequest:
    return TransactionRequest(
        transaction_id=tx_id,
        user_id=user_id,
        merchant_id=merchant_id,
        amount=amount,
        timestamp=timestamp or _T0,
    )


# ---------------------------------------------------------------------------
# UserStateStore unit tests
# ---------------------------------------------------------------------------


class TestUserStateStore:
    def setup_method(self):
        self.store = UserStateStore()

    def _commit(self, tx_id="txn-001", user_id="usr-001", amount=50_000,
                ts=None, merchant="mer-001", score=0.1, flags=None):
        self.store.commit(
            user_id=user_id,
            transaction_id=tx_id,
            amount_paise=amount,
            timestamp=ts or _T0,
            merchant_id=merchant,
            score=score,
            flags=flags or [],
        )

    def test_fresh_user_cold_start(self):
        snap = self.store.get_snapshot("new-user", _T0)
        assert snap.is_cold_start is True
        assert snap.total_tx_count == 0

    def test_commit_increments_tx_count(self):
        self._commit()
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.total_tx_count == 1

    def test_commit_updates_amount_stats(self):
        self._commit(amount=100_000)
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.total_amount_paise == 100_000
        assert snap.amount_mean == 100_000.0

    def test_welford_mean_two_transactions(self):
        """Two transactions of 100k and 200k → mean = 150k."""
        self._commit(tx_id="txn-001", amount=100_000)
        self._commit(tx_id="txn-002", amount=200_000)
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.total_tx_count == 2
        assert abs(snap.amount_mean - 150_000.0) < 1.0

    def test_commit_registers_merchant(self):
        self._commit(merchant="mer-A")
        snap = self.store.get_snapshot("usr-001", _T0)
        assert "mer-A" in snap.known_merchants

    def test_duplicate_commit_is_noop(self):
        self._commit(tx_id="txn-dup")
        self._commit(tx_id="txn-dup")  # second call — should not double-count
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.total_tx_count == 1  # not 2

    def test_get_cached_result_returns_stored(self):
        self._commit(tx_id="txn-001", score=0.42, flags=["TEST_FLAG"])
        result = self.store.get_cached_result("usr-001", "txn-001")
        assert result is not None
        score, flags = result
        assert abs(score - 0.42) < 1e-9
        assert "TEST_FLAG" in flags

    def test_is_duplicate_false_before_commit(self):
        assert self.store.is_duplicate("usr-001", "txn-001") is False

    def test_is_duplicate_true_after_commit(self):
        self._commit(tx_id="txn-001")
        assert self.store.is_duplicate("usr-001", "txn-001") is True

    def test_velocity_window(self):
        """Only tx within the velocity window should count."""
        # One old transaction (2 hours ago) and one recent (30 min ago)
        old_ts = _T0 - timedelta(hours=2)
        recent_ts = _T0 - timedelta(minutes=30)

        self._commit(tx_id="old", ts=old_ts, amount=10_000)
        self._commit(tx_id="recent", ts=recent_ts, amount=20_000)

        # Reference time = _T0, window = 1 hour
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.hourly_tx_count == 1, f"Got {snap.hourly_tx_count}"
        assert snap.hourly_amount_paise == 20_000

    def test_out_of_order_last_timestamp(self):
        """last_tx_timestamp should be the most recent event time, not insertion order."""
        earlier = _T0 - timedelta(hours=1)
        later = _T0 + timedelta(hours=1)

        # Insert LATER first, then EARLIER
        self._commit(tx_id="txn-later", ts=later)
        self._commit(tx_id="txn-earlier", ts=earlier)

        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.last_tx_timestamp == later

    def test_two_users_independent(self):
        self._commit(tx_id="txn-A", user_id="user-A", amount=50_000)
        self._commit(tx_id="txn-B", user_id="user-B", amount=999_999)

        snap_a = self.store.get_snapshot("user-A", _T0)
        snap_b = self.store.get_snapshot("user-B", _T0)

        assert snap_a.total_tx_count == 1
        assert snap_b.total_tx_count == 1
        assert snap_a.amount_mean != snap_b.amount_mean

    def test_reset_user_clears_state(self):
        self._commit()
        self.store.reset_user("usr-001")
        snap = self.store.get_snapshot("usr-001", _T0)
        assert snap.is_cold_start is True
        assert snap.total_tx_count == 0

    def test_idempotency_eviction_lru_style(self):
        """When seen_ids exceeds MAX_SEEN_IDS, oldest key is evicted."""
        store = UserStateStore()
        # Fill up to the limit
        for i in range(MAX_SEEN_IDS):
            store.commit(
                user_id="evict-user",
                transaction_id=f"tx-{i:06d}",
                amount_paise=1000,
                timestamp=_T0,
                merchant_id="mer-001",
                score=0.1,
                flags=[],
            )
        # Confirm first tx is still there
        assert store.is_duplicate("evict-user", "tx-000000") is True

        # One more — should evict tx-000000
        store.commit(
            user_id="evict-user",
            transaction_id="tx-overflow",
            amount_paise=1000,
            timestamp=_T0,
            merchant_id="mer-001",
            score=0.1,
            flags=[],
        )
        assert store.is_duplicate("evict-user", "tx-000000") is False
        assert store.is_duplicate("evict-user", "tx-overflow") is True

    def test_concurrent_commits_no_lost_updates(self):
        """50 threads each commit a unique tx → count should be exactly 50."""
        n_threads = 50
        errors = []

        def worker(i: int):
            try:
                self.store.commit(
                    user_id="concurrent-user",
                    transaction_id=f"tx-{i:04d}",
                    amount_paise=1000 * (i + 1),
                    timestamp=_T0 + timedelta(seconds=i),
                    merchant_id=f"mer-{i % 5}",
                    score=0.1,
                    flags=[],
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        snap = self.store.get_snapshot("concurrent-user", _T0 + timedelta(seconds=n_threads))
        assert snap.total_tx_count == n_threads


# ---------------------------------------------------------------------------
# FraudEngine integration tests
# ---------------------------------------------------------------------------


class TestFraudEngine:
    def setup_method(self):
        self.engine = FraudEngine()

    def test_first_call_not_duplicate(self):
        result = self.engine.evaluate(_req())
        assert result.is_duplicate is False

    def test_second_call_same_id_is_duplicate(self):
        self.engine.evaluate(_req(tx_id="txn-dup"))
        result2 = self.engine.evaluate(_req(tx_id="txn-dup"))
        assert result2.is_duplicate is True

    def test_duplicate_returns_same_score(self):
        r1 = self.engine.evaluate(_req(tx_id="txn-idem"))
        r2 = self.engine.evaluate(_req(tx_id="txn-idem"))
        assert r1.score == r2.score
        assert r1.risk_level == r2.risk_level

    def test_score_stored_after_first_call(self):
        r1 = self.engine.evaluate(_req(tx_id="txn-stored"))
        cached = self.engine.store.get_cached_result("usr-001", "txn-stored")
        assert cached is not None
        assert abs(cached[0] - r1.score) < 1e-9

    def test_different_users_independent(self):
        r_a = self.engine.evaluate(_req(tx_id="txn-A", user_id="user-alpha"))
        r_b = self.engine.evaluate(_req(tx_id="txn-B", user_id="user-beta"))
        # Both are cold-start, same amount — scores should be equal
        assert r_a.score == r_b.score

    def test_fraud_score_fields_valid(self):
        result = self.engine.evaluate(_req())
        assert result.transaction_id == "txn-001"
        assert result.user_id == "usr-001"
        assert 0.0 <= result.score <= 1.0
        assert result.scored_at.tzinfo is not None
        assert result.scored_at.utcoffset().total_seconds() == 0

    def test_established_user_scores_lower_than_cold_start(self):
        """After 30 normal transactions (spread across days), a new tx should
        score lower than a cold-start user's first transaction.

        We space history 2 hours apart so hourly velocity never triggers —
        isolating the cold-start vs. established-user discount signal.
        """
        engine = FraudEngine()
        # Build history spread over 30 * 2h = 60 hours (well outside 1-hour window)
        for i in range(30):
            engine.evaluate(_req(
                tx_id=f"txn-hist-{i}",
                user_id="user-established",
                amount=50_000,
                timestamp=_ts(i * 7200),   # 2 hours apart each
                merchant_id=f"mer-{i % 6}",  # 6 distinct merchants → discount fires
            ))
        # Cold-start user: first ever transaction
        cold_result = engine.evaluate(_req(tx_id="txn-cold", user_id="user-cold-new"))
        # Established user: 31st transaction, same amount, no velocity spike
        established_result = engine.evaluate(_req(
            tx_id="txn-new",
            user_id="user-established",
            amount=50_000,
            timestamp=_ts(30 * 7200 + 1),
        ))
        assert established_result.score < cold_result.score, (
            f"Established ({established_result.score:.4f}) should score lower "
            f"than cold-start ({cold_result.score:.4f})\n"
            f"  established flags: {established_result.flags}\n"
            f"  cold_start flags: {cold_result.flags}"
        )

    def test_high_velocity_detected(self):
        """10 rapid transactions in 1 hour should trigger high velocity flag."""
        engine = FraudEngine()
        for i in range(10):
            engine.evaluate(_req(
                tx_id=f"txn-vel-{i}",
                user_id="user-vel",
                timestamp=_ts(i * 60),  # 1 minute apart
            ))
        result = engine.evaluate(_req(
            tx_id="txn-vel-final",
            user_id="user-vel",
            timestamp=_ts(11 * 60),
        ))
        velocity_flags = [f for f in result.flags if "VELOCITY" in f]
        assert len(velocity_flags) > 0, f"Expected velocity flag, got: {result.flags}"
