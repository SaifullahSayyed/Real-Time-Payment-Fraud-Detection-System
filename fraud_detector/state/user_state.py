"""
fraud_detector/state/user_state.py
────────────────────────────────────
Per-user state store with atomic read-modify-write under concurrent access.

Design:
  • One threading.Lock per user_id (fine-grained locking, no global GIL reliance).
  • State is held in an in-memory dict; easily swappable for Redis.
  • All timestamps stored as UTC-aware datetimes.
  • Out-of-order events are handled: we always merge new data into the running
    statistics rather than assuming monotone timestamps.
  • Duplicates: if a transaction_id is already in `seen_ids`, we return the
    cached score without modifying state.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Tuneable constants (no magic numbers in callers)
# ---------------------------------------------------------------------------

VELOCITY_WINDOW_SECONDS: int = 3_600        # 1 hour velocity window
VELOCITY_WINDOW_DAILY_SECONDS: int = 86_400  # 24 hour velocity window
MAX_RECENT_TX: int = 500                     # rolling buffer per user
MAX_SEEN_IDS: int = 10_000                   # idempotency set per user


# ---------------------------------------------------------------------------
# Per-user state snapshot (immutable view handed to scorer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserStateSnapshot:
    """Immutable snapshot of a user's state at query time.

    The scorer receives this; it never mutates it.
    """

    user_id: str
    total_tx_count: int                         # lifetime transaction count
    total_amount_paise: int                     # lifetime sum of amounts
    hourly_tx_count: int                        # transactions in last 1 h
    hourly_amount_paise: int                    # amount sum in last 1 h
    daily_tx_count: int                         # transactions in last 24 h
    daily_amount_paise: int                     # amount sum in last 24 h

    # Running Welford mean/M2 for incremental variance (amount)
    amount_mean: float
    amount_m2: float                            # sum of squared deviations

    # Set of merchant_ids seen historically (frozen for snapshot)
    known_merchants: frozenset[str]

    # Timestamp of most recent successfully processed transaction
    last_tx_timestamp: Optional[datetime]

    # True if this is the user's very first transaction
    is_cold_start: bool

    # Geo and device signals
    known_devices: frozenset[str] = field(default_factory=frozenset)
    device_count_1h: int = 0
    last_latitude: Optional[float] = None
    last_longitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Mutable per-user state (internal)
# ---------------------------------------------------------------------------


@dataclass
class _UserState:
    """Mutable record kept in the store.  Access is serialised by a lock."""

    total_tx_count: int = 0
    total_amount_paise: int = 0

    # Rolling deque of (timestamp, amount_paise) for velocity windows
    recent_tx: deque = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_TX)
    )

    # Welford online algorithm state for incremental mean/variance
    amount_mean: float = 0.0
    amount_m2: float = 0.0       # sum of squared deviations from mean

    # Set of merchant_ids this user has transacted with before
    known_merchants: set = field(default_factory=set)

    # Timestamp of the most recent processed transaction
    last_tx_timestamp: Optional[datetime] = None

    # Idempotency: maps transaction_id → (score_float, flags)
    seen_ids: dict = field(default_factory=dict)

    # Geo and device state
    known_devices: set = field(default_factory=set)
    recent_devices: deque = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_TX)
    )
    last_latitude: Optional[float] = None
    last_longitude: Optional[float] = None


# ---------------------------------------------------------------------------
class IdempotencyConflictError(ValueError):
    """Raised when a transaction_id is resubmitted with conflicting payload fields."""
    pass


import hashlib


def compute_payload_hash(
    user_id: str, amount_paise: int, merchant_id: str, currency: str = ""
) -> str:
    """Compute deterministic cryptographic fingerprint of transaction payload."""
    key = f"{user_id}|{amount_paise}|{merchant_id}|{currency}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Thread-safe store
# ---------------------------------------------------------------------------


class UserStateStore:
    """In-memory, thread-safe per-user state store.

    Thread safety model:
      - A dict of per-user locks; the dict itself is protected by a
        lightweight meta-lock only for lock *creation* (one-time per user).
      - All reads and writes to a user's state go through that user's lock,
        so two concurrent requests for different users never contend.
      - Memory bound: capacity capped at max_users with LRU eviction.
    """

    def __init__(self, max_users: int = 100_000) -> None:
        self._max_users = max_users
        self._states: dict[str, _UserState] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lock(self, user_id: str) -> threading.Lock:
        """Return (and lazily create) the per-user lock."""
        # Fast path: lock already exists, no meta-lock needed.
        if user_id in self._locks:
            return self._locks[user_id]
        # Slow path: first access for this user.
        with self._meta_lock:
            if user_id not in self._locks:
                self._locks[user_id] = threading.Lock()
            return self._locks[user_id]

    def _get_or_create_state(self, user_id: str) -> _UserState:
        """Must be called while holding the per-user lock."""
        if user_id not in self._states:
            with self._meta_lock:
                if len(self._states) >= self._max_users and user_id not in self._states:
                    oldest_uid = next(iter(self._states))
                    del self._states[oldest_uid]
                    self._locks.pop(oldest_uid, None)
                self._states[user_id] = _UserState()
        return self._states[user_id]

    @staticmethod
    def _velocity(
        recent_tx: deque,
        now: datetime,
        window_seconds: int,
    ) -> tuple[int, int]:
        """Count transactions and sum amounts within the time window.

        Handles out-of-order events: scans the full buffer.
        Only counts transactions in [now - window_seconds, now], strictly
        excluding any future timestamps.
        """
        cutoff = now.timestamp() - window_seconds
        now_ts = now.timestamp()
        count = 0
        total = 0
        for ts, amount in recent_tx:
            t = ts.timestamp()
            if cutoff <= t <= now_ts:
                count += 1
                total += amount
        return count, total

    @staticmethod
    def _device_velocity(
        recent_devices: deque,
        now: datetime,
        window_seconds: int,
    ) -> int:
        """Count distinct device_ids used in the time window [now - window_seconds, now]."""
        cutoff = now.timestamp() - window_seconds
        now_ts = now.timestamp()
        seen = set()
        for ts, dev in recent_devices:
            t = ts.timestamp()
            if cutoff <= t <= now_ts:
                seen.add(dev)
        return len(seen)

    @staticmethod
    def _welford_update(
        count: int, mean: float, m2: float, value: float
    ) -> tuple[float, float]:
        """One step of Welford's online mean/variance algorithm.

        Returns (new_mean, new_m2).
        """
        count += 1  # count after this update
        delta = value - mean
        new_mean = mean + delta / count
        delta2 = value - new_mean
        new_m2 = m2 + delta * delta2
        return new_mean, new_m2

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_snapshot(self, user_id: str, reference_time: datetime) -> UserStateSnapshot:
        """Return an immutable state snapshot for the scorer.

        Does NOT modify state.  Safe to call concurrently.
        """
        lock = self._get_lock(user_id)
        with lock:
            st = self._get_or_create_state(user_id)
            hourly_count, hourly_amount = self._velocity(
                st.recent_tx, reference_time, VELOCITY_WINDOW_SECONDS
            )
            daily_count, daily_amount = self._velocity(
                st.recent_tx, reference_time, VELOCITY_WINDOW_DAILY_SECONDS
            )
            dev_1h = self._device_velocity(
                st.recent_devices, reference_time, VELOCITY_WINDOW_SECONDS
            )
            return UserStateSnapshot(
                user_id=user_id,
                total_tx_count=st.total_tx_count,
                total_amount_paise=st.total_amount_paise,
                hourly_tx_count=hourly_count,
                hourly_amount_paise=hourly_amount,
                daily_tx_count=daily_count,
                daily_amount_paise=daily_amount,
                amount_mean=st.amount_mean,
                amount_m2=st.amount_m2,
                known_merchants=frozenset(st.known_merchants),
                last_tx_timestamp=st.last_tx_timestamp,
                is_cold_start=(st.total_tx_count == 0),
                known_devices=frozenset(st.known_devices),
                device_count_1h=dev_1h,
                last_latitude=st.last_latitude,
                last_longitude=st.last_longitude,
            )

    def is_duplicate(self, user_id: str, transaction_id: str) -> bool:
        """Check idempotency without modifying state."""
        lock = self._get_lock(user_id)
        with lock:
            st = self._get_or_create_state(user_id)
            return transaction_id in st.seen_ids

    def get_cached_result(
        self, user_id: str, transaction_id: str, payload_hash: Optional[str] = None
    ) -> Optional[tuple[float, list[str]]]:
        """Return cached (score, flags) for a duplicate transaction, or None.

        Raises IdempotencyConflictError if payload_hash is provided and does
        not match the recorded payload for this transaction_id.
        """
        lock = self._get_lock(user_id)
        with lock:
            st = self._get_or_create_state(user_id)
            entry = st.seen_ids.get(transaction_id)
            if entry is None:
                return None
            if len(entry) >= 3 and payload_hash is not None:
                score, flags, stored_hash = entry[0], entry[1], entry[2]
                if stored_hash != payload_hash:
                    raise IdempotencyConflictError(
                        f"transaction_id {transaction_id!r} was previously submitted with a different payload"
                    )
                return score, flags
            return entry[0], entry[1]

    def commit(
        self,
        user_id: str,
        transaction_id: str,
        amount_paise: int,
        timestamp: datetime,
        merchant_id: str,
        score: float,
        flags: list[str],
        payload_hash: Optional[str] = None,
        device_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> None:
        """Atomically update state after a successful scoring.

        Idempotent: if transaction_id is already recorded, this is a no-op.
        """
        lock = self._get_lock(user_id)
        with lock:
            st = self._get_or_create_state(user_id)

            # Idempotency guard inside the lock
            if transaction_id in st.seen_ids:
                return

            # Update counters
            st.total_tx_count += 1
            st.total_amount_paise += amount_paise

            # Rolling velocity buffer
            st.recent_tx.append((timestamp, amount_paise))

            # Device tracking
            if device_id:
                st.known_devices.add(device_id)
                st.recent_devices.append((timestamp, device_id))

            # Geo tracking
            if latitude is not None and longitude is not None:
                if st.last_tx_timestamp is None or timestamp >= st.last_tx_timestamp:
                    st.last_latitude = latitude
                    st.last_longitude = longitude

            # Welford update (use count BEFORE this transaction)
            new_mean, new_m2 = self._welford_update(
                st.total_tx_count - 1,  # count prior to this update
                st.amount_mean,
                st.amount_m2,
                float(amount_paise),
            )
            st.amount_mean = new_mean
            st.amount_m2 = new_m2

            # Merchant history
            st.known_merchants.add(merchant_id)

            # Last timestamp (we track the actual wall-clock of the event,
            # not the commit order, to handle out-of-order correctly)
            if (
                st.last_tx_timestamp is None
                or timestamp > st.last_tx_timestamp
            ):
                st.last_tx_timestamp = timestamp

            # Compute payload hash if not provided
            if payload_hash is None:
                payload_hash = compute_payload_hash(user_id, amount_paise, merchant_id)

            # Record idempotency entry (evict oldest if over limit)
            if len(st.seen_ids) >= MAX_SEEN_IDS:
                # Remove the first-inserted key (dict is ordered in Py 3.7+)
                oldest_key = next(iter(st.seen_ids))
                del st.seen_ids[oldest_key]
            st.seen_ids[transaction_id] = (score, list(flags), payload_hash)

    def reset_user(self, user_id: str) -> None:
        """Reset state for a user.  Used in tests only."""
        lock = self._get_lock(user_id)
        with lock:
            self._states[user_id] = _UserState()

    def user_count(self) -> int:
        """Number of users currently in the store."""
        with self._meta_lock:
            return len(self._states)
