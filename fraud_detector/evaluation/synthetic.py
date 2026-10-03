"""
fraud_detector/evaluation/synthetic.py
────────────────────────────────────────
Deterministic synthetic dataset generator.

Produces a labelled dataset of TransactionRequest objects with ground-truth
fraud labels, using only stdlib random with a fixed seed — no external deps.

Design:
  • All fraud patterns match real-world fraud archetypes.
  • No label leakage: ground-truth labels are assigned BEFORE any model sees
    the data (labels are properties of the generation process, not derived
    from model scores).
  • Dataset is time-ordered by transaction timestamp (ascending).
  • Generator is deterministic: same seed → same dataset always.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from fraud_detector.models.transaction import Channel, TransactionRequest, TransactionType

# ---------------------------------------------------------------------------
# Synthetic dataset constants
# ---------------------------------------------------------------------------

_UTC = timezone.utc
_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)

_CURRENCIES = ["INR"] * 80 + ["USD"] * 10 + ["EUR"] * 5 + ["GBP"] * 5
_CHANNELS = [
    Channel.ONLINE, Channel.UPI, Channel.POS, Channel.APP,
    Channel.NEFT, Channel.IMPS, Channel.ATM,
]
_CHANNEL_WEIGHTS = [30, 25, 20, 10, 7, 5, 3]

# Merchant pools
_LEGIT_MERCHANTS = [f"mer-legit-{i:03d}" for i in range(50)]
_FRAUD_MERCHANTS = [f"mer-fraud-{i:03d}" for i in range(10)]

# Amount distributions (in paise)
_NORMAL_AMOUNT_MIN = 1_000        # ₹10
_NORMAL_AMOUNT_MAX = 500_000      # ₹5,000
_HIGH_AMOUNT_MIN = 1_000_000      # ₹10,000
_HIGH_AMOUNT_MAX = 50_000_000     # ₹5,00,000


# ---------------------------------------------------------------------------
# Labelled transaction
# ---------------------------------------------------------------------------


@dataclass
class LabelledTransaction:
    """A transaction with a ground-truth fraud label."""
    request: TransactionRequest
    is_fraud: bool
    fraud_type: str   # e.g. "card_testing", "account_takeover", "normal"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generates a labelled synthetic transaction stream.

    Parameters
    ----------
    n_users       : number of distinct user IDs
    n_transactions: total transactions to generate
    fraud_rate    : fraction of transactions that are fraudulent
    seed          : random seed for full reproducibility
    start_time    : UTC timestamp for the first transaction
    """

    def __init__(
        self,
        n_users: int = 200,
        n_transactions: int = 5_000,
        fraud_rate: float = 0.05,
        seed: int = 42,
        start_time: datetime | None = None,
        include_unseen_patterns: bool = False,
    ) -> None:
        self.n_users = n_users
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.rng = random.Random(seed)
        self.start_time = start_time or _EPOCH
        self.include_unseen_patterns = include_unseen_patterns

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _user_id(self, uid: int) -> str:
        return f"usr-{uid:05d}"

    def _tx_id(self, idx: int) -> str:
        return f"txn-{idx:08d}"

    def _normal_amount(self) -> int:
        # Log-normal-like: mostly small, occasionally large
        if self.rng.random() < 0.05:
            return self.rng.randint(_HIGH_AMOUNT_MIN, _HIGH_AMOUNT_MAX)
        return self.rng.randint(_NORMAL_AMOUNT_MIN, _NORMAL_AMOUNT_MAX)

    def _timestamp(self, base: datetime, offset_seconds: int) -> datetime:
        return base + timedelta(seconds=offset_seconds)

    def _random_channel(self) -> Channel:
        return self.rng.choices(_CHANNELS, weights=_CHANNEL_WEIGHTS, k=1)[0]

    def _random_currency(self) -> str:
        return self.rng.choice(_CURRENCIES)

    # ------------------------------------------------------------------
    # Fraud pattern generators
    # ------------------------------------------------------------------

    def _card_testing_burst(
        self, user_id: str, base_ts: datetime, idx: int
    ) -> list[LabelledTransaction]:
        """Card testing: 5-15 micro-transactions within minutes."""
        n = self.rng.randint(5, 15)
        txns = []
        for i in range(n):
            ts = self._timestamp(base_ts, i * 30)  # 30 seconds apart
            req = TransactionRequest(
                transaction_id=self._tx_id(idx + i),
                user_id=user_id,
                merchant_id=self.rng.choice(_FRAUD_MERCHANTS),
                amount=self.rng.randint(1, 99),   # sub-₹1 micro amounts
                currency="INR",
                timestamp=ts,
                channel=Channel.ONLINE,
                transaction_type=TransactionType.PURCHASE,
            )
            txns.append(LabelledTransaction(req, is_fraud=True, fraud_type="card_testing"))
        return txns

    def _account_takeover(
        self, user_id: str, base_ts: datetime, idx: int
    ) -> list[LabelledTransaction]:
        """Account takeover: sudden large transaction at unusual hour."""
        # Pick 2–3 AM UTC (high-risk hour)
        ts = base_ts.replace(hour=self.rng.randint(0, 4), minute=self.rng.randint(0, 59))
        req = TransactionRequest(
            transaction_id=self._tx_id(idx),
            user_id=user_id,
            merchant_id=self.rng.choice(_FRAUD_MERCHANTS),
            amount=self.rng.randint(5_000_000, 50_000_000),  # ₹50k–₹5L
            currency=self.rng.choice(["USD", "EUR", "GBP"]),  # foreign
            timestamp=ts,
            channel=Channel.ONLINE,
            transaction_type=TransactionType.TRANSFER,
        )
        return [LabelledTransaction(req, is_fraud=True, fraud_type="account_takeover")]

    def _velocity_fraud(
        self, user_id: str, base_ts: datetime, idx: int
    ) -> list[LabelledTransaction]:
        """Velocity fraud: many normal-looking txns in one hour."""
        n = self.rng.randint(12, 20)
        txns = []
        for i in range(n):
            ts = self._timestamp(base_ts, i * 120)  # 2 minutes apart
            req = TransactionRequest(
                transaction_id=self._tx_id(idx + i),
                user_id=user_id,
                merchant_id=self.rng.choice(_FRAUD_MERCHANTS),
                amount=self.rng.randint(10_000, 100_000),
                currency="INR",
                timestamp=ts,
                channel=self._random_channel(),
                transaction_type=TransactionType.PURCHASE,
            )
            txns.append(LabelledTransaction(req, is_fraud=True, fraud_type="velocity_fraud"))
        return txns

    def _slow_drip_drain(
        self, user_id: str, base_ts: datetime, idx: int
    ) -> list[LabelledTransaction]:
        """Slow-drip account drain: periodic modest amounts crafted to evade rules."""
        n = self.rng.randint(3, 5)
        txns = []
        for i in range(n):
            ts = self._timestamp(base_ts, i * 4 * 3600)  # 4 hours apart (evades hourly velocity)
            req = TransactionRequest(
                transaction_id=self._tx_id(idx + i),
                user_id=user_id,
                merchant_id="mer-drip-drain",
                amount=self.rng.randint(25_000, 38_000),  # ₹250-₹380 (evades micro & high amount)
                currency="INR",
                timestamp=ts,
                channel=Channel.UPI,
                transaction_type=TransactionType.PURCHASE,
            )
            txns.append(LabelledTransaction(req, is_fraud=True, fraud_type="slow_drip_drain"))
        return txns

    def _merchant_hopping(
        self, user_id: str, base_ts: datetime, idx: int
    ) -> list[LabelledTransaction]:
        """Merchant hopping: bursts across unfamiliar merchant identities."""
        n = self.rng.randint(3, 4)
        txns = []
        for i in range(n):
            ts = self._timestamp(base_ts, i * 15 * 60)  # 15 minutes apart
            req = TransactionRequest(
                transaction_id=self._tx_id(idx + i),
                user_id=user_id,
                merchant_id=f"mer-hop-{idx}-{i}",
                amount=self.rng.randint(30_000, 70_000),  # modest amount
                currency="INR",
                timestamp=ts,
                channel=Channel.ONLINE,
                transaction_type=TransactionType.PURCHASE,
            )
            txns.append(LabelledTransaction(req, is_fraud=True, fraud_type="merchant_hopping"))
        return txns

    def _normal_transaction(
        self, user_id: str, ts: datetime, idx: int
    ) -> LabelledTransaction:
        return LabelledTransaction(
            request=TransactionRequest(
                transaction_id=self._tx_id(idx),
                user_id=user_id,
                merchant_id=self.rng.choice(_LEGIT_MERCHANTS),
                amount=self._normal_amount(),
                currency=self._random_currency(),
                timestamp=ts,
                channel=self._random_channel(),
                transaction_type=TransactionType.PURCHASE,
            ),
            is_fraud=False,
            fraud_type="normal",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> list[LabelledTransaction]:
        """Generate the full dataset, sorted by timestamp (ascending).

        This is deterministic: same constructor args → same output.
        """
        transactions: list[LabelledTransaction] = []
        current_time = self.start_time
        tx_idx = 0

        for _ in range(self.n_transactions):
            # Advance time by 1–600 seconds (average ~5 minutes per tx)
            current_time = self._timestamp(
                current_time, self.rng.randint(1, 600)
            )
            user_id = self._user_id(self.rng.randint(0, self.n_users - 1))

            roll = self.rng.random()
            if self.include_unseen_patterns and roll < self.fraud_rate * 0.25:
                # Unseen pattern 1: Slow drip drain
                burst = self._slow_drip_drain(user_id, current_time, tx_idx)
                transactions.extend(burst)
                tx_idx += len(burst)
                current_time = burst[-1].request.timestamp
            elif self.include_unseen_patterns and roll < self.fraud_rate * 0.50:
                # Unseen pattern 2: Merchant hopping
                burst = self._merchant_hopping(user_id, current_time, tx_idx)
                transactions.extend(burst)
                tx_idx += len(burst)
                current_time = burst[-1].request.timestamp
            elif roll < self.fraud_rate * 0.4:
                # Card testing burst
                burst = self._card_testing_burst(user_id, current_time, tx_idx)
                transactions.extend(burst)
                tx_idx += len(burst)
                current_time = burst[-1].request.timestamp
            elif roll < self.fraud_rate * 0.7:
                # Account takeover
                txns = self._account_takeover(user_id, current_time, tx_idx)
                transactions.extend(txns)
                tx_idx += 1
            elif roll < self.fraud_rate:
                # Velocity fraud
                burst = self._velocity_fraud(user_id, current_time, tx_idx)
                transactions.extend(burst)
                tx_idx += len(burst)
                current_time = burst[-1].request.timestamp
            else:
                # Normal transaction
                txn = self._normal_transaction(user_id, current_time, tx_idx)
                transactions.append(txn)
                tx_idx += 1

        # Sort by timestamp (ascending) — ensures walk-forward correctness
        transactions.sort(key=lambda t: t.request.timestamp)

        # Re-number transaction IDs after sort to guarantee uniqueness
        seen_ids: set[str] = set()
        relabelled = []
        for i, lt in enumerate(transactions):
            new_id = f"txn-{i:08d}"
            # Use model_copy (Pydantic v2) to create a new frozen instance
            new_req = lt.request.model_copy(update={"transaction_id": new_id})
            if new_id not in seen_ids:
                seen_ids.add(new_id)
                relabelled.append(LabelledTransaction(
                    request=new_req,
                    is_fraud=lt.is_fraud,
                    fraud_type=lt.fraud_type,
                ))

        return relabelled
