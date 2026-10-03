"""
tests/test_suspected_issues.py
───────────────────────────────
Adversarial attack reproduction suite for suspected vulnerabilities:
  (a) same transaction_id with a different payload returns cached score instead of 409/422
  (b) far-future and epoch-zero timestamps alter velocity behavior for later normal events
  (c) unbounded growth of per-user state under many distinct user_ids
  (d) oversized request bodies
  (e) no rate limiting
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.engine import FraudEngine
from fraud_detector.models.transaction import TransactionRequest
from fraud_detector.state.user_state import UserStateStore


_UTC = timezone.utc


# ---------------------------------------------------------------------------
# (a) Same transaction_id with different payload
# ---------------------------------------------------------------------------
def test_issue_a_same_transaction_id_different_payload():
    """An attacker reuses an existing transaction_id with altered amount or user.

    Expected: The system must detect payload conflict and reject with 409 Conflict
    or 422 Unprocessable Entity, NEVER silently return the cached score.
    """
    app = create_app(engine=FraudEngine())
    with TestClient(app) as client:
        payload1 = {
            "transaction_id": "txn-reused-id",
            "user_id": "usr-victim",
            "merchant_id": "mer-001",
            "amount": 1000,
            "currency": "INR",
            "timestamp": "2024-06-15T10:00:00Z",
            "channel": "ONLINE",
            "transaction_type": "PURCHASE",
        }
        res1 = client.post("/v1/score", json=payload1)
        assert res1.status_code == 200
        original_score = res1.json()["score"]

        # Attacker tries to submit a 10 Lakh transaction under the same ID
        payload2 = dict(payload1)
        payload2["amount"] = 100_000_000  # Altered from 1000 to 1 Cr paise!

        res2 = client.post("/v1/score", json=payload2)
        assert res2.status_code in (409, 422), (
            f"Expected 409 Conflict or 422 Unprocessable Entity when transaction_id payload differs, "
            f"but got {res2.status_code} with body: {res2.text}"
        )


# ---------------------------------------------------------------------------
# (b) Far-future and epoch-zero timestamps
# ---------------------------------------------------------------------------
def test_issue_b_far_future_and_epoch_zero_timestamps():
    """Suspected issue:
    1. Far-future (e.g. year 2099) and epoch-zero (1970) timestamps should be rejected
       or bounded so they do not distort velocity calculations for subsequent normal events.
    2. A future event must NOT count as part of 'past 1 hour' velocity for a normal event at t0.
    """
    store = UserStateStore()
    engine = FraudEngine(store=store)

    # 1. Validation rejection of extreme timestamps (epoch zero or absurd future)
    with pytest.raises(Exception):
        # Epoch zero timestamp should be rejected during request validation
        TransactionRequest(
            transaction_id="txn-epoch-zero",
            user_id="usr-time-traveler",
            merchant_id="mer-001",
            amount=50000,
            timestamp=datetime(1970, 1, 1, 0, 0, 0, tzinfo=_UTC),
        )

    with pytest.raises(Exception):
        # Far-future timestamp should be rejected during request validation
        TransactionRequest(
            transaction_id="txn-far-future",
            user_id="usr-time-traveler",
            merchant_id="mer-001",
            amount=50000,
            timestamp=datetime(2099, 1, 1, 0, 0, 0, tzinfo=_UTC),
        )

    # 2. Velocity window integrity: even if stored, a future event (t > t_ref)
    # must NOT count in a normal event's 1-hour window [t_ref - 1h, t_ref].
    store.commit(
        user_id="usr-velocity-test",
        transaction_id="txn-future",
        amount_paise=50000,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=_UTC),
        merchant_id="mer-001",
        score=0.1,
        flags=[],
    )

    # Query snapshot for normal event at 2024-06-15 10:00:00 UTC
    normal_t0 = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)
    snap = store.get_snapshot("usr-velocity-test", reference_time=normal_t0)
    assert snap.hourly_tx_count == 0, (
        f"Velocity leakage! Future event at 2025 was counted in 2024 hourly window: "
        f"hourly_tx_count={snap.hourly_tx_count}"
    )


# ---------------------------------------------------------------------------
# (c) Unbounded growth of per-user state under many distinct user_ids
# ---------------------------------------------------------------------------
def test_issue_c_unbounded_user_state_growth():
    """An attacker floods the system with millions of unique user_ids.
    
    UserStateStore must enforce a maximum user capacity bound (MAX_USERS)
    and evict least recently active users rather than leaking memory unboundedly.
    """
    # Check that UserStateStore has a bounded capacity (e.g. MAX_USERS limit)
    # and prunes/evicts inactive users once capacity is reached.
    store = UserStateStore(max_users=50)
    t0 = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)

    # Insert 60 distinct users
    for i in range(60):
        store.commit(
            user_id=f"user-{i:05d}",
            transaction_id=f"txn-{i:05d}",
            amount_paise=1000,
            timestamp=t0,
            merchant_id="mer-001",
            score=0.1,
            flags=[],
        )

    assert store.user_count() <= 50, (
        f"Unbounded user state growth! User count {store.user_count()} exceeded max_users 50"
    )


# ---------------------------------------------------------------------------
# (d) Oversized request bodies
# ---------------------------------------------------------------------------
def test_issue_d_oversized_request_body():
    """An adversary sends a huge request payload (e.g. 5 MB) to consume server memory.
    
    Expected: The API must reject oversized request payloads with HTTP 413 Payload Too Large.
    """
    app = create_app()
    with TestClient(app) as client:
        # 2 MB payload of whitespace padding
        huge_payload = '{"amount": 1000' + (' ' * (2 * 1024 * 1024)) + '}'
        response = client.post(
            "/v1/score",
            content=huge_payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413, (
            f"Expected HTTP 413 Payload Too Large for 2MB request, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# (e) No rate limiting
# ---------------------------------------------------------------------------
def test_issue_e_rate_limiting():
    """An attacker floods the scoring endpoint with rapid-fire requests.

    Expected: Excessive burst requests from the same client IP must be throttled
    with HTTP 429 Too Many Requests.
    """
    app = create_app()
    with TestClient(app) as client:
        payload = {
            "transaction_id": "txn-rate-limit-",
            "user_id": "usr-rate-limit",
            "merchant_id": "mer-001",
            "amount": 10000,
            "currency": "INR",
            "timestamp": "2024-06-15T12:00:00Z",
            "channel": "ONLINE",
            "transaction_type": "PURCHASE",
        }
        status_codes = []
        # Rapidly send 100 requests
        for i in range(100):
            p = dict(payload, transaction_id=f"txn-rate-limit-{i}")
            resp = client.post("/v1/score", json=p)
            status_codes.append(resp.status_code)

        assert 429 in status_codes, (
            f"No rate limiting! All 100 requests returned status codes: {set(status_codes)}"
        )
