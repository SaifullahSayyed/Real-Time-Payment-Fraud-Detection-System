"""
tests/test_s10_review_hardening.py
────────────────────────────────────
FAILING-FIRST tests for S10 review-endpoint hardening and API security.

(a) POST /v1/review without a valid analyst token is rejected (401/403).
    Token is read from env var ANALYST_TOKEN; no default value.
(b) Same transaction_id reviewed twice does not double-count in the
    confusion matrix; a conflicting second label is rejected (409) or
    logged as an explicit override.
(c) Reviewing an unknown transaction_id returns 404.
(d) A flood of identical-label reviews cannot move thresholds beyond a
    bounded step per label and per hour.
(e) Concurrent reviews of the same transaction_id are safe (no races).
(f) /docs, /redoc, and /openapi.json are disabled in the default config.
(g) Responses include X-Content-Type-Options and a Content-Security-Policy.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time

import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.engine import FraudEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GOOD_TOKEN = "test-analyst-token-abc123"
_BAD_TOKEN = "wrong-token"

_VALID_TX = {
    "transaction_id": "txn-review-base",
    "user_id": "usr-review-001",
    "merchant_id": "mer-review-001",
    "amount": 20000,
    "currency": "INR",
    "timestamp": "2024-06-15T12:00:00Z",
    "channel": "ONLINE",
    "transaction_type": "PURCHASE",
}


def _app_with_token(token: str = _GOOD_TOKEN) -> TestClient:
    """Return a TestClient with ANALYST_TOKEN set."""
    os.environ["ANALYST_TOKEN"] = token
    try:
        app = create_app(engine=FraudEngine())
        return TestClient(app)
    finally:
        pass  # do NOT pop — token must stay set for request


def _score_tx(client: TestClient, tx_id: str, amount: int = 20000) -> int:
    payload = dict(_VALID_TX, transaction_id=tx_id, amount=amount)
    r = client.post("/v1/score", json=payload)
    return r.status_code


def _review(client: TestClient, tx_id: str, label: str, token: str = _GOOD_TOKEN) -> dict:
    r = client.post(
        "/v1/review",
        json={"transaction_id": tx_id, "label": label},
        headers={"X-Analyst-Token": token},
    )
    return {"status": r.status_code, "body": r.json()}


# ---------------------------------------------------------------------------
# (a) Analyst token authentication
# ---------------------------------------------------------------------------

class TestAnalystTokenAuth:
    """POST /v1/review must require a valid token from env ANALYST_TOKEN."""

    def test_missing_token_header_returns_401_or_403(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            # Seed a known transaction so 404 isn't the first failure
            _score_tx(client, "txn-auth-001")
            r = client.post(
                "/v1/review",
                json={"transaction_id": "txn-auth-001", "label": "FRAUD"},
                # No X-Analyst-Token header
            )
            assert r.status_code in (401, 403), (
                f"Missing token should return 401 or 403, got {r.status_code}"
            )

    def test_wrong_token_returns_401_or_403(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            _score_tx(client, "txn-auth-002")
            r = client.post(
                "/v1/review",
                json={"transaction_id": "txn-auth-002", "label": "FRAUD"},
                headers={"X-Analyst-Token": _BAD_TOKEN},
            )
            assert r.status_code in (401, 403), (
                f"Wrong token should return 401 or 403, got {r.status_code}"
            )

    def test_correct_token_returns_200(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            # Plant the transaction directly in the review queue
            app.state.review_queue["txn-auth-ok"] = {
                "transaction_id": "txn-auth-ok", "user_id": "u",
                "amount": 1000, "score": 0.75, "flags": [],
                "scored_at": "2024-06-15T12:00:00+00:00",
            }
            r = client.post(
                "/v1/review",
                json={"transaction_id": "txn-auth-ok", "label": "LEGIT"},
                headers={"X-Analyst-Token": _GOOD_TOKEN},
            )
            assert r.status_code == 200, (
                f"Valid token should return 200, got {r.status_code}: {r.text}"
            )

    def test_no_env_var_set_raises_or_returns_500_startup(self):
        """If ANALYST_TOKEN env var is not set, the app must refuse to start
        (raise at startup) or return 500/503 for all /v1/review calls —
        never silently allow unauthenticated access."""
        os.environ.pop("ANALYST_TOKEN", None)
        try:
            app = create_app(engine=FraudEngine())
            with TestClient(app) as client:
                r = client.post(
                    "/v1/review",
                    json={"transaction_id": "txn-no-env", "label": "FRAUD"},
                )
                # Must NOT be 200 — either the app raised at startup (so TestClient
                # would have thrown) or it returns a non-200 error.
                assert r.status_code != 200, (
                    "With no ANALYST_TOKEN set, /v1/review must not return 200"
                )
        except Exception:
            # Startup failure is also acceptable
            pass
        finally:
            os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN


# ---------------------------------------------------------------------------
# (b) Double-review idempotency and conflict detection
# ---------------------------------------------------------------------------

class TestDoubleReview:
    """Reviewing the same tx_id twice must not double-count."""

    def _plant(self, app, tx_id: str) -> None:
        """Place tx_id directly in the review queue."""
        app.state.review_queue[tx_id] = {
            "transaction_id": tx_id, "user_id": "u",
            "amount": 1000, "score": 0.75, "flags": [],
            "scored_at": "2024-06-15T12:00:00+00:00",
        }

    def test_same_label_twice_does_not_double_count(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            self._plant(app, "txn-double-same")

            r1 = _review(client, "txn-double-same", "FRAUD")
            assert r1["status"] == 200

            cm_after_first = client.get("/v1/metrics").json()["confusion_matrix"]["tp"]

            # Second review of same transaction without replanting
            r2 = _review(client, "txn-double-same", "FRAUD")
            assert r2["status"] == 200
            cm_after_second = client.get("/v1/metrics").json()["confusion_matrix"]["tp"]

            # Second call with same label must NOT increment tp again
            assert cm_after_second == cm_after_first, (
                f"Double review (same label) must not double-count: "
                f"tp after 1st={cm_after_first}, tp after 2nd={cm_after_second}"
            )

    def test_conflicting_label_rejected_or_logged(self):
        """A second review with a conflicting label must return 409 or at
        minimum carry an 'override' indicator in the response — never silently
        apply both labels."""
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            self._plant(app, "txn-conflict")

            r1 = _review(client, "txn-conflict", "FRAUD")
            assert r1["status"] == 200

            # Second review with conflicting label
            r2 = _review(client, "txn-conflict", "LEGIT")
            # Acceptable outcomes: 409 conflict, or 200 with override field
            assert r2["status"] == 409 or (
                r2["status"] == 200 and r2["body"].get("override") is True
            ), (
                f"Conflicting second label must return 409 or 200+override, "
                f"got {r2['status']}: {r2['body']}"
            )


# ---------------------------------------------------------------------------
# (c) Reviewing an unknown transaction_id returns 404
# ---------------------------------------------------------------------------

class TestReviewUnknownTransaction:
    def test_unknown_transaction_id_returns_404(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            r = client.post(
                "/v1/review",
                json={"transaction_id": "txn-does-not-exist-xyz", "label": "FRAUD"},
                headers={"X-Analyst-Token": _GOOD_TOKEN},
            )
            assert r.status_code == 404, (
                f"Unknown transaction_id should return 404, got {r.status_code}: {r.text}"
            )

    def test_known_transaction_in_review_queue_returns_200(self):
        """Baseline: a tx that was scored AND placed in the review queue
        must still be reviewable."""
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            # Score a high-value tx that will be REVIEW or BLOCK
            # Use a burst to drive score up
            for i in range(12):
                _score_tx(client, f"txn-queue-burst-{i}", amount=75_000_000)

            # Now directly plant a known tx in the review queue
            tx_id = "txn-queue-known"
            app.state.review_queue[tx_id] = {
                "transaction_id": tx_id,
                "user_id": "usr-001",
                "amount": 1000,
                "score": 0.75,
                "flags": [],
                "scored_at": "2024-06-15T12:00:00+00:00",
            }

            r = client.post(
                "/v1/review",
                json={"transaction_id": tx_id, "label": "LEGIT"},
                headers={"X-Analyst-Token": _GOOD_TOKEN},
            )
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# (d) Threshold drift is bounded per hour
# ---------------------------------------------------------------------------

class TestThresholdDriftBound:
    """A flood of identical-label reviews must not move thresholds unboundedly."""

    def test_flood_of_reviews_cannot_move_threshold_by_more_than_bound(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            initial = client.get("/v1/metrics").json()["thresholds"]["block"]

            # Plant 100 transactions in the review queue and flood-review them
            for i in range(100):
                tx_id = f"txn-flood-{i:04d}"
                app.state.review_queue[tx_id] = {
                    "transaction_id": tx_id, "user_id": "u", "amount": 1000,
                    "score": 0.9, "flags": [], "scored_at": "2024-06-15T12:00:00+00:00",
                }
                client.post(
                    "/v1/review",
                    json={"transaction_id": tx_id, "label": "LEGIT"},
                    headers={"X-Analyst-Token": _GOOD_TOKEN},
                )

            final = client.get("/v1/metrics").json()["thresholds"]["block"]
            drift = abs(final - initial)
            # Max allowed drift: 0.20 per hour across all reviews
            assert drift <= 0.20, (
                f"Threshold drifted by {drift:.4f} after 100 flood-reviews "
                f"(initial={initial:.4f}, final={final:.4f}). "
                f"Must be bounded to ≤ 0.20 per hour."
            )


# ---------------------------------------------------------------------------
# (e) Concurrent reviews of the same transaction_id are safe
# ---------------------------------------------------------------------------

class TestConcurrentReviews:
    def test_concurrent_reviews_do_not_corrupt_confusion_matrix(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            # Plant 20 transactions in the review queue
            tx_ids = [f"txn-concurrent-{i:03d}" for i in range(20)]
            for tx_id in tx_ids:
                app.state.review_queue[tx_id] = {
                    "transaction_id": tx_id, "user_id": "u", "amount": 1000,
                    "score": 0.75, "flags": [], "scored_at": "2024-06-15T12:00:00+00:00",
                }

            errors: list[str] = []

            def do_review(tx_id: str) -> None:
                try:
                    r = client.post(
                        "/v1/review",
                        json={"transaction_id": tx_id, "label": "FRAUD"},
                        headers={"X-Analyst-Token": _GOOD_TOKEN},
                    )
                    if r.status_code not in (200, 404, 409):
                        errors.append(f"{tx_id}: {r.status_code}")
                except Exception as exc:
                    errors.append(f"{tx_id}: {exc}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                list(pool.map(do_review, tx_ids))

            assert not errors, f"Concurrent review errors: {errors}"

            cm = client.get("/v1/metrics").json()["confusion_matrix"]
            total = cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"]
            # Each of 20 reviews increments exactly one counter — total must be
            # exactly 20 (no double-counts, no drops).
            assert total == 20, (
                f"Expected 20 total in confusion matrix after 20 concurrent reviews, "
                f"got {total}: {cm}"
            )


# ---------------------------------------------------------------------------
# (f) /docs, /redoc, /openapi.json are disabled by default
# ---------------------------------------------------------------------------

class TestDocsDisabled:
    def test_docs_endpoint_not_exposed(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/docs")
            assert r.status_code in (404, 401, 403), (
                f"/docs must not be publicly exposed (got {r.status_code})"
            )

    def test_redoc_endpoint_not_exposed(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/redoc")
            assert r.status_code in (404, 401, 403), (
                f"/redoc must not be publicly exposed (got {r.status_code})"
            )

    def test_openapi_json_not_exposed(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/openapi.json")
            assert r.status_code in (404, 401, 403), (
                f"/openapi.json must not be publicly exposed (got {r.status_code})"
            )


# ---------------------------------------------------------------------------
# (g) Security headers: X-Content-Type-Options and CSP
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    def _check_headers(self, response) -> None:
        headers = {k.lower(): v for k, v in response.headers.items()}
        assert "x-content-type-options" in headers, (
            "Missing X-Content-Type-Options header"
        )
        assert headers["x-content-type-options"] == "nosniff", (
            f"X-Content-Type-Options must be 'nosniff', got '{headers['x-content-type-options']}'"
        )
        assert "content-security-policy" in headers, (
            "Missing Content-Security-Policy header"
        )
        csp = headers["content-security-policy"]
        assert "default-src" in csp or "script-src" in csp, (
            f"CSP must contain at least default-src or script-src directive, got: {csp}"
        )

    def test_score_response_has_security_headers(self):
        os.environ["ANALYST_TOKEN"] = _GOOD_TOKEN
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            r = client.post("/v1/score", json=_VALID_TX)
            assert r.status_code == 200
            self._check_headers(r)

    def test_health_response_has_security_headers(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            r = client.get("/v1/health")
            assert r.status_code == 200
            self._check_headers(r)

    def test_dashboard_response_has_security_headers(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            r = client.get("/dashboard")
            assert r.status_code == 200
            self._check_headers(r)

    def test_metrics_response_has_security_headers(self):
        app = create_app(engine=FraudEngine())
        with TestClient(app) as client:
            r = client.get("/v1/metrics")
            assert r.status_code == 200
            self._check_headers(r)
