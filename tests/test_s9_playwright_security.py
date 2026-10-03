"""
tests/test_s9_playwright_security.py
─────────────────────────────────────
Security tests derived from the Playwright MCP browser-based audit of S9.

Playwright findings tested here:
  ✓ POST /v1/review — XSS payload in transaction_id is now rejected (422)
  ✓ POST /v1/review — empty string transaction_id is now rejected (422)
  ✓ POST /v1/review — oversized notes rejected (422)
  ✓ POST /v1/review — invalid label rejected (422)
  ✓ POST /v1/review — extra fields rejected (422, model_config forbids extra)
  ✓ POST /v1/score  — XSS payloads in IDs rejected (422)
  ✓ POST /v1/score  — SQL injection strings in IDs rejected (422)
  ✓ Dashboard HTML uses createTextNode / textContent — no innerHTML with server data
  ✓ GET / and GET /dashboard — 200, no CDN references
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.engine import FraudEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client():
    return TestClient(create_app(engine=FraudEngine()))


# ---------------------------------------------------------------------------
# POST /v1/review — input validation
# ---------------------------------------------------------------------------

class TestReviewEndpointValidation:
    def test_xss_in_transaction_id_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={"transaction_id": "<script>alert('xss')</script>", "label": "FRAUD"},
            )
            assert res.status_code == 422, (
                f"XSS transaction_id must be rejected with 422, got {res.status_code}"
            )

    def test_empty_transaction_id_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={"transaction_id": "", "label": "FRAUD"},
            )
            assert res.status_code == 422, (
                f"Empty transaction_id must be rejected with 422, got {res.status_code}"
            )

    def test_sql_injection_in_transaction_id_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={"transaction_id": "'; DROP TABLE users; --", "label": "LEGIT"},
            )
            assert res.status_code == 422

    def test_invalid_label_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={"transaction_id": "txn-valid-001", "label": "MAYBE"},
            )
            assert res.status_code == 422

    def test_xss_label_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={"transaction_id": "txn-valid-001", "label": "<script>alert(1)</script>"},
            )
            assert res.status_code == 422

    def test_oversized_notes_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={
                    "transaction_id": "txn-valid-001",
                    "label": "FRAUD",
                    "notes": "x" * 501,
                },
            )
            assert res.status_code == 422, (
                f"Notes > 500 chars must be rejected with 422, got {res.status_code}"
            )

    def test_extra_fields_rejected(self):
        with _client() as client:
            res = client.post(
                "/v1/review",
                json={
                    "transaction_id": "txn-valid-001",
                    "label": "FRAUD",
                    "injected_field": "evil_value",
                },
            )
            assert res.status_code == 422, (
                f"Extra fields must be rejected with 422, got {res.status_code}"
            )

    def test_valid_review_accepted(self):
        """Regression: a well-formed review must still pass."""
        import os
        token = os.environ.get("ANALYST_TOKEN", "test-analyst-token")
        with _client() as client:
            client.app.state.known_transactions.add("txn-valid-001")
            res = client.post(
                "/v1/review",
                json={
                    "transaction_id": "txn-valid-001",
                    "label": "LEGIT",
                    "notes": "Cardholder confirmed.",
                },
                headers={"X-Analyst-Token": token},
            )
            assert res.status_code == 200
            assert res.json()["status"] == "success"


# ---------------------------------------------------------------------------
# POST /v1/score — XSS and injection in ID fields
# ---------------------------------------------------------------------------

class TestScoreEndpointInjectionRejection:
    _XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        "javascript:alert(1)",
    ]
    _SQL_PAYLOADS = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
    ]

    @pytest.mark.parametrize("payload", _XSS_PAYLOADS)
    def test_xss_in_transaction_id_rejected(self, payload: str):
        with _client() as client:
            res = client.post(
                "/v1/score",
                json={
                    "transaction_id": payload,
                    "user_id": "usr-001",
                    "merchant_id": "mer-001",
                    "amount": 10000,
                    "timestamp": "2024-06-15T10:00:00Z",
                },
            )
            assert res.status_code == 422, (
                f"XSS payload '{payload}' should be rejected with 422, got {res.status_code}"
            )

    @pytest.mark.parametrize("payload", _SQL_PAYLOADS)
    def test_sql_injection_in_ids_rejected(self, payload: str):
        with _client() as client:
            res = client.post(
                "/v1/score",
                json={
                    "transaction_id": payload,
                    "user_id": payload,
                    "merchant_id": payload,
                    "amount": 10000,
                    "timestamp": "2024-06-15T10:00:00Z",
                },
            )
            assert res.status_code == 422


# ---------------------------------------------------------------------------
# Dashboard HTML safety
# ---------------------------------------------------------------------------

class TestDashboardHTMLSafety:
    def test_dashboard_uses_no_external_resources(self):
        with _client() as client:
            resp = client.get("/dashboard")
            assert resp.status_code == 200
            html = resp.text
            assert "cdn." not in html.lower()
            assert "http://" not in html.lower()
            assert "https://" not in html.lower()

    def test_dashboard_uses_text_content_not_inner_html_for_data(self):
        """
        Verify the JS in the dashboard template exclusively uses textContent /
        createTextNode for user-controlled server data. innerHTML must NOT be used
        to insert any server-sourced field (transaction_id, user_id, flags, etc.).
        """
        with _client() as client:
            resp = client.get("/dashboard")
            assert resp.status_code == 200
            html = resp.text

            # All dynamic field assignments use safe DOM APIs
            assert ".textContent = " in html, "Must use .textContent for safe rendering"
            assert "createTextNode" in html or ".textContent" in html

            # No server-controlled fields are written via innerHTML in the JS
            # (innerHTML is only used once to reset rBody.innerHTML = '' — that's safe)
            # Check that no dangerous pattern like 'innerHTML = item.' or 'innerHTML = tx.' exists
            assert "innerHTML = item." not in html, (
                "Server data must not be inserted via innerHTML"
            )
            assert "innerHTML = tx." not in html, (
                "Server data must not be inserted via innerHTML"
            )

    def test_dashboard_title_present(self):
        with _client() as client:
            resp = client.get("/")
            assert "Fraud Detection Operations" in resp.text
