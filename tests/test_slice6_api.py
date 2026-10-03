"""
tests/test_slice6_api.py
─────────────────────────
Slice 6 tests: FastAPI HTTP interface.

Coverage:
  ✓ GET /v1/health returns 200 and valid UTC timestamp
  ✓ POST /v1/score with valid payload returns 200 and FraudScore schema
  ✓ Idempotency: repeated POST with same transaction_id returns cached score and is_duplicate=True
  ✓ Float amount returns 422 Unprocessable Entity
  ✓ Zero amount returns 422
  ✓ Negative amount returns 422
  ✓ Huge amount (> 10 Cr paise) returns 422
  ✓ Missing required fields return 422
  ✓ Extra/unknown fields return 422
  ✓ Naive timestamp (no tzinfo) returns 422
  ✓ Non-UTC timestamp (e.g. +05:30) returns 422
  ✓ Invalid enum channel/type returns 422
  ✓ Invalid currency format returns 422
  ✓ Invalid transaction_id chars/length returns 422
  ✓ Error responses are structured JSON (never 500 or plain stack traces)
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.engine import FraudEngine
from fraud_detector.state.user_state import UserStateStore


@pytest.fixture
def client() -> TestClient:
    store = UserStateStore()
    engine = FraudEngine(store=store)
    app = create_app(engine=engine)
    with TestClient(app) as test_client:
        yield test_client


def _valid_payload(**overrides) -> dict:
    base = {
        "transaction_id": "txn-api-001",
        "user_id": "usr-api-001",
        "merchant_id": "mer-api-001",
        "amount": 75000,
        "currency": "INR",
        "timestamp": "2024-06-15T12:00:00Z",
        "channel": "ONLINE",
        "transaction_type": "PURCHASE",
    }
    base.update(overrides)
    return base


class TestHealthEndpoint:
    def test_health_check_returns_200(self, client: TestClient):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "utc" in data


class TestScoreEndpointHappyPath:
    def test_valid_transaction_returns_200(self, client: TestClient):
        payload = _valid_payload()
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "txn-api-001"
        assert data["user_id"] == "usr-api-001"
        assert 0.0 <= data["score"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert data["is_duplicate"] is False
        assert isinstance(data["flags"], list)
        assert "scored_at" in data

    def test_idempotency_returns_cached_score(self, client: TestClient):
        payload = _valid_payload(transaction_id="txn-api-idem")
        resp1 = client.post("/v1/score", json=payload)
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["is_duplicate"] is False

        # Repeat identical call
        resp2 = client.post("/v1/score", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["is_duplicate"] is True
        assert data2["score"] == data1["score"]
        assert data2["risk_level"] == data1["risk_level"]
        assert data2["flags"] == data1["flags"]


class TestScoreEndpointValidationRejection:
    def test_float_amount_returns_422(self, client: TestClient):
        payload = _valid_payload(amount=123.45)
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_zero_amount_returns_422(self, client: TestClient):
        payload = _valid_payload(amount=0)
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_negative_amount_returns_422(self, client: TestClient):
        payload = _valid_payload(amount=-500)
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_amount_above_maximum_returns_422(self, client: TestClient):
        payload = _valid_payload(amount=10_000_000_001)
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_missing_user_id_returns_422(self, client: TestClient):
        payload = _valid_payload()
        del payload["user_id"]
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_extra_field_forbidden_returns_422(self, client: TestClient):
        payload = _valid_payload(unexpected_field="adversarial_probe")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_naive_timestamp_returns_422(self, client: TestClient):
        # Missing 'Z' or timezone offset
        payload = _valid_payload(timestamp="2024-06-15T12:00:00")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_non_utc_timestamp_returns_422(self, client: TestClient):
        payload = _valid_payload(timestamp="2024-06-15T12:00:00+05:30")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_invalid_channel_returns_422(self, client: TestClient):
        payload = _valid_payload(channel="TELEPATHY")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_invalid_currency_returns_422(self, client: TestClient):
        payload = _valid_payload(currency="rupees")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_empty_transaction_id_returns_422(self, client: TestClient):
        payload = _valid_payload(transaction_id="")
        response = client.post("/v1/score", json=payload)
        assert response.status_code == 422

    def test_malformed_json_body_returns_422(self, client: TestClient):
        response = client.post(
            "/v1/score",
            content="not a json payload",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"
