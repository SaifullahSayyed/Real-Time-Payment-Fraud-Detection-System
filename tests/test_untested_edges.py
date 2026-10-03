"""
tests/test_untested_edges.py
────────────────────────────
Edge cases and uncovered lines:
  ✓ create_app() default lifespan with default engine
  ✓ UserStateStore.user_count()
  ✓ Precision@K when k exceeds scored list size (empty top_k handling)
  ✓ Generic exception handler returns 500 structured JSON
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.engine import FraudEngine
from fraud_detector.evaluation.synthetic import LabelledTransaction
from fraud_detector.evaluation.walk_forward import evaluate_walk_forward
from fraud_detector.models.transaction import TransactionRequest
from fraud_detector.state.user_state import UserStateStore


def test_create_app_default_lifespan():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        # Post a transaction to ensure default engine is active
        post_resp = client.post(
            "/v1/score",
            json={
                "transaction_id": "txn-default-engine",
                "user_id": "usr-default",
                "merchant_id": "mer-001",
                "amount": 25000,
                "timestamp": "2024-06-15T12:00:00Z",
            },
        )
        assert post_resp.status_code == 200


def test_user_state_store_user_count():
    store = UserStateStore()
    assert store.user_count() == 0
    store.commit(
        user_id="user-1",
        transaction_id="tx-1",
        amount_paise=1000,
        timestamp=TransactionRequest(
            transaction_id="tx-1",
            user_id="user-1",
            merchant_id="m1",
            amount=1000,
            timestamp="2024-01-01T00:00:00Z",
        ).timestamp,
        merchant_id="m1",
        score=0.1,
        flags=[],
    )
    assert store.user_count() == 1


def test_precision_at_k_empty_when_no_predictions():
    # evaluate_walk_forward with k_values on empty data edge
    from fraud_detector.evaluation.synthetic import SyntheticDataGenerator
    gen = SyntheticDataGenerator(n_users=5, n_transactions=10, seed=1)
    data = gen.generate()
    # K larger than test set
    res = evaluate_walk_forward(data, train_fraction=0.9, k_values=[1000])
    assert 1000 in res.precision_at_k


def test_generic_exception_handler_returns_500(monkeypatch):
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Patch FraudEngine.evaluate to raise unexpected Exception
        def boom(self, *args, **kwargs):
            raise RuntimeError("Database connection exploded")

        monkeypatch.setattr(FraudEngine, "evaluate", boom)

        response = client.post(
            "/v1/score",
            json={
                "transaction_id": "txn-boom",
                "user_id": "usr-boom",
                "merchant_id": "mer-001",
                "amount": 50000,
                "timestamp": "2024-06-15T12:00:00Z",
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_error"
        assert "Database connection exploded" not in data["detail"]


def test_get_engine_uninitialized_raises():
    from fraud_detector.api.app import _get_engine
    from unittest.mock import MagicMock
    req = MagicMock()
    req.app.state.engine = None
    import fraud_detector.api.app as app_mod
    orig_engine = app_mod._ENGINE
    try:
        app_mod._ENGINE = None
        with pytest.raises(RuntimeError, match="FraudEngine not initialised"):
            _get_engine(req)
    finally:
        app_mod._ENGINE = orig_engine


def test_evaluate_walk_forward_empty_data():
    res = evaluate_walk_forward([], train_fraction=0.7, k_values=[5])
    assert res.n_test == 0
    assert res.precision_at_k[5] == 0.0
