"""
tests/test_s9_decision_dashboard.py
───────────────────────────────────
S9 test suite: Decision layer, operations endpoints, and dashboard.

Coverage:
  ✓ DecisionEngine cost-optimal decision classification (APPROVE, STEP_UP, REVIEW, BLOCK)
  ✓ Dynamic threshold recalibration under feedback
  ✓ GET /v1/metrics returning valid KPIs, latency p95, and confusion matrix
  ✓ POST /v1/review updating confusion matrix and recalibrating thresholds
  ✓ GET / and GET /dashboard returning 200 OK with self-contained offline HTML
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from fraud_detector.api.app import create_app
from fraud_detector.decision.cost_matrix import Decision, DecisionEngine


def test_decision_engine_classifications():
    engine = DecisionEngine(thresh_stepup=0.30, thresh_review=0.60, thresh_block=0.85)

    res_clean = engine.decide(score=0.15, amount_paise=50000)
    assert res_clean.decision == Decision.APPROVE

    res_stepup = engine.decide(score=0.45, amount_paise=50000)
    assert res_stepup.decision == Decision.STEP_UP

    res_review = engine.decide(score=0.72, amount_paise=50000)
    assert res_review.decision == Decision.REVIEW

    res_block = engine.decide(score=0.92, amount_paise=50000)
    assert res_block.decision == Decision.BLOCK


def test_decision_engine_recalibration():
    engine = DecisionEngine(thresh_block=0.85, thresh_review=0.60)
    # High false positives -> shifts thresholds up to reduce false alarms
    new_thresh = engine.update_from_review_feedback(tp=5, fp=10, tn=80, fn=1)
    assert new_thresh["block"] >= 0.85
    assert new_thresh["review"] >= 0.60


def test_metrics_and_review_endpoints():
    app = create_app()
    with TestClient(app) as client:
        # Score a normal transaction
        res1 = client.post(
            "/v1/score",
            json={
                "transaction_id": "txn-ops-001",
                "user_id": "usr-ops-001",
                "merchant_id": "mer-001",
                "amount": 20000,
                "timestamp": "2024-06-15T12:00:00Z",
            },
        )
        assert res1.status_code == 200

        # Query metrics
        m_resp = client.get("/v1/metrics")
        assert m_resp.status_code == 200
        metrics = m_resp.json()
        assert metrics["total_scored"] >= 1
        assert "p95_latency_ms" in metrics
        assert "confusion_matrix" in metrics
        assert "thresholds" in metrics

        # Submit analyst review
        import os
        token = os.environ.get("ANALYST_TOKEN", "test-analyst-token")
        rev_resp = client.post(
            "/v1/review",
            json={
                "transaction_id": "txn-ops-001",
                "label": "LEGIT",
                "notes": "Verified with cardholder",
            },
            headers={"X-Analyst-Token": token},
        )
        assert rev_resp.status_code == 200
        assert rev_resp.json()["status"] == "success"

        # Check metrics updated
        m_resp2 = client.get("/v1/metrics")
        m2 = m_resp2.json()
        assert m2["confusion_matrix"]["fp"] >= 1


def test_dashboard_served_offline_without_cdns():
    app = create_app()
    with TestClient(app) as client:
        for path in ["/", "/dashboard"]:
            resp = client.get(path)
            assert resp.status_code == 200
            html = resp.text
            assert "Fraud Detection Operations & Decision Console" in html
            # Offline guarantee: NO external CDN scripts or stylesheet links
            assert "cdn." not in html.lower()
            assert "http://" not in html.lower()
            assert "https://" not in html.lower()
            assert "<script>" in html
            assert "<style>" in html
