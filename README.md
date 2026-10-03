# Real-Time Payment Fraud Detection System

A production-grade, adversarial-hardened, real-time payment fraud detection system built in Python with FastAPI, scikit-learn, and Pydantic. Designed to defend against adversarial attackers, race conditions, edge-case distortions, and unauthorized tampering.

---

## 🔒 Architectural Invariants & Threat Model

1. **Integer Money Representation**: All amounts are strictly positive 64-bit integers in minor currency units (paise, e.g. ₹100.00 = `10000`). Floats are rejected by Pydantic validators to prevent IEEE 754 rounding vulnerabilities.
2. **UTC Timestamps Only**: All timestamps must be timezone-aware UTC (`+00:00` or `Z`). Naive timestamps or offsets are rejected. Timestamps are bounded strictly to `[2020-01-01, 2035-01-01]`, preventing epoch-zero and far-future velocity skewing.
3. **Idempotency with Payload Verification**: Transaction scoring is deterministic and idempotent per `transaction_id`. If an attacker attempts to resubmit the same `transaction_id` with an altered payload (different amount or merchant), the system detects the SHA-256 fingerprint conflict and responds with **HTTP 409 Conflict**.
4. **Thread-Safe & Bounded State Store**: State updates per user are protected by fine-grained locks. Global state growth is bounded by an LRU eviction policy (`max_users`), preventing memory exhaustion under high-cardinality spoofing attacks.
5. **No Lookahead / Label Leakage**: Walk-forward temporal splits exclusively evaluate model performance; no future event or test label ever enters training or feature calculation.
6. **Hardened Operational Security**:
   - **Analyst Token Authentication**: `POST /v1/review` requires a secret token provided via the `ANALYST_TOKEN` environment variable and passed in `X-Analyst-Token` (HTTP 401/403 on failure).
   - **Double-Review Idempotency & Conflict Guard**: Reviewing the same transaction twice with the same label is idempotent; submitting conflicting labels is rejected with HTTP 409.
   - **Unknown Transaction Guard**: Reviewing an unknown transaction returns HTTP 404.
   - **Bounded Threshold Drift**: Analyst feedback cannot move decision thresholds beyond a bounded step per label (0.02) and per hour (0.10).
   - **Endpoint Hardening**: `/docs`, `/redoc`, and `/openapi.json` are disabled by default.
   - **Security Headers**: Every response returns `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and strict `Content-Security-Policy`.
   - **DOS Defense**: Request bodies are capped at 1 MB (HTTP 413) and incoming scoring requests are rate-limited via a sliding-window limiter (60 req/min, HTTP 429).
   - **Zero-CDN Offline Console**: The web dashboard is single-file, uses zero external CDNs or web fonts, and renders all dynamic server content via DOM `textContent` (blocking XSS).

---

## 🚀 Clean-Machine Quickstart

This repository has zero runtime external network dependencies and passes a clean virtual environment check.

### 1. Clone & Set Up Virtual Environment

#### On Windows (PowerShell / CMD):
```powershell
git clone https://github.com/SaifullahSayyed/Real-Time-Payment-Fraud-Detection-System.git
cd Real-Time-Payment-Fraud-Detection-System

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
git clone https://github.com/SaifullahSayyed/Real-Time-Payment-Fraud-Detection-System.git
cd Real-Time-Payment-Fraud-Detection-System

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite (255 Tests)

```bash
pytest
```
Expected output: **255 passed** (100% green).

### 3. Start the Server

```bash
# Set your analyst token for the review console
export ANALYST_TOKEN="analyst-secret-token"   # Linux/macOS
# or on Windows PowerShell:
$env:ANALYST_TOKEN="analyst-secret-token"

python -m uvicorn "fraud_detector.api.app:create_app" --factory --host 127.0.0.1 --port 8000
```

Access the **Offline Operations & Decision Dashboard** in your browser:
👉 [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## 📡 API Reference

### 1. Score Transaction: `POST /v1/score`
Evaluate a transaction in real-time.

```bash
curl -X POST http://127.0.0.1:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn-1001",
    "user_id": "usr-8821",
    "merchant_id": "mer-flipkart",
    "amount": 250000,
    "currency": "INR",
    "timestamp": "2026-10-03T12:00:00Z",
    "channel": "ONLINE",
    "transaction_type": "PURCHASE",
    "device_id": "dev-chrome-win",
    "latitude": 19.0760,
    "longitude": 72.8777
  }'
```

**Response (HTTP 200)**:
```json
{
  "transaction_id": "txn-1001",
  "user_id": "usr-8821",
  "score": 0.182,
  "risk_level": "LOW",
  "flags": [
    "COLD_START: no transaction history for user"
  ],
  "is_duplicate": false,
  "scored_at": "2026-10-03T12:00:00.042123Z"
}
```

### 2. Operational Metrics: `GET /v1/metrics`
Fetches live telemetry, p95 latency, confusion matrix, precision/recall, and review queue:
```bash
curl http://127.0.0.1:8000/v1/metrics
```

### 3. Analyst Review: `POST /v1/review`
Requires `X-Analyst-Token` header:
```bash
curl -X POST http://127.0.0.1:8000/v1/review \
  -H "Content-Type: application/json" \
  -H "X-Analyst-Token: analyst-secret-token" \
  -d '{
    "transaction_id": "txn-1001",
    "label": "LEGIT",
    "notes": "Verified with cardholder over SMS OTP"
  }'
```

---

## 🧩 Pipeline Architecture

```
Incoming Request (Pydantic Strict Validation)
       │
       ▼
Security Middleware (Body Size < 1MB, Rate Limiter < 60/min, Security Headers)
       │
       ▼
FraudEngine (Idempotency Check via SHA-256 Payload Hash)
       │
       ▼
UserStateStore (Atomic Per-User Lock, Welford Online Stats, LRU Eviction)
       │
       ▼
Feature Extraction (Velocity, Amount Z-Score, Hour-of-Day, Geo Haversine, Device Velocity)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Rule-Based Scorer (13 Rules)      Isolation Forest Anomaly Scorer
       │                                 │
       └────────────────┬────────────────┘
                        ▼
                Combined Risk Score
                        │
                        ▼
              Decision Engine (Cost Matrix)
        ┌───────────────┬────────────────┬──────────────┐
        ▼               ▼                ▼              ▼
     APPROVE         STEP_UP          REVIEW          BLOCK
     (<0.30)       (0.30-0.60)      (0.60-0.85)      (≥0.85)
```

---

## 🧪 Test Suite Structure

| Test File | Description | Test Count |
|-----------|-------------|------------|
| `test_slice1_models.py` | Pydantic model validation, paise amounts, UTC checks | 28 |
| `test_slice2_features.py` | Feature extractor, velocity windows, Welford updates | 25 |
| `test_slice3_scorer.py` | Rule engine scoring, determinism, cold start | 36 |
| `test_slice4_state_engine.py` | Concurrency, per-user lock, idempotency | 32 |
| `test_slice5_evaluation.py` | Synthetic generator, walk-forward splits, PR-AUC | 24 |
| `test_slice6_api.py` | FastAPI endpoints, structured error handling | 36 |
| `test_ss7_geo_device.py` | Impossible travel (>900 km/h), new device, device velocity | 9 |
| `test_s8_adaptivity.py` | Isolation Forest, unseen fraud patterns (slow drip, merchant hopping) | 6 |
| `test_s9_decision_dashboard.py` | Cost matrix decision engine, metrics, dashboard | 4 |
| `test_s9_playwright_security.py` | Playwright browser security audit, XSS, injection | 18 |
| `test_s10_review_hardening.py` | Token auth, double-review idempotency, bounded drift, headers | 17 |
| `test_suspected_issues.py` | Adversarial attacks: payload conflict, extreme timestamps, DOS | 5 |
| `test_untested_edges.py` | System edge cases, exception handlers | 15 |
| **Total** | **All Tests Passing** | **255** |
