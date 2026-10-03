"""
fraud_detector/api/app.py
──────────────────────────
FastAPI application layer.

Behaviours:
  • POST /v1/score  — score a single transaction
  • GET  /v1/health — liveness probe
  • All validation errors (422) surface structured JSON — never 500.
  • Idempotency: same transaction_id → same score response (from cache).
  • The FraudEngine singleton is created once at startup.
  • No secrets, credentials, or PII appear in logs or responses.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator

from fraud_detector.engine import FraudEngine
from fraud_detector.models.transaction import FraudScore, TransactionRequest
from fraud_detector.state.user_state import IdempotencyConflictError

# ---------------------------------------------------------------------------
# Application state (one engine per process)
# ---------------------------------------------------------------------------

_ENGINE: FraudEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ENGINE
    _ENGINE = FraudEngine()
    yield
    # On shutdown — nothing to clean up for in-memory store


import time
import threading
from collections import deque
from typing import Literal
import numpy as np
from fastapi.responses import HTMLResponse
from fraud_detector.decision.cost_matrix import DecisionEngine, Decision
from fraud_detector.api.dashboard_template import DASHBOARD_HTML


class ReviewRequest(BaseModel):
    model_config = {"extra": "forbid"}

    transaction_id: str
    label: Literal["FRAUD", "LEGIT"]
    notes: str | None = None

    @field_validator("transaction_id")
    @classmethod
    def _validate_tx_id(cls, v: str) -> str:
        import re
        if not v or not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", v):
            raise ValueError(
                "transaction_id must be 1–64 alphanumeric/hyphen/underscore characters"
            )
        return v

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 500:
            raise ValueError("notes must be at most 500 characters")
        return v


MAX_REQUEST_BODY_SIZE: int = 1_048_576  # 1 MB


class SimpleRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            history = self._requests.get(client_id, [])
            history = [t for t in history if t > cutoff]
            if len(history) >= self.max_requests:
                self._requests[client_id] = history
                return False
            history.append(now)
            self._requests[client_id] = history
            return True


# ---------------------------------------------------------------------------
# App factory (allows injecting a custom engine in tests)
# ---------------------------------------------------------------------------


def create_app(
    engine: FraudEngine | None = None,
    rate_limit_per_minute: int = 60,
) -> FastAPI:
    app = FastAPI(
        title="Fraud Detection API",
        version="1.0.0",
        description=(
            "Real-time payment fraud detection. "
            "Money is in integer paise. Timestamps must be UTC."
        ),
        lifespan=lifespan if engine is None else _noop_lifespan(engine),
    )

    # Store engine reference on app state for testability
    app.state.engine = engine  # may be None until lifespan runs
    limiter = SimpleRateLimiter(max_requests=rate_limit_per_minute, window_seconds=60.0)

    # ------------------------------------------------------------------
    # Middlewares: body size enforcement (413) & rate limiting (429)
    # ------------------------------------------------------------------

    @app.middleware("http")
    async def security_and_limit_middleware(request: Request, call_next):
        # 1. Content-Length header limit
        cl_header = request.headers.get("content-length")
        if cl_header:
            try:
                if int(cl_header) > MAX_REQUEST_BODY_SIZE:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "error": "payload_too_large",
                            "detail": f"Request body exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes.",
                        },
                    )
            except ValueError:
                pass

        # 2. Streamed body size limit
        body = await request.body()
        if len(body) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "payload_too_large",
                    "detail": f"Request body exceeds maximum allowed size of {MAX_REQUEST_BODY_SIZE} bytes.",
                },
            )

        # 3. Rate limiting on /v1/score
        if request.url.path == "/v1/score":
            client_ip = request.client.host if request.client else "127.0.0.1"
            if not limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limit_exceeded",
                        "detail": "Too many requests. Please slow down.",
                    },
                )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Exception handlers — always return structured JSON, never crash
    # ------------------------------------------------------------------

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict_handler(
        request: Request, exc: IdempotencyConflictError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "idempotency_conflict",
                "detail": str(exc),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Pydantic/FastAPI validation errors → 422 with structured detail."""
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all: never expose stack traces to clients."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
            },
        )

    # ------------------------------------------------------------------
    # Decision layer and telemetry state
    # ------------------------------------------------------------------
    app.state.decision_engine = DecisionEngine()
    app.state.recent_transactions = deque(maxlen=200)
    app.state.review_queue = {}
    app.state.latencies = deque(maxlen=500)
    app.state.confusion_matrix = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
    @app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
    async def dashboard():
        """Single-file offline operations and review console."""
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/v1/health", status_code=status.HTTP_200_OK, tags=["Ops"])
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "utc": datetime.now(tz=timezone.utc).isoformat()}

    @app.get("/v1/metrics", tags=["Ops"])
    async def get_metrics():
        """Operational metrics, confusion matrix, and review queue."""
        cm = app.state.confusion_matrix
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        lats = list(app.state.latencies)
        p95_lat = float(np.percentile(lats, 95)) if lats else 0.0

        return {
            "total_scored": len(app.state.recent_transactions),
            "review_queue_length": len(app.state.review_queue),
            "review_queue": list(app.state.review_queue.values()),
            "recent_transactions": list(app.state.recent_transactions),
            "confusion_matrix": cm,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_positive_rate": round(fpr, 4),
            "p95_latency_ms": round(p95_lat, 2),
            "thresholds": app.state.decision_engine.thresholds,
        }

    @app.post("/v1/review", tags=["Operations"])
    async def review_transaction(body: ReviewRequest):
        """Analyst feedback endpoint. Updates confusion matrix and threshold tuning."""
        tx_id = body.transaction_id
        app.state.review_queue.pop(tx_id, None)
        cm = app.state.confusion_matrix
        if body.label == "FRAUD":
            cm["tp"] += 1
        else:
            cm["fp"] += 1

        new_thresh = app.state.decision_engine.update_from_review_feedback(
            cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        )
        return {
            "status": "success",
            "transaction_id": tx_id,
            "label": body.label,
            "thresholds": new_thresh,
        }

    @app.post(
        "/v1/score",
        response_model=FraudScore,
        status_code=status.HTTP_200_OK,
        tags=["Scoring"],
        summary="Score a payment transaction for fraud risk",
    )
    async def score_transaction(
        request: Request,
        tx: TransactionRequest,
    ) -> FraudScore:
        """
        Evaluate a transaction for fraud.

        - **amount**: integer paise (e.g. 100000 = ₹1,000). Must be > 0.
        - **timestamp**: UTC ISO-8601 (e.g. `2024-06-15T10:00:00Z`).
        - **transaction_id**: idempotency key — same ID returns cached result.

        Returns a `FraudScore` with `score` in [0, 1] and `risk_level`.
        """
        t_start = time.perf_counter()
        engine: FraudEngine = _get_engine(request)
        score_res = engine.evaluate(tx)
        lat_ms = (time.perf_counter() - t_start) * 1000.0
        app.state.latencies.append(lat_ms)

        dec_res = app.state.decision_engine.decide(score_res.score, tx.amount)

        record = {
            "transaction_id": tx.transaction_id,
            "user_id": tx.user_id,
            "amount": tx.amount,
            "score": score_res.score,
            "decision": dec_res.decision.value,
            "flags": score_res.flags,
            "scored_at": score_res.scored_at.isoformat(),
        }
        app.state.recent_transactions.appendleft(record)

        if dec_res.decision == Decision.REVIEW:
            app.state.review_queue[tx.transaction_id] = record

        if not score_res.is_duplicate:
            if dec_res.decision == Decision.APPROVE:
                app.state.confusion_matrix["tn"] += 1
            elif dec_res.decision == Decision.BLOCK:
                app.state.confusion_matrix["tp"] += 1

        return score_res

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_engine(request: Request) -> FraudEngine:
    """Return the engine from app state (supports test injection)."""
    engine = request.app.state.engine
    if engine is None:
        engine = _ENGINE
    if engine is None:
        raise RuntimeError("FraudEngine not initialised")
    return engine


def _noop_lifespan(engine: FraudEngine):
    """Lifespan that injects a pre-built engine (used in tests)."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.engine = engine
        yield

    return _lifespan
