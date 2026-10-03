"""
tests/test_ss7_geo_device.py
─────────────────────────────
SS7 test suite: Geo and Device Signals.

Coverage:
  ✓ Haversine distance accuracy (known landmark distances)
  ✓ Impossible travel detected (Mumbai -> London in 15 minutes)
  ✓ Feasible travel permitted (Mumbai -> Pune in 3 hours)
  ✓ New device detected for established user
  ✓ Device velocity detected (3+ distinct devices within 1 hour)
  ✓ Reason codes present in score response
  ✓ Geo coordinate range validation (lat/lon bounds and co-dependence)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from pydantic import ValidationError

from fraud_detector.engine import FraudEngine
from fraud_detector.features.extractor import haversine_distance_km
from fraud_detector.models.transaction import TransactionRequest
from fraud_detector.state.user_state import UserStateStore


_UTC = timezone.utc
_T0 = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)

# Coordinates
MUMBAI_LAT, MUMBAI_LON = 19.0760, 72.8777
PUNE_LAT, PUNE_LON = 18.5204, 73.8567
LONDON_LAT, LONDON_LON = 51.5074, -0.1278


class TestGeoCoordinatesValidation:
    def test_lat_without_lon_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(
                transaction_id="tx-geo-1",
                user_id="usr-1",
                merchant_id="mer-1",
                amount=50000,
                timestamp=_T0,
                latitude=MUMBAI_LAT,
                longitude=None,
            )

    def test_lon_without_lat_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(
                transaction_id="tx-geo-2",
                user_id="usr-1",
                merchant_id="mer-1",
                amount=50000,
                timestamp=_T0,
                latitude=None,
                longitude=MUMBAI_LON,
            )

    def test_lat_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            TransactionRequest(
                transaction_id="tx-geo-3",
                user_id="usr-1",
                merchant_id="mer-1",
                amount=50000,
                timestamp=_T0,
                latitude=95.0,
                longitude=10.0,
            )


class TestHaversineDistance:
    def test_mumbai_to_pune_distance(self):
        # Mumbai to Pune is ~120-150 km
        dist = haversine_distance_km(MUMBAI_LAT, MUMBAI_LON, PUNE_LAT, PUNE_LON)
        assert 110.0 <= dist <= 150.0

    def test_mumbai_to_london_distance(self):
        # Mumbai to London is ~7,200 km
        dist = haversine_distance_km(MUMBAI_LAT, MUMBAI_LON, LONDON_LAT, LONDON_LON)
        assert 7000.0 <= dist <= 7500.0


class TestImpossibleTravel:
    def test_impossible_travel_detected(self):
        """Mumbai to London in 15 minutes is ~28,800 km/h, well over jet speed limit (900 km/h)."""
        engine = FraudEngine()
        # Txn 1 in Mumbai
        engine.evaluate(TransactionRequest(
            transaction_id="tx-mumbai",
            user_id="usr-traveler",
            merchant_id="mer-001",
            amount=50000,
            timestamp=_T0,
            latitude=MUMBAI_LAT,
            longitude=MUMBAI_LON,
        ))

        # Txn 2 in London 15 minutes later
        res2 = engine.evaluate(TransactionRequest(
            transaction_id="tx-london",
            user_id="usr-traveler",
            merchant_id="mer-002",
            amount=50000,
            timestamp=_T0 + timedelta(minutes=15),
            latitude=LONDON_LAT,
            longitude=LONDON_LON,
        ))

        impossible_flags = [f for f in res2.flags if "IMPOSSIBLE_TRAVEL" in f]
        assert len(impossible_flags) == 1
        assert "exceeds physical limit" in impossible_flags[0]

    def test_feasible_travel_permitted(self):
        """Mumbai to Pune in 3 hours is ~40 km/h, perfectly feasible."""
        engine = FraudEngine()
        engine.evaluate(TransactionRequest(
            transaction_id="tx-mumbai-ok",
            user_id="usr-driver",
            merchant_id="mer-001",
            amount=50000,
            timestamp=_T0,
            latitude=MUMBAI_LAT,
            longitude=MUMBAI_LON,
        ))

        res2 = engine.evaluate(TransactionRequest(
            transaction_id="tx-pune-ok",
            user_id="usr-driver",
            merchant_id="mer-002",
            amount=50000,
            timestamp=_T0 + timedelta(hours=3),
            latitude=PUNE_LAT,
            longitude=PUNE_LON,
        ))

        impossible_flags = [f for f in res2.flags if "IMPOSSIBLE_TRAVEL" in f]
        assert len(impossible_flags) == 0


class TestDeviceSignals:
    def test_new_device_detected_for_established_user(self):
        engine = FraudEngine()
        # Initial transaction with Device A
        engine.evaluate(TransactionRequest(
            transaction_id="tx-dev-1",
            user_id="usr-multi-dev",
            merchant_id="mer-001",
            amount=50000,
            timestamp=_T0,
            device_id="iphone-15-pro",
        ))

        # Next transaction with Device B
        res2 = engine.evaluate(TransactionRequest(
            transaction_id="tx-dev-2",
            user_id="usr-multi-dev",
            merchant_id="mer-001",
            amount=50000,
            timestamp=_T0 + timedelta(minutes=10),
            device_id="unseen-android-tablet",
        ))

        new_dev_flags = [f for f in res2.flags if "NEW_DEVICE" in f]
        assert len(new_dev_flags) == 1
        assert "unseen device fingerprint" in new_dev_flags[0]

    def test_device_velocity_detected(self):
        """Using 3+ distinct devices in 1 hour triggers device velocity flag."""
        engine = FraudEngine()
        for i in range(3):
            engine.evaluate(TransactionRequest(
                transaction_id=f"tx-vel-dev-{i}",
                user_id="usr-botnet",
                merchant_id="mer-001",
                amount=50000,
                timestamp=_T0 + timedelta(minutes=i * 5),
                device_id=f"device-fingerprint-{i}",
            ))

        res4 = engine.evaluate(TransactionRequest(
            transaction_id="tx-vel-dev-4",
            user_id="usr-botnet",
            merchant_id="mer-001",
            amount=50000,
            timestamp=_T0 + timedelta(minutes=20),
            device_id="device-fingerprint-99",
        ))

        dev_vel_flags = [f for f in res4.flags if "DEVICE_VELOCITY" in f]
        assert len(dev_vel_flags) == 1
