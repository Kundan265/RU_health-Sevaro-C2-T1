"""Tests for engine/offer_log.py - event-sourced offer management."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.offer_log import OfferLog, Offer, OfferStatus
from engine.rank import CandidateScore


def make_candidates(n: int = 5) -> list[CandidateScore]:
    """Create test candidates."""
    return [
        CandidateScore(
            physician_id=f"PHY-{i+1:03d}",
            name=f"Dr. Test{i+1}",
            total_score=1.0 - (i * 0.1),
            components={},
            reasons=[],
            gates=[],
            eligible=True,
            rank=i + 1,
        )
        for i in range(n)
    ]


def test_create_fan_out():
    """Test creating 3-5 ranked offers."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(5)
    hospital_names = {"SYN-001": "Test Hospital"}
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names=hospital_names,
        now=now,
    )
    
    assert len(offers) == 5, f"Expected 5 offers, got {len(offers)}"
    
    # First 3 should be pending_approval (primary)
    assert offers[0].status == OfferStatus.PENDING_APPROVAL
    assert offers[1].status == OfferStatus.PENDING_APPROVAL
    assert offers[2].status == OfferStatus.PENDING_APPROVAL
    
    # Last 2 should be pending_backup (pre-warmed)
    assert offers[3].status == OfferStatus.PENDING_BACKUP
    assert offers[4].status == OfferStatus.PENDING_BACKUP
    
    # Ranks should be correct
    for i, offer in enumerate(offers):
        assert offer.rank == i + 1
    
    print("✓ Fan-out creates 3 primary + 2 backup offers")


def test_approval_and_send():
    """Test approving an offer."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(3)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    # Approve first offer
    result = log.approve(offers[0].offer_id, "Ops Lead", now)
    assert result == True
    assert offers[0].status == OfferStatus.SENT
    assert offers[0].approved_by == "Ops Lead"
    assert offers[0].expires_at is not None
    
    print("✓ Approval sends offer with expiry")


def test_first_accept_wins():
    """Test that first acceptance wins under concurrency."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(3)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    # Approve and send first two
    log.approve(offers[0].offer_id, "Ops Lead", now)
    log.approve(offers[1].offer_id, "Ops Lead", now)
    
    # First acceptance
    success1, msg1 = log.accept(offers[0].offer_id, now)
    assert success1 == True
    assert "confirmed" in msg1.lower()
    assert offers[0].status == OfferStatus.ACCEPTED
    
    # Second offer should be cancelled (all related offers cancelled on accept)
    # When trying to accept a cancelled offer, it fails gracefully
    assert offers[1].status == OfferStatus.CANCELLED
    success2, msg2 = log.accept(offers[1].offer_id, now)
    assert success2 == False
    assert "no longer open" in msg2.lower() or "cancelled" in msg2.lower()
    
    print("✓ First-accept-wins under concurrency")


def test_decline_promotes_backup():
    """Test that declining promotes the next backup."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(4)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    # Approve and send first offer
    log.approve(offers[0].offer_id, "Ops Lead", now)
    
    # Fourth offer should be pending_backup
    assert offers[3].status == OfferStatus.PENDING_BACKUP
    
    # Decline first
    success, msg = log.decline(offers[0].offer_id, now)
    assert success == True
    assert offers[0].status == OfferStatus.DECLINED
    
    # Backup should be promoted to sent
    assert offers[3].status == OfferStatus.SENT
    
    print("✓ Decline promotes next backup")


def test_expiry_on_tick():
    """Test that tick expires old offers."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(2)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    # Approve and send
    log.approve(offers[0].offer_id, "Ops Lead", now)
    
    # Tick past expiry
    future = now + pd.Timedelta(minutes=15)  # Past 10 min expiry
    events = log.tick(future)
    
    assert offers[0].status == OfferStatus.EXPIRED
    assert any(e.event_type == "offer_expired" for e in events)
    
    print("✓ Tick expires old offers")


def test_approval_timeout():
    """Test that pending offers escalate to backup after timeout."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(1)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    assert offers[0].status == OfferStatus.PENDING_APPROVAL
    
    # Tick past approval timeout
    future = now + pd.Timedelta(minutes=6)  # Past 5 min timeout
    events = log.tick(future)
    
    assert offers[0].status == OfferStatus.PENDING_BACKUP
    assert any(e.event_type == "approval_timeout" for e in events)
    
    print("✓ Approval timeout escalates to pending_backup")


def test_event_sourcing():
    """Test that all state changes are logged as events."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(1)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Test Hospital"},
        now=now,
    )
    
    log.approve(offers[0].offer_id, "Ops Lead", now)
    log.accept(offers[0].offer_id, now)
    
    # Should have events for: create, approve, send, accept
    events = log.events_for(offers[0].offer_id)
    event_types = [e.event_type for e in events]
    
    assert "offer_created" in event_types
    assert "offer_approved" in event_types
    assert "offer_sent" in event_types
    assert "offer_accepted" in event_types
    
    print("✓ All state changes logged as events")


def test_accepted_shifts_format():
    """Test that accepted_shifts returns correct format."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(1)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001", "SYN-002"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Hospital 1", "SYN-002": "Hospital 2"},
        now=now,
    )
    
    log.approve(offers[0].offer_id, "Ops Lead", now)
    log.accept(offers[0].offer_id, now)
    
    shifts = log.accepted_shifts()
    assert len(shifts) == 1
    assert shifts[0]["physician_id"] == "PHY-001"
    assert len(shifts[0]["hospitals"]) == 2
    assert isinstance(shifts[0]["start"], pd.Timestamp)
    
    print("✓ accepted_shifts returns correct format")


def test_offer_counts():
    """Test offer counts for eligibility checking."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(2)
    
    # Create offers
    log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates[:1],
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Hospital"},
        now=now,
    )
    
    counts = log.offer_counts()
    assert "PHY-001" in counts
    assert counts["PHY-001"]["open"] == 1
    
    print("✓ Offer counts track correctly")


def test_reject_offer():
    """Test rejecting an offer."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(1)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Hospital"},
        now=now,
    )
    
    result = log.reject(offers[0].offer_id, "Clinical Lead", "Not appropriate")
    assert result == True
    assert offers[0].status == OfferStatus.REJECTED
    
    print("✓ Reject offer works")


def test_is_gap_filled():
    """Test gap filled detection."""
    log = OfferLog()
    now = pd.Timestamp.now()
    candidates = make_candidates(1)
    
    offers = log.create_fan_out(
        gap_id="GAP-001",
        hospital_ids=["SYN-001"],
        candidates=candidates,
        shift_start=now + pd.Timedelta(hours=1),
        shift_end=now + pd.Timedelta(hours=4),
        hospital_names={"SYN-001": "Hospital"},
        now=now,
    )
    
    assert not log.is_gap_filled("GAP-001")
    
    log.approve(offers[0].offer_id, "Ops Lead", now)
    log.accept(offers[0].offer_id, now)
    
    assert log.is_gap_filled("GAP-001")
    
    print("✓ Gap filled detection works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Offer Log Module")
    print("=" * 60)
    
    test_create_fan_out()
    test_approval_and_send()
    test_first_accept_wins()
    test_decline_promotes_backup()
    test_expiry_on_tick()
    test_approval_timeout()
    test_event_sourcing()
    test_accepted_shifts_format()
    test_offer_counts()
    test_reject_offer()
    test_is_gap_filled()
    
    print()
    print("=" * 60)
    print("All offer log tests passed! ✓")
    print("=" * 60)
