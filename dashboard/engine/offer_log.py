"""[NEW] Offer Log: Event-sourced store for split-shift offers.

This module provides:
1. Event-sourced offer state (all state changes as events)
2. 3-5 ranked fan-out (send offers to top candidates simultaneously)
3. Pre-warmed backup promotion (next candidate ready if first declines)
4. First-accept-wins under concurrency (idempotent acceptance)
5. Audit-ready event log

The offer lifecycle (from act.py):
  pending_approval -> (timeout) -> pending_backup
  pending_approval/pending_backup -> (approve) -> sent -> accepted/declined/expired
                                  -> (reject) -> rejected
  any open -> (gap filled) -> cancelled/no_longer_needed
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal
import uuid

import pandas as pd

from .rank import CandidateScore, rank_candidates


class OfferStatus(str, Enum):
    """All possible offer statuses."""
    PENDING_APPROVAL = "pending_approval"
    PENDING_BACKUP = "pending_backup"
    SENT = "sent"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    NO_LONGER_NEEDED = "no_longer_needed"


# Open statuses where offer can still change
OPEN_STATUSES = {
    OfferStatus.PENDING_APPROVAL,
    OfferStatus.PENDING_BACKUP,
    OfferStatus.SENT,
}

# Terminal statuses
TERMINAL_STATUSES = {
    OfferStatus.ACCEPTED,
    OfferStatus.DECLINED,
    OfferStatus.EXPIRED,
    OfferStatus.REJECTED,
    OfferStatus.CANCELLED,
    OfferStatus.NO_LONGER_NEEDED,
}


# ============================================================================
# EVENTS - All state changes as events
# ============================================================================

@dataclass
class OfferEvent:
    """An event in the offer log."""
    event_id: str
    timestamp: datetime
    offer_id: str
    event_type: str
    actor: str
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "offer_id": self.offer_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "details": self.details,
        }


# ============================================================================
# OFFER - The offer entity
# ============================================================================

@dataclass
class Offer:
    """A split-shift offer to a physician."""
    offer_id: str
    gap_id: str  # Which coverage gap this addresses
    hospital_ids: list[str]
    physician_id: str
    physician_name: str
    rank: int  # 1 = primary, 2+ = backup
    status: OfferStatus
    created_at: datetime
    
    # Timing
    shift_start: datetime
    shift_end: datetime
    online_by: datetime
    
    # Tracking
    sent_at: datetime | None = None
    expires_at: datetime | None = None
    responded_at: datetime | None = None
    approved_by: str | None = None
    
    # Message
    message: str = ""
    channel: str = "SMS"
    
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES
    
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


# ============================================================================
# OFFER LOG - Event-sourced store
# ============================================================================

class OfferLog:
    """Event-sourced offer management.
    
    Provides:
    - 3-5 ranked fan-out for simultaneous offers
    - Pre-warmed backup promotion
    - First-accept-wins idempotency
    - Full audit trail
    """
    
    # Configuration
    FAN_OUT = 5          # Max simultaneous offers per gap
    PRIMARY_SLOTS = 3    # Initial offers sent
    BACKUP_SLOTS = 2     # Pre-warmed backups
    APPROVAL_TIMEOUT_MIN = 5
    OFFER_EXPIRY_MIN = 10
    
    def __init__(self):
        self._events: list[OfferEvent] = []
        self._offers: dict[str, Offer] = {}
        self._gaps: dict[str, str] = {}  # gap_id -> first accepted offer_id
        self._event_counter = 0
    
    # ---------- Event sourcing ----------
    
    def _emit(self, offer_id: str, event_type: str, actor: str, **details) -> OfferEvent:
        """Emit an event to the log."""
        self._event_counter += 1
        event = OfferEvent(
            event_id=f"EVT-{self._event_counter:06d}",
            timestamp=datetime.now(),
            offer_id=offer_id,
            event_type=event_type,
            actor=actor,
            details=details,
        )
        self._events.append(event)
        return event
    
    def events_for(self, offer_id: str) -> list[OfferEvent]:
        """Get all events for an offer."""
        return [e for e in self._events if e.offer_id == offer_id]
    
    def events_since(self, since: datetime) -> list[OfferEvent]:
        """Get events since a timestamp."""
        return [e for e in self._events if e.timestamp >= since]
    
    def events_df(self) -> pd.DataFrame:
        """Get all events as a DataFrame."""
        return pd.DataFrame([e.to_dict() for e in self._events])
    
    # ---------- Fan-out creation ----------
    
    def create_fan_out(
        self,
        gap_id: str,
        hospital_ids: list[str],
        candidates: list[CandidateScore],
        shift_start: pd.Timestamp,
        shift_end: pd.Timestamp,
        hospital_names: dict[str, str],
        now: pd.Timestamp,
    ) -> list[Offer]:
        """Create 3-5 ranked offers for a coverage gap.
        
        The top PRIMARY_SLOTS candidates get sent immediately.
        The next BACKUP_SLOTS are pre-warmed (ready to promote if primary declines).
        
        Returns list of created offers (may be fewer if not enough candidates).
        """
        # Filter to eligible candidates
        eligible = [c for c in candidates if c.eligible][:self.FAN_OUT]
        
        if not eligible:
            return []
        
        offers = []
        for i, candidate in enumerate(eligible):
            rank = i + 1
            status = (OfferStatus.PENDING_APPROVAL if rank <= self.PRIMARY_SLOTS 
                      else OfferStatus.PENDING_BACKUP)
            
            online_by = shift_start + pd.Timedelta(minutes=15)  # 15 min to get online
            
            offer_id = f"OFR-{str(uuid.uuid4())[:8]}"
            offer = Offer(
                offer_id=offer_id,
                gap_id=gap_id,
                hospital_ids=hospital_ids,
                physician_id=candidate.physician_id,
                physician_name=candidate.name,
                rank=rank,
                status=status,
                created_at=now.to_pydatetime(),
                shift_start=shift_start.to_pydatetime(),
                shift_end=shift_end.to_pydatetime(),
                online_by=online_by.to_pydatetime(),
                message=self._build_message(hospital_ids, hospital_names, shift_start, shift_end, online_by),
                channel="SMS",  # Default channel
            )
            
            self._offers[offer_id] = offer
            self._emit(
                offer_id, "offer_created", "System",
                gap_id=gap_id, physician_id=candidate.physician_id,
                rank=rank, status=status.value, hospitals=hospital_ids,
            )
            
            offers.append(offer)
        
        return offers
    
    def _build_message(
        self,
        hospital_ids: list[str],
        hospital_names: dict[str, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        online_by: pd.Timestamp,
    ) -> str:
        """Build offer message (no PHI)."""
        names = [hospital_names.get(h, h) for h in hospital_ids]
        hours = (end - start).total_seconds() / 3600
        return (
            f"Neuro Ops split-shift offer: cover {', '.join(names)} "
            f"from {start:%H:%M} to {end:%H:%M} ET ({hours:g} h). "
            f"Please be online by {online_by:%H:%M}. "
            f"Reply within {self.OFFER_EXPIRY_MIN} min."
        )
    
    # ---------- Approval flow ----------
    
    def approve(self, offer_id: str, by: str, now: pd.Timestamp) -> bool:
        """Approve an offer and send it."""
        offer = self._offers.get(offer_id)
        if not offer or offer.status not in {OfferStatus.PENDING_APPROVAL, OfferStatus.PENDING_BACKUP}:
            return False
        
        offer.status = OfferStatus.SENT
        offer.sent_at = now.to_pydatetime()
        offer.expires_at = (now + pd.Timedelta(minutes=self.OFFER_EXPIRY_MIN)).to_pydatetime()
        offer.approved_by = by
        
        self._emit(
            offer_id, "offer_approved", by,
            expires_at=offer.expires_at.isoformat(),
        )
        self._emit(
            offer_id, "offer_sent", "System",
            channel=offer.channel, message=offer.message,
        )
        
        return True
    
    def reject(self, offer_id: str, by: str, reason: str = "") -> bool:
        """Reject a pending offer."""
        offer = self._offers.get(offer_id)
        if not offer or offer.status not in {OfferStatus.PENDING_APPROVAL, OfferStatus.PENDING_BACKUP}:
            return False
        
        offer.status = OfferStatus.REJECTED
        
        self._emit(offer_id, "offer_rejected", by, reason=reason)
        
        return True
    
    # ---------- Physician response ----------
    
    def accept(self, offer_id: str, now: pd.Timestamp) -> tuple[bool, str]:
        """Accept an offer. Returns (success, message).
        
        Implements first-accept-wins: if gap already filled, returns friendly message.
        """
        offer = self._offers.get(offer_id)
        if not offer:
            return False, "Offer not found."
        
        # Check if expired
        if offer.status == OfferStatus.SENT and offer.expires_at and now.to_pydatetime() >= offer.expires_at:
            offer.status = OfferStatus.EXPIRED
            self._emit(offer_id, "offer_expired_on_accept", offer.physician_name)
            return False, "This offer has expired."
        
        if offer.status != OfferStatus.SENT:
            return False, f"This offer is no longer open (status: {offer.status.value})."
        
        # First-accept-wins check
        if offer.gap_id in self._gaps:
            offer.status = OfferStatus.NO_LONGER_NEEDED
            self._emit(offer_id, "accepted_but_filled", offer.physician_name)
            return False, "Thank you, this gap has already been filled. No action needed."
        
        # Accept!
        offer.status = OfferStatus.ACCEPTED
        offer.responded_at = now.to_pydatetime()
        self._gaps[offer.gap_id] = offer_id
        
        self._emit(
            offer_id, "offer_accepted", offer.physician_name,
            hospitals=offer.hospital_ids,
        )
        
        # Cancel other offers for this gap
        self._cancel_related_offers(offer)
        
        return True, "Accepted. You're confirmed. Coverage has been updated."
    
    def decline(self, offer_id: str, now: pd.Timestamp) -> tuple[bool, str]:
        """Decline an offer."""
        offer = self._offers.get(offer_id)
        if not offer:
            return False, "Offer not found."
        
        if offer.status != OfferStatus.SENT:
            return False, f"This offer is no longer open (status: {offer.status.value})."
        
        offer.status = OfferStatus.DECLINED
        offer.responded_at = now.to_pydatetime()
        
        self._emit(offer_id, "offer_declined", offer.physician_name)
        
        # Promote backup
        self._promote_backup(offer.gap_id, now)
        
        return True, "Declined. Thank you."
    
    def _cancel_related_offers(self, accepted: Offer) -> None:
        """Cancel other offers for the same gap."""
        for offer in self._offers.values():
            if (offer.offer_id != accepted.offer_id and 
                offer.gap_id == accepted.gap_id and 
                offer.is_open()):
                offer.status = OfferStatus.CANCELLED
                self._emit(offer.offer_id, "cancelled_gap_filled", "System")
    
    def _promote_backup(self, gap_id: str, now: pd.Timestamp) -> Offer | None:
        """Promote next backup to sent status."""
        # Find highest-ranked pending_backup for this gap
        backups = [
            o for o in self._offers.values()
            if o.gap_id == gap_id and o.status == OfferStatus.PENDING_BACKUP
        ]
        
        if not backups:
            return None
        
        # Sort by rank
        backups.sort(key=lambda o: o.rank)
        promoted = backups[0]
        
        # Promote to sent
        promoted.status = OfferStatus.SENT
        promoted.sent_at = now.to_pydatetime()
        promoted.expires_at = (now + pd.Timedelta(minutes=self.OFFER_EXPIRY_MIN)).to_pydatetime()
        
        self._emit(
            promoted.offer_id, "backup_promoted", "System",
            from_rank=promoted.rank,
        )
        self._emit(
            promoted.offer_id, "offer_sent", "System",
            channel=promoted.channel, message=promoted.message,
        )
        
        return promoted
    
    # ---------- Timers ----------
    
    def tick(self, now: pd.Timestamp) -> list[OfferEvent]:
        """Process timeouts and expirations. Returns list of events generated."""
        events = []
        
        for offer in self._offers.values():
            # Approval timeout
            if (offer.status == OfferStatus.PENDING_APPROVAL and 
                (now - pd.Timestamp(offer.created_at)).total_seconds() / 60 >= self.APPROVAL_TIMEOUT_MIN):
                offer.status = OfferStatus.PENDING_BACKUP
                events.append(self._emit(
                    offer.offer_id, "approval_timeout", "System",
                    new_status="pending_backup",
                ))
            
            # Offer expiry
            if (offer.status == OfferStatus.SENT and 
                offer.expires_at and 
                now.to_pydatetime() >= offer.expires_at):
                offer.status = OfferStatus.EXPIRED
                events.append(self._emit(offer.offer_id, "offer_expired", "System"))
                
                # Promote backup
                promoted = self._promote_backup(offer.gap_id, now)
                if promoted:
                    events.extend(self.events_for(promoted.offer_id)[-2:])  # Get the promotion events
        
        return events
    
    # ---------- Queries ----------
    
    def get(self, offer_id: str) -> Offer | None:
        return self._offers.get(offer_id)
    
    def offers_for_gap(self, gap_id: str) -> list[Offer]:
        return [o for o in self._offers.values() if o.gap_id == gap_id]
    
    def open_offers(self) -> list[Offer]:
        return [o for o in self._offers.values() if o.is_open()]
    
    def pending_approvals(self) -> list[Offer]:
        return [o for o in self._offers.values() 
                if o.status in {OfferStatus.PENDING_APPROVAL, OfferStatus.PENDING_BACKUP}]
    
    def sent_offers(self) -> list[Offer]:
        return [o for o in self._offers.values() if o.status == OfferStatus.SENT]
    
    def accepted_shifts(self) -> list[dict]:
        """Get accepted shifts in the format expected by coverage calculation."""
        return [
            {
                "physician_id": o.physician_id,
                "start": pd.Timestamp(o.online_by),
                "end": pd.Timestamp(o.shift_end),
                "hospitals": o.hospital_ids,
            }
            for o in self._offers.values()
            if o.status == OfferStatus.ACCEPTED
        ]
    
    def offer_counts(self) -> dict[str, dict]:
        """Get offer counts per physician for eligibility checking."""
        counts: dict[str, dict] = {}
        for o in self._offers.values():
            c = counts.setdefault(o.physician_id, {"open": 0, "week": 0, "declined_today": 0})
            c["open"] += int(o.is_open())
            c["week"] += int(o.status not in {OfferStatus.PENDING_APPROVAL, OfferStatus.PENDING_BACKUP, OfferStatus.REJECTED})
            c["declined_today"] += int(o.status in {OfferStatus.DECLINED, OfferStatus.EXPIRED})
        return counts
    
    def is_gap_filled(self, gap_id: str) -> bool:
        """Check if a gap has been filled."""
        return gap_id in self._gaps
    
    def summary(self) -> dict:
        """Get summary stats."""
        by_status = {}
        for o in self._offers.values():
            by_status[o.status.value] = by_status.get(o.status.value, 0) + 1
        return {
            "total_offers": len(self._offers),
            "total_events": len(self._events),
            "gaps_filled": len(self._gaps),
            "by_status": by_status,
        }
