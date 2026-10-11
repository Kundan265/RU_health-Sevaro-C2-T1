"""[NEW] Pathway: Care pathway model with 4 task types over 10 stages.

This module defines:
1. Stage progression model (10 stages from Received to Signed Off)
2. Task types: contact, imaging_review, video, documentation
3. Critical stage clocks and SLA deadlines
4. Handoff protocol for task transitions
5. Minimum context packet (no PHI markers)

Based on benchmark definitions and the existing STAGE_LIMITS in detect.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

# ============================================================================
# STAGES - The 10 stages of a teleneurology consult
# ============================================================================

STAGES = [
    "Received",                    # 0: Consult request received
    "Triaged",                     # 1: Urgency assessed, awaiting assignment
    "Assigned",                    # 2: Doctor assigned, not yet contacted
    "Provider Contacted",          # 3: Initial contact made
    "CT/CTA or Imaging Pending",   # 4: Waiting for imaging results
    "Imaging Review",              # 5: Imaging back, needs review
    "Video Consult In Progress",   # 6: Live video with patient
    "Note Pending",                # 7: Video complete, awaiting note
    "Documentation",               # 8: Note in progress
    "Signed Off",                  # 9: Consult complete
]

STAGE_RANK = {s: i for i, s in enumerate(STAGES)}

# ============================================================================
# TASK TYPES - Work items derived from stage transitions
# ============================================================================

TaskType = Literal["contact", "imaging_review", "video", "documentation"]

TASK_TYPES = {
    "contact": "Initial physician contact and triage",
    "imaging_review": "Review CT/CTA imaging results",
    "video": "Video consultation with patient",
    "documentation": "Clinical notes and sign-off",
}

# Which stages correspond to which task type
STAGE_TO_TASK = {
    "Received": "contact",
    "Triaged": "contact",
    "Assigned": "contact",
    "Provider Contacted": "contact",
    "CT/CTA or Imaging Pending": "imaging_review",
    "Imaging Review": "imaging_review",
    "Video Consult In Progress": "video",
    "Note Pending": "documentation",
    "Documentation": "documentation",
    "Signed Off": None,  # Terminal stage
}

# ============================================================================
# CRITICAL STAGE CLOCKS - SLA deadlines by stage and consult type
# ============================================================================

# (stage, consult_type) -> critical_minutes
# From benchmark: triage10, Stroke assigned-contact8, imaging20, image-to-video12, notes Stroke45/routine90
CRITICAL_CLOCKS = {
    # All consult types
    ("Triaged", "All"): 10,           # Must be assigned within 10 min
    
    # Stroke-specific (more urgent)
    ("Assigned", "Stroke"): 8,         # Contact within 8 min of assignment
    ("CT/CTA or Imaging Pending", "Stroke"): 20,  # Imaging back within 20 min
    ("Imaging Review", "Stroke"): 12,  # Video started within 12 min
    ("Note Pending", "Stroke"): 45,    # Note signed within 45 min of video
    ("Documentation", "Stroke"): 45,
    
    # General Neuro (more relaxed)
    ("Note Pending", "General Neuro"): 90,
    ("Documentation", "General Neuro"): 90,
}

# Warning thresholds (before critical)
WARNING_RATIOS = {
    "Stroke": 0.5,       # Warn at 50% of critical for Stroke
    "General Neuro": 0.6,  # Warn at 60% for routine
}


def get_critical_minutes(stage: str, consult_type: str) -> int | None:
    """Get critical clock minutes for a stage/type combination."""
    # Try specific first, then "All"
    key = (stage, consult_type)
    if key in CRITICAL_CLOCKS:
        return CRITICAL_CLOCKS[key]
    key = (stage, "All")
    if key in CRITICAL_CLOCKS:
        return CRITICAL_CLOCKS[key]
    return None


def get_warning_minutes(stage: str, consult_type: str) -> int | None:
    """Get warning threshold minutes for a stage/type combination."""
    critical = get_critical_minutes(stage, consult_type)
    if critical is None:
        return None
    ratio = WARNING_RATIOS.get(consult_type, 0.6)
    return int(critical * ratio)


# ============================================================================
# CONTEXT PACKET - Minimum data for handoffs (no PHI)
# ============================================================================

@dataclass
class ContextPacket:
    """Minimum context for task handoff - excludes PHI.
    
    This packet contains ONLY operational data needed for task execution.
    Patient identifiers, names, DOB, and medical details are fetched
    separately via secure clinical APIs when needed.
    """
    # Identifiers (synthetic IDs, not PHI)
    consult_id: str
    task_id: str
    
    # Task metadata
    task_type: TaskType
    stage: str
    consult_type: str  # "Stroke" or "General Neuro"
    hospital_id: str
    
    # Timing (critical for SLA)
    stage_entered_at: datetime
    minutes_in_stage: float
    critical_deadline_min: int | None
    remaining_deadline_min: float | None
    
    # Routing context
    from_physician_id: str | None  # NULL if new assignment
    to_physician_id: str
    handoff_reason: str
    
    # History (for continuity)
    prior_physicians: list[str] = field(default_factory=list)
    routing_attempts: int = 0
    routing_history: list[dict] = field(default_factory=list)
    
    # Acknowledgement tracking
    packet_created_at: datetime = field(default_factory=datetime.now)
    ack_deadline_min: float = 5.0  # Routine default
    deadline_clock_start: datetime | None = None
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "consult_id": self.consult_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "stage": self.stage,
            "consult_type": self.consult_type,
            "hospital_id": self.hospital_id,
            "stage_entered_at": self.stage_entered_at.isoformat() if self.stage_entered_at else None,
            "minutes_in_stage": self.minutes_in_stage,
            "critical_deadline_min": self.critical_deadline_min,
            "remaining_deadline_min": self.remaining_deadline_min,
            "from_physician_id": self.from_physician_id,
            "to_physician_id": self.to_physician_id,
            "handoff_reason": self.handoff_reason,
            "prior_physicians": self.prior_physicians,
            "routing_attempts": self.routing_attempts,
            "routing_history": self.routing_history,
            "packet_created_at": self.packet_created_at.isoformat(),
            "ack_deadline_min": self.ack_deadline_min,
            "deadline_clock_start": self.deadline_clock_start.isoformat() if self.deadline_clock_start else None,
        }


# ============================================================================
# HANDOFF PROTOCOL - Rules for transferring work between physicians
# ============================================================================

@dataclass
class HandoffProtocol:
    """Handoff requirements based on task type and urgency."""
    
    # Acknowledgement deadlines (minutes)
    ACK_URGENT = 1.0   # Stroke
    ACK_ROUTINE = 5.0  # General Neuro
    
    # Maximum retries before escalation
    MAX_OFFER_ATTEMPTS = 3
    
    # Pre-warm backup slots
    BACKUP_SLOTS = 2
    
    @classmethod
    def ack_deadline_for(cls, consult_type: str) -> float:
        """Get acknowledgement deadline in minutes."""
        return cls.ACK_URGENT if consult_type == "Stroke" else cls.ACK_ROUTINE
    
    @classmethod
    def is_ack_overdue(cls, consult_type: str, elapsed_min: float) -> bool:
        """Check if acknowledgement is overdue."""
        return elapsed_min > cls.ack_deadline_for(consult_type)


# ============================================================================
# TASK CREATION - Create tasks from consult state
# ============================================================================

def create_task_for_stage(
    consult: dict,
    to_physician_id: str,
    handoff_reason: str,
    now: pd.Timestamp | datetime,
) -> ContextPacket:
    """Create a context packet for a task assignment.
    
    Args:
        consult: Consult dict from state['consults']
        to_physician_id: Physician being assigned
        handoff_reason: Why this handoff is happening
        now: Current timestamp
    
    Returns:
        ContextPacket with minimum context for the handoff
    """
    stage = consult.get("current_stage", "Received")
    ctype = consult.get("consult_type", "General Neuro")
    
    # Calculate deadline
    critical = get_critical_minutes(stage, ctype)
    stage_entered = consult.get("stage_entered_at")
    if stage_entered is None or pd.isna(stage_entered):
        stage_entered = pd.Timestamp(now) - pd.Timedelta(minutes=consult.get("minutes_in_stage", 0) or 0)
    stage_entered = pd.Timestamp(stage_entered)
    deadline_start = stage_entered
    note_start = consult.get("note_clock_start")
    if stage in ("Note Pending", "Documentation") and note_start is not None and pd.notna(note_start):
        deadline_start = pd.Timestamp(note_start)
    elapsed = (now - deadline_start).total_seconds() / 60
    remaining = critical - elapsed if critical is not None else None
    
    # Build prior physician list
    prior = list(consult.get("prior_physicians", []))
    current_holder = consult.get("physician_id")
    if current_holder and current_holder != "Unassigned" and current_holder not in prior:
        prior.append(current_holder)
    
    return ContextPacket(
        consult_id=consult.get("consult_id", ""),
        task_id=f"TSK-{consult.get('consult_id', 'X')[-6:]}-{stage[:3]}",
        task_type=STAGE_TO_TASK.get(stage, "contact"),
        stage=stage,
        consult_type=ctype,
        hospital_id=consult.get("hospital_id", ""),
        stage_entered_at=stage_entered.to_pydatetime(),
        minutes_in_stage=(now - stage_entered).total_seconds() / 60,
        critical_deadline_min=critical,
        remaining_deadline_min=remaining,
        from_physician_id=current_holder if current_holder != "Unassigned" else None,
        to_physician_id=to_physician_id,
        handoff_reason=handoff_reason,
        prior_physicians=prior,
        routing_attempts=consult.get("routing_failures", 0),
        routing_history=list(consult.get("routing_history", [])),
        ack_deadline_min=HandoffProtocol.ack_deadline_for(ctype),
        packet_created_at=pd.Timestamp(now).to_pydatetime(),
        deadline_clock_start=deadline_start.to_pydatetime(),
    )


def get_task_type(consult: dict) -> TaskType:
    """Get the current task type for a consult."""
    stage = consult.get("current_stage", "Received")
    return STAGE_TO_TASK.get(stage, "contact")


def stage_allows_reassignment(stage: str) -> bool:
    """Check if the stage allows reassignment (not in terminal or documentation)."""
    # Cannot reassign during video or after (documentation/signed off)
    terminal_stages = {"Video Consult In Progress", "Note Pending", "Documentation", "Signed Off"}
    return stage not in terminal_stages


def is_stage_critical(stage: str, consult_type: str, minutes_in_stage: float) -> bool:
    """Check if the stage is past critical deadline."""
    critical = get_critical_minutes(stage, consult_type)
    return critical is not None and minutes_in_stage >= critical


def is_stage_warning(stage: str, consult_type: str, minutes_in_stage: float) -> bool:
    """Check if the stage is past warning but not critical."""
    warning = get_warning_minutes(stage, consult_type)
    critical = get_critical_minutes(stage, consult_type)
    if warning is None:
        return False
    return minutes_in_stage >= warning and (critical is None or minutes_in_stage < critical)


# ============================================================================
# CONTINUITY TRACKING - Track physician continuity across a consult
# ============================================================================

def calculate_continuity_loss(consult: dict) -> int:
    """Count how many handoffs have occurred for this consult.
    
    Higher numbers indicate more continuity loss (more handoffs).
    Zero means same physician throughout.
    """
    routing_failures = consult.get("routing_failures", 0)
    # Each routing failure implies at least one attempted handoff
    # In a full implementation, we'd track actual physician changes
    return routing_failures


def is_continuity_at_risk(consult: dict, proposed_action: str) -> bool:
    """Check if the proposed action would cause continuity loss.
    
    Args:
        consult: Current consult state
        proposed_action: "reassign" or "split_shift"
    
    Returns:
        True if this would be the 2nd+ handoff (continuity loss)
    """
    if proposed_action == "escalate":
        return False  # Escalation doesn't change assignment
    
    current_loss = calculate_continuity_loss(consult)
    return current_loss >= 1  # Already had one handoff
