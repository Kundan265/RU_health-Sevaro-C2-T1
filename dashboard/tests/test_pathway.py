"""Tests for engine/pathway.py - care pathway model."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.pathway import (
    STAGES, STAGE_RANK, STAGE_TO_TASK, TASK_TYPES,
    get_critical_minutes, get_warning_minutes,
    create_task_for_stage, get_task_type,
    stage_allows_reassignment, is_stage_critical, is_stage_warning,
    HandoffProtocol, ContextPacket,
)


def test_stages_complete():
    """Test that all 10 stages are defined."""
    assert len(STAGES) == 10
    assert STAGES[0] == "Received"
    assert STAGES[-1] == "Signed Off"
    print("✓ All 10 stages defined")


def test_stage_ranks_consecutive():
    """Test that stage ranks are consecutive integers."""
    for i, stage in enumerate(STAGES):
        assert STAGE_RANK[stage] == i, f"Stage {stage} should have rank {i}"
    print("✓ Stage ranks are consecutive")


def test_task_types_map_all_stages():
    """Test that every stage maps to a task type (except Signed Off)."""
    for stage in STAGES:
        task_type = STAGE_TO_TASK.get(stage)
        if stage == "Signed Off":
            assert task_type is None, "Signed Off should have no task type"
        else:
            assert task_type in TASK_TYPES, f"Stage {stage} has invalid task type {task_type}"
    print("✓ All stages map to valid task types")


def test_critical_clocks():
    """Test critical clock values match benchmark definitions."""
    # From benchmark: triage10, Stroke assigned-contact8, imaging20, image-to-video12
    assert get_critical_minutes("Triaged", "Stroke") == 10
    assert get_critical_minutes("Triaged", "General Neuro") == 10
    assert get_critical_minutes("Assigned", "Stroke") == 8
    assert get_critical_minutes("CT/CTA or Imaging Pending", "Stroke") == 20
    assert get_critical_minutes("Imaging Review", "Stroke") == 12
    assert get_critical_minutes("Note Pending", "Stroke") == 45
    assert get_critical_minutes("Note Pending", "General Neuro") == 90
    print("✓ Critical clocks match benchmark")


def test_warning_before_critical():
    """Test that warning thresholds are before critical."""
    for stage in STAGES:
        for ctype in ("Stroke", "General Neuro"):
            warning = get_warning_minutes(stage, ctype)
            critical = get_critical_minutes(stage, ctype)
            if warning is not None and critical is not None:
                assert warning < critical, f"Warning {warning} should be < critical {critical} for {stage}/{ctype}"
    print("✓ Warning thresholds before critical")


def test_handoff_ack_deadlines():
    """Test acknowledgement deadline values."""
    assert HandoffProtocol.ack_deadline_for("Stroke") == 1.0
    assert HandoffProtocol.ack_deadline_for("General Neuro") == 5.0
    assert HandoffProtocol.is_ack_overdue("Stroke", 1.5)
    assert not HandoffProtocol.is_ack_overdue("Stroke", 0.5)
    print("✓ Handoff ack deadlines correct")


def test_create_task_packet():
    """Test context packet creation."""
    now = pd.Timestamp.now()
    consult = {
        "consult_id": "CON-12345",
        "current_stage": "Triaged",
        "consult_type": "Stroke",
        "hospital_id": "SYN-001",
        "stage_entered_at": now - pd.Timedelta(minutes=5),
        "minutes_in_stage": 5.0,
        "physician_id": "PHY-001",
        "routing_failures": 1,
    }
    
    packet = create_task_for_stage(consult, "PHY-002", "Shift ending", now)
    
    assert packet.consult_id == "CON-12345"
    assert packet.task_type == "contact"
    assert packet.stage == "Triaged"
    assert packet.to_physician_id == "PHY-002"
    assert packet.from_physician_id == "PHY-001"
    assert packet.prior_physicians == ["PHY-001"]
    assert packet.ack_deadline_min == 1.0  # Stroke
    assert packet.remaining_deadline_min is not None
    assert packet.remaining_deadline_min < 10  # 10 min critical - 5 min elapsed
    print("✓ Context packet created correctly")


def test_packet_no_phi():
    """Test that context packet has no PHI markers."""
    now = pd.Timestamp.now()
    consult = {
        "consult_id": "CON-12345",
        "current_stage": "Triaged",
        "consult_type": "Stroke",
        "hospital_id": "SYN-001",
        "stage_entered_at": now,
        "minutes_in_stage": 0.0,
        "physician_id": "Unassigned",
    }
    
    packet = create_task_for_stage(consult, "PHY-001", "Initial assignment", now)
    packet_dict = packet.to_dict()
    
    # Check that no common PHI field names are present
    phi_markers = ["patient_name", "dob", "mrn", "ssn", "address", "phone", "email"]
    for marker in phi_markers:
        assert marker not in packet_dict, f"PHI marker '{marker}' found in packet"
    print("✓ Context packet has no PHI markers")


def test_stage_reassignment_rules():
    """Test which stages allow reassignment."""
    # Early stages allow reassignment
    assert stage_allows_reassignment("Received")
    assert stage_allows_reassignment("Triaged")
    assert stage_allows_reassignment("Assigned")
    assert stage_allows_reassignment("CT/CTA or Imaging Pending")
    
    # Cannot reassign during/after video
    assert not stage_allows_reassignment("Video Consult In Progress")
    assert not stage_allows_reassignment("Note Pending")
    assert not stage_allows_reassignment("Documentation")
    assert not stage_allows_reassignment("Signed Off")
    print("✓ Stage reassignment rules correct")


def test_is_stage_critical():
    """Test critical stage detection."""
    assert is_stage_critical("Triaged", "Stroke", 11)  # Over 10 min
    assert not is_stage_critical("Triaged", "Stroke", 9)  # Under 10 min
    assert is_stage_critical("Assigned", "Stroke", 9)  # Over 8 min
    assert not is_stage_critical("Assigned", "Stroke", 7)  # Under 8 min
    print("✓ Critical stage detection correct")


def test_task_type_mapping():
    """Test get_task_type function."""
    assert get_task_type({"current_stage": "Received"}) == "contact"
    assert get_task_type({"current_stage": "Triaged"}) == "contact"
    assert get_task_type({"current_stage": "CT/CTA or Imaging Pending"}) == "imaging_review"
    assert get_task_type({"current_stage": "Video Consult In Progress"}) == "video"
    assert get_task_type({"current_stage": "Note Pending"}) == "documentation"
    print("✓ Task type mapping correct")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pathway Module")
    print("=" * 60)
    
    test_stages_complete()
    test_stage_ranks_consecutive()
    test_task_types_map_all_stages()
    test_critical_clocks()
    test_warning_before_critical()
    test_handoff_ack_deadlines()
    test_create_task_packet()
    test_packet_no_phi()
    test_stage_reassignment_rules()
    test_is_stage_critical()
    test_task_type_mapping()
    
    print()
    print("=" * 60)
    print("All pathway tests passed! ✓")
    print("=" * 60)
