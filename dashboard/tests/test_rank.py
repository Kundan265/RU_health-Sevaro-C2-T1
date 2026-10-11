"""Tests for engine/rank.py - candidate ranking."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.rank import (
    HardGate, check_hard_gates, all_gates_pass, failed_gates,
    score_candidate, rank_candidates, calculate_scarcity_cost,
    format_candidate_summary, CandidateScore, WEIGHTS,
)


def make_physician(pid="PHY-001", available=True, shift_remaining=120, consults=2):
    """Create a test physician dict."""
    return {
        "physician_id": pid,
        "name": f"Dr. {pid[-3:]}",
        "available_for_routing": available,
        "derived_state": "Available" if available else "On Call",
        "minutes_to_shift_end": shift_remaining,
        "open_consults": consults,
        "consults_this_shift": consults,
    }


def make_consult(cid="CON-001", hospital_id="SYN-001"):
    """Create a test consult dict."""
    return {
        "consult_id": cid,
        "hospital_id": hospital_id,
        "consult_type": "Stroke",
        "current_stage": "Triaged",
        "physician_id": "Unassigned",
    }


def make_ref(credentials=None):
    """Create test reference data."""
    if credentials is None:
        credentials = pd.DataFrame([
            {"physician_id": "PHY-001", "hospital_id": "SYN-001"},
            {"physician_id": "PHY-001", "hospital_id": "SYN-002"},
            {"physician_id": "PHY-002", "hospital_id": "SYN-001"},
            {"physician_id": "PHY-003", "hospital_id": "SYN-003"},
        ])
    return {"credentials": credentials}


def make_state(physicians=None, consults=None, now=None):
    """Create test state dict."""
    if now is None:
        now = pd.Timestamp.now()
    if physicians is None:
        physicians = pd.DataFrame([make_physician()])
    if consults is None:
        consults = pd.DataFrame([make_consult()])
    return {
        "now": now,
        "physicians": physicians,
        "consults": consults,
    }


def test_hard_gates_all_pass():
    """Test hard gates when all pass."""
    phys = make_physician(available=True, shift_remaining=120)
    consult = make_consult(hospital_id="SYN-001")
    ref = make_ref()
    
    gates = check_hard_gates(phys, consult, ref["credentials"], pd.Timestamp.now())
    
    assert all_gates_pass(gates), f"All gates should pass, failed: {failed_gates(gates)}"
    assert len(failed_gates(gates)) == 0
    print("✓ All hard gates pass when eligible")


def test_hard_gates_credential_fail():
    """Test hard gate fails when not credentialed."""
    phys = make_physician(pid="PHY-003")  # Only credentialed at SYN-003
    consult = make_consult(hospital_id="SYN-001")  # Need SYN-001
    ref = make_ref()
    
    gates = check_hard_gates(phys, consult, ref["credentials"], pd.Timestamp.now())
    
    assert not all_gates_pass(gates)
    assert "credentialed" in failed_gates(gates)
    print("✓ Credential gate fails when not credentialed")


def test_hard_gates_availability_fail():
    """Test hard gate fails when unavailable."""
    phys = make_physician(available=False)
    consult = make_consult()
    ref = make_ref()
    
    gates = check_hard_gates(phys, consult, ref["credentials"], pd.Timestamp.now())
    
    assert not all_gates_pass(gates)
    assert "available" in failed_gates(gates)
    print("✓ Availability gate fails when unavailable")


def test_hard_gates_shift_remaining_fail():
    """Test hard gate fails when shift ending too soon."""
    phys = make_physician(shift_remaining=5)  # Only 5 min left
    consult = make_consult()
    ref = make_ref()
    
    gates = check_hard_gates(phys, consult, ref["credentials"], pd.Timestamp.now())
    
    assert not all_gates_pass(gates)
    assert "shift_remaining" in failed_gates(gates)
    print("✓ Shift remaining gate fails when <10 min")


def test_score_ineligible():
    """Test scoring returns zero for ineligible candidates."""
    phys = make_physician(available=False)
    consult = make_consult()
    ref = make_ref()
    state = make_state()
    
    score = score_candidate(phys, consult, state, ref)
    
    assert not score.eligible
    assert score.total_score == float("-inf")
    assert "INELIGIBLE" in score.reasons[0]
    print("✓ Ineligible candidates score -inf (sort last)")


def test_score_eligible_components():
    """Test scoring includes expected components."""
    phys = make_physician(available=True, consults=3)
    consult = make_consult()
    ref = make_ref()
    state = make_state(physicians=pd.DataFrame([phys]))
    
    score = score_candidate(phys, consult, state, ref)
    
    assert score.eligible
    assert "assignments" in score.components
    assert len(score.reasons) > 0
    print("✓ Eligible candidates have scored components")


def test_score_continuity_bonus():
    """Test continuity bonus when physician already assigned."""
    phys = make_physician()
    consult = make_consult()
    consult["physician_id"] = "PHY-001"  # Already assigned
    ref = make_ref()
    state = make_state(physicians=pd.DataFrame([phys]))
    
    score = score_candidate(phys, consult, state, ref)
    
    assert score.eligible
    assert "continuity" in score.components
    assert score.components["continuity"] > 0
    assert any("continuity" in r.lower() for r in score.reasons)
    print("✓ Continuity bonus applied")


def test_ranking_order():
    """Test candidates are ranked by score."""
    candidates = [
        CandidateScore("PHY-001", "Dr. A", 0.5, {}, ["Reason A"], [], True),
        CandidateScore("PHY-002", "Dr. B", 0.8, {}, ["Reason B"], [], True),
        CandidateScore("PHY-003", "Dr. C", 0.3, {}, ["Reason C"], [], True),
    ]
    
    ranked = rank_candidates(candidates)
    
    assert ranked[0].physician_id == "PHY-002"  # Highest score first
    assert ranked[1].physician_id == "PHY-001"
    assert ranked[2].physician_id == "PHY-003"
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3
    print("✓ Candidates ranked by score (descending)")


def test_ranking_ineligible_last():
    """Test ineligible candidates are ranked last."""
    candidates = [
        CandidateScore("PHY-001", "Dr. A", 0.5, {}, ["Reason"], [], True),
        CandidateScore("PHY-002", "Dr. B", 0.0, {}, ["INELIGIBLE"], [], False),
        CandidateScore("PHY-003", "Dr. C", 0.8, {}, ["Reason"], [], True),
    ]
    
    ranked = rank_candidates(candidates)
    
    # Eligible first, ineligible last
    assert ranked[0].eligible
    assert ranked[1].eligible
    assert not ranked[2].eligible
    assert ranked[2].rank is None  # Ineligible have no rank
    print("✓ Ineligible candidates ranked last")


def test_scarcity_cost_single_hospital():
    """Test no scarcity cost when only credentialed at one hospital."""
    phys = make_physician(pid="PHY-003")  # Only SYN-003
    consult = make_consult(hospital_id="SYN-003")
    ref = make_ref()
    state = make_state(
        consults=pd.DataFrame([make_consult(hospital_id="SYN-003")])
    )
    
    cost = calculate_scarcity_cost(phys, consult, state, ref)
    
    assert cost == 0.0
    print("✓ No scarcity cost for single-hospital credential")


def test_scarcity_cost_multi_hospital():
    """Test scarcity cost when credentialed at multiple needy hospitals."""
    phys = make_physician(pid="PHY-001")  # SYN-001 and SYN-002
    consult = make_consult(hospital_id="SYN-001")
    ref = make_ref()
    
    # Both hospitals have waiting consults
    state = make_state(
        consults=pd.DataFrame([
            {"consult_id": "C1", "hospital_id": "SYN-001", "physician_id": "Unassigned", "routing_failures": 0},
            {"consult_id": "C2", "hospital_id": "SYN-002", "physician_id": "Unassigned", "routing_failures": 0},
        ])
    )
    
    cost = calculate_scarcity_cost(phys, consult, state, ref)
    
    assert cost > 0  # Should have scarcity cost
    print("✓ Scarcity cost for multi-hospital credential")


def test_format_candidate_summary():
    """Test human-readable summary formatting."""
    candidate = CandidateScore(
        "PHY-001", "Dr. Test", 0.5, 
        {"assignments": -0.006, "availability": 0.08},
        ["Load: 3 consults", "Availability: 80%"],
        [], True, rank=1
    )
    
    summary = format_candidate_summary(candidate)
    
    assert "Dr. Test" in summary
    assert "rank #1" in summary
    assert "Load: 3 consults" in summary
    print("✓ Candidate summary formatted correctly")


def test_hard_gates_never_overridden():
    """Test that hard gates are never overridden regardless of score."""
    # Even with perfect ML predictions, hard gates block
    phys = make_physician(available=False)  # Unavailable
    consult = make_consult()
    ref = make_ref()
    state = make_state(physicians=pd.DataFrame([phys]))
    
    # Simulate perfect ML predictions
    ml_preds = {"response_sla_prob": 0.0, "availability_15": 1.0}
    
    score = score_candidate(phys, consult, state, ref, ml_preds)
    
    # Still ineligible despite great predictions
    assert not score.eligible
    print("✓ Hard gates never overridden (even with perfect ML)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Ranking Module")
    print("=" * 60)
    
    test_hard_gates_all_pass()
    test_hard_gates_credential_fail()
    test_hard_gates_availability_fail()
    test_hard_gates_shift_remaining_fail()
    test_score_ineligible()
    test_score_eligible_components()
    test_score_continuity_bonus()
    test_ranking_order()
    test_ranking_ineligible_last()
    test_scarcity_cost_single_hospital()
    test_scarcity_cost_multi_hospital()
    test_format_candidate_summary()
    test_hard_gates_never_overridden()
    
    print()
    print("=" * 60)
    print("All ranking tests passed! ✓")
    print("=" * 60)
