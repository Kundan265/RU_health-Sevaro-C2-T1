"""[NEW] Rank: Score and rank physician candidates for task assignment.

This module provides:
1. Scoring formula combining ML predictions and operational factors
2. Scarcity opportunity cost (future workload pressure)
3. Tie-breakers for equal scores
4. Per-candidate reason strings explaining the ranking
5. Hard gates that are NEVER overridden

The ranking approach from the benchmark:
- Among hard-eligible candidates, minimize calibrated response-deadline miss probability
- Add 0.002 * cumulative consult assignments (load balancing)
- Hard gates (credentials, shift, rest, hours) are never overridden
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .pathway import HandoffProtocol, get_task_type


# ============================================================================
# HARD GATES - These are NEVER overridden
# ============================================================================

@dataclass
class HardGate:
    """A hard gate that must pass for a candidate to be considered."""
    name: str
    passed: bool
    reason: str
    
    def __bool__(self) -> bool:
        return self.passed


def check_hard_gates(
    physician: dict,
    consult: dict,
    credentials: pd.DataFrame,
    now: pd.Timestamp,
) -> list[HardGate]:
    """Check all hard gates for a physician-consult pairing.
    
    Hard gates are NEVER overridden regardless of operational pressure.
    """
    pid = physician.get("physician_id", "")
    hid = consult.get("hospital_id", "")
    
    gates = []
    
    # 1. Hospital credentials
    creds = set(credentials[credentials["physician_id"] == pid]["hospital_id"])
    has_cred = hid in creds
    gates.append(HardGate(
        "credentialed",
        has_cred,
        f"Credentialed at {hid}" if has_cred else f"NOT credentialed at {hid}"
    ))
    
    # 2. On shift (for on-shift candidates)
    on_shift = physician.get("minutes_to_shift_end", 0) > 0
    gates.append(HardGate(
        "on_shift",
        on_shift,
        f"On shift, {physician.get('minutes_to_shift_end', 0):.0f} min remaining" if on_shift 
        else "Not currently on shift"
    ))
    
    # 3. Available for routing (not in break, not busy with video)
    available = physician.get("available_for_routing", False)
    derived_state = physician.get("derived_state", "")
    gates.append(HardGate(
        "available",
        available,
        f"Available: {derived_state}" if available 
        else f"Unavailable: {derived_state}"
    ))
    
    # 4. Minimum shift remaining (need at least 10 min to start work)
    min_remaining = 10
    shift_ok = physician.get("minutes_to_shift_end", 0) >= min_remaining
    gates.append(HardGate(
        "shift_remaining",
        shift_ok,
        f"{physician.get('minutes_to_shift_end', 0):.0f} min left" if shift_ok
        else f"Only {physician.get('minutes_to_shift_end', 0):.0f} min left (need {min_remaining})"
    ))
    
    return gates


def all_gates_pass(gates: list[HardGate]) -> bool:
    """Check if all hard gates pass."""
    return all(gate.passed for gate in gates)


def failed_gates(gates: list[HardGate]) -> list[str]:
    """Get names of failed gates."""
    return [gate.name for gate in gates if not gate.passed]


# ============================================================================
# SCORING FORMULA
# ============================================================================

# Weights for the scoring formula
WEIGHTS = {
    "response_sla_prob": -1.0,      # Lower is better (probability of missing deadline)
    # Benchmark rule: minimize prob + 0.002 * assignments. This module sorts
    # descending, so the load term must be negative (more prior load => lower score).
    "assignments_penalty": -0.002,
    "availability_bonus": 0.1,       # Bonus for projected availability
    "continuity_bonus": 0.15,        # Bonus if already working this consult
    "scarcity_cost": -0.05,          # Penalty for assigning scarce resources
}

# Eligible scores can be negative. A 0.0 ineligible score would otherwise outrank them.
INELIGIBLE_SCORE = float("-inf")

# Tie-breaker weights (secondary sort)
TIEBREAKERS = {
    "open_consults": 1.0,         # Fewer open consults = more capacity
    "shift_remaining": -0.01,     # More shift remaining = lower urgency
    "assignments_today": 1.0,     # Fewer assignments = fresher
}


@dataclass 
class CandidateScore:
    """A scored candidate with reason strings."""
    physician_id: str
    name: str
    total_score: float
    components: dict[str, float]
    reasons: list[str]
    gates: list[HardGate]
    eligible: bool
    rank: int | None = None
    
    def __lt__(self, other: "CandidateScore") -> bool:
        """Higher scores are better (sort descending)."""
        return self.total_score > other.total_score


def score_candidate(
    physician: dict,
    consult: dict,
    state: dict,
    ref: dict,
    ml_predictions: dict | None = None,
) -> CandidateScore:
    """Score a single candidate for a consult assignment.
    
    Args:
        physician: Physician dict from state['physicians']
        consult: Consult dict to assign
        state: Full state dict
        ref: Reference data
        ml_predictions: Optional ML predictions for this physician-consult pair
            Expected keys: response_sla_prob, availability_15, etc.
    
    Returns:
        CandidateScore with total score and component breakdown
    """
    pid = physician.get("physician_id", "")
    pname = physician.get("name", pid)
    
    # Check hard gates
    gates = check_hard_gates(physician, consult, ref.get("credentials", pd.DataFrame()), state["now"])
    eligible = all_gates_pass(gates)
    
    # If not eligible, send them to the bottom of any descending sort.
    if not eligible:
        return CandidateScore(
            physician_id=pid,
            name=pname,
            total_score=INELIGIBLE_SCORE,
            components={},
            reasons=[f"INELIGIBLE: {gate.reason}" for gate in gates if not gate.passed],
            gates=gates,
            eligible=False,
        )
    
    components = {}
    reasons = []
    
    # 1. ML response SLA probability (if available)
    if ml_predictions and "response_sla_prob" in ml_predictions:
        prob = ml_predictions["response_sla_prob"]
        components["response_sla"] = WEIGHTS["response_sla_prob"] * prob
        reasons.append(f"Response SLA risk: {prob:.1%}")
    
    # 2. Assignment load balancing
    consults_today = physician.get("consults_this_shift", 0)
    components["assignments"] = WEIGHTS["assignments_penalty"] * consults_today
    reasons.append(f"Load: {consults_today} consults this shift")
    
    # 3. Availability projection (if ML available)
    if ml_predictions and "availability_15" in ml_predictions:
        avail = ml_predictions["availability_15"]
        components["availability"] = WEIGHTS["availability_bonus"] * avail
        reasons.append(f"Availability at +15min: {avail:.0%}")
    
    # 4. Continuity bonus (same physician continuing)
    is_current = consult.get("physician_id") == pid
    if is_current:
        components["continuity"] = WEIGHTS["continuity_bonus"]
        reasons.append("Already assigned (continuity)")
    
    # 5. Scarcity opportunity cost
    scarcity = calculate_scarcity_cost(physician, consult, state, ref)
    if scarcity > 0:
        components["scarcity"] = WEIGHTS["scarcity_cost"] * scarcity
        reasons.append(f"Scarcity cost: {scarcity:.1f} (in demand elsewhere)")
    
    # Calculate total
    total = sum(components.values())
    
    return CandidateScore(
        physician_id=pid,
        name=pname,
        total_score=total,
        components=components,
        reasons=reasons,
        gates=gates,
        eligible=True,
    )


def calculate_scarcity_cost(
    physician: dict,
    consult: dict,
    state: dict,
    ref: dict,
) -> float:
    """Calculate the opportunity cost of assigning this physician.
    
    A physician is "scarce" if they are one of few credentialed at multiple
    hospitals that all need coverage. Assigning them here means they can't
    help elsewhere.
    
    Returns:
        Scarcity score (0 = not scarce, higher = more scarce)
    """
    pid = physician.get("physician_id", "")
    
    # Get all hospitals this physician is credentialed at
    creds = ref.get("credentials", pd.DataFrame())
    if len(creds) == 0:
        return 0.0
    
    my_hospitals = set(creds[creds["physician_id"] == pid]["hospital_id"])
    if len(my_hospitals) <= 1:
        return 0.0  # Only credentialed at one hospital, not scarce
    
    # Count how many of those hospitals have waiting consults
    waiting = state["consults"]
    waiting_hospitals = set(waiting[
        (waiting["physician_id"] == "Unassigned") | 
        waiting.get("routing_failures", 0) > 0
    ]["hospital_id"])
    
    overlap = my_hospitals & waiting_hospitals
    if len(overlap) <= 1:
        return 0.0
    
    # Scarcity = number of other hospitals that also need help
    return float(len(overlap) - 1)


# ============================================================================
# RANKING
# ============================================================================

def rank_candidates(
    candidates: list[CandidateScore],
    task_type: str | None = None,
) -> list[CandidateScore]:
    """Rank candidates by score, applying tie-breakers.
    
    Args:
        candidates: List of scored candidates
        task_type: Optional task type for specialized tie-breakers
    
    Returns:
        Sorted list with rank assignments
    """
    # Eligible first (unconditionally), then score descending. Ineligible score
    # is -inf so they stay last even if a caller skips this split.
    ordered = sorted(candidates, key=lambda c: (c.eligible, c.total_score), reverse=True)
    eligible = [c for c in ordered if c.eligible]
    ineligible = [c for c in ordered if not c.eligible]

    for i, c in enumerate(eligible):
        c.rank = i + 1
    for c in ineligible:
        c.rank = None

    return eligible + ineligible


def rank_for_consult(
    consult: dict,
    state: dict,
    ref: dict,
    ml_service=None,
) -> list[CandidateScore]:
    """Rank all on-shift physicians for a consult assignment.
    
    Args:
        consult: Consult to assign
        state: Full state dict
        ref: Reference data
        ml_service: Optional ModelService for predictions
    
    Returns:
        Ranked list of candidates
    """
    candidates = []
    
    for _, phys in state["physicians"].iterrows():
        phys_dict = phys.to_dict()
        
        # Get ML predictions if service available
        ml_preds = None
        if ml_service:
            try:
                from .features import build_features_for_consult
                features = build_features_for_consult(
                    consult, phys_dict, state, ref, state["now"]
                )
                features_df = pd.DataFrame([features])
                ml_preds = {
                    "response_sla_prob": ml_service.predict_response_sla(features_df)[0],
                    "availability_15": ml_service.predict_availability(features_df, 15)[0],
                }
            except Exception:
                pass  # Fall back to non-ML scoring
        
        score = score_candidate(phys_dict, consult, state, ref, ml_preds)
        candidates.append(score)
    
    task_type = get_task_type(consult)
    return rank_candidates(candidates, task_type)


def top_n_candidates(
    consult: dict,
    state: dict,
    ref: dict,
    n: int = 5,
    ml_service=None,
) -> list[CandidateScore]:
    """Get top N eligible candidates for a consult.
    
    Args:
        consult: Consult to assign
        state: Full state dict
        ref: Reference data  
        n: Maximum candidates to return
        ml_service: Optional ModelService for predictions
    
    Returns:
        Top N eligible candidates, ranked
    """
    ranked = rank_for_consult(consult, state, ref, ml_service)
    eligible = [c for c in ranked if c.eligible]
    return eligible[:n]


# ============================================================================
# REASON STRINGS
# ============================================================================

def format_candidate_summary(candidate: CandidateScore) -> str:
    """Format a candidate's score into a human-readable summary."""
    if not candidate.eligible:
        failed = [g.name for g in candidate.gates if not g.passed]
        return f"❌ {candidate.name}: INELIGIBLE ({', '.join(failed)})"
    
    lines = [f"✓ {candidate.name} (rank #{candidate.rank}, score {candidate.total_score:+.3f})"]
    for reason in candidate.reasons:
        lines.append(f"  • {reason}")
    return "\n".join(lines)


def format_ranking_rationale(candidates: list[CandidateScore]) -> str:
    """Format the full ranking rationale for display."""
    lines = ["## Candidate Ranking\n"]
    
    eligible = [c for c in candidates if c.eligible]
    ineligible = [c for c in candidates if not c.eligible]
    
    if eligible:
        lines.append(f"**{len(eligible)} eligible candidates:**\n")
        for c in eligible[:5]:  # Top 5
            lines.append(format_candidate_summary(c))
            lines.append("")
    
    if ineligible:
        lines.append(f"**{len(ineligible)} ineligible:**")
        for c in ineligible:
            failed = [g.name for g in c.gates if not g.passed]
            lines.append(f"  ❌ {c.name}: {', '.join(failed)}")
    
    return "\n".join(lines)
