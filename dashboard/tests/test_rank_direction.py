"""Ranking direction: busier physicians rank lower; ineligible always last."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.rank import (
    INELIGIBLE_SCORE, WEIGHTS, CandidateScore, rank_candidates, score_candidate,
)


def _phys(pid, consults, available=True):
    return {
        "physician_id": pid,
        "name": pid,
        "available_for_routing": available,
        "derived_state": "Available" if available else "On Call",
        "minutes_to_shift_end": 120,
        "open_consults": consults,
        "consults_this_shift": consults,
    }


def _consult():
    return {
        "consult_id": "CON-001",
        "hospital_id": "SYN-001",
        "consult_type": "Stroke",
        "current_stage": "Triaged",
        "physician_id": "Unassigned",
    }


def _ctx(physicians):
    ref = {"credentials": pd.DataFrame([
        {"physician_id": p["physician_id"], "hospital_id": "SYN-001"} for p in physicians
    ])}
    state = {
        "now": pd.Timestamp("2026-03-17 18:20"),
        "physicians": pd.DataFrame(physicians),
        "consults": pd.DataFrame([_consult()]),
    }
    return state, ref


def test_assignments_penalty_is_negative():
    assert WEIGHTS["assignments_penalty"] == -0.002
    assert INELIGIBLE_SCORE == float("-inf")


def test_busier_physician_ranks_lower():
    idle = _phys("PHY-IDLE", 0)
    busy = _phys("PHY-BUSY", 10)
    state, ref = _ctx([idle, busy])
    scores = [
        score_candidate(idle, _consult(), state, ref),
        score_candidate(busy, _consult(), state, ref),
    ]
    ranked = rank_candidates(scores)
    assert ranked[0].physician_id == "PHY-IDLE"
    assert ranked[1].physician_id == "PHY-BUSY"
    assert ranked[0].total_score > ranked[1].total_score
    assert ranked[1].components["assignments"] == WEIGHTS["assignments_penalty"] * 10


def test_ineligible_always_sort_last():
    eligible_negative = CandidateScore(
        "PHY-NEG", "Neg", -0.5, {"assignments": -0.5}, ["load"], [], True)
    ineligible_zero = CandidateScore(
        "PHY-ZERO", "Zero", 0.0, {}, ["INELIGIBLE"], [], False)
    ineligible_high = CandidateScore(
        "PHY-HIGH", "High", 99.0, {}, ["INELIGIBLE"], [], False)
    ranked = rank_candidates([ineligible_high, eligible_negative, ineligible_zero])
    assert ranked[0].physician_id == "PHY-NEG"
    assert ranked[0].eligible
    assert all(not c.eligible for c in ranked[1:])
    assert ranked[0].rank == 1
    assert all(c.rank is None for c in ranked[1:])


if __name__ == "__main__":
    test_assignments_penalty_is_negative()
    test_busier_physician_ranks_lower()
    test_ineligible_always_sort_last()
    print("PASS test_rank_direction")
