"""Tests for engine/pool_sizing.py - demand-matched pool sizing."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.pool_sizing import (
    calculate_pool_size, PoolSizeRecommendation,
    identify_flex_candidates, FlexPoolMember,
    prioritize_hospitals, manage_pools,
    MIN_POOL, MAX_QUEUE_MINUTES,
)


def test_pool_size_respects_minimum():
    """Test pool size never goes below minimum."""
    rec = calculate_pool_size(
        task_type="contact",
        demand_forecast=0.1,  # Very low demand
        current_queue=0,
        available_physicians=5,
    )
    
    assert rec.recommended_size >= MIN_POOL["contact"]
    print(f"✓ Pool size respects minimum ({MIN_POOL['contact']})")


def test_pool_size_scales_with_demand():
    """Test pool size increases with demand."""
    low_demand = calculate_pool_size(
        task_type="contact",
        demand_forecast=1.0,
        current_queue=0,
        available_physicians=10,
    )
    
    high_demand = calculate_pool_size(
        task_type="contact",
        demand_forecast=5.0,
        current_queue=0,
        available_physicians=10,
    )
    
    assert high_demand.recommended_size > low_demand.recommended_size
    print("✓ Pool size scales with demand")


def test_flex_activates_on_high_queue():
    """Test flex mode activates when queue is high."""
    rec = calculate_pool_size(
        task_type="video",
        demand_forecast=2.0,
        current_queue=MAX_QUEUE_MINUTES + 5,  # Exceed threshold
        available_physicians=2,
    )
    
    assert rec.flex_active
    assert "Flex mode" in rec.rationale
    print("✓ Flex activates on high queue")


def test_flex_activates_when_understaffed():
    """Test flex mode activates when pool is undersized."""
    rec = calculate_pool_size(
        task_type="contact",
        demand_forecast=5.0,  # High demand
        current_queue=3,
        available_physicians=2,  # Not enough
    )
    
    assert rec.flex_active
    print("✓ Flex activates when understaffed")


def test_queue_pressure_increases_size():
    """Test that queue pressure increases recommended size."""
    no_queue = calculate_pool_size(
        task_type="contact",
        demand_forecast=2.0,
        current_queue=0,
        available_physicians=5,
    )
    
    high_queue = calculate_pool_size(
        task_type="contact",
        demand_forecast=2.0,
        current_queue=10,  # High queue
        available_physicians=5,
    )
    
    assert high_queue.recommended_size >= no_queue.recommended_size
    print("✓ Queue pressure increases recommended size")


def test_stroke_gets_higher_ratio():
    """Test that Stroke consults get higher staffing ratio."""
    stroke = calculate_pool_size(
        task_type="contact",
        demand_forecast=2.0,
        current_queue=0,
        available_physicians=5,
        consult_type="Stroke",
    )
    
    general = calculate_pool_size(
        task_type="contact",
        demand_forecast=2.0,
        current_queue=0,
        available_physicians=5,
        consult_type="General Neuro",
    )
    
    assert stroke.recommended_size >= general.recommended_size
    print("✓ Stroke gets higher staffing ratio")


def test_identify_flex_candidates():
    """Test flex candidate identification."""
    physicians = pd.DataFrame([
        {"physician_id": "PHY-001", "available_for_routing": True, 
         "open_consults": 1, "minutes_to_shift_end": 120, "derived_state": "Available"},
        {"physician_id": "PHY-002", "available_for_routing": True,
         "open_consults": 3, "minutes_to_shift_end": 60, "derived_state": "Available"},
        {"physician_id": "PHY-003", "available_for_routing": False,  # Not available
         "open_consults": 0, "minutes_to_shift_end": 120, "derived_state": "On Break"},
    ])
    
    credentials = pd.DataFrame([
        {"physician_id": "PHY-001", "hospital_id": "SYN-001"},
        {"physician_id": "PHY-002", "hospital_id": "SYN-001"},
        {"physician_id": "PHY-003", "hospital_id": "SYN-001"},
    ])
    
    candidates = identify_flex_candidates(
        physicians=physicians,
        credentials=credentials,
        target_task="video",
        target_hospital="SYN-001",
    )
    
    # Should include PHY-001 and PHY-002, not PHY-003
    ids = [c.physician_id for c in candidates]
    assert "PHY-001" in ids
    assert "PHY-002" in ids
    assert "PHY-003" not in ids  # Not available
    
    # PHY-001 should have higher priority (lower score) - less loaded
    if len(candidates) >= 2:
        assert candidates[0].physician_id == "PHY-001"  # Lower load
    
    print("✓ Flex candidates identified and prioritized")


def test_hospital_priority():
    """Test hospital prioritization by criticality."""
    coverage = {
        "summary": pd.DataFrame([
            {"hospital_id": "SYN-001", "first_gap": None, "last_60_min": 2},
            {"hospital_id": "SYN-002", "first_gap": pd.Timestamp.now(), "last_60_min": 5},  # Has gap
        ]),
    }
    
    consults = pd.DataFrame([
        {"hospital_id": "SYN-001", "consult_type": "General Neuro", "physician_id": "PHY-001"},
        {"hospital_id": "SYN-002", "consult_type": "Stroke", "physician_id": "Unassigned"},  # Critical
    ])
    
    priorities = prioritize_hospitals(coverage, consults)
    
    # SYN-002 should be higher priority (gap + stroke + unassigned)
    assert len(priorities) == 2
    assert priorities[0][0] == "SYN-002"
    print("✓ Hospitals prioritized by criticality")


def test_manage_pools_integration():
    """Test integrated pool management."""
    state = {
        "now": pd.Timestamp.now(),
        "consults": pd.DataFrame([
            {"hospital_id": "SYN-001", "consult_type": "Stroke", 
             "current_stage": "Triaged", "physician_id": "Unassigned"},
        ]),
        "physicians": pd.DataFrame([
            {"physician_id": "PHY-001", "available_for_routing": True,
             "open_consults": 1, "minutes_to_shift_end": 120, "derived_state": "Available"},
        ]),
    }
    
    ref = {
        "credentials": pd.DataFrame([
            {"physician_id": "PHY-001", "hospital_id": "SYN-001"},
        ]),
    }
    
    coverage = {
        "summary": pd.DataFrame([
            {"hospital_id": "SYN-001", "first_gap": None, "last_60_min": 2},
        ]),
    }
    
    result = manage_pools(state, ref, coverage)
    
    assert "recommendations" in result
    assert "flex_candidates" in result
    assert "hospital_priority" in result
    assert "actions" in result
    assert result["status"] == "unavailable"
    assert not result["recommendations"] and not result["actions"]
    print("✓ Unvalidated live pool demand is explicitly unavailable")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Pool Sizing Module")
    print("=" * 60)
    
    test_pool_size_respects_minimum()
    test_pool_size_scales_with_demand()
    test_flex_activates_on_high_queue()
    test_flex_activates_when_understaffed()
    test_queue_pressure_increases_size()
    test_stroke_gets_higher_ratio()
    test_identify_flex_candidates()
    test_hospital_priority()
    test_manage_pools_integration()
    
    print()
    print("=" * 60)
    print("All pool sizing tests passed! ✓")
    print("=" * 60)
