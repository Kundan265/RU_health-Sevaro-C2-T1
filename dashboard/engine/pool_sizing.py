"""[NEW] Pool Sizing: Demand-matched pool sizing with flex fallback.

This module addresses the blind cyclic rotation flaw identified in the benchmark.

The problem:
- Fixed pool sizes don't adapt to demand variations
- Cyclic rotation treats all time periods equally
- Results in over/under staffing depending on time of day

The fix:
1. Demand-matched sizing: Use ML demand predictions to size pools dynamically
2. Flex fallback: When primary pool is exhausted, fall back to cross-credentialed
3. Hospital priority: During shortages, prioritize by critical patient load
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


# ============================================================================
# DEMAND-BASED POOL SIZING
# ============================================================================

# Base staffing ratios
BASE_RATIO = {
    "Stroke": 1.5,       # 1.5 doctors per concurrent Stroke consult
    "General Neuro": 1.0, # 1.0 doctors per General Neuro consult
}

# Minimum pool sizes by task type
MIN_POOL = {
    "contact": 2,
    "imaging_review": 1,
    "video": 2,
    "documentation": 1,
}

# Maximum queue before flex is triggered
MAX_QUEUE_MINUTES = 10


@dataclass
class PoolSizeRecommendation:
    """Recommended pool size for a task type."""
    task_type: str
    current_size: int
    recommended_size: int
    demand_forecast: float  # Expected arrivals in next hour
    queue_depth: int
    flex_active: bool
    rationale: str
    
    @property
    def delta(self) -> int:
        return self.recommended_size - self.current_size


def calculate_pool_size(
    task_type: str,
    demand_forecast: float,
    current_queue: int,
    available_physicians: int,
    consult_type: str = "General Neuro",
) -> PoolSizeRecommendation:
    """Calculate recommended pool size based on demand forecast.
    
    Args:
        task_type: The task type (contact, imaging_review, video, documentation)
        demand_forecast: Expected arrivals in next hour
        current_queue: Current queue depth
        available_physicians: Number of physicians currently available
        consult_type: Stroke or General Neuro
    
    Returns:
        PoolSizeRecommendation with sizing and rationale
    """
    # Get base ratio
    ratio = BASE_RATIO.get(consult_type, 1.0)
    
    # Calculate demand-based size
    # Formula: max(min_pool, demand * ratio + queue_pressure)
    queue_pressure = max(0, current_queue - 2) * 0.5  # Add 0.5 per queued beyond 2
    demand_size = int(np.ceil(demand_forecast * ratio + queue_pressure))
    
    # Apply minimum
    min_size = MIN_POOL.get(task_type, 1)
    recommended = max(min_size, demand_size)
    
    # Check if flex is needed
    flex_active = current_queue > MAX_QUEUE_MINUTES or recommended > available_physicians
    
    # Build rationale
    if flex_active:
        rationale = (
            f"Flex mode active: demand ({demand_forecast:.1f}/h) exceeds pool capacity. "
            f"Cross-credentialed fallback enabled."
        )
    elif recommended > available_physicians:
        rationale = (
            f"Pool undersized: {available_physicians} available vs {recommended} recommended "
            f"for {demand_forecast:.1f}/h demand"
        )
    else:
        rationale = (
            f"Pool sized to demand: {recommended} physicians for {demand_forecast:.1f}/h expected"
        )
    
    return PoolSizeRecommendation(
        task_type=task_type,
        current_size=available_physicians,
        recommended_size=recommended,
        demand_forecast=demand_forecast,
        queue_depth=current_queue,
        flex_active=flex_active,
        rationale=rationale,
    )


# ============================================================================
# FLEX FALLBACK
# ============================================================================

@dataclass
class FlexPoolMember:
    """A physician who can flex into another pool."""
    physician_id: str
    primary_pool: str
    flex_to: list[str]
    current_load: int
    priority: float  # Lower = higher priority to flex
    
    def can_flex_to(self, task_type: str) -> bool:
        return task_type in self.flex_to


def identify_flex_candidates(
    physicians: pd.DataFrame,
    credentials: pd.DataFrame,
    target_task: str,
    target_hospital: str,
) -> list[FlexPoolMember]:
    """Identify physicians who can flex to help with a task type.
    
    Args:
        physicians: Current physician state
        credentials: Credential mapping
        target_task: Task type needing help
        target_hospital: Hospital needing coverage
    
    Returns:
        List of FlexPoolMember ordered by flex priority
    """
    # Get physicians credentialed at target hospital
    creds = credentials[credentials["hospital_id"] == target_hospital]
    credentialed_ids = set(creds["physician_id"])
    
    candidates = []
    for _, phys in physicians.iterrows():
        pid = phys["physician_id"]
        if pid not in credentialed_ids:
            continue
        if not phys.get("available_for_routing", False):
            continue
        
        # Determine flex eligibility based on current state
        current_pool = _infer_primary_pool(phys)
        if current_pool == target_task:
            continue  # Already in the target pool
        
        # Calculate flex priority (lower = flex first)
        # Prefer: low load, far from shift end, not in video
        load = phys.get("open_consults", 0)
        shift_remaining = phys.get("minutes_to_shift_end", 0)
        in_video = "Video" in phys.get("derived_state", "")
        
        priority = load * 10 - shift_remaining * 0.1 + (100 if in_video else 0)
        
        candidates.append(FlexPoolMember(
            physician_id=pid,
            primary_pool=current_pool,
            flex_to=[target_task],  # Simplified
            current_load=load,
            priority=priority,
        ))
    
    # Sort by priority
    candidates.sort(key=lambda c: c.priority)
    
    return candidates


def _infer_primary_pool(physician: dict) -> str:
    """Infer a physician's primary pool from their state."""
    state = physician.get("derived_state", "")
    if "Video" in state:
        return "video"
    if "Documentation" in state or "Note" in state:
        return "documentation"
    if "Imaging" in state:
        return "imaging_review"
    return "contact"


# ============================================================================
# HOSPITAL PRIORITY
# ============================================================================

def prioritize_hospitals(
    coverage: dict,
    consults: pd.DataFrame,
) -> list[tuple[str, float]]:
    """Prioritize hospitals by critical patient load.
    
    Returns list of (hospital_id, priority_score) sorted by priority (higher first).
    """
    priorities = []
    
    summary = coverage.get("summary", pd.DataFrame())
    if len(summary) == 0:
        return priorities
    
    for _, row in summary.iterrows():
        hid = row.get("hospital_id", "")
        
        # Count critical patients at this hospital
        hosp_consults = consults[consults["hospital_id"] == hid]
        stroke_count = len(hosp_consults[hosp_consults["consult_type"] == "Stroke"])
        unassigned = len(hosp_consults[hosp_consults["physician_id"] == "Unassigned"])
        
        # Check if gap exists
        has_gap = row.get("first_gap") is not None
        
        # Priority score
        # Higher = more urgent
        score = (
            stroke_count * 50 +
            unassigned * 30 +
            (100 if has_gap else 0) +
            row.get("last_60_min", 0) * 2  # Recent demand
        )
        
        priorities.append((hid, score))
    
    # Sort by score descending
    priorities.sort(key=lambda x: -x[1])
    
    return priorities


# ============================================================================
# INTEGRATED POOL MANAGEMENT
# ============================================================================

def manage_pools(
    state: dict,
    ref: dict,
    coverage: dict,
    ml_service=None,
) -> dict:
    """Integrated pool management with demand matching and flex fallback.
    
    Returns dict with:
    - recommendations: List of PoolSizeRecommendation per task type
    - flex_candidates: List of FlexPoolMember available to flex
    - hospital_priority: Ordered list of hospitals by criticality
    - actions: Suggested pool adjustments
    """
    # ALL-hospital dummy features and shared physician counts do not meet the
    # training or hospital-capacity contract. Keep sizing utilities available for
    # controlled experiments, but do not manufacture live demand or flex advice.
    return {"status": "unavailable",
            "note": "Forecast unavailable: hospital/task demand and shared credentialed capacity "
                    "are not validated. No assumed 0.5 arrivals/hour is substituted.",
            "recommendations": [], "flex_candidates": [], "actions": [],
            "hospital_priority": prioritize_hospitals(coverage, state.get("consults", pd.DataFrame())),
            "timestamp": state.get("now")}


def _stage_pattern(task_type: str) -> str:
    """Get stage matching pattern for a task type."""
    patterns = {
        "contact": "Received|Triaged|Assigned|Contacted",
        "imaging_review": "Imaging",
        "video": "Video",
        "documentation": "Note|Documentation",
    }
    return patterns.get(task_type, "")
