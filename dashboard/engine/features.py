"""[NEW] Features: Build the 31-column feature schema for ML predictions.

The schema matches the trained benchmark models exactly:
- 5 categorical: policy, hospital_id, task_type, consult_type, physician_id
- 26 numeric: time features, workload features, capacity features, signal features
"""
from __future__ import annotations
import json
from pathlib import Path
from .replay import shift_window
from .pathway import get_critical_minutes

import numpy as np
import pandas as pd

# Feature schema - must match the models exactly
CAT = ['policy', 'hospital_id', 'task_type', 'consult_type', 'physician_id']
NUM = [
    'hour', 'day_of_week', 'month', 'weekend',
    'arrivals_last15', 'arrivals_last60', 'releases_last60',
    'queue_length', 'eligible_capacity', 'busy_credentialed', 'active_work',
    'shift_remaining_min',
    'scheduled_on_shift', 'scheduled_on_shift_15', 'scheduled_on_shift_60', 'scheduled_on_shift_90',
    'stage_age_min', 'remaining_deadline_min',
    'agent_offline', 'external_camera', 'signal_freshness_min',
    'response_history_rate', 'response_history_count',
    'packet_complete', 'ack_deadline_min', 'baseline_hourly_count'
]
FEATURES = CAT + NUM

# Task type mappings for the model
TASK_TYPES = ['contact', 'imaging_review', 'video', 'documentation', 'arrival']


def build_features_for_consult(
    consult: dict,
    physician: dict | None,
    state: dict,
    ref: dict,
    now: pd.Timestamp,
    policy: str = "as_is"
) -> dict:
    """Build a single feature record for predicting wait/SLA for a consult.
    
    Args:
        consult: Consult dict from state['consults']
        physician: Physician dict or None if predicting for pool
        state: Full state dict from replay
        ref: Reference data dict
        now: Current timestamp
        policy: 'as_is' or 'specialized'
    
    Returns:
        Dict with all 31 features
    """
    # Time features
    hour = now.hour
    day_of_week = now.dayofweek
    month = now.month
    weekend = int(now.dayofweek >= 5)
    
    # Hospital info
    hid = consult.get('hospital_id', '')
    ctype = consult.get('consult_type', 'General Neuro')
    
    # Map stage to task type
    stage = consult.get('current_stage', 'Received')
    task_type = _stage_to_task_type(stage)
    
    # Physician features
    if physician is not None:
        pid = physician.get('physician_id', '')
        shift_remaining = physician.get('minutes_to_shift_end', 0) or 0
        scheduled_on_shift = 1
        agent_offline = int(physician.get('desktop_agent', 'Online') == 'Offline')
        external_camera = int('Camera' in physician.get('derived_state', ''))
        signal_freshness = np.nan  # No measured signal-age contract in this feed
        active_work = physician.get('open_consults', 0)
    else:
        pid = 'POOL'
        shift_remaining = 0
        scheduled_on_shift = 0
        agent_offline = 0
        external_camera = 0
        signal_freshness = np.nan
        active_work = 0
    
    # Workload features from state
    consults_df = state['consults']
    physicians_df = state['physicians']
    
    # Completed consults remain in the deduplicated received-event history.
    events = state.get("events", pd.DataFrame())
    if len(events):
        received = events[(events.event_type == "consult_received") & (events.hospital_id == hid)
                          & (events.event_time <= now)].drop_duplicates("consult_id")
        arrivals_15 = int((received.event_time > now - pd.Timedelta(minutes=15)).sum())
        arrivals_60 = int((received.event_time > now - pd.Timedelta(minutes=60)).sum())
    else:
        arrivals_15 = arrivals_60 = np.nan

    # Queue length (open consults at this hospital)
    queue_length = len(consults_df[consults_df['hospital_id'] == hid])
    
    # Capacity features
    creds = ref.get('credentials', pd.DataFrame())
    if len(creds):
        credentialed_at_hosp = set(creds[creds['hospital_id'] == hid]['physician_id'])
        eligible = physicians_df[
            physicians_df['physician_id'].isin(credentialed_at_hosp) &
            physicians_df['available_for_routing']
        ]
        eligible_capacity = len(eligible)
        busy_credentialed = len(physicians_df[
            physicians_df['physician_id'].isin(credentialed_at_hosp) &
            ~physicians_df['available_for_routing']
        ])
    else:
        eligible_capacity = len(physicians_df[physicians_df['available_for_routing']])
        busy_credentialed = len(physicians_df) - eligible_capacity
    
    # Stage timing
    stage_age = consult.get('minutes_in_stage', 0) or 0
    
    # Remaining deadline (based on consult type and stage)
    critical_time = get_critical_minutes(stage, ctype)
    clock_age = stage_age
    note_start = consult.get("note_clock_start")
    if stage in ("Note Pending", "Documentation") and note_start is not None and pd.notna(note_start):
        clock_age = (now - note_start).total_seconds() / 60
    remaining_deadline = critical_time - clock_age if critical_time is not None else np.nan

    # Baseline hourly count from reference data
    baseline = ref.get('baseline', pd.DataFrame())
    hour_str = f"{hour:02d}:00"
    if len(baseline):
        hosp_base = baseline[(baseline['hospital_id'] == hid) & 
                             (baseline['hour_start_et'] == hour_str)]
        if "dow_type" in hosp_base.columns:
            matching = hosp_base[hosp_base.dow_type == ("weekend" if weekend else "weekday")]
            hosp_base = matching  # Missing day-specific live demand stays unknown.
        baseline_hourly = float(hosp_base['mean_consults'].iloc[0]) if len(hosp_base) else np.nan
    else:
        baseline_hourly = np.nan
    
    def scheduled(minutes):
        if physician is None:
            return np.nan
        extra = next((s for s in state.get("extra_shifts", []) if s["physician_id"] == pid), None)
        start, end = (extra["start"], extra["end"]) if extra else shift_window(pid, ref, now)
        return int(start <= now + pd.Timedelta(minutes=minutes) < end)
    scheduled_on_shift = scheduled(0)
    scheduled_15, scheduled_60, scheduled_90 = (scheduled(m) for m in (15, 60, 90))
    
    return {
        'policy': policy,
        'hospital_id': hid,
        'task_type': task_type,
        'consult_type': ctype,
        'physician_id': pid,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'weekend': weekend,
        'arrivals_last15': arrivals_15,
        'arrivals_last60': arrivals_60,
        'releases_last60': np.nan,  # Task-release semantics not yet validated
        'queue_length': queue_length,
        'eligible_capacity': eligible_capacity,
        'busy_credentialed': busy_credentialed,
        'active_work': active_work,
        'shift_remaining_min': shift_remaining,
        'scheduled_on_shift': scheduled_on_shift,
        'scheduled_on_shift_15': scheduled_15,
        'scheduled_on_shift_60': scheduled_60,
        'scheduled_on_shift_90': scheduled_90,
        'stage_age_min': stage_age,
        'remaining_deadline_min': remaining_deadline,
        'agent_offline': agent_offline,
        'external_camera': external_camera,
        'signal_freshness_min': signal_freshness,
        'response_history_rate': np.nan,
        'response_history_count': np.nan,
        'packet_complete': np.nan,
        'ack_deadline_min': 1.0 if ctype == 'Stroke' else 5.0,
        'baseline_hourly_count': baseline_hourly,
    }


def build_features_for_physician(
    physician: dict,
    state: dict,
    ref: dict,
    now: pd.Timestamp,
    policy: str = "as_is",
    task_type: str = "contact"
) -> dict:
    """Build features for predicting physician availability."""
    pid = physician.get('physician_id', '')
    shift_remaining = physician.get('minutes_to_shift_end', 0) or 0
    
    # Basic time features
    hour = now.hour
    day_of_week = now.dayofweek
    month = now.month
    weekend = int(now.dayofweek >= 5)
    
    # Physician-specific features
    agent_offline = int(physician.get('desktop_agent', 'Online') == 'Offline')
    external_camera = int('Camera' in physician.get('derived_state', ''))
    active_work = physician.get('open_consults', 0)
    scheduled_on_shift = 1 if shift_remaining > 0 else 0
    
    return {
        'policy': policy,
        'hospital_id': 'ALL',
        'task_type': task_type,
        'consult_type': 'unknown',
        'physician_id': pid,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'weekend': weekend,
        'arrivals_last15': 0,
        'arrivals_last60': 0,
        'releases_last60': 0,
        'queue_length': 0,
        'eligible_capacity': len(state['physicians']),
        'busy_credentialed': len(state['physicians'][~state['physicians']['available_for_routing']]),
        'active_work': active_work,
        'shift_remaining_min': shift_remaining,
        'scheduled_on_shift': scheduled_on_shift,
        'scheduled_on_shift_15': int(shift_remaining > 15),
        'scheduled_on_shift_60': int(shift_remaining > 60),
        'scheduled_on_shift_90': int(shift_remaining > 90),
        'stage_age_min': 0,
        'remaining_deadline_min': 120,
        'agent_offline': agent_offline,
        'external_camera': external_camera,
        'signal_freshness_min': np.nan,
        'response_history_rate': np.nan,
        'response_history_count': np.nan,
        'packet_complete': np.nan,
        'ack_deadline_min': 5.0,
        'baseline_hourly_count': np.nan,
    }


def build_batch_features(
    state: dict,
    ref: dict,
    now: pd.Timestamp,
    policy: str = "as_is"
) -> pd.DataFrame:
    """Build feature dataframe for all current consults and physicians."""
    records = []
    
    # Features for each open consult
    for _, consult in state['consults'].iterrows():
        # Find assigned physician if any
        pid = consult.get('physician_id')
        physician = None
        if pid and pid != 'Unassigned':
            phys_match = state['physicians'][state['physicians']['physician_id'] == pid]
            if len(phys_match):
                physician = phys_match.iloc[0].to_dict()
        
        features = build_features_for_consult(
            consult.to_dict(), physician, state, ref, now, policy
        )
        features['record_id'] = f"consult-{consult['consult_id']}"
        records.append(features)
    
    # Features for each on-shift physician (availability predictions)
    for _, phys in state['physicians'].iterrows():
        features = build_features_for_physician(
            phys.to_dict(), state, ref, now, policy
        )
        features['record_id'] = f"phys-{phys['physician_id']}"
        records.append(features)
    
    if not records:
        # Return empty frame with correct columns
        return pd.DataFrame(columns=['record_id'] + FEATURES)
    
    return pd.DataFrame(records)


def _stage_to_task_type(stage: str) -> str:
    """Map consult stage to task type for the model."""
    stage_map = {
        'Received': 'contact',
        'Triaged': 'contact',
        'Assigned': 'contact',
        'Provider Contacted': 'contact',
        'CT/CTA or Imaging Pending': 'imaging_review',
        'Imaging Review': 'imaging_review',
        'Video Consult In Progress': 'video',
        'Note Pending': 'documentation',
        'Documentation': 'documentation',
    }
    return stage_map.get(stage, 'contact')


def validate_features(features: pd.DataFrame) -> None:
    """Validate that feature dataframe has all required columns."""
    missing = set(FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"Missing required features: {', '.join(sorted(missing))}")
