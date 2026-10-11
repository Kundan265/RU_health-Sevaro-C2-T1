"""Feature schema from benchmark, needed for model unpickling."""

# Categorical features
CAT = ['policy', 'hospital_id', 'task_type', 'consult_type', 'physician_id']

# Numeric features  
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
