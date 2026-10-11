"""Model classes from benchmark, needed for unpickling trained models.

The Predictor and Baseline dataclasses must match the benchmark's structure
exactly for joblib.load() to reconstruct the pickled objects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURES, CAT, NUM


TARGETS = {
    'demand': ('demand', 'label_count', 'count', 60),
    'availability_15': ('availability', 'label_available_15', 'binary', 15),
    'availability_60': ('availability', 'label_available_60', 'binary', 60),
    'availability_90': ('availability', 'label_available_90', 'binary', 90),
    'queue_wait': ('queue_wait', 'label_wait_capped_min', 'quantiles', 120),
    'response_sla': ('response_sla', 'label_response_missed', 'binary', '1 urgent / 5 routine'),
    'stage_sla': ('stage_sla', 'label_stage_sla_missed', 'binary', 'source critical deadline'),
    'handoff': ('handoff', 'label_handoff_delayed', 'binary', '1 urgent / 5 routine'),
    'escalation': ('escalation', 'label_escalation', 'binary', 30),
}

DEFINITIONS = {
    'demand': 'Number of hospital arrivals or task releases in the next 60 minutes',
    'queue_wait': 'First service-start delay capped at 120 minutes, with unresolved and censoring flags retained',
    'response_sla': 'Offered physician response exceeds urgent 1-minute or routine 5-minute acknowledgement deadline; timely decline is not a miss',
    'stage_sla': 'Applicable source critical-stage clock breached: triage10, Stroke assigned-contact8, imaging20, image-to-video12, notes Stroke45/routine90 minutes',
    'handoff': 'Receiving clinician fails to acknowledge context by packet-ready plus urgent1/routine5 minutes',
    'escalation': 'First new operational leadership escalation in next30 minutes, conditional on unresolved and no prior escalation',
    'availability_15': 'Physician routing-free at +15 minutes, on shift and outside break/cooldown/busy/dropout/external camera; hospital and task gates apply separately',
    'availability_60': 'Physician routing-free at +60 minutes, on shift and outside break/cooldown/busy/dropout/external camera; hospital and task gates apply separately',
    'availability_90': 'Physician routing-free at +90 minutes, on shift and outside break/cooldown/busy/dropout/external camera; hospital and task gates apply separately',
}


@dataclass
class Predictor:
    """A trained ML predictor for one target.
    
    This dataclass is serialized by joblib when saving models.
    Structure must match the benchmark's Predictor exactly.
    """
    target: str
    kind: str
    models: Any  # sklearn pipeline or list of pipelines for quantiles
    metadata: dict
    
    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Make predictions on a feature dataframe.
        
        Returns:
            For binary: probabilities (0-1)
            For quantiles: array of [p50, p90] predictions
            For count: expected counts
        """
        missing = [c for c in FEATURES if c not in frame.columns]
        if missing:
            raise ValueError('Missing required features: ' + ', '.join(missing))
        
        x = frame[FEATURES].copy()
        for c in CAT:
            x[c] = x[c].fillna('unknown').astype(str)
        for c in NUM:
            x[c] = pd.to_numeric(x[c], errors='raise')
        
        if self.kind == 'binary':
            probabilities = self.models.predict_proba(x)
            if probabilities.shape[1] == 2:
                return probabilities[:, 1]
            return np.full(len(x), float(self.models.classes_[0]))
        
        if self.kind == 'quantiles':
            predictions = np.column_stack([m.predict(x) for m in self.models])
            return np.sort(np.clip(predictions, 0, 120), axis=1)
        
        return np.maximum(0, self.models.predict(x))


@dataclass
class Baseline:
    """A baseline predictor using historical rates.
    
    This dataclass is serialized by joblib when saving models.
    Structure must match the benchmark's Baseline exactly.
    """
    target: str
    kind: str
    lookup: dict
    fallback: Any
    metadata: dict
    
    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Make predictions using historical rates."""
        if self.target.startswith('availability'):
            h = self.target.split('_')[-1]
            return np.where(frame['scheduled_on_shift_' + h].to_numpy(bool), 0.98, 0.02)
        
        if self.kind == 'count':
            return frame.baseline_hourly_count.to_numpy() * np.array([
                self.lookup.get((p, k), 1) 
                for p, k in zip(frame.policy, frame.task_type)
            ])
        
        pred = np.array([
            self.lookup.get((p, k, c), self.fallback) 
            for p, k, c in zip(frame.policy, frame.task_type, frame.consult_type)
        ])
        
        if self.target == 'stage_sla':
            pred = np.where(
                (frame.eligible_capacity == 0) | (frame.remaining_deadline_min <= 0),
                0.98,
                pred
            )
        
        return pred
