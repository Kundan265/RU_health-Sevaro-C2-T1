"""[NEW] Predictor: Load and use the trained ML models for predictions.

Loads the 9 selected model bundles and provides prediction methods.
Models are cached using Streamlit's cache_resource for efficiency.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from .features import FEATURES, CAT, NUM

# Model directory relative to this file
MODEL_DIR = Path(__file__).parent.parent / "models"

# Target definitions
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


@dataclass
class ModelBundle:
    """A loaded model with its metadata."""
    target: str
    kind: str
    model: Any
    metadata: dict
    threshold: float | None = None
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on a feature dataframe.
        
        Returns:
            For binary: probabilities (0-1)
            For quantiles: array of [p50, p90] predictions
            For count: expected counts
        """
        # The loaded model is either:
        # 1. A benchmark Predictor dataclass with predict_frame method
        # 2. A raw sklearn pipeline
        
        if hasattr(self.model, 'predict_frame'):
            # Benchmark Predictor dataclass - use its method directly
            return self.model.predict_frame(features)
        
        # Fallback: raw sklearn pipeline
        # Validate and prepare features
        missing = [c for c in FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f'Missing required features: {", ".join(missing)}')
        
        x = features[FEATURES].copy()
        for c in CAT:
            x[c] = x[c].fillna('unknown').astype(str)
        for c in NUM:
            x[c] = pd.to_numeric(x[c], errors='coerce').fillna(0)
        
        if self.kind == 'binary':
            probabilities = self.model.predict_proba(x)
            if probabilities.shape[1] == 2:
                return probabilities[:, 1]
            return np.full(len(x), float(self.model.classes_[0]))
        
        if self.kind == 'quantiles':
            # queue_wait model returns [p50, p90]
            predictions = np.column_stack([m.predict(x) for m in self.model])
            return np.sort(np.clip(predictions, 0, 120), axis=1)
        
        # count model
        return np.maximum(0, self.model.predict(x))
    
    def predict_single(self, features: dict) -> float | tuple[float, float]:
        """Make prediction for a single record."""
        df = pd.DataFrame([features])
        result = self.predict(df)
        if self.kind == 'quantiles':
            return tuple(result[0])  # (p50, p90)
        return float(result[0])
    
    def is_above_threshold(self, features: pd.DataFrame) -> np.ndarray:
        """For binary models, return True where probability exceeds threshold."""
        if self.kind != 'binary' or self.threshold is None:
            raise ValueError("is_above_threshold only works for binary models with thresholds")
        probs = self.predict(features)
        return probs >= self.threshold


def _load_model(target: str) -> ModelBundle:
    """Load a single model bundle from disk."""
    import joblib
    
    model_path = MODEL_DIR / f"{target}_selected.joblib"
    metadata_path = MODEL_DIR / f"{target}_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    
    # Determine kind
    _, _, kind, _ = TARGETS.get(target, (target, '', 'binary', 0))
    
    # Get threshold for binary classifiers
    threshold = metadata.get('alert_threshold')
    
    return ModelBundle(
        target=target,
        kind=kind,
        model=model,
        metadata=metadata,
        threshold=threshold
    )


class ModelService:
    """Service class to manage all models."""
    
    def __init__(self):
        self._models: dict[str, ModelBundle] = {}
        self._loaded = False
    
    def load_all(self) -> None:
        """Load all models."""
        if self._loaded:
            return
        
        for target in TARGETS:
            try:
                self._models[target] = _load_model(target)
            except FileNotFoundError:
                print(f"Warning: Model not found for {target}")
        
        self._loaded = True
    
    def get(self, target: str) -> ModelBundle | None:
        """Get a loaded model by target name."""
        if not self._loaded:
            self.load_all()
        return self._models.get(target)
    
    def predict_stage_sla(self, features: pd.DataFrame) -> np.ndarray:
        """Predict P(stage SLA breach) for consults."""
        model = self.get('stage_sla')
        if model is None:
            return np.full(len(features), 0.5)
        return model.predict(features)
    
    def predict_escalation(self, features: pd.DataFrame) -> np.ndarray:
        """Predict P(escalation in next 30 min)."""
        model = self.get('escalation')
        if model is None:
            return np.full(len(features), 0.5)
        return model.predict(features)
    
    def predict_response_sla(self, features: pd.DataFrame) -> np.ndarray:
        """Predict P(physician response misses deadline)."""
        model = self.get('response_sla')
        if model is None:
            return np.full(len(features), 0.3)
        return model.predict(features)
    
    def predict_availability(self, features: pd.DataFrame, horizon: int = 15) -> np.ndarray:
        """Predict P(physician available at horizon)."""
        target = f'availability_{horizon}'
        model = self.get(target)
        if model is None:
            return np.full(len(features), 0.8)
        return model.predict(features)
    
    def predict_queue_wait(self, features: pd.DataFrame) -> np.ndarray:
        """Predict wait time (returns [p50, p90] for each row)."""
        model = self.get('queue_wait')
        if model is None:
            return np.full((len(features), 2), [30.0, 60.0])
        return model.predict(features)
    
    def predict_demand(self, features: pd.DataFrame) -> np.ndarray:
        """Predict arrivals in next 60 minutes."""
        model = self.get('demand')
        if model is None:
            return np.full(len(features), 1.0)
        return model.predict(features)
    
    def predict_handoff(self, features: pd.DataFrame) -> np.ndarray:
        """Predict P(handoff acknowledgement delayed)."""
        model = self.get('handoff')
        if model is None:
            return np.full(len(features), 0.2)
        return model.predict(features)
    
    def get_threshold(self, target: str) -> float | None:
        """Get the alert threshold for a binary model."""
        model = self.get(target)
        if model is None:
            return None
        return model.threshold


# Create cached singleton for Streamlit
if HAS_STREAMLIT:
    @st.cache_resource
    def get_model_service() -> ModelService:
        """Get or create the cached model service."""
        service = ModelService()
        service.load_all()
        return service
else:
    _service_instance = None
    
    def get_model_service() -> ModelService:
        """Get or create the model service singleton."""
        global _service_instance
        if _service_instance is None:
            _service_instance = ModelService()
            _service_instance.load_all()
        return _service_instance


def predict_for_problem(problem: dict, state: dict, ref: dict, now: pd.Timestamp) -> dict:
    """Make ML predictions relevant to a problem.
    
    Returns dict with prediction results that can be added to the problem.
    """
    from .features import build_features_for_consult
    
    service = get_model_service()
    predictions = {}
    
    # For consult problems, predict SLA risk
    if problem.get('kind') == 'consult':
        consult_id = problem.get('consult_id')
        consults_df = state['consults']
        consult_row = consults_df[consults_df['consult_id'] == consult_id]
        
        if len(consult_row):
            consult = consult_row.iloc[0].to_dict()
            holder = state["physicians"][state["physicians"].physician_id == consult.get("physician_id")]
            physician = holder.iloc[0].to_dict() if len(holder) else None
            features = build_features_for_consult(consult, physician, state, ref, now)
            missing = [name for name in NUM if pd.isna(features.get(name))]
            # Schema compatibility alone does not establish training-semantic fidelity.
            unvalidated = ["queue_length", "active_work", "eligible_capacity", "busy_credentialed"]
            if missing or unvalidated:
                return {"status": "experimental", "missing_features": missing,
                        "unvalidated_features": unvalidated,
                        "note": "Models loaded; live risk/wait predictions suppressed. Missing measurements: "
                                + ", ".join(missing) + ". Task/capacity semantics still experimental."}
            features_df = pd.DataFrame([features])
            
            # Stage SLA prediction
            stage_sla_prob = service.predict_stage_sla(features_df)[0]
            predictions['stage_sla_probability'] = float(stage_sla_prob)
            predictions['stage_sla_threshold'] = service.get_threshold('stage_sla')
            
            # Escalation prediction  
            esc_prob = service.predict_escalation(features_df)[0]
            predictions['escalation_probability'] = float(esc_prob)
            predictions['escalation_threshold'] = service.get_threshold('escalation')
            
            # Wait time prediction
            wait_pred = service.predict_queue_wait(features_df)[0]
            predictions['wait_p50'] = float(wait_pred[0])
            predictions['wait_p90'] = float(wait_pred[1])
    
    return predictions
