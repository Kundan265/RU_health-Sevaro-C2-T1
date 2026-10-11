"""Advisory ML annotations. Never changes a recommendation.

Builds the 31-column feature frame and scores each problem for stage-SLA risk,
escalation risk, and capped wait (p50/p90). Failures degrade to
``status="unavailable"`` so a missing joblib cannot break Detect → Decide → Act.
"""
from __future__ import annotations

import pandas as pd


def annotate_problem(problem: dict, state: dict, ref: dict, now: pd.Timestamp) -> dict:
    """Return per-problem risk annotations. Safe if models or features fail."""
    try:
        from .predictor import get_model_service, predict_for_problem

        service = get_model_service()
        if not getattr(service, "_models", None):
            return {"status": "unavailable", "note": "Models unavailable"}
        preds = predict_for_problem(problem, state, ref, now)
        if not preds:
            return {"status": "none", "note": "No consult-level features for this problem"}
        preds.setdefault("status", "ok")
        return preds
    except Exception as exc:  # missing joblib, schema drift, import error
        return {"status": "unavailable", "note": f"Models unavailable: {exc}"}


def experimental_pools(state: dict, ref: dict, coverage: dict) -> dict:
    """Demand-matched pool sizing. Experimental; never fed into recommend()."""
    try:
        from .pool_sizing import manage_pools
        return manage_pools(state, ref, coverage)
    except Exception as exc:
        return {"status": "unavailable", "note": f"Forecast unavailable: {exc}",
                "recommendations": [], "actions": []}
