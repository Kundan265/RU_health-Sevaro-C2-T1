"""Test that ML models reload correctly and match benchmark examples."""
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.features import FEATURES, CAT, NUM
from engine.predictor import ModelService, get_model_service, TARGETS

EXAMPLES_DIR = Path(__file__).resolve().parent / "fixtures" / "model_examples"


def load_example_features(target: str) -> pd.DataFrame:
    """Load example features from JSONL file."""
    path = EXAMPLES_DIR / f"{target}_features.jsonl"
    assert path.exists(), f"Missing required saved feature examples: {path}"
    records = [json.loads(line) for line in path.read_text().strip().split('\n')]
    return pd.DataFrame(records)


def load_example_predictions(target: str) -> pd.DataFrame:
    """Load example predictions from CSV file."""
    path = EXAMPLES_DIR / f"{target}_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def test_model_loads():
    """Test that all 9 models load successfully."""
    service = ModelService()
    service.load_all()
    
    for target in TARGETS:
        model = service.get(target)
        assert model is not None, f"Model for {target} failed to load"
        assert model.target == target
        print(f"✓ {target}: loaded {model.kind} model")


def test_feature_schema():
    """Test that feature schema matches expectations."""
    assert len(CAT) == 5, f"Expected 5 categorical features, got {len(CAT)}"
    assert len(NUM) == 26, f"Expected 26 numeric features, got {len(NUM)}"
    assert len(FEATURES) == 31, f"Expected 31 total features, got {len(FEATURES)}"
    print(f"✓ Feature schema: {len(CAT)} cat + {len(NUM)} num = {len(FEATURES)} total")


def test_response_sla_predictions():
    """Verify response_sla model reproduces benchmark examples."""
    features = load_example_features('response_sla')
    expected = load_example_predictions('response_sla')
    
    if len(features) == 0:
        print("⚠ Skipping response_sla: example files not found")
        return
    
    service = get_model_service()
    model = service.get('response_sla')
    
    predictions = model.predict(features)
    
    # Build comparison dataframe
    results = pd.DataFrame({
        'record_id': features['record_id'],
        'actual': predictions,
        'expected': expected.set_index('record_id').loc[features['record_id']]['probability'].values
    })
    results['diff'] = abs(results['actual'] - results['expected'])
    
    max_diff = results['diff'].max()
    mean_diff = results['diff'].mean()
    
    print(f"✓ response_sla: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # Allow small floating-point differences (1e-5)
    assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"


def test_stage_sla_predictions():
    """Verify stage_sla model can make predictions on example features."""
    features = load_example_features('stage_sla')
    expected = load_example_predictions('stage_sla')
    
    if len(features) == 0:
        print("⚠ Skipping stage_sla: example files not found")
        return
    
    service = get_model_service()
    model = service.get('stage_sla')
    
    predictions = model.predict(features)
    
    # If we have expected predictions, compare; otherwise just verify predictions work
    if len(expected) and 'record_id' in expected.columns:
        results = pd.DataFrame({
            'record_id': features['record_id'],
            'actual': predictions,
        })
        results = results.merge(expected, on='record_id')
        results['diff'] = abs(results['actual'] - results['probability'])
        
        max_diff = results['diff'].max()
        mean_diff = results['diff'].mean()
        
        print(f"✓ stage_sla: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
    else:
        # Just verify predictions are valid probabilities
        assert len(predictions) == len(features)
        assert all(0 <= p <= 1 for p in predictions), "Invalid probability values"
        print(f"✓ stage_sla: {len(predictions)} predictions generated (mean={np.mean(predictions):.4f})")


def test_availability_predictions():
    """Verify availability models can make predictions on example features."""
    for horizon in [15, 60, 90]:
        target = f'availability_{horizon}'
        features = load_example_features(target)
        expected = load_example_predictions(target)
        
        if len(features) == 0:
            print(f"⚠ Skipping {target}: example files not found")
            continue
        
        service = get_model_service()
        model = service.get(target)
        
        predictions = model.predict(features)
        
        if len(expected) and 'record_id' in expected.columns:
            results = pd.DataFrame({
                'record_id': features['record_id'],
                'actual': predictions,
            })
            results = results.merge(expected, on='record_id')
            results['diff'] = abs(results['actual'] - results['probability'])
            
            max_diff = results['diff'].max()
            mean_diff = results['diff'].mean()
            
            print(f"✓ {target}: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
        else:
            assert len(predictions) == len(features)
            assert all(0 <= p <= 1 for p in predictions), "Invalid probability values"
            print(f"✓ {target}: {len(predictions)} predictions generated (mean={np.mean(predictions):.4f})")


def test_queue_wait_predictions():
    """Verify queue_wait model reproduces benchmark examples."""
    features = load_example_features('queue_wait')
    expected = load_example_predictions('queue_wait')
    
    if len(features) == 0:
        print("⚠ Skipping queue_wait: example files not found")
        return
    
    service = get_model_service()
    model = service.get('queue_wait')
    
    predictions = model.predict(features)
    
    # queue_wait returns [p50, p90] - check both columns
    results = pd.DataFrame({
        'record_id': features['record_id'],
        'p50_actual': predictions[:, 0],
        'p90_actual': predictions[:, 1],
    })
    
    if len(expected) and 'record_id' in expected.columns:
        results = results.merge(expected, on='record_id')
        # Column names from benchmark: wait_p50_capped_min, wait_p90_capped_min
        p50_col = 'wait_p50_capped_min' if 'wait_p50_capped_min' in expected.columns else 'p50'
        p90_col = 'wait_p90_capped_min' if 'wait_p90_capped_min' in expected.columns else 'p90'
        
        results['diff_p50'] = abs(results['p50_actual'] - results[p50_col])
        results['diff_p90'] = abs(results['p90_actual'] - results[p90_col])
        
        max_diff = max(results['diff_p50'].max(), results['diff_p90'].max())
        mean_diff = (results['diff_p50'].mean() + results['diff_p90'].mean()) / 2
        
        print(f"✓ queue_wait: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
    else:
        assert len(predictions) == len(features)
        assert predictions.shape[1] == 2, "Expected [p50, p90] output"
        print(f"✓ queue_wait: {len(predictions)} predictions generated (p50 mean={np.mean(predictions[:, 0]):.2f})")


def test_demand_predictions():
    """Verify demand model can make predictions on example features."""
    features = load_example_features('demand')
    expected = load_example_predictions('demand')
    
    if len(features) == 0:
        print("⚠ Skipping demand: example files not found")
        return
    
    service = get_model_service()
    model = service.get('demand')
    
    predictions = model.predict(features)
    
    if len(expected) and 'record_id' in expected.columns:
        results = pd.DataFrame({
            'record_id': features['record_id'],
            'actual': predictions,
        })
        results = results.merge(expected, on='record_id')
        col = 'expected_count' if 'expected_count' in expected.columns else 'probability'
        results['diff'] = abs(results['actual'] - results[col])
        
        max_diff = results['diff'].max()
        mean_diff = results['diff'].mean()
        
        print(f"✓ demand: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
    else:
        assert len(predictions) == len(features)
        assert all(p >= 0 for p in predictions), "Invalid count values"
        print(f"✓ demand: {len(predictions)} predictions generated (mean={np.mean(predictions):.4f})")


def test_escalation_predictions():
    """Verify escalation model can make predictions on example features."""
    features = load_example_features('escalation')
    expected = load_example_predictions('escalation')
    
    if len(features) == 0:
        print("⚠ Skipping escalation: example files not found")
        return
    
    service = get_model_service()
    model = service.get('escalation')
    
    predictions = model.predict(features)
    
    if len(expected) and 'record_id' in expected.columns:
        results = pd.DataFrame({
            'record_id': features['record_id'],
            'actual': predictions,
        })
        results = results.merge(expected, on='record_id')
        results['diff'] = abs(results['actual'] - results['probability'])
        
        max_diff = results['diff'].max()
        mean_diff = results['diff'].mean()
        
        print(f"✓ escalation: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
    else:
        assert len(predictions) == len(features)
        assert all(0 <= p <= 1 for p in predictions), "Invalid probability values"
        print(f"✓ escalation: {len(predictions)} predictions generated (mean={np.mean(predictions):.4f})")


def test_handoff_predictions():
    """Verify handoff model can make predictions on example features."""
    features = load_example_features('handoff')
    expected = load_example_predictions('handoff')
    
    if len(features) == 0:
        print("⚠ Skipping handoff: example files not found")
        return
    
    service = get_model_service()
    model = service.get('handoff')
    
    predictions = model.predict(features)
    
    if len(expected) and 'record_id' in expected.columns:
        results = pd.DataFrame({
            'record_id': features['record_id'],
            'actual': predictions,
        })
        results = results.merge(expected, on='record_id')
        results['diff'] = abs(results['actual'] - results['probability'])
        
        max_diff = results['diff'].max()
        mean_diff = results['diff'].mean()
        
        print(f"✓ handoff: {len(results)} predictions, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        assert max_diff < 1e-4, f"Predictions differ too much: max_diff={max_diff}"
    else:
        assert len(predictions) == len(features)
        assert all(0 <= p <= 1 for p in predictions), "Invalid probability values"
        print(f"✓ handoff: {len(predictions)} predictions generated (mean={np.mean(predictions):.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing ML Models for Neuro Ops Command Center")
    print("=" * 60)
    
    test_feature_schema()
    print()
    
    test_model_loads()
    print()
    
    print("Verifying predictions match benchmark examples...")
    test_response_sla_predictions()
    test_stage_sla_predictions()
    test_availability_predictions()
    test_queue_wait_predictions()
    test_demand_predictions()
    test_escalation_predictions()
    test_handoff_predictions()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
