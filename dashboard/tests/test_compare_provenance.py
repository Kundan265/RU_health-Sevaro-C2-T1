"""Provenance for the Compare tab: real scheduling_replay.json, no invented wins."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.compare import BENCHMARK_PATH, ReplayCell, load_benchmark_comparison

COMPARE_SRC = Path(__file__).resolve().parents[1] / "engine" / "compare.py"


def test_no_hardcoded_metric_constants():
    src = COMPARE_SRC.read_text()
    for banned in ("28.3", "26.8", "is_better", "simulated benchmark data"):
        assert banned not in src, f"invented Compare constant still present: {banned}"


def test_loaded_values_equal_json_full_precision():
    report = load_benchmark_comparison()
    assert report is not None
    raw = json.loads(BENCHMARK_PATH.read_text())
    for algo in ("deterministic", "ml_augmented"):
        for policy in ("as_is", "specialized"):
            cell = report.cell(algo, policy)
            expected = ReplayCell.from_raw(raw["comparison"][algo][policy])
            assert cell == expected
            src = raw["comparison"][algo][policy]
            assert cell.mean_capped_120min_wait == src["mean_capped_120min_wait"]
            assert cell.ownership_changing_handoffs == src["ownership_changing_handoffs"]
            assert cell.continuity_loss_rate == src["continuity_loss_rate"]
            assert cell.task_stage_breach_rate == src["task_stage_breach_rate"]
            assert cell.response_miss_rate == src["response_miss_rate"]
            assert cell.leadership_alerts == src["leadership_alerts"]


def test_specialized_vs_as_is_is_degradation():
    report = load_benchmark_comparison()
    delta = report.policy_delta("deterministic")
    assert delta["specialized_is_worse"] is True
    assert delta["wait_delta_min"] > 0
    as_is = report.cell("deterministic", "as_is")
    spec = report.cell("deterministic", "specialized")
    assert spec.mean_capped_120min_wait > as_is.mean_capped_120min_wait


def test_missing_file_returns_none(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_benchmark_comparison(missing) is None


if __name__ == "__main__":
    test_no_hardcoded_metric_constants()
    test_loaded_values_equal_json_full_precision()
    test_specialized_vs_as_is_is_degradation()
    print("PASS test_compare_provenance")
