"""Tests for engine/rebase.py - mapping the frozen sample day onto the real clock."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.loader import ET, load_events, load_reference
from engine.pipeline import analyze
from engine.rebase import (
    SCENARIO_ANCHOR, compute_delta, rebase_events, rebase_reference,
    rotate_hour, shift_block, shift_time_of_day,
)

# A wall clock deliberately in the small hours, so every rebased shift and coverage
# block straddles midnight. That is the case the frozen clock never exercised.
WALL = pd.Timestamp("2026-10-11 01:05:00", tz=ET)


def test_delta_round_trips():
    full, tod = compute_delta(WALL)
    assert SCENARIO_ANCHOR + full == WALL
    assert tod == full % pd.Timedelta(days=1)
    assert full.microseconds == 0, "delta should be floored to whole seconds"
    print(f"✓ Delta round-trips: anchor + {full} == wall clock")


def test_time_of_day_wraps_midnight():
    tod = pd.Timedelta(hours=6, minutes=45)
    assert shift_time_of_day("08:00", tod) == "14:45"
    assert shift_time_of_day("20:00", tod) == "02:45"   # wraps
    assert shift_time_of_day("23:59", tod) == "06:44"
    assert shift_time_of_day("08:00", pd.Timedelta(0)) == "08:00"
    print("✓ Time-of-day shifting wraps past midnight")


def test_block_shifting():
    tod = pd.Timedelta(hours=6, minutes=45)
    assert shift_block("07:00-16:00", tod) == "13:45-22:45"
    assert shift_block("22:00-07:00", tod) == "04:45-13:45"
    print("✓ Coverage blocks shift on both ends")


def test_baseline_stays_on_the_hour():
    """coverage.detect_surges matches hour_start_et with an exact 'HH:00' string,
    so these labels must never carry minutes."""
    for minutes in (0, 14, 29, 31, 45, 59):
        tod = pd.Timedelta(hours=6, minutes=minutes)
        for hour in ("00:00", "09:00", "18:00", "23:00"):
            out = rotate_hour(hour, tod)
            assert out.endswith(":00"), f"{hour} + {tod} -> {out} is not on the hour"
            assert 0 <= int(out[:2]) <= 23
    assert rotate_hour("18:00", pd.Timedelta(hours=7)) == "01:00"
    assert rotate_hour("20:00", pd.Timedelta(hours=7)) == "03:00"   # wraps
    print("✓ Baseline hours rotate but stay on the hour")


def test_rebased_events_land_at_now():
    """The sample's last event is 18:19:40, 20s before the 18:20 anchor. After
    rebasing it must sit just behind the wall clock, not days in the past."""
    full, _ = compute_delta(WALL)
    events = rebase_events(load_events(), full)
    lag = (WALL - events["ingested_at"].max()).total_seconds()
    assert 0 <= lag < 120, f"last ingested event is {lag:.0f}s behind now"
    print(f"✓ Rebased stream ends {lag:.0f}s before now (was ~208 days)")


def test_rebase_reference_does_not_mutate_input():
    """app.py feeds this a @st.cache_data bundle; mutating it would poison the cache
    and compound the shift on every rerun."""
    full, tod = compute_delta(WALL)
    ref = load_reference()
    before_shift = ref["on_shift"]["shift_start_et"].tolist()
    before_block = ref["coverage"]["time_block_et"].tolist()
    before_hour = ref["baseline"]["hour_start_et"].tolist()
    before_next = ref["off_shift"]["next_shift_start"].tolist()

    out = rebase_reference(ref, full, tod)

    assert ref["on_shift"]["shift_start_et"].tolist() == before_shift
    assert ref["coverage"]["time_block_et"].tolist() == before_block
    assert ref["baseline"]["hour_start_et"].tolist() == before_hour
    assert ref["off_shift"]["next_shift_start"].tolist() == before_next
    assert out["on_shift"]["shift_start_et"].tolist() != before_shift, "output should differ"
    print("✓ rebase_reference leaves the cached reference untouched")


def test_all_five_surfaces_move():
    full, tod = compute_delta(WALL)
    ref = load_reference()
    out = rebase_reference(ref, full, tod)

    assert out["on_shift"]["shift_start_et"].tolist() != ref["on_shift"]["shift_start_et"].tolist()
    assert out["on_shift"]["shift_end_et"].tolist() != ref["on_shift"]["shift_end_et"].tolist()
    assert out["coverage"]["time_block_et"].tolist() != ref["coverage"]["time_block_et"].tolist()
    assert out["baseline"]["hour_start_et"].tolist() != ref["baseline"]["hour_start_et"].tolist()
    assert (out["off_shift"]["next_shift_start"] - ref["off_shift"]["next_shift_start"] == full).all()

    events = rebase_events(load_events(), full)
    assert (events["event_time"] - load_events()["event_time"] == full).all()
    print("✓ All five time-bearing surfaces shift by one coherent delta")


def test_midnight_crossing_shift_is_live():
    """Rebasing to 01:05 makes every day shift straddle midnight. A shift running
    now started yesterday, so anchoring only to today would drop the whole roster."""
    full, tod = compute_delta(WALL)
    ref = rebase_reference(load_reference(), full, tod)
    events = rebase_events(load_events(), full)

    state = analyze(events, ref, WALL, [], {}, "Ops Lead")["state"]
    assert len(state["physicians"]) == 8, f"expected 8 on shift, got {len(state['physicians'])}"
    assert (state["physicians"]["minutes_to_shift_end"] > 0).all()
    print(f"✓ {len(state['physicians'])} physicians stay on shift across midnight")


def test_rebased_matches_frozen_scenario():
    """The headline invariant: rebasing must change *when* the scenario plays, not
    *what* it says. Same problems, same ranks, same recommendations."""
    frozen_ref, frozen_events = load_reference(), load_events()
    frozen = analyze(frozen_events, frozen_ref, SCENARIO_ANCHOR, [], {}, "Ops Lead")

    full, tod = compute_delta(WALL)
    live = analyze(rebase_events(load_events(), full),
                   rebase_reference(load_reference(), full, tod),
                   WALL, [], {}, "Ops Lead")

    assert len(live["state"]["consults"]) == len(frozen["state"]["consults"])
    assert len(live["state"]["physicians"]) == len(frozen["state"]["physicians"])

    fp = [(p["problem_id"], p["rank"], p["severity"], p["rec"]["action"]) for p in frozen["problems"]]
    lp = [(p["problem_id"], p["rank"], p["severity"], p["rec"]["action"]) for p in live["problems"]]
    assert lp == fp, (
        "rebased scenario diverged from frozen\n"
        f"  frozen: {fp}\n  live:   {lp}"
    )
    print(f"✓ Rebased clock reproduces all {len(fp)} problems identically (ranks + actions)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Clock Rebase Module")
    print("=" * 60)

    test_delta_round_trips()
    test_time_of_day_wraps_midnight()
    test_block_shifting()
    test_baseline_stays_on_the_hour()
    test_rebased_events_land_at_now()
    test_rebase_reference_does_not_mutate_input()
    test_all_five_surfaces_move()
    test_midnight_crossing_shift_is_live()
    test_rebased_matches_frozen_scenario()

    print()
    print("=" * 60)
    print("All rebase tests passed! ✓")
    print("=" * 60)
