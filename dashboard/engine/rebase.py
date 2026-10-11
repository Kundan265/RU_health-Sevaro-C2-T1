"""[0] Clock rebasing: map the frozen sample day onto the real wall clock.

The sample data is a single frozen day (2026-03-17) whose story is told from
18:20 ET. To run the dashboard on the real clock we shift every time-bearing
surface by one delta, so the 18:20 state becomes the state *right now* and
relative timing is preserved exactly.

Five surfaces carry time, in two different representations:

  absolute timestamps       -> shift by the full delta
    event_stream.csv          (event_time, ingested_at)
    physicians_off_shift_*    (last_shift_end, next_shift_start)

  time-of-day strings       -> shift by the delta modulo 24 h
    physicians_on_shift.csv   (shift_start_et, shift_end_et)   "08:00"
    coverage_requirements     (time_block_et)                  "07:00-16:00"
    baseline_hourly_volume    (hour_start_et)                  "18:00"

baseline_hourly_volume is the exception: coverage.detect_surges looks it up with
an exact f"{now.hour:02d}:00" match, so its labels must stay on the hour. Those
rows rotate by whole hours instead, which costs at most 30 minutes of alignment
against the other surfaces and keeps the lookup contract intact.

The loader stays untouched, so check_snapshot.py still validates the raw 18:20
snapshots.
"""
import pandas as pd

from .loader import ET

# The moment in the sample data that the dashboard narrates from.
SCENARIO_ANCHOR = pd.Timestamp("2026-03-17 18:20:00", tz=ET)

DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


def compute_delta(wall_now: pd.Timestamp) -> tuple[pd.Timedelta, pd.Timedelta]:
    """Return (full, time_of_day) offsets from the scenario anchor to `wall_now`.

    Freeze this once per session. Recomputing it on every Streamlit rerun would
    re-pin the anchor to the current instant, and the scenario clock would never
    advance.
    """
    full = (wall_now - SCENARIO_ANCHOR).floor("s")
    return full, full % DAY


def shift_time_of_day(hhmm: str, tod: pd.Timedelta) -> str:
    """'08:00' shifted by 6h45m -> '14:45'. Wraps past midnight."""
    total = (pd.Timedelta(f"{hhmm.strip()}:00") + tod) % DAY
    minutes = int(total.total_seconds() // 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def shift_block(block: str, tod: pd.Timedelta) -> str:
    """'07:00-16:00' shifted by 6h45m -> '13:45-22:45'."""
    start, end = block.split("-")
    return f"{shift_time_of_day(start, tod)}-{shift_time_of_day(end, tod)}"


def rotate_hour(hhmm: str, tod: pd.Timedelta) -> str:
    """'18:00' rotated by ~6h45m -> '01:00'. Stays on the hour for exact matching."""
    hours = int(round(tod / HOUR)) % 24
    return f"{(int(hhmm[:2]) + hours) % 24:02d}:00"


def rebase_events(events: pd.DataFrame, full: pd.Timedelta) -> pd.DataFrame:
    """Shift an event_stream-shaped frame onto the real clock."""
    out = events.copy()
    for col in ("event_time", "ingested_at"):
        out[col] = out[col] + full
    return out


def rebase_reference(ref: dict, full: pd.Timedelta, tod: pd.Timedelta) -> dict:
    """Shift the reference tables onto the real clock.

    Returns a new dict with copies of the frames it touches, so the caller's
    @st.cache_data reference bundle is never mutated.
    """
    out = dict(ref)

    on_shift = ref["on_shift"].copy()
    for col in ("shift_start_et", "shift_end_et"):
        on_shift[col] = on_shift[col].map(lambda v: shift_time_of_day(v, tod))
    out["on_shift"] = on_shift

    off_shift = ref["off_shift"].copy()
    for col in ("last_shift_end", "next_shift_start"):
        off_shift[col] = off_shift[col] + full
    out["off_shift"] = off_shift

    coverage = ref["coverage"].copy()
    coverage["time_block_et"] = coverage["time_block_et"].map(lambda v: shift_block(v, tod))
    out["coverage"] = coverage

    baseline = ref["baseline"].copy()
    baseline["hour_start_et"] = baseline["hour_start_et"].map(lambda v: rotate_hour(v, tod))
    out["baseline"] = baseline

    return out
