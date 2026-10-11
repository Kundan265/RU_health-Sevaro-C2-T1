"""Atomic upload preparation: content identity, validation, and event-id accounting."""
import hashlib
import io
import pandas as pd
from .loader import parse_events
from .rebase import rebase_events


def prepare_upload(payload: bytes, events: pd.DataFrame, ref: dict, delta) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    new = parse_events(pd.read_csv(io.BytesIO(payload)))
    allowed_types = set(events.event_type)
    unknown_types = set(new.event_type) - allowed_types
    hospitals = set(ref["hospitals"].hospital_id)
    physicians = set(ref["credentials"].physician_id)
    unknown_hospitals = set(new.hospital_id) - hospitals - {""}
    unknown_physicians = set(new.physician_id) - physicians - {""}
    if unknown_types or unknown_hospitals or unknown_physicians:
        raise ValueError(f"Unknown event types {sorted(unknown_types)}, hospitals "
                         f"{sorted(unknown_hospitals)}, physicians {sorted(unknown_physicians)}")
    new = rebase_events(new, delta)
    fresh = int((~new.event_id.isin(set(events.event_id))).groupby(new.event_id).any().sum())
    return {"digest": digest, "events": pd.concat([events, new], ignore_index=True),
            "rows": len(new), "fresh": fresh, "duplicates": len(new) - fresh,
            "latest": new.event_time.max().ceil("min")}
