"""Replay comparison: load versioned benchmark outputs, never invented numbers.

Two independent questions (N28):
  1. Care-policy effect: as_is vs specialized, algorithm held fixed
  2. Algorithm effect:  deterministic vs ml_augmented, policy held fixed

Replay cells use the saved-file definitions (N32):
  mean_capped_120min_wait, ownership_changing_handoffs, continuity_loss_rate,
  task_stage_breach_rate, response_miss_rate.

Live-dashboard snapshots use different, labeled proxies and must not be mixed
with replay cells.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "scheduling_replay.json"

ALGORITHMS = ("deterministic", "ml_augmented")
POLICIES = ("as_is", "specialized")


# ============================================================================
# REPLAY CELLS (from scheduling_replay.json)
# ============================================================================

@dataclass(frozen=True)
class ReplayCell:
    """One (algorithm, policy) cell. Field names match the saved file exactly."""
    mean_capped_120min_wait: float
    started_task_wait_p50: float
    started_task_wait_p90: float
    ownership_changing_handoffs: int
    continuity_loss_rate: float
    applicable_task_stage_breaches: int
    task_stage_breach_rate: float
    response_deadline_misses: int
    response_miss_rate: float
    leadership_alerts: int
    utilization_service_minutes_over_scheduled_minutes: float
    consults: int
    signed_off: int
    unresolved_consults: int

    @classmethod
    def from_raw(cls, raw: dict) -> "ReplayCell":
        return cls(
            mean_capped_120min_wait=float(raw["mean_capped_120min_wait"]),
            started_task_wait_p50=float(raw["started_task_wait_p50"]),
            started_task_wait_p90=float(raw["started_task_wait_p90"]),
            ownership_changing_handoffs=int(raw["ownership_changing_handoffs"]),
            continuity_loss_rate=float(raw["continuity_loss_rate"]),
            applicable_task_stage_breaches=int(raw["applicable_task_stage_breaches"]),
            task_stage_breach_rate=float(raw["task_stage_breach_rate"]),
            response_deadline_misses=int(raw["response_deadline_misses"]),
            response_miss_rate=float(raw["response_miss_rate"]),
            leadership_alerts=int(raw["leadership_alerts"]),
            utilization_service_minutes_over_scheduled_minutes=float(
                raw["utilization_service_minutes_over_scheduled_minutes"]),
            consults=int(raw["consults"]),
            signed_off=int(raw["signed_off"]),
            unresolved_consults=int(raw["unresolved_consults"]),
        )


@dataclass
class ReplayReport:
    """Full 2x2 replay plus the file's own provenance strings."""
    period: str
    episodes: int
    matching: str
    ml_rule: str
    permissions: str
    limitations: str
    cells: dict[tuple[str, str], ReplayCell]  # (algorithm, policy)

    def cell(self, algorithm: str, policy: str) -> ReplayCell:
        return self.cells[(algorithm, policy)]

    def policy_delta(self, algorithm: str = "deterministic") -> dict:
        """specialized minus as_is (positive wait delta = specialized is worse)."""
        a = self.cell(algorithm, "as_is")
        s = self.cell(algorithm, "specialized")
        return {
            "wait_delta_min": s.mean_capped_120min_wait - a.mean_capped_120min_wait,
            "handoff_delta": s.ownership_changing_handoffs - a.ownership_changing_handoffs,
            "continuity_delta": s.continuity_loss_rate - a.continuity_loss_rate,
            "stage_breach_rate_delta": s.task_stage_breach_rate - a.task_stage_breach_rate,
            "alert_delta": s.leadership_alerts - a.leadership_alerts,
            "specialized_is_worse": s.mean_capped_120min_wait > a.mean_capped_120min_wait,
        }

    def algorithm_delta(self, policy: str = "as_is") -> dict:
        """ml_augmented minus deterministic (positive wait delta = ML is worse)."""
        d = self.cell("deterministic", policy)
        m = self.cell("ml_augmented", policy)
        return {
            "wait_delta_min": m.mean_capped_120min_wait - d.mean_capped_120min_wait,
            "alert_delta": m.leadership_alerts - d.leadership_alerts,
            "stage_breach_rate_delta": m.task_stage_breach_rate - d.task_stage_breach_rate,
        }

    def policy_table(self, algorithm: str = "deterministic") -> pd.DataFrame:
        a = self.cell(algorithm, "as_is")
        s = self.cell(algorithm, "specialized")
        d = self.policy_delta(algorithm)
        rows = [
            ("Mean capped wait (120 min cap)",
             f"{a.mean_capped_120min_wait:.4f} min",
             f"{s.mean_capped_120min_wait:.4f} min",
             f"{d['wait_delta_min']:+.4f} min"),
            ("Started-task wait p50 (uncapped)",
             f"{a.started_task_wait_p50:.2f} min",
             f"{s.started_task_wait_p50:.2f} min", ""),
            ("Ownership-changing handoffs",
             str(a.ownership_changing_handoffs),
             str(s.ownership_changing_handoffs),
             f"{d['handoff_delta']:+d}"),
            ("Continuity-loss rate",
             f"{a.continuity_loss_rate:.4f}",
             f"{s.continuity_loss_rate:.4f}",
             f"{d['continuity_delta']:+.4f}"),
            ("Stage-SLA breach rate",
             f"{a.task_stage_breach_rate:.4f}",
             f"{s.task_stage_breach_rate:.4f}",
             f"{d['stage_breach_rate_delta']:+.4f}"),
            ("Response-deadline misses",
             str(a.response_deadline_misses),
             str(s.response_deadline_misses), ""),
            ("Leadership alerts",
             str(a.leadership_alerts),
             str(s.leadership_alerts),
             f"{d['alert_delta']:+d}"),
            ("Utilization (service / scheduled)",
             f"{a.utilization_service_minutes_over_scheduled_minutes:.4f}",
             f"{s.utilization_service_minutes_over_scheduled_minutes:.4f}", ""),
            ("Signed off / unresolved",
             f"{a.signed_off} / {a.unresolved_consults}",
             f"{s.signed_off} / {s.unresolved_consults}", ""),
        ]
        return pd.DataFrame(rows, columns=["Metric", "as_is", "specialized", "Delta (spec − as_is)"])

    def algorithm_table(self, policy: str = "as_is") -> pd.DataFrame:
        d = self.cell("deterministic", policy)
        m = self.cell("ml_augmented", policy)
        delta = self.algorithm_delta(policy)
        rows = [
            ("Mean capped wait (120 min cap)",
             f"{d.mean_capped_120min_wait:.4f} min",
             f"{m.mean_capped_120min_wait:.4f} min",
             f"{delta['wait_delta_min']:+.4f} min"),
            ("Leadership alerts",
             str(d.leadership_alerts),
             str(m.leadership_alerts),
             f"{delta['alert_delta']:+d}"),
            ("Stage-SLA breach rate",
             f"{d.task_stage_breach_rate:.4f}",
             f"{m.task_stage_breach_rate:.4f}",
             f"{delta['stage_breach_rate_delta']:+.4f}"),
            ("Ownership-changing handoffs",
             str(d.ownership_changing_handoffs),
             str(m.ownership_changing_handoffs), ""),
            ("Continuity-loss rate",
             f"{d.continuity_loss_rate:.4f}",
             f"{m.continuity_loss_rate:.4f}", ""),
        ]
        return pd.DataFrame(rows, columns=["Metric", "deterministic", "ml_augmented", "Delta (ML − det)"])


def load_benchmark_comparison(path: Path | None = None) -> ReplayReport | None:
    """Load the vendored scheduling_replay.json. Returns None if the file is missing."""
    src = path or BENCHMARK_PATH
    if not src.exists():
        return None
    raw = json.loads(src.read_text())
    cells = {}
    for algo in ALGORITHMS:
        for policy in POLICIES:
            cells[(algo, policy)] = ReplayCell.from_raw(raw["comparison"][algo][policy])
    return ReplayReport(
        period=raw["period"],
        episodes=int(raw["episodes"]),
        matching=raw["matching"],
        ml_rule=raw["ml_rule"],
        permissions=raw["permissions"],
        limitations=raw["limitations"],
        cells=cells,
    )


# ============================================================================
# LIVE-SNAPSHOT PROXIES (labeled; not comparable to replay cells)
# ============================================================================

@dataclass
class SnapshotProxies:
    """Current-dashboard approximations. Not first-service waits or ownership transfers."""
    minutes_since_received_mean: float = 0.0
    routing_failure_count: int = 0
    consults_with_routing_failures: int = 0
    open_consults: int = 0
    busy_fraction: float = 0.0
    note: str = (
        "Snapshot proxies only: minutes_since_received is not first-service wait; "
        "routing_failures are not ownership-changing handoffs."
    )


def snapshot_proxies(consults: pd.DataFrame, physicians: pd.DataFrame) -> SnapshotProxies:
    """Labelled live-state approximations. Do not mix with ReplayCell values."""
    out = SnapshotProxies()
    if len(consults):
        out.open_consults = len(consults)
        if "minutes_since_received" in consults.columns:
            out.minutes_since_received_mean = float(consults["minutes_since_received"].mean())
        if "routing_failures" in consults.columns:
            out.routing_failure_count = int(consults["routing_failures"].sum())
            out.consults_with_routing_failures = int((consults["routing_failures"] > 0).sum())
    if len(physicians) and "available_for_routing" in physicians.columns:
        out.busy_fraction = float((~physicians["available_for_routing"]).mean())
    return out
