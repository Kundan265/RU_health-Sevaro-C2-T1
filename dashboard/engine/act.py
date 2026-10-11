"""[8] Action store: the only stateful part. Runs the Act loop and keeps an
append-only, machine-readable audit log (mentors: used for audit, process
improvement, and as triggers for messaging/paging systems).

Split-shift offer lifecycle:
  pending_approval --(no answer in APPROVAL_TIMEOUT_MIN)--> pending_backup
  pending_approval|pending_backup --approve--> sent --> accepted | declined | expired
                                   --reject--> rejected
  any open offer --(gap filled by someone else)--> cancelled / no_longer_needed
"""
import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .pathway import create_task_for_stage

APPROVAL_TIMEOUT_MIN = 5
OFFER_EXPIRY_MIN = 10
OPEN_STATUSES = {"pending_approval", "pending_backup", "sent"}
PHI_MARKERS = ("CON-", "stroke", "patient", "NIHSS")
DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[1] / "logs" / "audit.jsonl"


def build_offer_message(hospital_names: list[str], start, end, online_at) -> str:
    """Hospital names, hours and start time only. No patient information."""
    msg = (f"Sevaro split-shift offer: cover {', '.join(hospital_names)} "
           f"from {start:%H:%M} to {end:%H:%M} ET ({(end - start).total_seconds() / 3600:g} h). "
           f"Please be online by {online_at:%H:%M}. You were asked because you are credentialed at "
           f"these hospitals, opted in to split shifts, and are rested. Reply within {OFFER_EXPIRY_MIN} min.")
    assert not any(m.lower() in msg.lower() for m in PHI_MARKERS), "offer message must not carry patient details"
    return msg


def _consult_from_problem(problem: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    hid = problem.get("hospital_id") or extra.get("hospital_id") or ""
    if not hid and extra.get("hospitals"):
        hid = extra["hospitals"][0]
    return {
        "consult_id": problem.get("consult_id") or extra.get("consult_id") or problem.get("problem_id", ""),
        "hospital_id": hid,
        "consult_type": problem.get("consult_type") or extra.get("consult_type") or "General Neuro",
        "current_stage": problem.get("stage") or extra.get("stage") or "Triaged",
        "physician_id": problem.get("physician_id") or extra.get("physician_id") or "Unassigned",
        "minutes_in_stage": problem.get("minutes_in_stage", 0.0) or 0.0,
        "stage_entered_at": problem.get("stage_entered_at") or extra.get("stage_entered_at"),
        "routing_failures": problem.get("routing_failures", 0) or 0,
        "note_clock_start": problem.get("note_clock_start"),
        "prior_physicians": problem.get("prior_physicians", []),
        "routing_history": problem.get("routing_history", []),
    }


class ActionStore:
    def __init__(self, audit_path: Path | None = DEFAULT_AUDIT_PATH):
        self.proposals: dict[str, dict] = {}
        self.log: list[dict] = []
        self.synthetic_events: list[dict] = []
        self.accepted_shifts: list[dict] = []
        self.handoffs: dict[str, dict] = {}
        self.alerted: set[str] = set()
        self._ids = itertools.count(1)
        self.audit_path = audit_path

    # ---------- logging ----------
    def _log(self, now, actor, action, proposal=None, **details):
        row = {
            "sim_time": now.strftime("%H:%M:%S"), "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "actor": actor, "action": action,
            "proposal_id": proposal["id"] if proposal else "",
            "kind": proposal["kind"] if proposal else "",
            "physician_id": (proposal or {}).get("physician_id") or details.pop("physician_id", ""),
            "hospitals": ", ".join((proposal or {}).get("hospitals", [])) or details.pop("hospitals", ""),
            "details": json.dumps(details, default=str),
        }
        self.log.append(row)
        self._persist(row)

    def _persist(self, row: dict) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=["sim_time", "logged_at", "actor", "action", "proposal_id",
                                               "kind", "physician_id", "hospitals", "details"])

    # ---------- queries ----------
    def open_for(self, problem_id: str) -> dict | None:
        return next((p for p in self.proposals.values()
                     if p["problem_id"] == problem_id and p["status"] in OPEN_STATUSES), None)

    def offer_counts(self, now=None) -> dict:
        """week = created in the last 7 days; declined_today = last 24 hours.

        If ``now`` is omitted, use the latest proposal created_at so scenario-clock
        callers still window correctly.
        """
        times = [p.get("created_at") for p in self.proposals.values() if p.get("created_at") is not None]
        if now is None:
            now = max(times) if times else pd.Timestamp.now()
        week_ago = now - pd.Timedelta(days=7)
        day_ago = now - pd.Timedelta(hours=24)
        counts: dict[str, dict] = {}
        for p in self.proposals.values():
            if p["kind"] != "split_shift":
                continue
            created = p.get("created_at")
            c = counts.setdefault(p["physician_id"], {"open": 0, "week": 0, "declined_today": 0})
            c["open"] += p["status"] in OPEN_STATUSES
            if created is not None and created >= week_ago:
                c["week"] += p["status"] not in {"pending_approval", "pending_backup", "rejected"}
            if created is not None and created >= day_ago:
                c["declined_today"] += p["status"] in {"declined", "expired"}
        return counts

    def pending_approvals(self) -> list[dict]:
        return [p for p in self.proposals.values() if p["status"] in {"pending_approval", "pending_backup"}]

    def sent_offers(self) -> list[dict]:
        return [p for p in self.proposals.values() if p["kind"] == "split_shift" and p["status"] == "sent"]

    def _log_handoff_packet(self, now, actor: str, problem: dict, to_pid: str,
                            reason: str, proposal=None, extra=None):
        packet = create_task_for_stage(_consult_from_problem(problem, extra), to_pid, reason, now)
        self._log(now, actor, "context_packet", proposal, context_packet=packet.to_dict(),
                  ack_deadline_min=packet.ack_deadline_min, to_physician=to_pid)
        hid = f"HND-{next(self._ids):04d}"
        self.handoffs[hid] = {"id": hid, "packet": packet.to_dict(), "status": "pending",
                             "created_at": now, "ack_due_at": now + pd.Timedelta(minutes=packet.ack_deadline_min)}
        return packet

    def acknowledge_handoff(self, handoff_id: str, physician_id: str, now):
        handoff = self.handoffs[handoff_id]
        if physician_id != handoff["packet"]["to_physician_id"]:
            raise ValueError("Only the receiving physician can acknowledge this simulated handoff")
        if handoff["status"] == "acknowledged":
            return
        late = now > handoff["ack_due_at"]
        handoff.update(status="acknowledged", acknowledged_at=now, acknowledged_by=physician_id, late=late)
        self._log(now, physician_id, "handoff_acknowledged", handoff_id=handoff_id,
                  consult_id=handoff["packet"]["consult_id"], late=late)

    # ---------- automatic, low-risk ----------
    def auto_alert(self, problem: dict, now):
        if problem["problem_id"] not in self.alerted:
            self.alerted.add(problem["problem_id"])
            self._log(now, "System", "internal_alert", problem_id=problem["problem_id"],
                      title=problem["title"], detail=problem["rec"]["detail"])

    # ---------- coordinator ----------
    def reassign(self, problem: dict, to_pid: str, now):
        cid = problem["consult_id"]
        self.synthetic_events.append({
            "event_id": f"SIM-{next(self._ids):04d}", "event_time": now, "ingested_at": now,
            "consult_id": cid, "event_type": "physician_assigned", "physician_id": to_pid,
            "hospital_id": problem["hospital_id"], "description": "Consult reassigned by coordinator"})
        self._log(now, "Coordinator", "reassigned", physician_id=to_pid, hospitals=problem["hospital_id"],
                  consult_id=cid, from_physician=problem.get("physician_id") or "Unassigned")
        self._log_handoff_packet(now, "Coordinator", problem, to_pid, "Coordinator reassignment")
        if problem.get("physician_id"):
            self._log(now, "System", "informed_original_physician", physician_id=problem["physician_id"],
                      consult_id=cid, note="Informed after reassignment (mentor answer Q2)")

    def propose_split_shift(self, problem: dict, candidate: dict, eligibility_df: pd.DataFrame,
                            hospital_names: dict, now, approver_role: str) -> dict:
        hosp = list(candidate["covers"])
        p = {"id": f"OFR-{next(self._ids):04d}", "kind": "split_shift", "problem_id": problem["problem_id"],
             "title": f"Split shift for {', '.join(hospital_names[h] for h in hosp)}",
             "physician_id": candidate["physician_id"], "physician_name": candidate["name"],
             "hospitals": hosp, "start": candidate["offer_start"], "end": candidate["offer_end"],
             "online_at": candidate["online_at"], "channel": candidate["channel"],
             "message": build_offer_message([hospital_names[h] for h in hosp], candidate["offer_start"],
                                            candidate["offer_end"], candidate["online_at"]),
             "rationale": problem["rec"]["rationale"] + problem["rec"]["why_not"],
             "considered": [{"physician_id": r.physician_id, "eligible": r.eligible,
                             "failed": [f"{k}: {r.rules[k][1]}" for k in r.failed_rules]}
                            for r in eligibility_df.itertuples()],
             "status": "pending_approval", "created_at": now, "approver": approver_role,
             "source_problem": {
                 "problem_id": problem.get("problem_id"),
                 "consult_id": problem.get("consult_id"),
                 "hospital_id": problem.get("hospital_id"),
                 "consult_type": problem.get("consult_type"),
                 "stage": problem.get("stage"),
                 "physician_id": problem.get("physician_id"),
             }}
        self.proposals[p["id"]] = p
        self._log(now, "Coordinator", "proposed_split_shift", p, considered=p["considered"],
                  window=f"{p['start']:%H:%M}-{p['end']:%H:%M}", approver=approver_role)
        p["coverage_context"] = {"hospitals": hosp, "online_at": candidate["online_at"],
                                 "end": candidate["offer_end"], "routing_restricted": True}
        self._log(now, "Coordinator", "coverage_context", p, coverage_context=p["coverage_context"])
        return p

    def propose_escalation(self, problem: dict, now, approver_role: str) -> dict:
        p = {"id": f"ESC-{next(self._ids):04d}", "kind": "escalate", "problem_id": problem["problem_id"],
             "title": problem["title"], "physician_id": "", "hospitals": [problem.get("hospital_id") or ""],
             "rationale": problem["why"] + problem["rec"]["why_not"], "status": "pending_approval",
             "created_at": now, "approver": approver_role}
        self.proposals[p["id"]] = p
        self._log(now, "Coordinator", "escalated", p, problem=problem["title"], approver=approver_role)
        return p

    # ---------- configured approver / backup ----------
    def approve(self, pid: str, now, by: str):
        p = self.proposals[pid]
        if p["status"] not in {"pending_approval", "pending_backup"}:
            return
        expected = p["approver"]
        if by != expected and not (p["status"] == "pending_backup" and by.startswith("Backup approver")):
            raise ValueError(f"Approval requires {expected}")
        self._log(now, by, "approved", p)
        if p["kind"] == "split_shift":
            p.update(status="sent", sent_at=now, expires_at=now + pd.Timedelta(minutes=OFFER_EXPIRY_MIN),
                     approved_by=by)
            self._log(now, "System", "offer_sent_simulated", p, channel=p["channel"], message=p["message"])
        else:
            p.update(status="authorized", approved_by=by)

    def reject(self, pid: str, now, by: str, reason: str = ""):
        p = self.proposals[pid]
        p["status"] = "rejected"
        self._log(now, by, "rejected", p, reason=reason)

    # ---------- off-shift physician ----------
    def respond(self, pid: str, accept: bool, now) -> str:
        p = self.proposals[pid]
        if p["status"] == "expired" or (p["status"] == "sent" and now >= p["expires_at"]):
            p["status"] = "expired"
            self._log(now, p["physician_name"], "late_reply_ignored", p, reply="accept" if accept else "decline")
            return "This offer has expired."
        if p["status"] in {"cancelled", "no_longer_needed"}:
            # Someone else accepted first (or the gap closed): tell them, log it, change nothing.
            self._log(now, p["physician_name"], "accepted_but_already_filled" if accept else "reply_after_close", p)
            return "Thank you, this gap has already been filled. No action needed."
        if p["status"] != "sent":
            self._log(now, p["physician_name"], "reply_after_close", p, status=p["status"])
            return "This offer is no longer open."
        if not accept:
            p["status"] = "declined"
            self._log(now, p["physician_name"], "declined", p)
            return "Declined. Thank you."
        # First acceptance wins: if the gap is already covered, the second acceptor is told so.
        if any(set(s["hospitals"]) >= set(p["hospitals"]) for s in self.accepted_shifts):
            p["status"] = "no_longer_needed"
            self._log(now, p["physician_name"], "accepted_but_already_filled", p)
            return "Thank you, this gap has already been filled. No action needed."
        p["status"] = "accepted"
        self.accepted_shifts.append({"physician_id": p["physician_id"], "start": p["online_at"],
                                     "end": p["end"], "hospitals": p["hospitals"]})
        self._log(now, p["physician_name"], "accepted", p, online_at=f"{p['online_at']:%H:%M}",
                  routing_limited_to=p["hospitals"])
        self._log(now, p["physician_name"], "coverage_activated", p,
                  coverage_context=p["coverage_context"])
        for other in self.proposals.values():  # stop asking once the gap is filled
            if other is not p and other["kind"] == "split_shift" and other["status"] in OPEN_STATUSES \
                    and set(other["hospitals"]) <= set(p["hospitals"]):
                other["status"] = "cancelled"
                self._log(now, "System", "cancelled_gap_filled", other)
        return "Accepted. You're confirmed. Coverage has been updated."

    # ---------- timers ----------
    def tick(self, now):
        for handoff in self.handoffs.values():
            if handoff["status"] == "pending" and now > handoff["ack_due_at"]:
                handoff["status"] = "overdue"
                self._log(now, "System", "handoff_ack_overdue", handoff_id=handoff["id"],
                          consult_id=handoff["packet"]["consult_id"],
                          to_physician=handoff["packet"]["to_physician_id"])
        for p in self.proposals.values():
            if p["status"] == "pending_approval" and now >= p["created_at"] + pd.Timedelta(minutes=APPROVAL_TIMEOUT_MIN):
                p.update(status="pending_backup", approver="Backup approver (leadership chain)")
                self._log(now, "System", "approval_timeout_escalated_to_backup", p)
            if p["status"] == "sent" and now >= p["expires_at"]:
                p["status"] = "expired"
                self._log(now, "System", "offer_expired", p)
