"""Neuro Ops Command Center: Detect -> Decide -> Act.

Demo - Synthetic Data Only

Run:  streamlit run app.py
"""
import html
import os
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.act import ActionStore
from engine.ingestion import prepare_upload
from engine.compare import load_benchmark_comparison, snapshot_proxies
from engine.decide import RACI
from engine.detect import STALE_DATA_MIN
from engine.loader import ET, EVENT_COLUMNS, load_events, load_reference
from engine.pipeline import analyze
from engine.rebase import SCENARIO_ANCHOR, compute_delta, rebase_events, rebase_reference
from ui_theme import apply_theme, eyebrow, header

LIVE_REFRESH_SECONDS = 20
SEV = {"critical": "🔴 CRITICAL", "warning": "🟠 WARNING", "info": "🔵 INFO"}
GAP_BG, GAP_FG = "#d03b3b", "#ffffff"   # status palette: critical
FREE_LOW_BG = "#fab219"                  # status palette: warning

st.set_page_config(page_title="Neuro Ops Command Center", page_icon="🧠", layout="wide")
apply_theme()


# ---------------------------------------------------------------- state
@st.cache_data
def reference():
    return load_reference()


def init_state(force=False, reference_clock=False):
    if force or "events" not in st.session_state:
        # Pin the sample day onto the real clock once, at session start. Recomputing
        # this per rerun would re-anchor 18:20 to the current instant every time and
        # the scenario would never move.
        anchor = SCENARIO_ANCHOR if reference_clock else pd.Timestamp.now(tz=ET).floor("s")
        full, tod = compute_delta(anchor)
        st.session_state.frozen_now = anchor
        st.session_state.scenario_anchor = anchor
        st.session_state.delta_full = full
        st.session_state.delta_tod = tod
        st.session_state.events = rebase_events(load_events(), full)
        audit_override = os.environ.get("SEVARO_AUDIT_PATH")
        st.session_state.store = ActionStore(audit_path=Path(audit_override)) if audit_override else ActionStore()
        st.session_state.offset_min = 0
        st.session_state.live_clock = False
        st.session_state.loaded_files = set()
        st.session_state.phone_reply = {}
        st.session_state.stale_ack = False
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


init_state()
ref = rebase_reference(reference(), st.session_state.delta_full, st.session_state.delta_tod)
store: ActionStore = st.session_state.store
HOSP = ref["hospitals"].set_index("hospital_id")["hospital_name"].to_dict()
PNAME = pd.concat([ref["on_shift"], ref["off_shift"]]).set_index("physician_id")["name"].to_dict()

# Default approver role (configurable via sidebar)
if "approver_role" not in st.session_state:
    st.session_state.approver_role = "Clinical Lead"


def all_events() -> pd.DataFrame:
    ev = st.session_state.events
    if store.synthetic_events:
        ev = pd.concat([ev, pd.DataFrame(store.synthetic_events)[EVENT_COLUMNS]], ignore_index=True)
    return ev


def base_now() -> pd.Timestamp:
    """Paused scenario time or elapsed live time, advanced to uploaded event times."""
    latest = st.session_state.events["event_time"].max().ceil("min")
    clock = st.session_state.frozen_now
    if st.session_state.live_clock:
        clock = st.session_state.clock_scenario_anchor + (pd.Timestamp.now(tz=ET) - st.session_state.clock_wall_anchor)
    return max(clock, latest - pd.Timedelta(minutes=st.session_state.offset_min))


def clock_changed():
    if st.session_state.live_clock:
        st.session_state.clock_scenario_anchor = st.session_state.frozen_now
        st.session_state.clock_wall_anchor = pd.Timestamp.now(tz=ET)
    else:
        st.session_state.frozen_now = st.session_state.clock_scenario_anchor + (
            pd.Timestamp.now(tz=ET) - st.session_state.clock_wall_anchor)


# ---------------------------------------------------------------- help
@st.dialog("Tutorial & Help", width="large")
def show_help():
    st.markdown("""
**What this tool does.** It reads Synapse events, works out what needs attention now and in the
next 2 hours (**Detect**), recommends the fix and who approves it (**Decide**), and runs the fix
through approval to the doctor's reply (**Act**).

#### Workflow in 6 steps
1. **🧭 Operations coordinator tab: read the top card.** *Act on this first* is the #1 problem: what, where,
   why it matters, the recommended fix, who approves it, and why that fix beats the others.
2. **Scan the ranked list** below it. 🔴 Critical → 🟠 Warning → 🔵 Info. Expand any item for details
   and its action button.
3. **Check the coverage grid.** Each cell is *credentialed doctors on shift / required*.
   **Red ▲ short** = a projected gap. **Amber** in the *Now* column = nobody credentialed is free right now.
   ⚡ = demand surge.
4. **Act.** *Reassign* runs immediately (the coordinator owns it). *Propose offer* and *Escalate* go to
   the **🩺 Approver** tab (Clinical Lead by default) for approval. Low-risk alerts are sent automatically.
5. **🩺 Approver tab: approve or reject.** It shows the exact message the doctor will get (no patient
   details) and every off-shift doctor considered. If nobody answers within 5 min, it goes to a backup approver.
6. **📱 Off-shift neurologist tab: accept or decline.** Accept → the doctor is added to coverage (only at the
   short hospitals), the red cells clear, and other offers for that gap stop. Decline → next eligible
   doctor or escalate. Everything is recorded in the **📜 Audit log** (downloadable CSV).

#### Sidebar controls
- **Now (ET):** simulation time — the sample day is rebased onto the real clock so the 18:20
  scenario plays live. **+1 / +5 min** moves time forward to show timers, offer expiry and approval timeouts.
- **Load new events:** drop a CSV in the `event_stream` format. It is added to the day and everything
  is recalculated, with no restart. Duplicate and late records are handled automatically.
- **Reset demo:** reloads the sample data and clears all offers and logs.

#### Terms
| Term | Meaning |
|---|---|
| Consult | One hospital's request for a neurologist for one patient |
| Stroke / General Neuro | Consult types; stroke has the tightest time limits |
| Credentialed | Approved to take consults from that hospital |
| Derived state | What a doctor is actually doing, inferred from call, video, camera and agent signals (not their self-set status) |
| Agent offline | Desktop helper stopped reporting, so availability is unverified |
| Coverage gap | Fewer credentialed doctors than the hospital's minimum, now or soon |
| Surge | Last-60-min consults above the hospital's normal-busy level (p90) for that hour; raises the minimum by 1 |
| p90 | Normal-busy level: 9 in 10 past weeks were at or below it |
| Split shift | An off-duty doctor works a few extra hours to cover a gap |
| Split-shift trap | Routing would send a newly joined doctor almost every call; we limit them to the short hospitals |
| Escalate | Hand the problem to the clinical lead when the normal fix isn't available |
| RACI | Responsible · Accountable (approves) · Consulted · Informed |

#### Time limits (warning / critical)
Triaged, no doctor: 5 / 10 min · Assigned, not contacted (stroke): 3 / 8 · Waiting for CT/CTA (stroke): 12 / 20 ·
Scans back, no video (stroke): 6 / 12 · Note after call/video: stroke 30 / 45, general 45 / 90 ·
2+ open notes within 60 min of shift end: critical.

#### Split-shift rules (all must pass)
Opted in · credentialed at a short hospital · 10 h rest since last shift · 7-day hours + offer ≤ 60 ·
no clash with a scheduled shift · worth ≥ 2 h after time to get online · one open offer and ≤ 3 offers a week.
Offers expire after 10 min; the first acceptance wins.
""")
    st.caption("All data is synthetic. Thresholds are illustrative, not Sevaro's production settings.")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    eyebrow("Scenario controls")
    now = base_now() + pd.Timedelta(minutes=st.session_state.offset_min)
    st.metric("Now (ET)", now.strftime("%H:%M:%S"),
              help="Scenario time stays fixed while paused. Use +1/+5 to advance, or enable Live clock.")
    c1, c2, c3 = st.columns(3)
    if c1.button("+1 min"):
        st.session_state.offset_min += 1; st.rerun()
    if c2.button("+5 min"):
        st.session_state.offset_min += 5; st.rerun()
    if c3.button("Reset"):
        st.session_state.offset_min = 0; st.rerun()
    st.toggle(
        "Live clock", key="live_clock", on_change=clock_changed,
        help=f"Re-runs the page every {LIVE_REFRESH_SECONDS}s so the clock advances on its own. "
             "Turn off to hold the picture still while you talk through it.")
    st.caption(f"Sample day rebased by {st.session_state.delta_full.days}d "
               f"{int(st.session_state.delta_full.seconds // 3600)}h "
               f"{int(st.session_state.delta_full.seconds % 3600 // 60)}m "
               f"from {SCENARIO_ANCHOR:%Y-%m-%d %H:%M}")

    st.subheader("Load new events")
    files = st.file_uploader("Drop an event_stream-format CSV", type="csv", accept_multiple_files=True,
                             key=f"uploader_{st.session_state.uploader_key}")
    for f in files or []:
        sig = hashlib.sha256(f.getvalue()).hexdigest()
        if sig in st.session_state.loaded_files:
            continue
        try:
            # Uploads are written against the sample day, so they get the same shift.
            upload = prepare_upload(f.getvalue(), st.session_state.events, ref,
                                    st.session_state.delta_full)
        except Exception as exc:  # bad file: tell the user, keep running
            st.error(f"{f.name}: {exc}")
            continue
        fresh = upload["fresh"]
        st.session_state.events = upload["events"]
        # Uploaded future events advance the paused clock, without adding the manual offset twice.
        st.session_state.frozen_now = max(st.session_state.frozen_now,
            upload["latest"] - pd.Timedelta(minutes=st.session_state.offset_min))
        if st.session_state.live_clock and upload["latest"] > now:
            st.session_state.clock_scenario_anchor = st.session_state.frozen_now
            st.session_state.clock_wall_anchor = pd.Timestamp.now(tz=ET)
        st.session_state.loaded_files.add(sig)
        store._log(now, "System", "data_loaded", file=f.name, rows=upload["rows"], new_events=int(fresh),
                   duplicates=upload["duplicates"], content_sha256=sig)
        st.toast(f"Loaded {f.name}: {fresh} new events, {upload['duplicates']} duplicates ignored")
        st.rerun()

    ev = st.session_state.events
    st.caption(f"{ev['event_id'].nunique():,} unique events · last ingested "
               f"{ev['ingested_at'].max():%H:%M:%S}")
    st.button("Reference 18:20 scenario", type="secondary", on_click=init_state,
              kwargs={"force": True, "reference_clock": True})
    st.button("Reset demo", type="secondary", on_click=init_state, kwargs={"force": True})
    if st.button("❓ Tutorial / Help", type="tertiary", help="Opens a guide over this page"):
        show_help()
    
    st.divider()
    eyebrow("Approver role")
    st.selectbox(
        "Who approves split-shift offers?",
        ["Ops Lead", "Clinical Lead"],
        key="approver_role",
        help="Challenge default: Clinical Lead approves split-shift offers. "
             "Ops Lead remains available as an explicit alternative configuration."
    )

# ---------------------------------------------------------------- live clock
@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def live_clock_tick():
    """Re-run the whole page on a timer so the wall clock advances unattended.
    A fragment keeps session state (offers, approvals, audit log) intact; a browser
    reload would start a new session and reset the demo.

    The elapsed-time guard matters: the fragment body also runs during the initial
    script pass, and rerunning there would restart the script before it can finish
    painting, forever.
    """
    if not st.session_state.get("live_clock", True):
        return
    since = (pd.Timestamp.now(tz=ET) - st.session_state.last_full_run).total_seconds()
    if since >= LIVE_REFRESH_SECONDS:
        st.rerun(scope="app")


st.session_state.last_full_run = pd.Timestamp.now(tz=ET)
live_clock_tick()

# ---------------------------------------------------------------- engine
now = base_now() + pd.Timedelta(minutes=st.session_state.offset_min)
store.tick(now)
A = analyze(all_events(), ref, now, store.accepted_shifts, store.offer_counts(now),
            st.session_state.approver_role)
problems, cov, state, elig = A["problems"], A["coverage"], A["state"], A["eligibility"]
feed_lag_min = (now - state["quality"]["last_ingested"]).total_seconds() / 60
feed_stale = feed_lag_min > STALE_DATA_MIN
if not feed_stale:
    st.session_state.stale_ack = False
if "stale_ack" not in st.session_state:
    st.session_state.stale_ack = False
for p in problems:
    if p["rec"]["action"] == "alert" and p["severity"] == "critical":
        store.auto_alert(p, now)


# ---------------------------------------------------------------- shared renderers
def open_split_for(hospital_id):
    return next((o for o in store.proposals.values() if o["kind"] == "split_shift"
                 and o["status"] in {"pending_approval", "pending_backup", "sent", "accepted"}
                 and hospital_id in o["hospitals"]), None)


def actions_allowed() -> bool:
    return (not feed_stale) or bool(st.session_state.get("stale_ack"))


def action_area(p, key):
    rec = p["rec"]
    act = rec["action"]
    if feed_stale and not actions_allowed():
        st.caption("Act buttons are gated until you acknowledge that the event feed is stale.")
        return
    if act == "reassign":
        if st.button(f"Reassign to {PNAME.get(rec['target'])}", key=f"ra_{key}", type="primary"):
            store.reassign(p, rec["target"], now); st.rerun()
    elif act == "split_shift":
        existing = open_split_for(p["hospital_id"])
        if existing:
            st.info(f"Offer {existing['id']} to {existing['physician_name']}: **{existing['status'].replace('_', ' ')}**")
        elif st.button(f"Propose offer to {PNAME.get(rec['target'])} → {st.session_state.approver_role.lower()}", key=f"ss_{key}", type="primary"):
            cand = elig[elig["physician_id"] == rec["target"]].iloc[0]
            store.propose_split_shift(p, cand, elig, HOSP, now, st.session_state.approver_role); st.rerun()
    elif act == "escalate":
        existing = store.open_for(p["problem_id"]) or next(
            (o for o in store.proposals.values() if o["problem_id"] == p["problem_id"] and o["status"] == "authorized"), None)
        if existing:
            st.info(f"Escalation {existing['id']}: **{existing['status'].replace('_', ' ')}**")
        elif st.button(f"Escalate to {st.session_state.approver_role.lower()}", key=f"es_{key}", type="primary"):
            store.propose_escalation(p, now, st.session_state.approver_role); st.rerun()
    else:
        st.caption("🔔 Internal alert: sent automatically (low-risk, no approval needed)")


def problem_body(p, key):
    rec = p["rec"]
    st.markdown(f"**Where:** {p['where']}")
    st.markdown("**Why it matters:**\n" + "\n".join(f"- {w}" for w in p["why"]))
    st.markdown(f"**Do this:** {rec['label']}: {rec['detail']}  \n**Who approves:** {rec['approver']}")
    if rec["rationale"] or rec["why_not"]:
        st.markdown("**Why this fix:**\n" + "\n".join(f"- {r}" for r in rec["rationale"] + rec["why_not"]))
    if rec.get("candidates"):
        with st.expander("Eligible reassignment alternatives and scarce coverage"):
            for candidate in rec["candidates"]:
                pid = candidate["physician_id"]
                affected = ", ".join(HOSP.get(h, h) for h in candidate["scarce_hospitals"]) or "none"
                st.markdown(f"**{PNAME.get(pid, pid)} ({pid})**: "
                            f"{candidate['scarcity_cost']} other hospital(s) lose their last eligible alternative "
                            f"({affected}); {candidate['open_consults']} open consults, "
                            f"{candidate['consults_this_shift']} consults this shift.")
            st.caption("Every candidate passed credentials, routing availability, different owner and >30 min "
                       "remaining shift. Lower scarcity cost preserves the last eligible doctor at other at-risk hospitals.")
    r = rec["raci"]
    st.caption(f"RACI · Responsible: {r['R']} · Accountable: {r['A']} · Consulted: {r['C']} · Informed: {r['I']}")
    _render_advisory(p)
    action_area(p, key)


def _render_advisory(p):
    st.caption("Advisory — does not change the recommendation")
    ml = p.get("ml") or {}
    status = ml.get("status")
    if status == "ok":
        a1, a2, a3, a4 = st.columns(4)
        if "stage_sla_probability" in ml:
            a1.metric("Stage-SLA risk", f"{ml['stage_sla_probability']:.0%}")
        if "escalation_probability" in ml:
            a2.metric("Escalation risk", f"{ml['escalation_probability']:.0%}")
        if "wait_p50" in ml:
            a3.metric("Wait p50 (capped 120)", f"{ml['wait_p50']:.0f} min")
        if "wait_p90" in ml:
            a4.metric("Wait p90 (capped 120)", f"{ml['wait_p90']:.0f} min")
    elif status in {"unavailable", "experimental"}:
        st.caption("Live risk and wait estimates are unavailable; rules still guide this recommendation.")
        with st.expander("Advisory validation details"):
            st.caption(ml.get("note", "Models unavailable"))
    else:
        st.caption(ml.get("note", "No ML annotation for this problem type"))


def _render_pool_advisory(pools):
    with st.expander("Advisory pool sizing — experimental, does not change the recommendation"):
        if not pools or pools.get("status") == "unavailable":
            st.caption((pools or {}).get("note", "Pool sizing unavailable"))
            return
        recs = pools.get("recommendations") or []
        if recs:
            st.dataframe(pd.DataFrame([{
                "Task": r.task_type,
                "Current": r.current_size,
                "Recommended": r.recommended_size,
                "Demand forecast": f"{r.demand_forecast:.2f}",
                "Queue": r.queue_depth,
                "Flex": "yes" if r.flex_active else "no",
                "Rationale": r.rationale,
            } for r in recs]), hide_index=True, width="stretch")
        actions = pools.get("actions") or []
        if actions:
            st.caption("Suggested (not applied): " + "; ".join(actions))
        else:
            st.caption("No flex action suggested. This panel is experimental annotation only.")


def coverage_table():
    g = cov["grid"].copy()
    times = sorted(g["time"].unique())
    surge = cov["summary"].set_index("hospital_id")["surge"]
    labels = {t: (f"Now {t:%H:%M:%S %Z}" if t == now else f"Sched {t:%H:%M:%S %Z}") for t in times}

    def cell(r):
        if r["time"] == now:
            return f"{int(r['free_now'])} free · {r['staffed']}/{r['required']}"
        return f"{r['staffed']}/{r['required']}" + ("  ▲ short" if r["gap"] else "")

    g["cell"] = g.apply(cell, axis=1)
    g["row"] = g["hospital_id"].map(lambda h: HOSP[h] + ("  ⚡surge" if surge[h] else ""))
    text = g.pivot(index="row", columns="time", values="cell")
    gap = g.pivot(index="row", columns="time", values="gap")
    free_short = g[g["time"] == now].set_index("row")["short_now"]
    order = g.groupby("row")["gap"].any().sort_values(ascending=False).index
    text, gap = text.loc[order], gap.loc[order]
    text.index.name = "Hospital"

    def colour(_):
        out = pd.DataFrame("", index=text.index, columns=text.columns)
        for col in text.columns:
            for row in text.index:
                if gap.loc[row, col]:
                    out.loc[row, col] = f"background-color:{GAP_BG};color:{GAP_FG};font-weight:600"
                elif col == now and bool(free_short.get(row, False)):
                    out.loc[row, col] = f"background-color:{FREE_LOW_BG};color:#1a1a19"
        return out

    styled = text.rename(columns=labels).style.apply(
        lambda d: colour(d).rename(columns=labels), axis=None)
    st.dataframe(styled, width="stretch")
    st.caption("Now column: credentialed doctors on shift / required, plus who is free right now. "
               "Future columns are scheduled coverage (credentialed headcount on the roster) — "
               "usable capacity is unmodelled. Red ▲ = projected short. Amber in Now = nobody "
               "credentialed is free. Surge adds 1 to required. Accepted split shifts count only at their gap hospitals.")


# ---------------------------------------------------------------- header + tabs
header(now.strftime("%H:%M"), sum(p["severity"] == "critical" for p in problems),
       sum(p["kind"] == "gap" for p in problems))
st.caption(f"Simulation time {now:%Y-%m-%d %H:%M:%S} ET (sample day rebased onto the wall clock). "
           "Demo — synthetic data only.")
if feed_stale:
    st.warning(
        f"Event feed is stale ({feed_lag_min:.0f} min since last ingest). "
        "Offline-agent availability is unverified. Act buttons stay gated until you acknowledge.")
    st.session_state.stale_ack = st.checkbox(
        "I acknowledge that inputs may be out of date",
        value=bool(st.session_state.get("stale_ack")),
        key="stale_ack_box")

approver_label = st.session_state.approver_role
tab_coord, tab_lead, tab_phone, tab_log, tab_compare = st.tabs(
    ["🧭 Operations coordinator", f"🩺 {approver_label} ({len(store.pending_approvals())})",
     f"📱 Off-shift neurologist ({len(store.sent_offers())})", "📜 Audit log", "📊 Compare"])

with tab_coord:
    crit = [p for p in problems if p["severity"] == "critical"]
    waiting = [p for p in problems if p["kind"] == "consult" and p.get("physician_id") is None
               and p.get("consult_type") == "Stroke"]
    gaps = [p for p in problems if p["kind"] == "gap"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Critical", len(crit))
    k2.metric("Stroke · no doctor", len(waiting))
    k3.metric("Short in 2 h", len(gaps))
    k4.metric("Open consults", len(state["consults"]))

    if problems:
        top = problems[0]
        with st.container(border=True, key="topcard"):
            eyebrow(f"Act on this first · {SEV[top['severity']]}")
            st.markdown(f"### {top['title']}")
            problem_body(top, "top")

    left, right = st.columns([1.05, 1])
    with left:
        eyebrow("Detect")
        st.subheader("Everything that needs attention, ranked")
        for p in problems[1:]:
            with st.expander(f"#{p['rank']} · {SEV[p['severity']]} · {p['title']}",
                             expanded=p["rec"]["action"] in {"split_shift"} and not open_split_for(p.get("hospital_id"))):
                problem_body(p, p["problem_id"])
    with right:
        eyebrow("Coverage projection")
        st.subheader("Credentialed neurologists: now → next 2 hours")
        coverage_table()

        eyebrow("Decide")
        st.subheader("Split shift: who can be called in")
        if not A["gap_hospitals"]:
            st.caption("No uncovered short hospitals.")
        else:
            st.caption("Short hospitals: " + ", ".join(HOSP[h] for h in A["gap_hospitals"]))
        st.caption("Demand assumptions use the supplied synthetic weekday baseline when a weekend baseline is absent.")
        if len(elig):
            rule_names = list(elig.iloc[0]["rules"].keys())
            tbl = pd.DataFrame([{
                "Doctor": f"{r['name']} ({r['physician_id']})",
                "Eligible": f"✅ #{int(r['rank'])}" if r["eligible"] else ("⏳ offer open" if r["offer_open"] else "❌"),
                **{k: ("✅ " if r["rules"][k][0] else "❌ ") + r["rules"][k][1] for k in rule_names},
            } for _, r in elig.iterrows()])
            st.dataframe(tbl, hide_index=True, width="stretch")
            if A["gap_hospitals"] and not (elig["eligible"].any() or elig["offer_open"].any()):
                st.warning("Nobody passes every rule → escalate to the clinical lead (never bend a rule).")
        if store.accepted_shifts:
            for s in store.accepted_shifts:
                st.success(f"✅ {PNAME[s['physician_id']]} joins {s['start']:%H:%M}–{s['end']:%H:%M}, "
                           f"routable only at {', '.join(HOSP[h] for h in s['hospitals'])}")

        t = A["trap"]
        if t:
            st.subheader("Split-shift trap: keeping the new doctor from being swamped")
            j = PNAME[A["joiner"]["physician_id"]]
            m1, m2 = st.columns(2)
            m1.metric(f"Default routing: {j}", f"{t['joiner_unrestricted']} of {t['total_calls']} calls",
                      f"{len(t['joiner_unrestricted_hospitals'])} hospital(s)", delta_color="off")
            m2.metric("Our plan: gap hospitals only + seeded count",
                      f"{t['joiner_restricted']} of {t['total_calls']} calls",
                      f"{len(t['joiner_restricted_hospitals'])} hospital(s)", delta_color="off")
            st.caption(f"Simulated next {t['total_calls']} calls (baseline demand, surge rate where surging). "
                       f"Default: {j} starts at 0 consults, so routing rule 4 sends them calls from "
                       f"{', '.join(HOSP[h] for h in t['joiner_unrestricted_hospitals'])}. "
                       f"Our plan limits them to the short hospitals and seeds their count at the team median "
                       f"({t['seed']}), so they share the load instead of absorbing it.")
            both = t["unrestricted"][["physician_id", "calls"]].merge(
                t["restricted"][["physician_id", "calls"]], on="physician_id", suffixes=(" (default)", " (our plan)"))
            both.insert(1, "Doctor", both["physician_id"].map(PNAME))
            st.dataframe(both.drop(columns="physician_id"), hide_index=True, width="stretch")

    if store.handoffs:
        st.subheader("Consult handoffs — simulated receiving physician acknowledgment")
        for handoff in store.handoffs.values():
            packet = handoff["packet"]
            with st.container(border=True):
                st.markdown(f"**{handoff['id']} · {packet['consult_id']} · {handoff['status']}**")
                st.caption(f"{packet['from_physician_id'] or 'Unassigned'} → {packet['to_physician_id']} · "
                           f"{packet['stage']} · original stage entry {packet['stage_entered_at']} · "
                           f"remaining at transfer {packet['remaining_deadline_min']} min · "
                           f"ack due {handoff['ack_due_at']:%H:%M:%S}")
                if packet["remaining_deadline_min"] is not None:
                    remaining_now = packet["remaining_deadline_min"] - (now - handoff["created_at"]).total_seconds() / 60
                    st.caption(f"Original deadline: {remaining_now:.1f} min remaining now")
                if handoff.get("late"):
                    st.warning("Receiving physician acknowledged after the deadline; overdue outcome remains in the audit log.")
                with st.expander("Complete handoff context"):
                    st.json(packet)
                if handoff["status"] != "acknowledged" and st.button(
                        f"Simulate acknowledgment by {packet['to_physician_id']}", key=f"ack_{handoff['id']}"):
                    store.acknowledge_handoff(handoff["id"], packet["to_physician_id"], now); st.rerun()

    with st.expander("Doctor activity (derived from signals, not self-reported)"):
        ph = state["physicians"].copy()
        ph.insert(1, "name", ph["physician_id"].map(PNAME))
        ph["risk_flags"] = ph["risk_flags"].apply(lambda f: "; ".join(f))
        ph["availability"] = ph.apply(
            lambda r: "Unverified (agent offline)" if r["desktop_agent"] == "Offline"
            else ("Routable" if r["available_for_routing"] else "Not routable"), axis=1)
        st.dataframe(ph[["physician_id", "name", "shift_et", "derived_state", "self_reported_status",
                         "desktop_agent", "availability", "open_consults", "consults_this_shift",
                         "minutes_to_shift_end", "risk_flags"]], hide_index=True, width="stretch")
        st.caption("Offline desktop agent: availability is unverified. The physician can still appear "
                   "routable from other signals; treat that as unverified, not confirmed free.")
    with st.expander("Open consults: lifecycle and time in stage"):
        cs = state["consults"].copy()
        cs["hospital"] = cs["hospital_id"].map(HOSP)
        cs["stage_entered_at"] = cs["stage_entered_at"].dt.strftime("%H:%M:%S")
        st.dataframe(cs[["consult_id", "hospital", "consult_type", "physician_id", "current_stage",
                         "stage_entered_at", "minutes_in_stage", "minutes_since_received", "routing_failures"]],
                     hide_index=True, width="stretch")
        st.caption("Snapshot proxies: minutes_since_received is time since the consult arrived, not "
                   "first-service wait. routing_failures are failed routing attempts, not "
                   "ownership-changing handoffs. Replay metrics live on the Compare tab.")
    q = state["quality"]
    st.caption(f"Data handling: {len(q['duplicates'])} duplicate event(s) removed "
               f"({', '.join(q['duplicates']['event_id'])}); {len(q['late'])} late event(s) placed by event_time "
               f"({', '.join(q['late']['event_id'])}).")

with tab_lead:
    eyebrow(f"{approver_label} on duty")
    st.subheader("What puts stroke care at risk")
    risk = [p for p in problems if p.get("consult_type") == "Stroke" and p["severity"] == "critical"] + \
           [p for p in problems if p["kind"] == "gap"]
    for p in risk:
        st.markdown(f"- {SEV[p['severity']]} **{p['title']}**: {p['why'][0]}")

    # Forward-looking analytics
    eyebrow("Forward-Looking Analytics")
    st.subheader("Patterns for process improvement")
    
    # Coverage gaps by hospital
    g = cov["grid"]
    gaps_by_hosp = g[g["gap"] == True].groupby("hospital_id").size().sort_values(ascending=False)
    if len(gaps_by_hosp):
        st.markdown("**Projected coverage gaps by hospital** (next 2 hours):")
        gap_chart_data = gaps_by_hosp.rename(index=HOSP).reset_index()
        gap_chart_data.columns = ["Hospital", "Gaps"]
        st.bar_chart(gap_chart_data, x="Hospital", y="Gaps", height=200)
    
    # Stroke response time distribution
    stroke_consults = state["consults"][state["consults"]["consult_type"] == "Stroke"]
    if len(stroke_consults):
        st.markdown("**Stroke consult timing:**")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Open Stroke consults", len(stroke_consults))
        avg_wait = stroke_consults["minutes_since_received"].mean()
        mc2.metric("Avg time since received (snapshot proxy)", f"{avg_wait:.0f} min")
        mc3.metric("Max time in current stage", f"{stroke_consults['minutes_in_stage'].max():.0f} min")
    
    # Shift ending analysis
    ending_soon = state["physicians"][state["physicians"]["minutes_to_shift_end"] <= 60]
    if len(ending_soon):
        st.markdown("**Shifts ending in next 60 minutes:**")
        for _, p in ending_soon.iterrows():
            open_consults = p["open_consults"]
            status = "🔴" if open_consults >= 2 else ("🟠" if open_consults >= 1 else "🟢")
            st.markdown(f"- {status} **{PNAME.get(p['physician_id'], p['physician_id'])}**: "
                       f"{p['minutes_to_shift_end']:.0f} min left, {open_consults} open consults")
    
    # Escalation patterns (from log)
    esc_count = len([o for o in store.proposals.values() if o["kind"] == "escalate"])
    split_count = len([o for o in store.proposals.values() if o["kind"] == "split_shift"])
    st.markdown("**This session:**")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Split-shift proposals", split_count)
    sc2.metric("Escalations", esc_count)
    accepted = len([o for o in store.proposals.values() if o["status"] == "accepted"])
    sc3.metric("Shifts filled", accepted)

    eyebrow("Act")
    st.subheader("Waiting for your decision")
    st.caption("Each proposal retains its original required approver when the sidebar configuration changes.")
    pend = store.pending_approvals()
    if not pend:
        st.success("Nothing waiting for approval.")
    for o in pend:
        with st.container(border=True):
            backup = o["status"] == "pending_backup"
            if backup:
                st.warning(f"The {approver_label.lower()} did not respond within 5 min, so this was escalated to the backup approver "
                           "(leadership chain).")
            st.markdown(f"**{o['id']} · {'Split-shift offer' if o['kind'] == 'split_shift' else 'Escalation'}**: {o['title']}")
            if o["kind"] == "split_shift":
                st.markdown(f"**Doctor:** {o['physician_name']} · **Hours:** {o['start']:%H:%M}–{o['end']:%H:%M} ET "
                            f"· **Online by:** {o['online_at']:%H:%M} · **Channel:** {o['channel']}")
                st.markdown("**Message the doctor will receive (no patient details):**")
                st.markdown(f'<div class="sv-msg">{html.escape(o["message"])}</div>', unsafe_allow_html=True)
            st.markdown("**Rationale:**\n" + "\n".join(f"- {r}" for r in o["rationale"]))
            if o["kind"] == "split_shift":
                with st.expander("Everyone considered, and why each was or wasn't eligible"):
                    st.dataframe(pd.DataFrame([{"Doctor": PNAME[c["physician_id"]],
                                                "Eligible": "✅" if c["eligible"] else "❌",
                                                "Failed rules": "; ".join(c["failed"]) or "—"}
                                               for c in o["considered"]]), hide_index=True, width="stretch")
            who = "Backup approver" if backup else o["approver"]
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button(f"Approve ({who})", key=f"ap_{o['id']}", type="primary"):
                store.approve(o["id"], now, who); st.rerun()
            if b2.button("Reject", key=f"rj_{o['id']}"):
                store.reject(o["id"], now, who); st.rerun()

    done = [o for o in store.proposals.values() if o not in pend]
    if done:
        st.subheader("Decided")
        st.dataframe(pd.DataFrame([{"ID": o["id"], "What": o["title"], "Status": o["status"].replace("_", " "),
                                    "Doctor": o.get("physician_name", "")} for o in done]),
                     hide_index=True, width="stretch")

with tab_phone:
    offers = store.sent_offers()
    choices = [o["physician_id"] for o in offers] + [p for p in ref["off_shift"]["physician_id"]
                                                     if p not in {o["physician_id"] for o in offers}]
    pid = st.selectbox("View the phone of", choices, format_func=lambda p: f"{PNAME[p]} ({p})")
    _, phone, _ = st.columns([1, 1.1, 1])
    with phone:
        with st.container(border=True, key="phone"):
            st.markdown(f'<div class="sv-phone-top"><span>{now:%H:%M}</span><span>Sevaro Synapse</span></div>'
                        f'<div class="sv-phone-hello">Good evening, {PNAME[pid]}</div>',
                        unsafe_allow_html=True)
            mine = next((o for o in offers if o["physician_id"] == pid), None)
            if mine:
                left_min = max(0, int((mine["expires_at"] - now).total_seconds() // 60))
                st.markdown("#### New split-shift offer")
                st.markdown(f"**Hospitals:** {', '.join(HOSP[h] for h in mine['hospitals'])}")
                st.markdown(f"**Hours:** {mine['start']:%H:%M} – {mine['end']:%H:%M} ET")
                st.markdown(f"**Start:** be online by {mine['online_at']:%H:%M}")
                st.markdown("**Why you:** credentialed at these hospitals, opted in, rested.")
                st.caption(f"Approved by {mine.get('approved_by', 'clinical lead')} · expires in {left_min} min")
                a, d = st.columns(2)
                if a.button("Accept", key=f"acc_{mine['id']}", type="primary", width="stretch"):
                    st.session_state.phone_reply[pid] = store.respond(mine["id"], True, now); st.rerun()
                if d.button("Decline", key=f"dec_{mine['id']}", width="stretch"):
                    st.session_state.phone_reply[pid] = store.respond(mine["id"], False, now); st.rerun()
            else:
                st.markdown(st.session_state.phone_reply.get(pid, "No offers right now."))

with tab_log:
    eyebrow("Record")
    st.subheader("Audit log (append-only)")
    st.caption("Every offer: who was considered, who was excluded and why, who approved, and the outcome. "
               "Appended to logs/audit.jsonl (survives Reset) and downloadable as CSV. "
               "sim_time is simulation time.")
    log = store.log_frame()
    st.dataframe(log, hide_index=True, width="stretch", column_config={
        "sim_time": st.column_config.TextColumn("Scenario time", width="small"),
        "logged_at": st.column_config.TextColumn("Logged at", width="small"),
        "actor": st.column_config.TextColumn("Actor", width="small"),
        "action": st.column_config.TextColumn("Action", width="medium"),
        "proposal_id": st.column_config.TextColumn("Offer / escalation", width="small"),
        "kind": st.column_config.TextColumn("Type", width="small"),
        "physician_id": st.column_config.TextColumn("Neurologist", width="small"),
        "hospitals": st.column_config.TextColumn("Hospitals", width="small"),
        "details": st.column_config.TextColumn("Details (JSON)", width="large"),
    })
    st.download_button("Download log (CSV)", log.to_csv(index=False), "offer_audit_log.csv", "text/csv")

with tab_compare:
    _render_pool_advisory(A.get("pools"))
    eyebrow("Benchmark replay")
    st.subheader("Two independent questions, one saved file")
    st.caption("Simulation time: 2026-11-07 through 2026-12-31 (plus 3-day outcome drain). "
               "Waits are mean_capped_120min_wait — queue waits capped at 120 minutes, not 45-second "
               "access claims and not live-dashboard minutes_since_received.")

    report = load_benchmark_comparison()
    if report is None:
        st.error("benchmark/scheduling_replay.json is missing. Compare will not invent numbers.")
    else:
        as_is = report.cell("deterministic", "as_is")
        spec = report.cell("deterministic", "specialized")
        ml_as_is = report.cell("ml_augmented", "as_is")
        policy = report.policy_delta("deterministic")
        algo = report.algorithm_delta("as_is")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("as_is wait (capped 120)", f"{as_is.mean_capped_120min_wait:.2f} min")
        k2.metric("specialized wait (capped 120)", f"{spec.mean_capped_120min_wait:.2f} min",
                  f"{policy['wait_delta_min']:+.2f} min", delta_color="inverse")
        k3.metric("ML as_is wait (capped 120)", f"{ml_as_is.mean_capped_120min_wait:.2f} min",
                  f"{algo['wait_delta_min']:+.2f} min", delta_color="inverse")
        k4.metric("Leadership alerts (ML − det)", f"{ml_as_is.leadership_alerts:,}",
                  f"{algo['alert_delta']:+d}", delta_color="inverse")

        st.warning(
            f"**Measured result: specialized care policy degraded wait.** "
            f"Deterministic as_is {as_is.mean_capped_120min_wait:.4f} min vs specialized "
            f"{spec.mean_capped_120min_wait:.4f} min ({policy['wait_delta_min']:+.4f} min). "
            f"ML-augmented as_is changed wait by {algo['wait_delta_min']:+.4f} min while "
            f"leadership alerts moved {as_is.leadership_alerts:,} → {ml_as_is.leadership_alerts:,}. "
            "This is not an improvement claim."
        )

        st.markdown("### Care-policy effect (algorithm held fixed: deterministic)")
        st.caption("as_is vs specialized. Same arrivals, shifts, credentials, imaging waits.")
        st.dataframe(report.policy_table("deterministic"), hide_index=True, width="stretch")

        st.markdown("### Algorithm effect (policy held fixed: as_is)")
        st.caption("deterministic vs ml_augmented. ML only reorders already-eligible candidates. "
                   "Brier-score gains are not operational claims.")
        st.dataframe(report.algorithm_table("as_is"), hide_index=True, width="stretch")

        with st.expander("Provenance (from scheduling_replay.json)"):
            st.markdown(f"**Period:** {report.period}")
            st.markdown(f"**Episodes:** {report.episodes}")
            st.markdown(f"**Matching:** {report.matching}")
            st.markdown(f"**ML rule:** {report.ml_rule}")
            st.markdown(f"**Permissions:** {report.permissions}")
            st.markdown(f"**Limitations:** {report.limitations}")

        proxies = snapshot_proxies(state["consults"], state["physicians"])
        with st.expander("Live-snapshot proxies (not comparable to replay cells)"):
            s1, s2, s3 = st.columns(3)
            s1.metric("Mean minutes since received", f"{proxies.minutes_since_received_mean:.1f}")
            s2.metric("Routing-failure count", f"{proxies.routing_failure_count}")
            s3.metric("Open consults", f"{proxies.open_consults}")
            st.caption(proxies.note)

    with st.expander("Scope and limits"):
        st.markdown(
            "Twenty-one nuances cannot be cleared by a dashboard patch "
            "(identity, EHR, notifications, accreditation, clinical outcomes, and others). "
            "See [SCOPE_AND_LIMITS.md](SCOPE_AND_LIMITS.md) in this folder for each ID, "
            "its integration boundary, and the evidence that would close it."
        )
        st.caption("N17/N20 are deferred: a Streamlit session cannot authenticate users, "
                   "and in-memory lists cannot give atomic reservation.")
