# Design decisions

Rules the engine and dashboard follow, with their source. HB = Challenge 2 Handbook, Brief = Challenge 2 brief v2, Mentor = Sevaro mentor answers (10 Oct 2026), Client = clarification answers (11 Oct 2026).

## RACI per intervention (Client Answers #2, Mentor Q1, Q2, Q3, Q6)

| Intervention | Responsible | Accountable (approves) | Consulted | Informed |
|---|---|---|---|---|
| Internal alert | System (automatic) | none | none | Coordinator |
| Reassign consult | Coordinator | Coordinator (no approval) | none | Original physician (auto, after), Quality Leadership |
| Split-shift offer | Coordinator (proposes) | **Ops Lead on Duty** (default, configurable) | none | Clinical Lead, Quality Leadership |
| Escalation | Coordinator | **Ops Lead on Duty** (default, configurable) | none | Clinical Lead, Quality Leadership |
| Approval not answered in time | System | Next tier in the leadership chain | none | Coordinator |

**Key changes from brief default (Client Answer #2):**
- **Ops Leaders are the real-time decision makers.** They approve split-shift offers and authorize escalations.
- **Clinical Leaders are informed**, not consulted. They analyze patterns and make forward-looking changes, not real-time calls.
- The approver role is **configurable** via a sidebar toggle. Flip to "Clinical Lead" if challenged by a judge; the brief's exact wording is quoted in the tooltip.

## Coverage gaps handled uniformly (Client Answer #1)
- **A coverage gap triggers the same intervention whether the waiting consults are Stroke or General Neuro.**
- Consult type affects only the time thresholds (Stroke clocks are tighter), never the intervention path.
- A gap affecting only General Neuro consults **still triggers a split-shift offer**, not just an alert.

## Offer log = structured event log (Client Answer #3, Mentor Q7)
- Every offer status change is recorded as **its own timestamped event**: proposed, approved, rejected, sent, accepted, declined, expired, superseded, cancelled_gap_filled.
- This drives real-time paging triggers, auditing, and process-improvement analysis.
- Each record holds: who was considered, who was excluded and why, the approver, timestamps, and the outcome (HB p.15).

## Original physician auto-notified (Client Answer #4, Mentor Q2)
- When a consult is reassigned, the original physician is **automatically notified** with only the consult ID and time.
- Example: "CON-2110 was reassigned from you at 18:25."
- No coordinator confirmation required; the notice is sent immediately upon reassignment.

## Hospitals are not in the loop (Mentor Q4)
- Hospitals are not told about reassignments or incoming doctors. Live insight is for clinical and ops leadership only.
- Hospitals see who completed a consult only in a monthly report. **No hospital-facing notifications.**

## Root cause to frame the pitch around (Mentor Q8)
- The main problem is **staffing during unforeseen surges concentrated in a few hospital systems, where credentialing limits who can help.**
- Clearer ownership helps as the network scales.
- What this means for the dashboard: show **credential depth** per hospital (how many credentialed doctors, and how many will still be on after shift ends), not just raw headcount.

## Engine defaults for open questions (change here if mentors answer differently)

| Question | Default in code | Where |
|---|---|---|
| N3: what is "now" after new data? | `max(wall clock, latest event_time)`, plus the +1/+5 min clock buttons | `app.py: base_now()` |
| N3a: how does the sample day reach the wall clock? | The session pins the sample's 18:20 anchor to the moment it starts and shifts all five time-bearing surfaces by that one delta, so the scenario plays live | `engine/rebase.py` |
| N4: do scheduled night starts count? | Yes, from their start time (12 h shift) | `coverage.py: shift_roster()` |
| N1: repeat calls | Dedupe only by `event_id`; never merge different `consult_id`s | `replay.py` |
| B5: fresh approval after a decline? | Yes. A decliner is excluded for the day, and the next candidate needs a new approval. Nobody left → escalate | `decide.py`, `act.py` |
| E1: ranking eligible doctors | Most gap hospitals covered → fastest online → fewest hours in 7 days → fewest recent declines | `decide.py: eligibility()` |
| E4: two doctors accept | First acceptance wins; the second is told "already filled"; both are logged | `act.py: respond()` |
| Offer length | 3 h from now (rounded to 5 min); value = 3 h minus time to get online | `decide.py: OFFER_HOURS` |
| Approval timeout | 5 min, then goes to a backup approver (leadership chain) | `act.py: APPROVAL_TIMEOUT_MIN` |
| Surge in projection | A surge seen now is assumed to persist across the 2 h horizon | `coverage.py` |
| Temporarily Unavailable | Not counted toward coverage until their status changes | `coverage.py` |
| Split-shift trap | Joiner is routable only at gap hospitals and seeded at the on-shift median consult count | `routing_sim.py`, `act.py` |
| Documentation stage | Interruptible, so the doctor counts as available for routing | `replay.py: derive_state()` |
