# Eight required demonstrations

Source: supplied `Challenge_2_Handbook.html`, chapter 4, “What the brief asks you to show.” Local acceptance uses the supplied kit and controlled regressions. This is not certification of arbitrary unseen inputs.

| # | Requirement | Status | Verification and limit |
| --- | --- | --- | --- |
| 1 | Rebuild 18:20 and handle late/duplicate records | Pass | `check_snapshot.py`: 18 consults, eight physicians, zero mismatches; EVT-00529 deduplicated, EVT-01161 late. Reference-clock button verified in real browser and AppTest. |
| 2 | Separate physician activity, consult stage and timers | Pass | Reference replay matches separate tables; interface shows independent frames and elapsed deadlines. Nine focused regressions include true stage/deadline preservation. |
| 3 | Rank and explain flagged problems | Pass | CON-2105 top issue in reference scenario. AppTest exercises actions; real browser shows CON-2200, selected PHY-003, alternatives and scarcity cost in the controlled scenario. |
| 4 | Look ahead ≥90 min with shift ends and normal demand | Pass | 120-minute projection includes exact endpoint, shifts, hospital-rule boundaries, midnight/DST handling and baseline demand. Missed SYN-008 08:07 gap reproduced and repaired. Scheduled headcount is explicitly labeled; weekday demand is disclosed as fallback when synthetic weekend baseline is absent. |
| 5 | Recommend top fix; show every off-duty doctor's rule outcomes | Pass | Top CON-2105 escalation and candidate gates checked in engine; split-shift eligibility table rendered in browser. Controlled active reassignment preserves hard gates and explains alternatives. |
| 6 | Propose → approve → simulated send → accept/decline → update | Pass | Real browser and AppTest verify Clinical Lead approval, simulated send, acceptance and 2 → 0 future gaps. Engine checks cover decline, rejection, expiry, duplicate acceptance and backup. Actual browser proof does not cover every alternative branch. |
| 7 | Protect newly joined doctor from overload | Pass, modeled | Interface displays unrestricted versus restricted/seeded routing. AppTest verifies protection display and accepted gap-hospital restrictions; deterministic 20-call scenario is 10 → 5 joiner calls. No production routing or observed throughput claim. |
| 8 | Update with the judge CSV without restart | Pass | Actual browser file chooser: original file → 1,172 unique events; changed same-name/same-size bytes → 1,173; repeated identical bytes remain 1,173; invalid file leaves count, clock and working offer intact. No server restart between these upload cases. AppTest injection is separate. |

Additional gates: all eleven original scripts pass; all nine new focused scenarios pass; 30/30 final AppTest assertions pass. Model examples are loaded from the genuine saved files. Handoffs expose pending, overdue and late acknowledgment without resetting deadlines. Original research, data, saved models and benchmark remain unchanged. Human timing and personal four-minute rehearsal are pending, outside these machine acceptance results.
