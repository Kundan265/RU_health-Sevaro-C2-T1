# Four-minute presentation script — rehearsal still pending

Lead with: **“We turn an explained operational problem into an approved intervention while preserving scarce coverage.”**

Before presenting: launch the updated dashboard, click **Reference 18:20 scenario**, pause the clock, set **Clinical Lead**, and have the original judge CSV ready. Keep the controlled credential fixture available as a separate counterexample. A reset clears the session's offers; it does not erase persisted audit history. Do not promise a four-minute completion until you personally rehearse.

| Time | Action | What to say or show |
| --- | --- | --- |
| 0:00–0:35 | Detect | Show 18 consults and eight doctors, late/duplicate handling and CON-2105. Explain its elapsed stage limit and why escalation is recommended. Physician signals and consult stage remain separate. |
| 0:35–1:10 | Decide | Upload `dashboard/demo/credential_preservation_events.csv` in the reference scenario. Expand CON-2200 and eligible alternatives. Explain PHY-003 cost 1 versus less-loaded PHY-001 cost 2; privileges and >30 min remaining shift still gate both. This is a controlled supplied-credential counterexample. |
| 1:10–2:05 | Act | Reset to **Reference 18:20 scenario**. Show all off-shift candidate gates; propose PHY-009 for SYN-005/SYN-008 and the waiting hospital SYN-010. Approve as Clinical Lead, then accept on the simulated phone. Point out future gaps 2 → 0 and online time 18:35; the immediate urgent consult still needs attention. |
| 2:05–2:30 | Protect | Show the routing comparison: 10/20 versus 5/20 modeled calls, restricted gap hospitals and seeded workload. Explicitly call it a routing model. |
| 2:30–3:15 | Adapt | Upload `dashboard/demo/judge_test_events.csv` through the actual file chooser. Show 18:29, new CON-2119 and CON-2109 assignment to PHY-007. Explain automatic recomputation, byte-based identity and duplicate-event handling. |
| 3:15–4:00 | Prove | Show the eight-demo checklist, final test summary and controlled outcomes. Human task-time savings are unmeasured. If you later complete the solo study, present its individual observations and per-task medians with limitations. Live ML is suppressed until features are faithful; the genuine saved benchmark includes negative pool results. |

Optional follow-up outside the four-minute slot: reset, free a credentialed doctor using the controlled event fixture, reassign a consult, show its original stage time and remaining deadline in the handoff, advance +5 minutes and simulate acknowledgment to expose overdue/late states. Do not portray a simulated acknowledgment as a real clinician response.

Launch and verification commands are in [README.md](README.md). Files for content identity, duplicate uploads and invalid-batch demonstrations are in `upload_cases/`. Avoid favorable constants, clinical-outcome claims, or a promised 95/100 score.
