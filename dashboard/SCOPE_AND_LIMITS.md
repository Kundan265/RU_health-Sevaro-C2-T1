# Scope and limits

This dashboard is a hackathon operations demo on synthetic data. Twenty-one nuances
from the Sevaro comparison cannot be cleared by a patch in this repository. Each
row names the integration boundary and the evidence that would close it.

Recommendations remain rule-based, with active credential-preservation ranking.
All nine saved model bundles load, but live estimates are suppressed when input
measurements or training-feature semantics are incomplete. Models never change
the operational decision. Experimental pool sizing reports unavailable.

N17 and N20 are explicitly deferred: a Streamlit session cannot authenticate
users, and in-memory lists cannot give atomic reservation. Documenting them
is honest; faking them is not.

| ID | What is out of reach | Integration boundary | Evidence that would close it |
|---|---|---|---|
| N02 | Operating maturity and scale (12 hospitals / 8+6 doctors vs hundreds of sites) | Production roster, hospital graph, and SLA contract | Live census and staffing feeds covering the real network, with measured scale tests |
| N13 | Privilege lifecycle (static CSV, no expiration or reappointment) | Credentialing / privileging system of record | Approved credential feed with expiration, license, and reappointment checks before dispatch |
| N17 | Role access enforcement (tabs + sidebar selector, one shared session) | Identity provider / SSO | Authenticated users, role-bound sessions, and audit of the real actor — not a widget |
| N19 | Real notification delivery (`offer_sent_simulated`, physician tab) | Paging / native triage / SMS or voice | Delivery receipts, failed-send handling, and physician responses from the real channel |
| N20 | Concurrent acceptance and confirmed presence (in-memory first-accept) | Durable offer store with atomic reservation | Cross-session locking plus confirmed-online before a joiner counts as ready |
| N26 | Ambient note creation | Nirvana Notes / documentation product | Documentation status consumed from Sevaro; rebuilding notes is out of scope |
| N34 | Event transport and response speed (CSV upload, 20 s refresh) | Live Synapse / OneCall event adapter | Ingestion-to-action latency on the production bus, separate from dashboard refresh |
| N35 | EHR interoperability and chart write-back | EHR integration boundary | Approved operational events in; clinical write-back remains outside the hackathon |
| N36 | Video, imaging, and PACS | Existing clinical tools | Readiness events from video / PACS; those tools stay integration dependencies |
| N37 | Native mobile triage and call failover | Physician mobile app | Offers and receipts through the existing app if Sevaro exposes an approved interface |
| N38 | Security assurance (no SOC 2 / production access controls here) | Sevaro security review | Integration requirements, attested deployment boundary, and access control evidence |
| N39 | Clinical accreditation and governance | Joint Commission / Sevaro governance | A pilot aligned with Sevaro clinical governance — this prototype is not a service |
| N41 | Clinical outcomes vs operational completion | Outcome registry / real cohorts | Patient outcomes, treatment times, and transfers measured separately from Signed Off |
| N42 | National quality benchmarking | Governed KPI / registry mapping | National standards mapping; keep synthetic operational thresholds separate |
| N43 | Financial impact | Cost and ROI model with finance | Operator time, staffing cost, and avoided delay before any dollar claim |
| N44 | Remote EEG workflow | Distinct EEG operational queue | Add only after the core command-center loop is validated on Stroke / General Neuro |
| N45 | Inpatient rounding and neurocritical care | Service-line operational contracts | Separate rounding / ICU queues if requested; current lifecycle is episodic consults |
| N46 | Outpatient and post-discharge continuity | Ambulatory / referral workflows | Link sign-off to existing follow-up systems if the project expands past acute ops |
| N47 | Rehabilitation and caregiver coordination | Rehab / caregiver programs | Optional integration; not part of the core Detect → Decide → Act demo |
| N48 | Hybrid staffing supply | Workforce / locum supply | Ranking cannot create qualified staff; integrate supply and credentialing processes |
| N49 | Specialty expansion | Runtime specialty configuration | Needed only if another service line is requested |
| N50 | Funding and company maturity | Not an engineering close | Financing explains enterprise expectations; it does not score this algorithm |
| N51 | Market chronology and valuations | Dated company / investor evidence | Prefer primary sources; do not invent valuations or treat announcement dates as close dates |
| N52 | Public sentiment and internal competitive claims | Sevaro demo + anonymized operational pilot | No Reddit or superiority claim is supported from this folder |

N50–N52 are not product defects. They are listed so the demo does not borrow
company-scale, market, or sentiment claims it cannot evidence.
