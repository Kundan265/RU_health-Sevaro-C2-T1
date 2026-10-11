# Dashboard

Run from the **repository root** so `.streamlit/config.toml` is loaded consistently locally and on Streamlit Community Cloud:

```sh
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
python dashboard/verify_hackathon.py --output /tmp/sevaro-verification
```

Use Python 3.11. See the [root README](../README.md) for the demo, deployment configuration and evidence.

The engine rebuilds consult and physician state from the supplied synthetic CSVs, projects scheduled coverage, ranks actions with hard eligibility gates and credential preservation, and runs simulated approval/offer/handoff workflows. `data/`, `models/`, `benchmark/`, `demo/`, and repository-local model test fixtures are all included.

Nine saved model bundles load, but incomplete live feature semantics suppress live risk/wait estimates. Experimental pool sizing reports unavailable. The active recommendation remains rule-based. Consult handoffs preserve clocks and routing history; staffing offers carry separate coverage context. See [SCOPE_AND_LIMITS.md](SCOPE_AND_LIMITS.md).
