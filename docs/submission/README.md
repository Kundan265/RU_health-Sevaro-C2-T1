# Submission evidence

This folder contains selected evidence from the October 11, 2026 local review. The disclosed reviewer score is **90/100**, not an official judging score. Human task timing and personal rehearsal remain pending.

- [Eight demonstrations](acceptance_checklist.md)
- [Four-minute presentation script](demo_script.md)
- [Reviewer assessment and limitations](hackathon_readiness_review_2026-10-11.md)
- [Automated checks](test_results.json) and [UI assertions](ui_checks.json)
- [Verification summary](verification_summary.json)
- [Controlled credential-preservation result](scarcity_outcome.json)

Screenshots document local browser verification on synthetic data. They do not establish a live deployment or clinical outcome. The repository's GitHub Actions run provides fresh Linux verification. `upload_cases/` reproduces changed same-name files, invalid batches and repeated content. The full original research remains outside this submission repository.

Run from the repository root:

```sh
python dashboard/verify_hackathon.py --output /tmp/sevaro-verification
```

Deployment settings and startup commands are in the [root README](../../README.md).
