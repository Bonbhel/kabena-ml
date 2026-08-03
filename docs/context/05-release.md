# 05 — Release process

Reference: `Runbook_Deploiement_kabena_2_1_0.docx` (the human reference).

## Hard rules

1. **Never bump a published PyPI version.** Always a new patch. A published
   version can never be withdrawn.
2. **The tag and push that trigger publication stay a human action.** An agent
   may prepare everything — tests, build, checks — then stop and ask.

## Publishing

Trusted Publishing (OIDC) + Sigstore attestations. **No API tokens.**

## Pre-release sequence

```
python tests/test_core.py            # 15/15 OK
QUICK=1 python validation/run_all.py
python -m build                      # dist/*.whl + dist/*.tar.gz
twine check dist/*                   # PASSED for both
```

Slash command: `/pre-release-check` (stops at a FAIL and before the push).

## Version state

- 1.2.0 → 2.1.0 (deliberate jump, 6 July 2026), then 2.1.1 (DX patch).
- Current branch: `fix/2.1.1`.

## Repo hygiene — OUTSTANDING since 2.1.0

`.venv/`, `.DS_Store`, `kabena.egg-info/` must be removed from tracking
(`git rm --cached`) and covered by `.gitignore`. `.ipynb_checkpoints/` and
`__pycache__` were handled during 2.1.1. Slash command: `/gitignore-audit`.

Also untracked and therefore invisible to any other machine or session:
`scripts/check_bilingual.py`.

## Known past incidents (so they are recognised, not re-debugged)

CI failures, lint errors, build-backend misconfiguration, a critical
label-indexing bug in `KabenaWrapper`, and SSH issues (ed25519 key, GPG/SSH
confusion, port 22 → 443).
