# CLAUDE.md

> Loaded automatically at every session. Keep it SHORT and STABLE.
> Deep context lives in `docs/context/` — read those files on demand, not by default.

## Project

kabena (K-ABENA v3) — selective gradient computation, design-unbiased via
Horvitz-Thompson reweighting. Same accuracy as full-batch, ~28% less compute.

- Paper: https://arxiv.org/abs/2607.05903 (arXiv:2607.05903, cs.LG)
- Repo: https://github.com/Bonbhel/kabena-ml (MIT)
- PyPI: https://pypi.org/project/kabena — current version 2.1.1
- Author: M. Bonbhel — NeuroSoft IA x YekoElite University

## Environment (check before running anything)

Always run in the repo venv: `source .venv/bin/activate`
Verify before any test, build or experiment:

```
python -c "import sys, kabena; print(sys.prefix, kabena.__version__)"
```

Expected: the repo's `.venv` prefix and `2.1.1`. If it shows a system interpreter
or another prefix, STOP and tell me — do not run tests or build.
Never create a venv inside `kabena/` (the package directory).

Pinned: torch 2.2.2 (Mac Intel caps torch<2.3) which requires numpy<2 (ABI).
No GPU on this machine — CPU only.

## Core architecture

- `kabena/core/filter.py` — `Kabena` class, strategy dispatch, `select()`
- `kabena/core/sampling.py` — `select_v1` / `select_v2` / `select_v3` (the sampling plans)
- `kabena/core/config.py`, `kabena/core/gate.py` — params, v2→v3 safety gate
- `kabena/integrations/` — sklearn, torch, keras, huggingface (importable without the dep)
- `validation/` — Tables 1-5 reproducing the paper; `QUICK=1` for smoke test
- `tutorials/` — 4 families x 3 levels (notebook, script, MLOps)
- `tests/test_core.py` — 15 tests, home-made auto-discovery runner

Public API is two lines and must stay that way:

```python
kb = Kabena(N=0.3)             # signature: N, strategy, seed, k_percentile, alpha, min_active
active, weights = kb.select(losses)   # -> (bool mask, float weights)
```

There is NO `K` kwarg — the threshold is computed internally from `k_percentile`
on every `.select()` call.

## Invariants — do not break these without asking

1. **Design-unbiasedness** is the paper's headline claim. Any change to the
   sampling plan or the weights must preserve it, or be flagged loudly.
2. **The `(active, weights)` contract** is identical for v1/v2/v3 — that is what
   makes strategies swappable for callers.
3. **N is a PROPORTION in (0,1)**, never a count of examples.
4. **Never bump a published PyPI version** — always a new patch. A published
   version can never be withdrawn.
5. **Tag + push that trigger a PyPI release stay a human action.** Prepare
   everything, then stop and ask.

## Before any commit

```
python tests/test_core.py          # must show 15/15 OK
QUICK=1 python validation/run_all.py
```

Adding a test = define a module-level `test_*` function; the runner discovers it
automatically via `globals()`. No manual registration.

## Language rule (EN default, EN/FR when possible)

Docs, code comments and script output default to ENGLISH, then English/French
when possible. This is a goal, not a blocking requirement. Identifiers and
trivial inline comments are English-only. See `docs/context/04-conventions.md`.

## Deeper context — read on demand

| File | Read it when |
|---|---|
| `docs/context/01-history.md` | Touching v1/v2/v3 semantics, or writing about the method's genesis |
| `docs/context/02-theory.md` | Touching sampling, weights, bias, or the convergence argument |
| `docs/context/03-research-log.md` | Working on the arXiv v2 / the AMSGrad-variance question |
| `docs/context/04-conventions.md` | Writing docs, README, docstrings, user-facing strings |
| `docs/context/05-release.md` | Releasing, versioning, CI, packaging |
| `docs/context/00-pitfalls.md` | **Read this one early.** Mistakes already made on this project — do not repeat them |

## Working style

- Verify every number programmatically before writing it down. This project has
  already been burned by plausible-looking figures that were never computed.
- When a result looks too clean (identical to 4+ decimals across independent
  seeds), suspect the measurement before believing the finding.
- Prefer describing a correction in chat over regenerating a whole deliverable.
