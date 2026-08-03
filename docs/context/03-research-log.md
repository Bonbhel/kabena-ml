# 03 — Research log: the HT-variance investigation (closed)

State as of 2 Aug 2026. Read before doing any experimental work on the
AMSGrad-variance question, or before drafting the arXiv v2.

---

## Question

Do the HT variance spikes, engraved by the AMSGrad ratchet, cause a real problem —
and can any change to the algorithm reduce them?

**Non-negotiable acceptance filter (set by the author):** K-ABENA's strategic
advantage is the BALANCE — same accuracy as full-batch AND ~28% compute saving.
Any fix that degrades accuracy is rejected even if it does reduce variance. The
success criterion is "variance reduced AT preserved efficiency AND preserved
saving", not "variance reduced".

Five approaches were tested in an imposed order: A1, A2, C1, C2, B — then D.

---

## Setup used

- Canonical fraud replica: `make_fraud(n=20000, share=0.0017, d=15, seed=0, sep=1.2)`
  in `validation/_common.py`. A home-made generator (two Gaussians, no `flip_y`),
  NOT `sklearn.make_classification`.
- Note `table2_extreme_imbalance.py` trains with plain weighted SGD, so the
  AMSGrad collision does not appear there. Experiments used
  `torch.optim.Adam(amsgrad=True)`, beta2=0.999, lr=0.05, logistic regression,
  200 epochs, 5 seeds.
- Non-regression benchmark: Table 2 → base 0.9998 / v1 0.56 / v2 0.53 / v3 0.9991.
- Key metric: **brake inflation `v̂/v`** (see pitfall P5).

---

## Verdicts

| Lever | What was tried | Verdict |
|---|---|---|
| **A1** | Raise the defensive floor alpha (0.1 → 1.0) | **No purchase.** Mechanism works (w_max varies monotonically with alpha) but is temporally DECOUPLED from the peak |
| **A2** | Rigorous unequal-probability design without replacement instead of `np.random.choice` | **No purchase on the peak** — but fixed a real sampling bias (see below) |
| **C1** | Robust `v̂`: max over a smoothed statistic (median w3/w5/w11, short EMA b0.5/b0.8) | **No purchase.** The peak is PERSISTENT, not a transient artifact |
| **C2** | Warm-up of alpha (high early, released later) | **No purchase.** Confirms A1 independently, via the alpha=1.0 bound |
| **B** | Truncated / winsorized HT weights | **Only measurable purchase**, but rejected for production |

### Why A1 and A2 fail — the unified explanation

`v̂` is a slow EMA (beta2=0.999) starting from zero; it takes hundreds of steps to
reach its equilibrium level, while the raw squared gradient DECREASES as the model
converges (~0.003-0.004 at step 0-5, ~1000-2000x smaller by step 80). The peak
detected around step ~80 is simply where the two curves cross — but its ceiling was
built from the very first steps, when alpha (and the sampling plan) have no
differentiating grip yet. So the ceiling is essentially independent of both levers,
by construction of the EMA + ratchet dynamics.

### A2's side finding — keep this, it has independent value

`np.random.choice(replace=False, p=...)` does NOT satisfy Assumption 1(b). The
deviation is systematic and directional: empirical `pi` is BELOW assumed `pi` on
the highest-`p_i` units (max deviation 0.039). Measured consequence: **+2.7% bias**
on a test HT sum (37.42 vs true 36.43), and higher variance.

The **pivotal method** (Deville-Tillé) fixes both: deviation drops to 0.004
(residual bias +0.08%), variance −5 to −8%, and it runs in 10.6 ms per call at the
real regime. Implementation: `select_v3_pivotal` + `_pivotal_draw`
(`sampling_v3_sampford.py`, not yet in the repo).

This upgrades the implementation from "approximately conforming to Assumption 1(b)"
to "exactly conforming", and it is what makes hypothesis (H3) of Théorème 1′
actually hold. Worth shipping in v2 on its own merits, independently of the
variance question.

### B's verdict in detail

Caps 5.0 / 3.0 / 2.0 / 1.5, each with and without renormalization.

- Canonical: real, monotone, dose-dependent effect (inflation 1.071 → 1.055), NOT
  a mass-amputation artifact (renormalized variants show the same drop), AUC intact.
  Best case erases ~24% of the over-braking.
- Stress (n=300, sep=0.5): effect **inverted**, monotone the other way
  (1.052 → 1.067) — aggressive capping makes over-braking WORSE exactly where the
  theoretical problem lives.
- At cap<=2.0, 16.67% of weights are capped = ALL the reweighted minority units.
  That is not clipping outliers, it is flattening the HT structure.

Two opposite monotone trends means a real effect whose SIGN depends on the data
regime — unshippable. Rejected for production; kept as a sensitivity note.

---

## Conclusion (this is the v2 deliverable, "piste D")

The over-braking is **real, measured for the first time, and small**: inflation
1.071 canonical / 1.052 stress, i.e. **5-7%**. It is structural and persistent —
insensitive to four independent levers — and the most brutal lever available
(flattening the whole HT structure) moves it by only ±1.5 points.

This is quantitatively consistent with the O(sqrt(log T)) bound of Théorème 1′ and
with the null empirical results. "Piste D" is therefore no longer the cheap
fallback: it is the **positively demonstrated conclusion**.

**Next action:** draft the v2 robustness note — Théorème 1′ plus the A/C/B
cross-validation, with the pivotal correction presented as a contribution in its
own right. Agreed process: propose an OUTLINE before writing any LaTeX.

---

## Scripts produced (currently outside the repo, in `Dev/piste 0/`)

`piste0_grid_real_package.py`, `piste_A1_alpha_sweep.py`,
`piste_A2_pivotal_vs_choice.py`, `sampling_v3_sampford.py`,
`piste_C1_robust_vhat.py`, `piste_C2_alpha_warmup.py`,
`piste_B_truncated_weights.py`.

Pending decision: bring them into the repo (e.g. `research/variance/`) so the
agent and future readers can see them, or leave them out to keep the published
repo minimal.

---

## Remaining v2 prerequisites

1. Canonical tests with the real fraud replica — **done** (this log).
2. The author must validate the v2 mathematical derivations, ideally with
   Dr Fendji (the cs.LG endorser).
