# 02 — Theory: what is proved, what is not

Read before touching sampling, weights, or any claim about convergence.

---

## What the paper proves

- **Lemmas 2-3** — the v3 estimator is (quasi) design-unbiased; the residual is
  the O(1/m) self-normalization (Hájek) bias, measured at 0.004-0.037.
- **Lemma 5** — v2 is biased by construction.
- **Assumption 1(b)** — `pi_i = m * p_i` (truncated at 1). The HT weights `1/pi_i`
  are only correct insofar as the sampling plan actually realises these inclusion
  probabilities. See the A2 finding in `03-research-log.md`: `np.random.choice`
  does NOT realise them exactly.
- **Theorem 1** — convergence, for optimizers that are LINEAR in the gradient
  estimate. This is why L1 excludes Adam / AdamW / Lion: they divide by
  `sqrt(v)`, which is non-linear, so the unbiasedness of the estimator does not
  transfer to the update.

## Proof-work already done

- **Lemma 1** was rebuilt after an invalid intermediate assumption was found. A
  final exact two-term bound requiring no extra assumptions was derived, plus a
  weighted variant **Lemma 1'**.
- The auto-switch threshold for extreme imbalance was corrected empirically from
  `k/n > 0.95` to **`n - k < 15`** (Lemma 1 violations are quasi-systematic when
  `n - k <= 5`).
- A peer-review simulation caught and fixed C1 (semantic contradiction in the N
  definition) and C2 (the Lemma 1 gap above).

---

## Théorème 1′ — the AMSGrad x HT result

This is the project's own contribution closing the Theorem 1 gap for AMSGrad.
LaTeX already delivered: `appendix_amsgrad_concentration.tex` (+ compiled PDF,
`references_appendix.bib`), written in the preprint's style, `\input`-ready.

**Hypotheses**

- (H1) per-coordinate bounded gradients, constant G
- (H2) Assumption 1(b) with the defensive floor ⟹ `1/pi_i <= k/(m·alpha)`
- (H3) **negatively associated** sampling indicators — satisfied by rejective /
  Sampford / pivotal designs, NOT by `np.random.choice` draw-by-draw

**Result**

Hoeffding-under-negative-association applied to numerator and denominator
separately, then a ratio decomposition (which absorbs the Hájek bias):

```
P(|ĝ⁽ʲ⁾ − ḡ⁽ʲ⁾| > eps) <= 4·exp(−eps²n²m²alpha² / (8G²k³))
```

Uniform over T steps and d coordinates:

```
eps(delta) = 2·sqrt(2)·G·k^{3/2} / (n·m·alpha) · sqrt(log(4dT/delta))
```

so the AMSGrad ratchet inflation is **at most O(sqrt(log T))**, of amplitude
`O(G·k^{3/2}·sqrt(log(dT/delta))/(n·m·alpha))`. With the paper's defaults this is
**O(1/sqrt(n))** — vanishing with dataset size, consistent with the null empirical
result at n=20000.

**Scale correction worth remembering:** the exponent is `k^{3/2}/(n·m·alpha)`,
NOT `k/(m·alpha)` as an earlier sketch had it.

**Honest framing (keep this in any write-up):** concentration gives an UPPER
bound. That is the favourable direction — it closes the gap benignly. A lower
bound showing the inflation actually occurs would require anti-concentration:
separate, harder, and less useful.

Optional refinement available: Bernstein-under-NA replaces `k³` by `k²·variance`.

References already in the bib: Joag-Dev & Proschan 1983, Dubhashi & Panconesi
2009, Reddi et al. 2018, Sampford 1967, Brändén & Jonasson 2012.

---

## The mechanism behind the spikes (condensed)

The defensive floor bounds `pi_i >= m·alpha/k`, hence `1/pi_i <= k/(m·alpha)`.
With small-scale defaults (k=8, m=2, alpha=0.3) the max weight is 13.3, so a
minority gradient of 0.5 becomes 6.67 and its square 44.4 (versus g²=0.25 without
reweighting). Those squared values are the spikes.

Collision with AMSGrad: the ratchet `v̂ = max(v̂, v)` engraves a spike FOREVER
(brake permanently too tight), where Adam's EMA forgets it. The asymmetry matters:
AMSGrad is over-sensitive to PEAKS but immune to TROUGHS. Any fix aimed at peaks
must not destroy the benign immunity to troughs.

**Empirical status: no measured degradation.** At n=20000, vanilla Adam and
AMSGrad both show all deltas within noise (max |Δ| = 0.0012, and ±0.0001 on the
canonical fraud replica). The problem is a proof gap, not an observed failure.
The measured over-braking is 5-7% (see `03-research-log.md`).
