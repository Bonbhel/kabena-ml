# 01 — History: how v1, v2 and v3 came to be

Needed whenever you touch strategy semantics, or write about the method for
teaching or publication. Getting this wrong produces confident, wrong prose.

---

## v1 — historical (pre-2.0)

Deterministic exclusion of the easy examples closest to the threshold K.

Common misconception to avoid: **v1 did not simply "ignore" the easy examples.**
It already had a parameter N which recovered the N easy examples FURTHEST from
the threshold.

Implementation: `select_v1` in `kabena/core/sampling.py`.

---

## v2 — regularized mode (still shipped, optional)

Lower half-domain only, `p_i ∝ eps_i`, no reweighting. **Biased by construction**
(Lemma 5) — a possible multiclass bonus, but contraindicated outside its validity
zone (see `kabena/core/gate.py`). Requires N <= 0.5.

v2 tested FOUR selection strategies for the N easy examples:

- (a) the N closest to the threshold
- (b) the N most central
- (c) the N going from the centre outwards, toward the furthest from the threshold
- (d) the N going from the centre toward the closest to the threshold

**(d) was retained**, with limited bias.

For teaching material: illustrate the four strategies with a GRAPH (it shows the
research process), but restrict numerical worked examples to the retained
strategy (d) only. Do not attempt to rank (a)-(d) by bias numerically.

Contraindications: minority signal ≳5%, noise ≲25%, N<=0.5.

---

## v3 — canonical (Definition 1 of the paper)

Draw `m = N·k` over ALL of 𝕄_K, with

```
p_i = alpha/k + (1 - alpha) * eps_i / sum(eps)        (alpha = 0.3, defensive floor)
pi_i = min(1, m * p_i)                                 (Assumption 1(b))
w_i  = 1 / pi_i                                        (Horvitz-Thompson)
```

then self-normalized (Hájek). Design-unbiased. N ∈ (0,1). Compute saving
`(1-N)·k/n`, ~28% at defaults. At `alpha=1`, `pi = m/k` is EXACT (SRSWOR).

Measured: bias divided by ~38 versus uncompensated variants; parity on real
datasets (Breast Cancer 0.9706, Digits 0.9609, Wine 0.9822).

---

## Key numbers to quote (measured, from the paper)

| Quantity | Value |
|---|---|
| Bias, uncompensated variants (Remark 1) | 0.128 - 0.151 |
| Bias, v3 | 0.004 |
| Reduction factor (Remark 2) | ~38 |
| Table 2, fraud 0.17%: base / v1 / v2 / v3 | 0.9998 / 0.56 / 0.53 / 0.9991 |
| Compute saving at N=0.3 | ~28% |
| N default / small-n recommendation | 0.3 / 0.4 if n<1000 |

---

## Parameters

- `N` — proportion of easy examples recovered, in (0,1). Optimum 0.3; use 0.4 if
  n < 1000.
- `k_percentile` (40.0) — defines the threshold K internally at each `.select()`.
- `alpha` (0.3) — defensive floor in the sampling probability. Bounds the maximum
  HT weight at `k/(m·alpha)`.
- `min_active` (1) — floor on the active set size.
- `strategy` — `'auto'` by default; the gate refuses v2 outside its validity zone
  unless `.force()` is called.

---

## Known limitations register

The paper carries L1-L8; the internal register runs to L16.

| ID | Content |
|---|---|
| L1 | Adam/AdamW/Lion excluded from the convergence proof (linearity) — see `03-research-log.md` |
| L2 | BatchNorm |
| L3 | Calibration |
| L4 | NumPy-only core; rivals benchmarked on CPU |
| L5 | Noise (v2 reaches 0.386 at 40% noise) |
| L6, L11 | Dissolved by v3 |
| L7 | Removed in v3 |
| L12 | Resolved by v3+SGD (0.9991 vs 0.53) |
| L13 | HT variance x10 |
| L14 | Parity |
| L15 | O(1/m) self-normalization bias, measured 0.004-0.037 |
| L16 | RNG |
