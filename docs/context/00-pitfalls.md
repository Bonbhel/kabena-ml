# 00 — Pitfalls: mistakes already made on this project

Every item below actually happened. They are recorded so the same hours are not
spent twice. Read this before any experimental or numerical work.

---

## P1 — Do not reimplement the v3 mechanism from memory

A from-scratch NumPy reimplementation of v3 failed the paper's own published
extreme-imbalance result (reached ~0.78-0.83 where the paper reports 0.9991).
Root cause: an incorrect self-normalization formula written from memory. Even
after correcting it against Definition 1 / Eq. 3 fetched from the arXiv HTML, the
reimplementation still did not reproduce the baseline (0.94 vs 0.9998).

**Rule:** always call the installed `kabena` package. If a reimplementation is
unavoidable (e.g. a custom optimizer that torch cannot express), it MUST pass an
explicit parity test against the reference implementation before any result from
it is interpreted. See P6.

---

## P2 — A worrying number that was never measured

A figure of "AMSGrad+v3 AUC 0.806 vs SGD+v3 0.9991" circulated for a while and
motivated an entire investigation. It was a **misattribution** — an error made
while reconstructing compacted context, most likely confused with OHEM's 0.4464.
The actual record (a `torch_utils.py` docstring from the session concerned)
documents Adam+K-ABENA as *empirically stable*; the real issue was always a
THEORY GAP (no proof), never a measured degradation.

**Rule:** before acting on an alarming number, find where it was measured. If the
provenance cannot be traced to an actual run, treat it as unverified.

---

## P3 — Do not present v1 → v2 as decreasing bias

Illustrative bias figures (+0.49 for v1, +0.73 for v2) were once computed on a
SINGLE draw of a 10-example toy set. They measured sampling noise, not structural
bias (which is an expectation over many draws).

The paper does NOT rank v1 and v2 by bias. Section 5 groups them in the same
family of uncompensated methods. Remark 1 gives the measured range for
uncompensated variants: **0.128-0.151**, versus **0.004** for v3 (factor ~38).
The break happens at v3, not between v1 and v2.

**Rule:** any illustrative single-draw number must carry an honesty note and cite
the real measured values alongside it.

---

## P4 — Classic algorithms can fail at this project's scale

Sampford rejective sampling works fine at k=30, m=9. At the real regime
(k~8000, m~2400) it never converges — 0 successes in 20 000 rejection attempts.
It was replaced by the **pivotal method** (Deville-Tillé): same guarantees
(exact π_i, negative association), O(n), no rejection.

A first pivotal implementation then took **3.9 s per call** because it refiltered
a pending list at every step (effectively O(k²)). Rewritten as a single-pointer
sequential sweep: **10.6 ms per call**, ~370x faster.

**Rule:** test any sampling/statistical algorithm at the real regime (n=20000,
k~8000) before adopting it, and profile it before running a full grid.

---

## P5 — Choose a metric that can distinguish the things you are comparing

The peak of `max(v̂)` over a training run cannot tell Adam from AMSGrad: for Adam
`v̂ = v`, for AMSGrad `v̂` is the running max, so the maximum over the trace is
`max_t(v_t)` in BOTH cases. It is a mathematical identity, not a finding.

Worse, smoothing `v` (median window, short EMA) mechanically lowers the max of
the smoothed series whether or not the spikes are transient — which produced a
fake ~15-25% "improvement".

The metric that actually works is the **brake inflation** `v̂ / v`: exactly 1.000
for Adam by definition, >1 for AMSGrad, and it measures the real over-braking.

**Rule:** before running a grid, check that the metric returns the expected value
on a known control case.

---

## P6 — Parity thresholds: strict on mechanism, loose on outcome

A home-made AMSGrad matched `torch.optim.Adam(amsgrad=True)` to 3.5e-8 on the
`v̂` trace but differed by 1.7e-5 on final AUC. That AUC gap is floating-point
rounding accumulated over 200 steps (torch uses fused ops; an equivalent explicit
sequence rounds differently) — not a logic error.

**Rule:** keep the strict threshold (1e-6) on the mechanism being validated, and
allow a looser one (5e-4) on downstream outcomes.

---

## P7 — Venv traps on this machine

Two successive venvs broke the environment: the first created BY MISTAKE inside
`kabena/` (the package directory), the second retaining internal paths to a
previous directory. Symptom: `pip` reports "already satisfied" while Python
raises `ModuleNotFoundError`.

Diagnosis — these three must agree and point at the repo's `.venv`:

```
which python
which pip
python -c "import sys; print(sys.prefix)"
```

The repo path contains spaces (`00 - KABENA`) — quote it in zsh. Also note `<`
and `>` are redirection operators in zsh: never paste a `<placeholder>` literally.

---

## P8 — Results that look too clean

Identical peaks to 4-5 decimals across 5 independent seeds AND 7 parameter values
is not a null result — it is a measurement artifact until proven otherwise. In
this project it revealed P5 twice.

**Rule:** when independent runs agree suspiciously well, instrument the mechanism
at the lowest level (raw weights, raw gradients) before concluding.
