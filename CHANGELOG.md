# Changelog — kabena-ml

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] — 2026-06-30

### Added
- `calibrate_K_noisy()` — noise-adaptive K calibration for label/feature noise (core.filter)
- 2 new notebooks: `13_noise_robustness_pytorch.ipynb`, `14_noise_robustness_tf_hf.ipynb`
  (covers PyTorch, TensorFlow/Keras, and HuggingFace `Trainer` integration)
- `KabenaTrainerHF` reference implementation — HuggingFace Trainer subclass with
  K-ABENA filtering in `compute_loss` (+3 lines vs standard Trainer)
- Noise injection utilities: symmetric/asymmetric label noise, Gaussian feature noise
- Reproduction scripts for book Chapter 16B (noise robustness case study)

### Known limitation — please read before using `calibrate_K_noisy()`
- The default `noise_factor=2.0` and `q_cap=25.0` in `calibrate_K_noisy()` were observed
  as a reasonable tradeoff on **two configurations only** (CIFAR-10, SST-2) — this is
  **not** a calibrated universal constant, unlike `N=0.3` which is validated by ablation
  across 8 datasets. Documented as **Limitation L10** in the companion book (ch.18.10.2).
  **Before production use on a new dataset**, validate `noise_factor` via a grid search
  `[1.0, 1.5, 2.0, 2.5, 3.0]` on a held-out validation set with known noise rate.
- K-ABENA's noise self-protection mechanism (Section 16B.3) is empirically observed and
  analyzed but not yet formally proven. See Adam compatibility exploration notes for a
  related open theoretical question.

## [1.0.0] — 2026-06-29

### Added
- Core algorithm: `kabena_filter()`, `calibrate_K()`, `kabena_safe()`
- PyTorch integration: `kabena_filter_torch()`, `KabenaTrainer`
- TensorFlow/Keras integration: `KabenaCallback`, `KabenaTFTrainer`
- scikit-learn integration: `KabenaWrapper` (drop-in replacement)
- MLOps support: `KabenaConfig` (YAML/JSON/dict), Hydra + MLflow
- `KabenaScheduler`: adaptive warm-up schedule (warm-up, percentile, exponential)
- 12 Jupyter notebooks (3 levels: N1 beginner / N2 scripts / N3 MLOps)
- Reproduction scripts for book chapters 14–17
- Full test suite (pytest, 90%+ coverage)

### Notes
- Results marked `†` in the companion paper are simulations; GPU benchmarks planned for v1.1.0
- Adam + K-ABENA convergence proof is an open problem (see paper Section 5)
