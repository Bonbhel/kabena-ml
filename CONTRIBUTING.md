# Contributing to kabena-ml

Thank you for your interest in contributing to K-ABENA!

## Quick start

```bash
git clone https://github.com/yekoelite/kabena-ml.git
cd kabena-ml
pip install -e ".[dev]"
pytest tests/ -v
```

## Contribution types

- **Bug reports**: open an issue with a minimal reproducible example
- **New integrations**: HuggingFace, Flax, JAX — PRs welcome
- **Low-resource case studies**: document your use case in GitHub Discussions
- **Theoretical extensions**: federated K-ABENA, token-level LLMs, meta-learning K and N

## Code style

```bash
black kabena_ml/ tests/
ruff check kabena_ml/ tests/
```

## Contact

Jean-François Bonbhel — contact@neurosoft-ia.com
YekoElite University × NeuroSoft IA
