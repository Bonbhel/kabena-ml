# Contributing to kabena

Thank you for your interest in contributing to K-ABENA!

## Quick start

```bash
git clone https://github.com/Bonbhel/kabena-ml.git
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
black kabena/ tests/
ruff check kabena/ tests/
```

## Contact

Jean-François Bonbhel — contact@neurosoft-ia.com
YekoElite University × NeuroSoft IA
