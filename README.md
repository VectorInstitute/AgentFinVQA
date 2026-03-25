# AgentFinVQA

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/AgentFinVQA/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/AgentFinVQA/graph/badge.svg?token=83MYFZ3UPA)](https://codecov.io/github/VectorInstitute/AgentFinVQA)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/AgentFinVQA)

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```

### Running pre-commit hooks

```bash
uv run pre-commit run --all-files
```

> **Note for Vector Institute HPC users:** The Compute Canada pip configuration
> (set via `PIP_CONFIG_FILE`) interferes with pre-commit's environment setup,
> causing source builds of Rust-based tools (ruff, typos) instead of downloading
> pre-built wheels. To avoid this, either run with:
>
> ```bash
> PIP_CONFIG_FILE=/dev/null uv run pre-commit run --all-files
> ```
>
> Or add `export PIP_CONFIG_FILE=/dev/null` to your `~/.bashrc`.
