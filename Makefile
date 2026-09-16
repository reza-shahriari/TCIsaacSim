# Interpreter used for every target. The project interpreter is Isaac Sim's bundled Python
# (see docs/decisions/0002-dependency-floors-and-interpreter.md), e.g.
#   make check PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
# Any CPython >= 3.10 with the dev extras installed also works for the engine-free core.
PYTHON ?= python

# `make ci`: the GPU-free gate on a plain CPython venv, the same job .github/workflows/check.yml
# runs. Proves the engine-free core needs neither Isaac Sim nor CUDA.
CI_PYTHON ?= python3.10
CI_VENV ?= .venv-ci

.PHONY: install test test-all lint fmt typecheck check ci luts golden-update clean next

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/unit tests/golden -q --durations=10

test-all:
	$(PYTHON) -m pytest tests -q -m ""

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

fmt:
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

typecheck:
	$(PYTHON) -m mypy src/irsim src/irsim_isaac src/irsim_eval

# What to start now, and republish the queue the roadmap shows. `make next` regenerates it;
# `make check` only verifies it, so a stale queue fails the gate instead of misleading a reader.
next:
	$(PYTHON) scripts/next_step.py
	@$(PYTHON) scripts/next_step.py --write

check: lint typecheck test
	@$(PYTHON) scripts/next_step.py --check
	@echo "OK — safe to commit"

ci:
	$(CI_PYTHON) -m venv $(CI_VENV)
	$(CI_VENV)/bin/python -m pip install -q --upgrade pip
	$(CI_VENV)/bin/python -m pip install -q -e ".[dev]"
	$(MAKE) check PYTHON=$(CI_VENV)/bin/python

luts:
	$(PYTHON) scripts/generate_luts.py --configs configs/sensors --out data/lut --data data

golden-update:
	$(PYTHON) -m pytest tests/golden -q --update-golden

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build $(CI_VENV)
	find . -name __pycache__ -type d -exec rm -rf {} +
