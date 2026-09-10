# Interpreter used for every target. The project interpreter is Isaac Sim's bundled Python
# (see docs/decisions/0002-dependency-floors-and-interpreter.md), e.g.
#   make check PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
# Any CPython >= 3.10 with the dev extras installed also works for the engine-free core.
PYTHON ?= python

.PHONY: install test test-all lint fmt typecheck check luts golden-update clean

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
	$(PYTHON) -m mypy src/irsim

check: lint typecheck test
	@echo "OK — safe to commit"

luts:
	$(PYTHON) scripts/generate_luts.py --configs configs/sensors --out data/lut

golden-update:
	$(PYTHON) -m pytest tests/golden -q --update-golden

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name __pycache__ -type d -exec rm -rf {} +
