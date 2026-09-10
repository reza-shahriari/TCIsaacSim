.PHONY: install test test-all lint fmt typecheck check luts golden-update clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest tests/unit -q

test-all:
	pytest tests -q -m ""

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src/irsim

check: lint typecheck test
	@echo "OK — safe to commit"

luts:
	python scripts/generate_luts.py --configs configs/sensors --out data/lut

golden-update:
	pytest tests/golden -q --update-golden

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	find . -name __pycache__ -type d -exec rm -rf {} +
