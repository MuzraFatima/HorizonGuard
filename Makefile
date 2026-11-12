PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: help install lint format test data train eval paper all clean

help:
	@echo "install  install the package with dev extras"
	@echo "lint     ruff lint and format check"
	@echo "format   apply ruff formatting and safe fixes"
	@echo "test     run the test suite"
	@echo "data     generate trajectories and fork labels"
	@echo "train    train hazard and effect models"
	@echo "eval     run evaluation and write tables"
	@echo "paper    regenerate figures and tables for the papers"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m horizonguard.cli data --config $(CONFIG)

train:
	$(PYTHON) -m horizonguard.cli train --config $(CONFIG)

eval:
	$(PYTHON) -m horizonguard.cli eval --config $(CONFIG)

paper:
	$(PYTHON) -m horizonguard.cli paper --config $(CONFIG)

all: data train eval paper

clean:
	rm -rf build dist .pytest_cache .ruff_cache runs outputs
