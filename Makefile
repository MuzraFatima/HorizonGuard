PYTHON ?= python
DATA ?= data/driftbench-cf

.PHONY: help install lint format test data train eval paper dashboard serve all clean

help:
	@echo "install    install the package with dev extras and dashboard dependencies"
	@echo "lint       ruff lint and format check, plus dashboard ESLint"
	@echo "test       run the Python test suite"
	@echo "data       generate DriftBench-CF trajectories and fork labels"
	@echo "train      fit and cache the live monitor artifacts"
	@echo "eval       regenerate every report in docs/ from the benchmark"
	@echo "paper      copy the regenerated tables into paper/tables"
	@echo "dashboard  build the dashboard"
	@echo "serve      run the API and dashboard on http://127.0.0.1:8765"
	@echo "all        data, train, eval, paper and dashboard in order"

install:
	$(PYTHON) -m pip install -e ".[dev]"
	cd dashboard && npm ci

lint:
	ruff check .
	ruff format --check .
	cd dashboard && npm run lint

format:
	ruff check --fix .
	ruff format .

test:
	$(PYTHON) -m pytest

data:
	horizonguard data --out $(DATA)
	horizonguard audit --data $(DATA) --write docs/label_noise.md

train:
	$(PYTHON) -c "from horizonguard.monitor.live import load_or_build; load_or_build('$(DATA)')"

eval:
	horizonguard reproduce --data $(DATA)

paper:
	mkdir -p paper/tables
	cp docs/baselines.md docs/hazard.md docs/effects.md docs/guarantees.md docs/transfer.md paper/tables/

dashboard:
	cd dashboard && npm run build

serve:
	horizonguard serve --data $(DATA)

all: data train eval paper dashboard

clean:
	rm -rf build dist .pytest_cache .ruff_cache runs outputs dashboard/dist
