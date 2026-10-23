LOG ?= results/logs/samples.parquet

.PHONY: install test lint synthetic fake part-a part-b

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src scripts tests

synthetic:
	python scripts/synthetic_check.py

fake:
	python scripts/make_fake_log.py --out results/logs/fake_samples.parquet

part-a:
	python scripts/part_a.py --log $(LOG)

part-b:
	python scripts/part_b.py --log $(LOG)
