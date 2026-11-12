.PHONY: setup lint test eda experiment-rq1 experiment-rq2 gate replay clean

setup:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest -v tests/

eda:
	python -m weatherpipe.eda --raw-dir data/raw --out results/eda_summary.md

experiment-rq1:
	python -m weatherpipe.experiment_rq1 --city "San Francisco"

experiment-rq2:
	python -m weatherpipe.experiment_rq2 --city "San Francisco"

gate:
	python -m weatherpipe.accuracy_gate --results results/rq1_forecast_accuracy.csv

replay:
	python -m weatherpipe.replay --weeks 4

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache mlruns

