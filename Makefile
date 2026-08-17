.PHONY: setup audit backtest train predict validate test all

VENV := .venv/bin
PY := $(VENV)/python

setup:
	python3.14 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip --quiet
	$(PY) -m pip install -r requirements.txt --quiet
	$(PY) -m ipykernel install --user --name retailcast-venv --display-name "Python (retailcast venv)"

audit:
	$(VENV)/jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=retailcast-venv --ExecutePreprocessor.timeout=180 \
		notebooks/01_audit.ipynb

backtest:
	$(PY) src/backtest_report.py

train predict:
	$(PY) src/predict.py

validate:
	$(PY) validate_format.py --submission submission.csv

test:
	$(PY) -m pytest tests/ -q

all: backtest predict validate test
