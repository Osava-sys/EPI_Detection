# Raccourcis de developpement.
#
# Sous Windows, les scripts PowerShell de scripts/ sont la voie principale ;
# ce Makefile s'adresse aux environnements disposant de `make` (Git Bash, WSL,
# Linux, macOS) et cible le meme interpreteur de venv.

PYTHON ?= .venv/Scripts/python.exe
DATA ?= data.yaml
DETECTION_DATA ?= artifacts/dataset_detection/data.yaml
WEIGHTS ?= artifacts/models/best.pt

.PHONY: help setup audit clean-dataset smoke train evaluate predict api ui export test lint typecheck check format all

help:
	@echo "Cibles disponibles :"
	@echo "  setup          Installe le projet et ses dependances de developpement"
	@echo "  audit          Audite le dataset original"
	@echo "  clean-dataset  Construit le dataset de detection normalise"
	@echo "  smoke          Smoke test d'entrainement (2 epoques)"
	@echo "  train          Entrainement complet"
	@echo "  evaluate       Evaluation sur le split de test"
	@echo "  predict        Inference sur SOURCE=<chemin>"
	@echo "  api            Demarre l'API REST"
	@echo "  ui             Demarre l'interface Streamlit"
	@echo "  export         Export ONNX avec verification"
	@echo "  test           Tests unitaires"
	@echo "  lint           Ruff"
	@echo "  typecheck      Mypy"
	@echo "  check          lint + typecheck + test"

setup:
	$(PYTHON) -m pip install -e ".[api,ui,export,audit,dev]"

audit:
	$(PYTHON) -m ppe_detection.dataset_audit --data $(DATA) \
		--output artifacts/reports/dataset_audit_original.json

clean-dataset:
	$(PYTHON) -m ppe_detection.dataset_cleaner --source $(DATA) \
		--output artifacts/dataset_detection --mode copy

smoke:
	$(PYTHON) -m ppe_detection.train --config configs/train.yaml --smoke

train:
	$(PYTHON) -m ppe_detection.train --config configs/train.yaml

evaluate:
	$(PYTHON) -m ppe_detection.evaluate --weights $(WEIGHTS) \
		--data $(DETECTION_DATA) --split test

predict:
	$(PYTHON) -m ppe_detection.predict --weights $(WEIGHTS) \
		--source "$(SOURCE)" --save --save-json

api:
	$(PYTHON) -m uvicorn ppe_detection.api:app --host 127.0.0.1 --port 8000

ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py

export:
	$(PYTHON) -m ppe_detection.export --weights $(WEIGHTS) \
		--format onnx --imgsz 640 --simplify

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check src tests app

format:
	$(PYTHON) -m ruff check src tests app --fix

typecheck:
	$(PYTHON) -m mypy

check: lint typecheck test
