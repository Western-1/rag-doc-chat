IMAGE_NAME := rag-service:local
VENV_BIN := ./venv/bin
PYTHON := $(VENV_BIN)/python
UVICORN := $(VENV_BIN)/uvicorn

.PHONY: help install dev lint eval build up down stop k8s-deploy k8s-delete k8s-logs k8s-forward clean

# --- Local Development ---

install:
	# 1. Створюємо venv саме з назвою "venv" і без питань
	uv venv venv --allow-existing
	# 2. Встановлюємо пакети САМЕ В ЦЕ оточення (--python venv/bin/python)
	uv pip install --python venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python venv/bin/python -r requirements.txt

dev:
	$(UVICORN) src.main:app --reload --host 0.0.0.0 --port 8000

lint:
	$(VENV_BIN)/ruff check .

eval:
	$(PYTHON) evaluation/run_eval.py

# --- Docker ---

build:
	docker build -t $(IMAGE_NAME) .

up:
	docker compose up --build -d

down:
	docker compose down

stop:
	docker compose down

# --- Kubernetes (K8s) ---

k8s-deploy:
	kubectl apply -f k8s/qdrant-statefulset.yaml
	kubectl apply -f k8s/qdrant-service.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

k8s-delete:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/qdrant-service.yaml
	kubectl delete -f k8s/qdrant-statefulset.yaml

k8s-forward:
	kubectl port-forward service/rag-service 8000:8000

k8s-logs:
	kubectl logs -f deployment/rag-deployment

# --- Utils ---

clean:
	rm -rf __pycache__ .pytest_cache venv .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'