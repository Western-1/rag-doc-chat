IMAGE_NAME := rag-service:local
VENV_BIN := ./venv/bin
PYTHON := $(VENV_BIN)/python
UVICORN := $(VENV_BIN)/uvicorn
COMPOSE := docker compose

.PHONY: help install dev lint eval ui clean-db rebuild build up down stop restart ps logs logs-qdrant logs-api logs-streamlit exec-qdrant exec-api exec-streamlit clean-volumes clean-all k8s-deploy k8s-delete k8s-logs k8s-forward clean

# --- Local Development ---

install:  ## Install dependencies in virtual environment
	uv venv venv --allow-existing
	uv pip install --python venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python venv/bin/python -r requirements.txt

dev:  ## Run development server with hot reload
	$(UVICORN) src.main:app --reload --host 0.0.0.0 --port 8000

lint:  ## Run linter (ruff)
	$(VENV_BIN)/ruff check .

eval:  ## Run evaluation script
	$(PYTHON) evaluation/run_eval.py

ui:  ## Run Streamlit UI
	PYTHONPATH=. $(VENV_BIN)/streamlit run src/app.py

clean-db:  ## Clean Qdrant database by deleting collection
	@echo "🧹 Deleting collection documents_v3_multilingual..."
	curl -X DELETE "http://localhost:6333/collections/documents_v3_multilingual"
	@echo "\n✅ Done! Database is clean."

# --- Docker Management ---

rebuild:  ## Rebuild Docker image and services from scratch
	$(COMPOSE) down
	docker builder prune -f
	docker rmi $(IMAGE_NAME) || true
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

build:  ## Build Docker image
	docker build -t $(IMAGE_NAME) .

up:  ## Start Docker services (build if needed)
	$(COMPOSE) up --build -d

down:  ## Stop and remove Docker services
	$(COMPOSE) down

stop: down  ## Alias for down

restart: down up  ## Restart all services

ps:  ## Show status of Docker services
	$(COMPOSE) ps -a

logs:  ## Tail logs for all services
	$(COMPOSE) logs -f

logs-qdrant:  ## Tail logs for Qdrant service
	$(COMPOSE) logs -f qdrant

logs-api:  ## Tail logs for API service
	$(COMPOSE) logs -f api

logs-streamlit:  ## Tail logs for Streamlit service
	$(COMPOSE) logs -f streamlit

exec-qdrant:  ## Exec into Qdrant container shell
	docker exec -it talk_to_your_docs_rag_system-qdrant-1 sh

exec-api:  ## Exec into API container shell
	docker exec -it talk_to_your_docs_rag_system-api-1 sh

exec-streamlit:  ## Exec into Streamlit container shell
	docker exec -it talk_to_your_docs_rag_system-streamlit-1 sh

clean-volumes:  ## Remove Docker volumes (data loss warning!)
	$(COMPOSE) down -v

clean-all: down  ## Clean everything: stop, remove images, volumes, prune
	docker rmi $(IMAGE_NAME) || true
	$(COMPOSE) down -v --rmi all
	docker system prune -f

# --- Kubernetes (K8s) ---

k8s-deploy:  ## Deploy to Kubernetes
	kubectl apply -f k8s/qdrant-statefulset.yaml
	kubectl apply -f k8s/qdrant-service.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

k8s-delete:  ## Delete Kubernetes resources
	kubectl delete -f k8s/deployment.yaml || true
	kubectl delete -f k8s/service.yaml || true
	kubectl delete -f k8s/qdrant-service.yaml || true
	kubectl delete -f k8s/qdrant-statefulset.yaml || true

k8s-forward:  ## Port forward Kubernetes service
	kubectl port-forward service/rag-service 8000:8000

k8s-logs:  ## Tail logs for Kubernetes deployment
	kubectl logs -f deployment/rag-deployment

# --- Utils ---

clean:  ## Clean Python caches and virtual env
	rm -rf __pycache__ .pytest_cache venv .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'