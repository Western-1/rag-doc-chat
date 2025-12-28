IMAGE_NAME=rag-service:local

# --- Docker Commands ---
build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8000:8000 --env-file .env -v "$(PWD)/qdrant_db:/app/qdrant_db" $(IMAGE_NAME)

stop:
	docker stop $$(docker ps -q --filter ancestor=$(IMAGE_NAME)) 2>/dev/null || true

# --- Kubernetes Commands ---
k8s-deploy:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

k8s-forward:
	@echo "🌐 Forwarding port 8000..."
	kubectl port-forward service/rag-service 8000:8000

k8s-delete:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/service.yaml

k8s-logs:
	kubectl logs -f deployment/rag-deployment

# --- Cleanup ---
clean:
	rm -rf __pycache__ .pytest_cache