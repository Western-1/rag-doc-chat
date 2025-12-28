IMAGE_NAME=rag-service:local

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker compose up --build

stop:
	docker compose down

k8s-deploy:
	kubectl apply -f k8s/qdrant-statefulset.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

k8s-forward:
	kubectl port-forward service/rag-service 8000:8000

k8s-delete:
	kubectl delete -f k8s/deployment.yaml
	kubectl delete -f k8s/service.yaml
	kubectl delete -f k8s/qdrant-statefulset.yaml

k8s-logs:
	kubectl logs -f deployment/rag-deployment

clean:
	rm -rf __pycache__ .pytest_cache
