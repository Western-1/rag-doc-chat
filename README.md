
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-red" alt="Qdrant">
  <img src="https://img.shields.io/badge/Docker-Containerized-blue" alt="Docker">
</p>

# Talk to Your Docs – RAG Inference Service

## Description
This repository contains a **production‑oriented Retrieval‑Augmented Generation (RAG) service** designed to ingest PDF documents, index them into a vector database, and answer user queries using Large Language Models (LLMs).

The service is implemented as a containerized microservice and follows **MLOps‑ready** design principles: reproducibility, persistence, clear configuration, and API‑first interaction.

## Architecture Overview
The system follows a standard RAG pipeline:

1. **Document ingestion** (PDF upload)
2. **Text extraction & chunking**
3. **Embedding generation**
4. **Vector storage (persistent)**
5. **Similarity search**
6. **LLM‑based answer generation**

## Technology Stack
- **Language:** Python 3.12
- **Web Framework:** FastAPI
- **RAG Framework:** LangChain (LCEL)
- **LLM Provider:** Google Gemini 2.5 Flash
- **Embedding Model:** Google Text Embedding 004
- **Vector Database:** Qdrant (local persistent mode)
- **Containerization:** Docker (slim image)
- **API UI:** Swagger + LangServe Playground

## Key Features
- Persistent vector storage across container restarts
- Optimized retrieval configuration for factual accuracy
- Stateless application layer
- Docker‑ready for local, staging, and cloud deployment
- Interactive playground for debugging and validation
- Clear separation between ingestion and inference

## Project Structure
```
├── app/
│   └── main.py          # FastAPI app and RAG pipeline logic
├── qdrant_db/           # Persistent vector database storage
├── tests/               # API and integration tests
├── Dockerfile           # Container build configuration
├── .dockerignore        # Docker ignore rules
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (not committed)
```

## Configuration
The service requires a Google API key for LLM and embeddings.

Create a `.env` file in the project root (or Copy .env.example to .env and fill in your keys.):
```bash
GOOGLE_API_KEY=your_google_api_key
```

## Build and Run (Docker)
### Build image
```bash
docker build -t rag-service:local .
```

### Run container with persistent storage
```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v "${PWD}/qdrant_db:/app/qdrant_db" \
  rag-service:local
```

The volume mount ensures vector indices persist across restarts.

## API Endpoints
Once running, the following endpoints are available:

| Endpoint | Description |
|--------|------------|
| `/docs` | Swagger UI for API testing |
| `/chat/playground` | LangServe UI for RAG interaction |
| `POST /ingest` | Upload and index PDF documents |

![ingest](images/ingest.png)

## Data Ingestion Flow
- PDFs are parsed using `PdfReader`
- Text is split using `RecursiveCharacterTextSplitter`
- Chunk size: `600`
- Retrieval top‑k: `10`
- Embeddings are generated and stored in Qdrant
- Cosine similarity is used for nearest‑neighbor search

![LLM Chat answer](images/playground_chat.png)

## MLOps Considerations
- **Reproducibility:** Fully containerized environment
- **Persistence:** Vector database stored outside container
- **Scalability:** Stateless API layer, replaceable vector backend
- **Observability (recommended):**
  - Add structured logging
  - Export metrics (Prometheus)
  - Monitor Qdrant storage growth
- **Security (recommended):**
  - Protect ingestion endpoints
  - Store secrets in a secret manager
- **CI/CD (recommended):**
  - Linting and tests on PR
  - Docker image build & push
  - Deployment to cloud (AWS / GCP / Azure)

## Common Issues
- **Docker build fails:** verify Docker daemon is running and base image is available
- **Permission issues with qdrant_db:** ensure correct filesystem permissions
- **Empty responses:** verify documents were ingested successfully

## Production Recommendations
- Use managed Qdrant or external vector DB for scale
- Add authentication and rate limiting
- Implement document versioning
- Add backup and retention policy for vector data

## License

MIT License

Copyright (c) 2025 Andriy Vlonha

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.