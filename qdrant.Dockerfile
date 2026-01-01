FROM qdrant/qdrant:latest

USER root
RUN apt-get update && apt-get install -y curl && apt-get clean && rm -rf /var/lib/apt/lists/*
USER $USER_ID