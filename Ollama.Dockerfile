FROM ollama/ollama:latest AS builder

ARG OLLAMA_MODEL=qwen2.5:1.5B
ENV OLLAMA_MODELS=/baked-models

RUN mkdir -p /baked-models && chmod 777 /baked-models && \
    ollama serve > /dev/null 2>&1 & \
    sleep 5 && \
    until ollama list > /dev/null 2>&1; do echo "Waiting for Ollama..." && sleep 1; done && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill -SIGTERM ollama && \
    sleep 5

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/bin/ollama /usr/bin/ollama

COPY --from=builder /usr/lib/ollama/ /usr/lib/ollama/

COPY --from=builder /baked-models /baked-models

ENV OLLAMA_MODELS=/baked-models
ENV OLLAMA_HOST=0.0.0.0

EXPOSE 11434

ENTRYPOINT ["ollama", "serve"]
