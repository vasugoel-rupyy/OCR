FROM ollama/ollama:latest

ARG OLLAMA_MODEL=qwen2.5:1.5B

ENV OLLAMA_MODELS=/baked-models

RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama
