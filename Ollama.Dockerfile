FROM ollama/ollama:latest

# Define the model to bake in (defaults to qwen, overrideable at build time)
ARG OLLAMA_MODEL=qwen3.5:0.8B

# CRITICAL FIX: The base ollama image declares /root/.ollama as a VOLUME, 
# which means anything downloaded to it during Docker build gets instantly discarded.
# We bypass this by downloading the models to a custom directory!
ENV OLLAMA_MODELS=/baked-models

# Start the Ollama background daemon, wait for it to boot, pull the model, and gracefully exit.
# This permanently bakes the multi-GB model weights into the container image layer for production without internet requirements.
RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull ${OLLAMA_MODEL} && \
    pkill ollama
