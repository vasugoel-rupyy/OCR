#!/bin/bash
# Development mode - rebuilds on code changes

echo "🔨 Starting OCR Pipeline (with rebuild)..."
docker compose up --build
