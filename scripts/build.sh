#!/bin/bash
# Build all Docker images

set -e

echo "Building TTS Docker images..."

docker build -t tts-gateway:latest ./gateway
echo "✓ Gateway built"

docker build -t tts-chatterbox:latest ./models/chatterbox
echo "✓ Chatterbox built"

docker build -t tts-kokkoro:latest ./models/kokkoro
echo "✓ Kokkoro built"

docker build -t tts-coqui:latest ./models/coqui
echo "✓ Coqui built"

echo "✓ All images built successfully!"
