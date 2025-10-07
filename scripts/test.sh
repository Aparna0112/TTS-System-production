#!/bin/bash
# Run tests

set -e

echo "Running TTS System tests..."

# Run pytest
pytest tests/test_integration.py -v

# Run integration tests if service is running
if curl -s http://localhost:8080/health &> /dev/null; then
    echo "Running integration tests..."
    python3 tests/test_client.py
else
    echo "⚠ Service not running, skipping integration tests"
fi

echo "✓ Tests complete!"
