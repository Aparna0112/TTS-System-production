#!/bin/bash
# Setup script for TTS system

set -e

echo "Setting up TTS System..."

# Create directories
mkdir -p gateway models/chatterbox models/kokkoro models/coqui tests scripts docs

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file"
fi

# Install Python dependencies for testing
if command -v python3 &> /dev/null; then
    pip3 install pytest pytest-asyncio httpx requests
    echo "✓ Installed Python dependencies"
fi

echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Run: docker-compose up --build"
