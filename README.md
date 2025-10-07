# 🎙️ Multi-Model TTS System

Production-ready Text-to-Speech API system supporting multiple models with centralized authentication and routing.

## Features

- 🤖 Multiple TTS Models (Chatterbox, Kokkoro, Coqui XTTS)
- 🔐 JWT Authentication
- ⚡ GPU Acceleration
- 🚀 Serverless Ready (RunPod)
- 📦 In-Memory Processing
- 🎵 MP3 Output

## Quick Start
```bash
# 1. Clone repository
git clone https://github.com/yourusername/tts-system.git
cd tts-system

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start services
docker-compose up --build

# 4. Test API
curl -X POST "http://localhost:8080/auth/token?user_id=test" | jq -r '.token'
