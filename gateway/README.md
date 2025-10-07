# TTS Gateway Service

FastAPI-based gateway that routes TTS requests to appropriate model endpoints.

## Features

- JWT authentication
- Request routing
- Model health monitoring
- Error handling

## Environment Variables

- `JWT_SECRET`: Secret key for JWT tokens
- `CHATTERBOX_ENDPOINT`: Chatterbox model endpoint
- `KOKKORO_ENDPOINT`: Kokkoro model endpoint
- `COQUI_ENDPOINT`: Coqui model endpoint

## Running
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
