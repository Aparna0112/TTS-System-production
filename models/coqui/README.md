# Coqui XTTS Model

Coqui XTTS-v1 multilingual TTS model endpoint.

## Voices

33 high-quality voices including:
- Female: Claribel Dervla, Daisy Studious, Gracie Wise, etc.
- Male: Andrew Chipper, Badr Odhiambo, Viktor Eka, etc.

## Running
```bash
docker build -t tts-coqui .
docker run --gpus all -p 8000:8000 tts-coqui
