# Chatterbox TTS Model

ResembleAI Chatterbox model endpoint.

## Voices

- default
- female_1
- male_1
- neutral

## Running
```bash
docker build -t tts-chatterbox .
docker run --gpus all -p 8000:8000 tts-chatterbox
