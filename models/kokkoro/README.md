# Kokkoro TTS Model

Hexgrad Kokoro-82M lightweight TTS model endpoint.

## Voices

American Female: af_bella, af_nicole, af_sarah
American Male: am_adam, am_michael
British Female: bf_emma, bf_isabella
British Male: bm_george, bm_lewis

## Running
```bash
docker build -t tts-kokkoro .
docker run --gpus all -p 8000:8000 tts-kokkoro
