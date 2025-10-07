"""
Kokkoro TTS Model Endpoint
Serves hexgrad/Kokoro-82M model via FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Kokkoro TTS Endpoint")

# Global model variables
model = None
device = None

# Available voices
AVAILABLE_VOICES = [
    "af_bella",
    "af_nicole",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis"
]

class SynthesisRequest(BaseModel):
    text: str
    voice: str = "af_bella"

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device
    
    logger.info("Loading Kokkoro model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "hexgrad/Kokoro-82M",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = model.to(device)
        model.eval()
        
        logger.info("Kokkoro model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def text_to_audio(text: str, voice: str) -> bytes:
    """Convert text to audio"""
    global model, device
    
    try:
        with torch.no_grad():
            # Generate audio (adjust based on actual Kokkoro API)
            audio_array = model.generate(
                text=text,
                voice=voice,
                speed=1.0,
                device=device
            )
            
            if torch.is_tensor(audio_array):
                audio_array = audio_array.cpu().numpy()
            
            audio_array = audio_array / np.max(np.abs(audio_array))
            audio_array = (audio_array * 32767).astype(np.int16)
        
        wav_buffer = BytesIO()
        sample_rate = 24000
        wavfile.write(wav_buffer, sample_rate, audio_array)
        wav_buffer.seek(0)
        
        audio = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="192k")
        mp3_buffer.seek(0)
        
        return mp3_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        raise

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """Synthesize speech from text"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Available: {AVAILABLE_VOICES}"
        )
    
    try:
        logger.info(f"Synthesizing with voice: {request.voice}")
        audio_bytes = text_to_audio(request.text, request.voice)
        
        from fastapi.responses import Response
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg"
        )
        
    except Exception as e:
        logger.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": "kokkoro",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": AVAILABLE_VOICES,
        "model": "kokkoro",
        "voice_details": {
            "af_*": "American Female voices",
            "am_*": "American Male voices",
            "bf_*": "British Female voices",
            "bm_*": "British Male voices"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
