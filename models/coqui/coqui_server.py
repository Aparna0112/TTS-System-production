```python
"""
Coqui XTTS Model Endpoint
Serves coqui/XTTS-v1 model via FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from TTS.api import TTS
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coqui XTTS Endpoint")

# Global model variables
tts = None
device = None

# Available voices
AVAILABLE_VOICES = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios"
]

class SynthesisRequest(BaseModel):
    text: str
    voice: str = "Claribel Dervla"

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global tts, device
    
    logger.info("Loading Coqui XTTS model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v1",
            progress_bar=False,
            gpu=(device == "cuda")
        )
        
        logger.info("Coqui XTTS model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def text_to_audio(text: str, voice: str) -> bytes:
    """Convert text to audio"""
    global tts, device
    
    try:
        wav_buffer = BytesIO()
        
        # Generate speech
        wav_data = tts.tts(
            text=text,
            speaker=voice,
            language="en"
        )
        
        if isinstance(wav_data, list):
            wav_data = np.array(wav_data)
        
        wav_data = wav_data / np.max(np.abs(wav_data))
        wav_data = (wav_data * 32767).astype(np.int16)
        
        sample_rate = 22050
        wavfile.write(wav_buffer, sample_rate, wav_data)
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
    
    if not tts:
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
        "model": "coqui-xtts",
        "device": device,
        "model_loaded": tts is not None
    }

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": AVAILABLE_VOICES,
        "model": "coqui-xtts",
        "total_voices": len(AVAILABLE_VOICES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
