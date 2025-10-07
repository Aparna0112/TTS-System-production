

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatterbox TTS Endpoint")

# Global model variables
model = None
tokenizer = None
device = None

# Available voices
AVAILABLE_VOICES = [
    "default",
    "female_1",
    "male_1",
    "neutral"
]

class SynthesisRequest(BaseModel):
    text: str
    voice: str = "default"

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, device
    
    logger.info("Loading Chatterbox model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model_name = "ResembleAI/chatterbox"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            model = model.to(device)
        
        model.eval()
        logger.info("Chatterbox model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def text_to_audio(text: str, voice: str) -> bytes:
    """Convert text to audio"""
    global model, tokenizer, device
    
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1000,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
            
            audio_data = outputs.cpu().numpy()
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = BytesIO()
        sample_rate = 22050
        wavfile.write(wav_buffer, sample_rate, audio_data.flatten())
        wav_buffer.seek(0)
        
        audio = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="128k")
        mp3_buffer.seek(0)
        
        return mp3_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        raise

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """Synthesize speech from text"""
    
    if not model or not tokenizer:
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
        "model": "chatterbox",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": AVAILABLE_VOICES,
        "model": "chatterbox"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
