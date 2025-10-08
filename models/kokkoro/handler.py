"""
RunPod Serverless Handler for Kokkoro
"""
import runpod
import torch
from transformers import AutoModel
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
device = None

def initialize():
    """Load model on cold start"""
    global model, device
    
    logger.info("Initializing Kokkoro model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model = AutoModel.from_pretrained(
            "hexgrad/Kokoro-82M",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = model.to(device)
        model.eval()
        logger.info("Kokkoro model loaded!")
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        raise

# Initialize on import
initialize()

def handler(event):
    """RunPod handler"""
    try:
        input_data = event['input']
        text = input_data['text']
        voice = input_data.get('voice', 'af_bella')
        
        logger.info(f"Processing: {text[:50]}...")
        
        with torch.no_grad():
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
        wavfile.write(wav_buffer, 24000, audio_array)
        wav_buffer.seek(0)
        
        audio = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="192k")
        
        audio_base64 = base64.b64encode(mp3_buffer.getvalue()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "format": "mp3"
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
