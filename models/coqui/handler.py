"""
RunPod Serverless Handler for Coqui XTTS
"""
import runpod
import torch
from TTS.api import TTS
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
tts = None
device = None

def initialize():
    """Load model on cold start"""
    global tts, device
    
    logger.info("Initializing Coqui XTTS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v1",
            progress_bar=False,
            gpu=(device == "cuda")
        )
        logger.info("Coqui XTTS loaded!")
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
        voice = input_data.get('voice', 'Claribel Dervla')
        
        logger.info(f"Processing: {text[:50]}...")
        
        wav_data = tts.tts(
            text=text,
            speaker=voice,
            language="en"
        )
        
        if isinstance(wav_data, list):
            wav_data = np.array(wav_data)
        
        wav_data = wav_data / np.max(np.abs(wav_data))
        wav_data = (wav_data * 32767).astype(np.int16)
        
        wav_buffer = BytesIO()
        wavfile.write(wav_buffer, 22050, wav_data)
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
