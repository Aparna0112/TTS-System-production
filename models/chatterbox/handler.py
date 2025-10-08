
import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
tokenizer = None
device = None

def initialize():
    """Load model on cold start"""
    global model, tokenizer, device
    
    logger.info("Initializing Chatterbox model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model_name = "ResembleAI/chatterbox"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Initialize on import
initialize()

def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event['input']
        text = input_data['text']
        voice = input_data.get('voice', 'default')
        
        logger.info(f"Processing: {text[:50]}...")
        
        # Generate audio
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1000)
            audio_data = outputs.cpu().numpy()
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to MP3
        wav_buffer = BytesIO()
        wavfile.write(wav_buffer, 22050, audio_data.flatten())
        wav_buffer.seek(0)
        
        audio = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = BytesIO()
        audio.export(mp3_buffer, format="mp3", bitrate="128k")
        
        audio_base64 = base64.b64encode(mp3_buffer.getvalue()).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "format": "mp3"
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
