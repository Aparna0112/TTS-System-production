import runpod
from kokoro import KPipeline
from io import BytesIO
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None

def initialize():
    global pipeline
    logger.info("Initializing Kokoro TTS...")
    
    try:
        pipeline = KPipeline(lang_code='a')
        logger.info("Kokoro loaded!")
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        raise

initialize()

def handler(event):
    try:
        input_data = event['input']
        text = input_data['text']
        voice = input_data.get('voice', 'af_bella')
        
        logger.info(f"Processing: {text[:50]}... with voice: {voice}")
        
        # Generate audio
        generator = pipeline(text, voice=voice)
        
        # Collect audio chunks
        audio_chunks = []
        for _, _, audio_chunk in generator:
            audio_chunks.append(audio_chunk)
        
        if not audio_chunks:
            return {"error": "No audio generated"}
        
        # Concatenate chunks
        audio_array = np.concatenate(audio_chunks)
        
        # Log original audio stats
        logger.info(f"Original audio - dtype: {audio_array.dtype}, min: {audio_array.min():.4f}, max: {audio_array.max():.4f}")
        
        # Normalize to full int16 range for maximum volume
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            max_val = np.max(np.abs(audio_array))
            
            if max_val > 0:
                # Normalize to -1.0 to 1.0 range
                audio_array = audio_array / max_val
                # Convert to int16 (full volume)
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                logger.error("Audio is silent!")
                return {"error": "Generated audio is silent"}
        
        logger.info(f"Normalized audio - dtype: {audio_array.dtype}, min: {audio_array.min()}, max: {audio_array.max()}")
        
        # Save as WAV
        sample_rate = 24000
        wav_buffer = BytesIO()
        wavfile.write(wav_buffer, sample_rate, audio_array)
        wav_buffer.seek(0)
        
        # Convert to MP3 with volume boost
        audio_segment = AudioSegment.from_wav(wav_buffer)
        
        # Boost volume by 6dB
        audio_segment = audio_segment + 6
        
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate="192k")
        mp3_buffer.seek(0)
        
        mp3_bytes = mp3_buffer.getvalue()
        logger.info(f"MP3 size: {len(mp3_bytes)} bytes")
        
        audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "format": "mp3",
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})