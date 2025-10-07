"""
RunPod Serverless Handler for Kokkoro
"""

import runpod
from kokkoro_server import text_to_audio, load_model
import base64
import asyncio

# Load model on cold start
asyncio.run(load_model())

def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event['input']
        text = input_data['text']
        voice = input_data.get('voice', 'af_bella')
        
        # Generate audio
        audio_bytes = text_to_audio(text, voice)
        
        # Encode as base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "audio": audio_base64,
            "format": "mp3"
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
