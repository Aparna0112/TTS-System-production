"""
FastAPI Gateway for TTS System
Centralized entry point for routing TTS requests to model endpoints
"""

from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import httpx
import jwt
from datetime import datetime, timedelta
import os
from io import BytesIO
import logging
import asyncio
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model TTS Gateway",
    description="Production TTS API supporting multiple models",
    version="1.0.0"
)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# Model endpoint URLs
MODEL_ENDPOINTS = {
    "chatterbox": os.getenv("CHATTERBOX_ENDPOINT", "http://chatterbox:8000"),
    "kokkoro": os.getenv("KOKKORO_ENDPOINT", "http://kokkoro:8000"),
    "coqui": os.getenv("COQUI_ENDPOINT", "http://coqui:8000"),
}

# Check if using RunPod endpoints
IS_RUNPOD = any("runpod.ai" in url for url in MODEL_ENDPOINTS.values())

# Request/Response Models
class TTSRequest(BaseModel):
    model: Literal["chatterbox", "kokkoro", "coqui"] = Field(
        ..., description="TTS model to use"
    )
    voice: str = Field(..., description="Voice identifier for the model")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    jwt_token: str = Field(..., description="JWT authentication token")

class HealthResponse(BaseModel):
    status: str
    models: dict

# JWT Authentication
def verify_jwt(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def create_jwt(user_id: str, expires_in_hours: int = 24) -> str:
    """Helper function to create JWT tokens"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def call_runpod_endpoint(endpoint_url: str, payload: dict) -> bytes:
    """Call RunPod serverless endpoint and return audio bytes"""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Submit job
        response = await client.post(
            f"{endpoint_url}/run",
            json={"input": payload},
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"RunPod submission failed: {response.text}"
            )
        
        job_data = response.json()
        job_id = job_data['id']
        
        # Poll for results
        max_attempts = 60  # 2 minutes timeout
        for attempt in range(max_attempts):
            status_response = await client.get(
                f"{endpoint_url}/status/{job_id}",
                headers=headers
            )
            
            if status_response.status_code != 200:
                continue
                
            status_data = status_response.json()
            
            if status_data['status'] == 'COMPLETED':
                # Decode base64 audio
                audio_base64 = status_data['output']['audio']
                audio_bytes = base64.b64decode(audio_base64)
                return audio_bytes
            elif status_data['status'] == 'FAILED':
                error_msg = status_data.get('error', 'Unknown error')
                raise HTTPException(status_code=500, detail=f"Job failed: {error_msg}")
            
            await asyncio.sleep(2)
        
        raise HTTPException(status_code=504, detail="Job timeout")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of gateway and model endpoints"""
    model_status = {}
    
    if IS_RUNPOD:
        # For RunPod, just mark as available if configured
        for model_name in MODEL_ENDPOINTS:
            model_status[model_name] = "configured"
    else:
        async with httpx.AsyncClient(timeout=5.0) as client:
            for model_name, endpoint in MODEL_ENDPOINTS.items():
                try:
                    response = await client.get(f"{endpoint}/health")
                    model_status[model_name] = "healthy" if response.status_code == 200 else "unhealthy"
                except Exception as e:
                    model_status[model_name] = f"unreachable: {str(e)}"
    
    return {
        "status": "healthy",
        "models": model_status
    }

# Main TTS generation endpoint
@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text using specified model"""
    # Verify JWT token
    try:
        user_payload = verify_jwt(request.jwt_token)
        logger.info(f"Request from user: {user_payload.get('user_id')}")
    except HTTPException as e:
        raise e
    
    # Get model endpoint
    endpoint = MODEL_ENDPOINTS.get(request.model)
    if not endpoint:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {request.model}"
        )
    
    # Prepare request to model endpoint
    model_request = {
        "voice": request.voice,
        "text": request.text
    }
    
    try:
        logger.info(f"Forwarding request to {request.model} endpoint")
        
        if IS_RUNPOD and "runpod.ai" in endpoint:
            # Call RunPod serverless endpoint
            audio_data = await call_runpod_endpoint(endpoint, model_request)
        else:
            # Call direct HTTP endpoint
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{endpoint}/synthesize",
                    json=model_request,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Model endpoint error: {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Model endpoint error: {error_detail}"
                    )
                
                audio_data = response.content
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=speech_{request.model}.mp3",
                "X-Model-Used": request.model,
                "X-Voice-Used": request.voice
            }
        )
    
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Request to {request.model} endpoint timed out"
        )
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to {request.model} endpoint"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Token generation endpoint
@app.post("/auth/token")
async def generate_token(user_id: str):
    """Generate JWT token for testing purposes"""
    token = create_jwt(user_id)
    return {
        "token": token,
        "user_id": user_id,
        "expires_in_hours": 24
    }

# List available voices
@app.get("/models/{model_name}/voices")
async def list_voices(model_name: str):
    """Get available voices for a specific model"""
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    endpoint = MODEL_ENDPOINTS[model_name]
    
    # If RunPod, return cached voice list
    if IS_RUNPOD and "runpod.ai" in endpoint:
        # Return pre-defined voice lists
        voices = {
            "chatterbox": ["default", "female_1", "male_1", "neutral"],
            "kokkoro": ["af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael", 
                       "bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
            "coqui": ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Andrew Chipper",
                     "Badr Odhiambo", "Dionisio Schuyler", "Viktor Eka"]
        }
        return {"voices": voices.get(model_name, []), "model": model_name}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{endpoint}/voices")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Could not fetch voices from model endpoint"
                )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to model endpoint: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GATEWAY_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
