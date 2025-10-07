"""
Test Client for TTS System
Example usage and integration tests
"""

import requests
import json
from pathlib import Path
import time

class TTSClient:
    """Client for interacting with TTS API"""
    
    def __init__(self, base_url: str, jwt_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.jwt_token = jwt_token
        self.session = requests.Session()
    
    def generate_token(self, user_id: str) -> str:
        """Generate JWT token"""
        response = self.session.post(
            f"{self.base_url}/auth/token",
            params={"user_id": user_id}
        )
        response.raise_for_status()
        data = response.json()
        self.jwt_token = data['token']
        return self.jwt_token
    
    def synthesize(
        self,
        text: str,
        model: str,
        voice: str,
        output_file: str = None
    ) -> bytes:
        """Synthesize speech"""
        if not self.jwt_token:
            raise ValueError("JWT token not set")
        
        payload = {
            "model": model,
            "voice": voice,
            "text": text,
            "jwt_token": self.jwt_token
        }
        
        print(f"Generating with {model}, voice: {voice}")
        start_time = time.time()
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=120
        )
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f}s")
        
        response.raise_for_status()
        audio_data = response.content
        
        if output_file:
            Path(output_file).write_bytes(audio_data)
            print(f"Saved to {output_file}")
        
        return audio_data
    
    def list_voices(self, model: str) -> list:
        """List voices"""
        response = self.session.get(f"{self.base_url}/models/{model}/voices")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        """Check health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def run_tests():
    """Run integration tests"""
    
    BASE_URL = "http://localhost:8080"
    USER_ID = "test_user_123"
    
    print("=" * 60)
    print("TTS System Integration Tests")
    print("=" * 60)
    
    client = TTSClient(BASE_URL)
    
    # Test 1: Health Check
    print("\n[TEST 1] Health Check")
    try:
        health = client.health_check()
        print(f"✓ Status: {health['status']}")
        for model, status in health['models'].items():
            print(f"  {model}: {status}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Test 2: Generate Token
    print("\n[TEST 2] Generate Token")
    try:
        token = client.generate_token(USER_ID)
        print(f"✓ Token: {token[:20]}...")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Test 3: List Voices
    print("\n[TEST 3] List Voices")
    models = ["chatterbox", "kokkoro", "coqui"]
    voice_map = {}
    
    for model in models:
        try:
            voices_data = client.list_voices(model)
            voices = voices_data.get('voices', [])
            voice_map[model] = voices[0] if voices else None
            print(f"✓ {model}: {len(voices)} voices")
        except Exception as e:
            print(f"✗ {model} failed: {e}")
    
    # Test 4: Generate Speech
    print("\n[TEST 4] Generate Speech")
    test_text = "Hello! This is a test."
    
    for model in models:
        if model not in voice_map or not voice_map[model]:
            continue
        
        try:
            audio = client.synthesize(
                text=test_text,
                model=model,
                voice=voice_map[model],
                output_file=f"test_{model}.mp3"
            )
            print(f"✓ {model}: {len(audio)} bytes")
        except Exception as e:
            print(f"✗ {model}: {e}")
    
    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
