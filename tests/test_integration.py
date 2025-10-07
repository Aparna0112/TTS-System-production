"""
Integration and Unit Tests
Run with: pytest test_integration.py -v
"""

import pytest
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gateway'))

from main import app, verify_jwt, create_jwt, JWT_SECRET, JWT_ALGORITHM

client = TestClient(app)

TEST_USER_ID = "test_user_123"
SAMPLE_TEXT = "This is a test."


class TestAuthentication:
    """Test JWT authentication"""
    
    def test_create_jwt_token(self):
        """Test token creation"""
        token = create_jwt(TEST_USER_ID, expires_in_hours=1)
        assert token is not None
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        assert payload["user_id"] == TEST_USER_ID
    
    def test_verify_valid_jwt(self):
        """Test valid JWT"""
        token = create_jwt(TEST_USER_ID)
        payload = verify_jwt(token)
        assert payload["user_id"] == TEST_USER_ID
    
    def test_verify_expired_jwt(self):
        """Test expired JWT"""
        expired_payload = {
            "user_id": TEST_USER_ID,
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            verify_jwt(expired_token)
        
        assert exc.value.status_code == 401
    
    def test_token_endpoint(self):
        """Test token generation endpoint"""
        response = client.post(f"/auth/token?user_id={TEST_USER_ID}")
        assert response.status_code == 200
        data = response.json()
        assert "token" in data


class TestHealthCheck:
    """Test health endpoints"""
    
    def test_health_endpoint(self):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models" in data


class TestSynthesis:
    """Test synthesis requests"""
    
    def setup_method(self):
        self.token = create_jwt(TEST_USER_ID)
    
    def test_synthesis_structure(self):
        """Test request structure"""
        payload = {
            "model": "coqui",
            "voice": "Claribel Dervla",
            "text": SAMPLE_TEXT,
            "jwt_token": self.token
        }
        
        response = client.post("/generate", json=payload)
        assert response.status_code in [200, 503, 504]
    
    def test_missing_token(self):
        """Test missing token"""
        payload = {
            "model": "coqui",
            "voice": "Claribel Dervla",
            "text": SAMPLE_TEXT
        }
        
        response = client.post("/generate", json=payload)
        assert response.status_code == 422
    
    def test_invalid_model(self):
        """Test invalid model"""
        payload = {
            "model": "invalid",
            "voice": "test",
            "text": SAMPLE_TEXT,
            "jwt_token": self.token
        }
        
        response = client.post("/generate", json=payload)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
