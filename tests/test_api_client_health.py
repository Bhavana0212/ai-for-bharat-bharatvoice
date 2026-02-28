"""
Unit tests for BharatVoiceAPIClient.check_health() method

Tests the health check functionality including:
- Successful health check (200 status)
- Failed health check (non-200 status)
- Connection errors
- Timeout errors

Requirements: 7.1
"""

import pytest
from unittest.mock import Mock, patch
import requests
import sys
import os

# Add parent directory to path to import app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import BharatVoiceAPIClient


class TestCheckHealth:
    """Test suite for check_health() method"""
    
    def test_health_check_success(self):
        """Test successful health check returns True"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock successful response
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is True
            mock_get.assert_called_once_with('http://localhost:8000/api/health', timeout=5)
    
    def test_health_check_failure_non_200(self):
        """Test health check with non-200 status returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock failed response (500 status)
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False
    
    def test_health_check_connection_error(self):
        """Test health check with connection error returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock connection error
        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False
    
    def test_health_check_timeout(self):
        """Test health check with timeout returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock timeout error
        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False
    
    def test_health_check_http_error(self):
        """Test health check with HTTP error returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock HTTP error
        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = requests.exceptions.HTTPError("HTTP error")
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False
    
    def test_health_check_uses_5_second_timeout(self):
        """Test health check uses 5 second timeout"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000', timeout=30)
        
        # Mock successful response
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Act
            client.check_health()
            
            # Assert - verify timeout is 5 seconds, not the client's default timeout
            mock_get.assert_called_once_with('http://localhost:8000/api/health', timeout=5)
    
    def test_health_check_with_trailing_slash_in_base_url(self):
        """Test health check works with trailing slash in base URL"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000/')
        
        # Mock successful response
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is True
            # Verify URL doesn't have double slashes
            mock_get.assert_called_once_with('http://localhost:8000/api/health', timeout=5)
    
    def test_health_check_with_404_status(self):
        """Test health check with 404 status returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock 404 response
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False
    
    def test_health_check_with_503_status(self):
        """Test health check with 503 (service unavailable) returns False"""
        # Arrange
        client = BharatVoiceAPIClient(base_url='http://localhost:8000')
        
        # Mock 503 response
        with patch.object(client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response
            
            # Act
            result = client.check_health()
            
            # Assert
            assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
