#!/usr/bin/env python3
"""
Validation script for Task 3.5: Implement health check method

This script validates the check_health() method implementation by:
1. Testing successful health check (200 status)
2. Testing failed health check (non-200 status)
3. Testing connection errors
4. Testing timeout errors
5. Verifying 5-second timeout is used

Requirements: 7.1
"""

import sys
import os
from unittest.mock import Mock, patch
import requests

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import BharatVoiceAPIClient


def test_health_check_success():
    """Test successful health check returns True"""
    print("Test 1: Successful health check (200 status)...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000')
    
    # Mock successful response
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = client.check_health()
        
        if result is True:
            print("  ✅ PASS: Returns True for 200 status")
        else:
            print("  ❌ FAIL: Should return True for 200 status")
            return False
        
        # Verify correct URL and timeout
        call_args = mock_get.call_args
        if call_args[0][0] == 'http://localhost:8000/api/health' and call_args[1]['timeout'] == 5:
            print("  ✅ PASS: Correct URL and 5-second timeout")
        else:
            print(f"  ❌ FAIL: Expected URL 'http://localhost:8000/api/health' with timeout=5")
            print(f"           Got: {call_args}")
            return False
    
    return True


def test_health_check_failure_non_200():
    """Test health check with non-200 status returns False"""
    print("\nTest 2: Failed health check (500 status)...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000')
    
    # Mock failed response (500 status)
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = client.check_health()
        
        if result is False:
            print("  ✅ PASS: Returns False for 500 status")
        else:
            print("  ❌ FAIL: Should return False for 500 status")
            return False
    
    return True


def test_health_check_connection_error():
    """Test health check with connection error returns False"""
    print("\nTest 3: Connection error...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000')
    
    # Mock connection error
    with patch.object(client.session, 'get') as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("Cannot connect")
        
        result = client.check_health()
        
        if result is False:
            print("  ✅ PASS: Returns False for connection error")
        else:
            print("  ❌ FAIL: Should return False for connection error")
            return False
    
    return True


def test_health_check_timeout():
    """Test health check with timeout returns False"""
    print("\nTest 4: Timeout error...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000')
    
    # Mock timeout error
    with patch.object(client.session, 'get') as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = client.check_health()
        
        if result is False:
            print("  ✅ PASS: Returns False for timeout")
        else:
            print("  ❌ FAIL: Should return False for timeout")
            return False
    
    return True


def test_health_check_uses_5_second_timeout():
    """Test health check uses 5 second timeout"""
    print("\nTest 5: Verify 5-second timeout (not client's default)...")
    
    # Create client with 30-second default timeout
    client = BharatVoiceAPIClient(base_url='http://localhost:8000', timeout=30)
    
    # Mock successful response
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        client.check_health()
        
        # Verify timeout is 5 seconds, not 30
        call_args = mock_get.call_args
        if call_args[1]['timeout'] == 5:
            print("  ✅ PASS: Uses 5-second timeout (not client's 30-second default)")
        else:
            print(f"  ❌ FAIL: Should use 5-second timeout, got {call_args[1]['timeout']}")
            return False
    
    return True


def test_health_check_with_trailing_slash():
    """Test health check works with trailing slash in base URL"""
    print("\nTest 6: Trailing slash in base URL...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000/')
    
    # Mock successful response
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = client.check_health()
        
        if result is True:
            print("  ✅ PASS: Returns True")
        else:
            print("  ❌ FAIL: Should return True")
            return False
        
        # Verify URL doesn't have double slashes
        call_args = mock_get.call_args
        url = call_args[0][0]
        if '//api' not in url:
            print("  ✅ PASS: No double slashes in URL")
        else:
            print(f"  ❌ FAIL: URL has double slashes: {url}")
            return False
    
    return True


def test_health_check_404_status():
    """Test health check with 404 status returns False"""
    print("\nTest 7: 404 status...")
    
    client = BharatVoiceAPIClient(base_url='http://localhost:8000')
    
    # Mock 404 response
    with patch.object(client.session, 'get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = client.check_health()
        
        if result is False:
            print("  ✅ PASS: Returns False for 404 status")
        else:
            print("  ❌ FAIL: Should return False for 404 status")
            return False
    
    return True


def main():
    """Run all validation tests"""
    print("=" * 70)
    print("Task 3.5 Validation: check_health() Method")
    print("=" * 70)
    
    tests = [
        test_health_check_success,
        test_health_check_failure_non_200,
        test_health_check_connection_error,
        test_health_check_timeout,
        test_health_check_uses_5_second_timeout,
        test_health_check_with_trailing_slash,
        test_health_check_404_status
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"  ⚠️  Test failed")
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Task 3.5 implementation is correct.")
        print("\nThe check_health() method:")
        print("  ✅ Returns True for 200 status code")
        print("  ✅ Returns False for non-200 status codes")
        print("  ✅ Returns False for connection errors")
        print("  ✅ Returns False for timeout errors")
        print("  ✅ Uses 5-second timeout for quick health checks")
        print("  ✅ Handles trailing slashes in base URL correctly")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
