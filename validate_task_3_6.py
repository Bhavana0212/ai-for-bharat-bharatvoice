#!/usr/bin/env python3
"""
Validation script for Task 3.6: Write property tests for API client

This script validates the property-based tests for the BharatVoiceAPIClient:
- Property 4: Language Propagation to API
- Property 6: Speech Recognition API Integration
- Property 31: JSON Response Validation

Requirements: 3.1, 2.3, 12.1, 12.2
"""

import sys
import os
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def run_pytest():
    """Run pytest on the property tests"""
    print("=" * 70)
    print("Task 3.6 Validation: API Client Property Tests")
    print("=" * 70)
    print()
    
    # Try to run pytest
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/test_api_client_properties.py', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("\n" + "=" * 70)
        if result.returncode == 0:
            print("🎉 All property tests passed!")
            print("\nValidated Properties:")
            print("  ✅ Property 4: Language Propagation to API")
            print("     - Language included in speech recognition requests")
            print("     - Language included in response generation requests")
            print("     - Language included in TTS requests")
            print()
            print("  ✅ Property 6: Speech Recognition API Integration")
            print("     - Audio data included in multipart/form-data")
            print("     - Language parameter included in form data")
            print("     - enable_code_switching parameter included")
            print("     - Correct endpoint /api/voice/recognize used")
            print()
            print("  ✅ Property 31: JSON Response Validation")
            print("     - Transcription responses validated for structure")
            print("     - Response generation responses validated")
            print("     - TTS responses validated for audio_url")
            print("     - Malformed JSON raises appropriate errors")
            print("=" * 70)
            return True
        else:
            print("❌ Some property tests failed")
            print("=" * 70)
            return False
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install: pip install pytest hypothesis")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_test_file_exists():
    """Check if the test file exists"""
    test_file = 'tests/test_api_client_properties.py'
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"✅ Test file exists: {test_file}")
    
    # Check file size
    file_size = os.path.getsize(test_file)
    print(f"   File size: {file_size} bytes")
    
    if file_size < 1000:
        print("   ⚠️  Warning: File seems small, may be incomplete")
    
    return True


def check_test_structure():
    """Check if the test file has the expected structure"""
    print("\nChecking test structure...")
    
    test_file = 'tests/test_api_client_properties.py'
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required test classes
        required_classes = [
            'TestProperty4_LanguagePropagationToAPI',
            'TestProperty6_SpeechRecognitionAPIIntegration',
            'TestProperty31_JSONResponseValidation'
        ]
        
        for class_name in required_classes:
            if class_name in content:
                print(f"  ✅ Found test class: {class_name}")
            else:
                print(f"  ❌ Missing test class: {class_name}")
                return False
        
        # Check for hypothesis decorators
        if '@given' in content:
            print("  ✅ Uses @given decorator for property-based testing")
        else:
            print("  ❌ Missing @given decorator")
            return False
        
        # Check for language strategy
        if 'language_strategy' in content:
            print("  ✅ Defines language_strategy")
        else:
            print("  ❌ Missing language_strategy")
            return False
        
        # Check for audio data strategy
        if 'audio_data_strategy' in content:
            print("  ✅ Defines audio_data_strategy")
        else:
            print("  ❌ Missing audio_data_strategy")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading test file: {e}")
        return False


def main():
    """Run all validation checks"""
    print("Validating Task 3.6: API Client Property Tests\n")
    
    # Check test file exists
    if not check_test_file_exists():
        return False
    
    # Check test structure
    if not check_test_structure():
        return False
    
    print("\n" + "=" * 70)
    print("Running property-based tests...")
    print("=" * 70)
    print()
    
    # Run pytest
    success = run_pytest()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
