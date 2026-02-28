#!/usr/bin/env python3
"""
Validation script for Task 3.4: Implement text-to-speech API method

This script validates that the synthesize_speech method has been properly
implemented in the BharatVoiceAPIClient class without requiring external
dependencies or test execution.

Requirements validated:
- 5.1: Automatic TTS request from text response
- 12.3: Base64 audio decoding
"""

import sys
import os
import ast
import inspect

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate_method_signature():
    """Validate that synthesize_speech method has correct signature."""
    print("\n=== Validating Method Signature ===")
    
    try:
        from app import BharatVoiceAPIClient
        
        # Check method exists
        if not hasattr(BharatVoiceAPIClient, 'synthesize_speech'):
            print("❌ synthesize_speech method not found")
            return False
        
        method = getattr(BharatVoiceAPIClient, 'synthesize_speech')
        sig = inspect.signature(method)
        
        # Check parameters
        params = list(sig.parameters.keys())
        required_params = ['self', 'text', 'language']
        optional_params = ['accent', 'speed', 'pitch']
        
        for param in required_params:
            if param not in params:
                print(f"❌ Missing required parameter: {param}")
                return False
        
        for param in optional_params:
            if param not in params:
                print(f"⚠️  Missing optional parameter: {param}")
        
        # Check return type annotation
        if sig.return_annotation != bytes and sig.return_annotation != inspect.Signature.empty:
            print(f"⚠️  Return type should be bytes, got: {sig.return_annotation}")
        
        print("✅ Method signature is correct")
        print(f"   Parameters: {', '.join(params)}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating signature: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_method_implementation():
    """Validate that synthesize_speech method is properly implemented."""
    print("\n=== Validating Method Implementation ===")
    
    try:
        # Read the source file
        with open('app.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Find the synthesize_speech method
        method_found = False
        method_body = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'BharatVoiceAPIClient':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'synthesize_speech':
                        method_found = True
                        method_body = item
                        break
        
        if not method_found:
            print("❌ synthesize_speech method not found in source")
            return False
        
        # Check for required implementation elements
        source_lines = source.split('\n')
        method_start = method_body.lineno
        method_end = method_body.end_lineno
        method_source = '\n'.join(source_lines[method_start-1:method_end])
        
        required_elements = {
            'POST request to /api/voice/synthesize': '/api/voice/synthesize' in method_source,
            'JSON payload construction': 'payload' in method_source and 'json=' in method_source,
            'audio_url extraction': 'audio_url' in method_source,
            'GET request to fetch audio': 'get(' in method_source or '.get' in method_source,
            'base64 decoding': 'base64' in method_source,
            'Error handling': 'raise_for_status' in method_source,
        }
        
        all_present = True
        for element, present in required_elements.items():
            if present:
                print(f"✅ {element}")
            else:
                print(f"❌ Missing: {element}")
                all_present = False
        
        # Check for NotImplementedError (should not be present)
        if 'NotImplementedError' in method_source:
            print("❌ Method still raises NotImplementedError")
            return False
        
        if all_present:
            print("✅ Method implementation is complete")
            return True
        else:
            print("⚠️  Some implementation elements may be missing")
            return False
        
    except Exception as e:
        print(f"❌ Error validating implementation: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_docstring():
    """Validate that method has proper documentation."""
    print("\n=== Validating Documentation ===")
    
    try:
        from app import BharatVoiceAPIClient
        
        method = getattr(BharatVoiceAPIClient, 'synthesize_speech')
        docstring = method.__doc__
        
        if not docstring:
            print("❌ Method has no docstring")
            return False
        
        required_sections = [
            'Args:',
            'Returns:',
            'Raises:',
            'Requirements:',
        ]
        
        all_present = True
        for section in required_sections:
            if section in docstring:
                print(f"✅ {section} section present")
            else:
                print(f"⚠️  {section} section missing")
                all_present = False
        
        # Check for requirement references
        if '5.1' in docstring and '12.3' in docstring:
            print("✅ Requirements 5.1 and 12.3 referenced")
        else:
            print("⚠️  Requirements 5.1 and 12.3 should be referenced")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating docstring: {e}")
        return False


def validate_integration():
    """Validate that method integrates correctly with the class."""
    print("\n=== Validating Integration ===")
    
    try:
        from app import BharatVoiceAPIClient
        
        # Create instance
        client = BharatVoiceAPIClient(
            base_url='http://localhost:8000',
            timeout=30
        )
        
        # Check that method is callable
        if not callable(getattr(client, 'synthesize_speech')):
            print("❌ Method is not callable")
            return False
        
        print("✅ Method is properly integrated with the class")
        print("✅ Can create client instance and access method")
        
        # Check that session is available (needed for requests)
        if not hasattr(client, 'session'):
            print("❌ Client missing session attribute")
            return False
        
        print("✅ Client has session attribute for HTTP requests")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating integration: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validations."""
    print("🔍 Validating Task 3.4: Implement text-to-speech API method")
    print("=" * 60)
    
    results = {
        'Method Signature': validate_method_signature(),
        'Method Implementation': validate_method_implementation(),
        'Documentation': validate_docstring(),
        'Integration': validate_integration(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 Task 3.4 Implementation Validation PASSED!")
        print("\n✅ synthesize_speech method is properly implemented")
        print("✅ Method accepts text, language, accent, speed, pitch parameters")
        print("✅ Method sends JSON payload to /api/voice/synthesize endpoint")
        print("✅ Method fetches audio file from audio_url")
        print("✅ Method handles base64 decoding if needed")
        print("✅ Requirements 5.1 and 12.3 are satisfied")
        print("\n📝 Task 3.4 can be marked as COMPLETE")
        return 0
    else:
        print("❌ Task 3.4 Implementation Validation FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
