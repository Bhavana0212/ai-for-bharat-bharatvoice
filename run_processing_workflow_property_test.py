#!/usr/bin/env python3
"""
Test runner for processing workflow property tests.
Validates the main processing workflow including automatic response generation,
automatic TTS request, graceful TTS degradation, and action logging completeness.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Run processing workflow property tests."""
    print("🚀 Running Processing Workflow Property Tests...")
    print("=" * 70)
    
    try:
        # Import the test module
        from tests import test_processing_workflow_properties
        
        # Run each test function
        tests = [
            ("Property 11: Automatic Response Generation", 
             test_processing_workflow_properties.test_automatic_response_generation),
            ("Property 14: Automatic TTS Request",
             test_processing_workflow_properties.test_automatic_tts_request),
            ("Property 16: Graceful TTS Degradation",
             test_processing_workflow_properties.test_graceful_tts_degradation),
            ("Property 10: Action Logging Completeness",
             test_processing_workflow_properties.test_action_logging_completeness),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n📝 Testing: {test_name}")
            print("-" * 70)
            
            try:
                # Run the test with hypothesis
                test_func()
                print(f"✅ PASSED: {test_name}")
                passed += 1
            except Exception as e:
                print(f"❌ FAILED: {test_name}")
                print(f"   Error: {str(e)}")
                failed += 1
        
        # Print summary
        print("\n" + "=" * 70)
        print(f"📊 Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
        print("=" * 70)
        
        if failed == 0:
            print("🎉 All processing workflow property tests passed!")
            return True
        else:
            print(f"⚠️  {failed} test(s) failed. Please review the errors above.")
            return False
    
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install pytest hypothesis streamlit requests")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
