#!/usr/bin/env python3
"""
Test runner for audio input property tests.
"""

import sys
import subprocess

def main():
    """Run audio input property tests."""
    print("🚀 Running Audio Input Property Tests...")
    
    try:
        # Run pytest with property marker
        result = subprocess.run(
            [sys.executable, "-m", "pytest", 
             "tests/test_audio_input_properties.py", 
             "-v", "-m", "property",
             "--tb=short"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ All audio input property tests passed!")
            return True
        else:
            print("\n❌ Some tests failed.")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
