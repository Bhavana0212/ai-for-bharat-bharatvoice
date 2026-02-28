#!/usr/bin/env python3
"""
Test runner for Streamlit session state property tests.

This script runs property-based tests for session state management
in the Streamlit Web Interface.
"""

import sys
import os
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the Streamlit session state property tests."""
    print("🚀 Starting Streamlit Session State Property Tests...")
    print("=" * 70)
    
    # Run pytest with the specific test file
    test_file = "tests/test_streamlit_session_state_properties.py"
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "-m", "property",
        "--hypothesis-show-statistics"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        print("=" * 70)
        if result.returncode == 0:
            print("✅ All property tests passed!")
            return True
        else:
            print("❌ Some tests failed. Check output above for details.")
            return False
            
    except FileNotFoundError:
        print("❌ Error: pytest not found. Please install pytest:")
        print("   pip install pytest hypothesis")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
