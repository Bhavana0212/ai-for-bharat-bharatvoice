"""
Script to run language selector property tests.

This script runs the property-based tests for the language selector component
with proper configuration and reporting.
"""

import subprocess
import sys
import os

def main():
    """Run language selector property tests."""
    
    print("=" * 80)
    print("Running Language Selector Property Tests")
    print("=" * 80)
    print()
    
    # Set environment variables for testing
    os.environ['PYTHONPATH'] = os.getcwd()
    
    # Run pytest with specific test file
    cmd = [
        sys.executable,
        '-m', 'pytest',
        'tests/test_language_selector_properties.py',
        '-v',
        '--tb=short',
        '--hypothesis-show-statistics',
        '-m', 'property'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    print()
    print("=" * 80)
    if result.returncode == 0:
        print("✅ All language selector property tests passed!")
    else:
        print("❌ Some language selector property tests failed.")
    print("=" * 80)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
