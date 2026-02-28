"""
Validation script for Task 13.4: Integration tests for main workflow

This script validates:
- Property 34: API Data Round-Trip Consistency
- Property 32: Base64 Audio Decoding

Requirements validated: 12.3, 12.5
"""

import subprocess
import sys


def run_tests():
    """Run the integration tests"""
    print("=" * 80)
    print("Task 13.4: Integration Tests for Main Workflow")
    print("=" * 80)
    print()
    
    print("Running property-based tests...")
    print("-" * 80)
    
    # Run pytest with verbose output
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/test_main_workflow_integration.py",
            "-v",
            "--tb=short",
            "-s"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print()
    print("=" * 80)
    
    if result.returncode == 0:
        print("✅ All integration tests passed!")
        print()
        print("Validated Properties:")
        print("  - Property 34: API Data Round-Trip Consistency")
        print("  - Property 32: Base64 Audio Decoding")
        print()
        print("Requirements Validated:")
        print("  - Requirement 12.3: Audio response handling and base64 decoding")
        print("  - Requirement 12.5: Round-trip data consistency for all API requests")
        return True
    else:
        print("❌ Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
