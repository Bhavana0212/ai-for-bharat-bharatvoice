#!/usr/bin/env python3
"""
Test runner for Deployment and Integration Tests.

This script runs comprehensive deployment and integration tests including:
- End-to-end voice interaction testing
- Multilingual conversation flow testing
- Indian service integration validation
- Offline/online mode transitions
- Performance under realistic Indian network conditions
"""

import sys
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List


def print_header(title: str):
    """Print formatted test section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted test subsection header."""
    print(f"\n--- {title} ---")


def run_test_suite(test_file: str, test_class: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run a specific test suite and return results.
    
    Args:
        test_file: Path to test file
        test_class: Specific test class to run (optional)
        verbose: Enable verbose output
        
    Returns:
        Dictionary with test results
    """
    cmd = ["python", "-m", "pytest"]
    
    if test_class:
        cmd.append(f"{test_file}::{test_class}")
    else:
        cmd.append(test_file)
    
    cmd.extend([
        "-v" if verbose else "-q",
        "--tb=short",
        "-m", "integration",
        "--durations=10",
        "--disable-warnings"
    ])
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    return {
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "duration": duration,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0
    }


def run_deployment_integration_tests():
    """Run all deployment and integration tests."""
    print_header("BHARATVOICE DEPLOYMENT & INTEGRATION TESTS")
    print("Testing comprehensive deployment scenarios and system integration")
    print(f"Test file: tests/test_deployment_integration.py")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if test file exists
    test_file = Path("tests/test_deployment_integration.py")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    # Test suites to run
    test_suites = [
        {
            "name": "End-to-End Voice Interaction Tests",
            "class": "TestEndToEndVoiceInteraction",
            "description": "Complete voice workflows from audio input to response synthesis"
        },
        {
            "name": "Indian Service Integration Tests", 
            "class": "TestIndianServiceIntegration",
            "description": "Integration with Indian Railways, weather, and government services"
        },
        {
            "name": "Offline/Online Transition Tests",
            "class": "TestOfflineOnlineTransitions", 
            "description": "Seamless switching between offline and online modes"
        },
        {
            "name": "Performance Under Indian Network Conditions",
            "class": "TestPerformanceUnderIndianNetworkConditions",
            "description": "Performance testing with realistic Indian network scenarios"
        },
        {
            "name": "Deployment Health Checks",
            "class": "TestDeploymentHealthChecks",
            "description": "System readiness and health validation"
        }
    ]
    
    results = []
    total_duration = 0
    
    for suite in test_suites:
        print_subheader(f"{suite['name']}")
        print(f"Description: {suite['description']}")
        
        result = run_test_suite(
            str(test_file),
            suite["class"],
            verbose=True
        )
        
        results.append({
            "suite": suite,
            "result": result
        })
        
        total_duration += result["duration"]
        
        if result["success"]:
            print(f"✅ {suite['name']} - PASSED ({result['duration']:.2f}s)")
        else:
            print(f"❌ {suite['name']} - FAILED ({result['duration']:.2f}s)")
            print("STDOUT:", result["stdout"][-500:] if result["stdout"] else "None")
            print("STDERR:", result["stderr"][-500:] if result["stderr"] else "None")
    
    # Summary
    print_header("TEST EXECUTION SUMMARY")
    
    passed_suites = [r for r in results if r["result"]["success"]]
    failed_suites = [r for r in results if not r["result"]["success"]]
    
    print(f"Total test suites: {len(results)}")
    print(f"Passed: {len(passed_suites)}")
    print(f"Failed: {len(failed_suites)}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Success rate: {len(passed_suites)/len(results)*100:.1f}%")
    
    if failed_suites:
        print("\n❌ FAILED TEST SUITES:")
        for failed in failed_suites:
            suite_name = failed["suite"]["name"]
            print(f"  - {suite_name}")
    
    if passed_suites:
        print("\n✅ PASSED TEST SUITES:")
        for passed in passed_suites:
            suite_name = passed["suite"]["name"]
            duration = passed["result"]["duration"]
            print(f"  - {suite_name} ({duration:.2f}s)")
    
    # Detailed results for failed tests
    if failed_suites:
        print_header("DETAILED FAILURE ANALYSIS")
        
        for failed in failed_suites:
            suite_name = failed["suite"]["name"]
            result = failed["result"]
            
            print(f"\n🔍 {suite_name}:")
            print(f"Command: {result['command']}")
            print(f"Return code: {result['return_code']}")
            print(f"Duration: {result['duration']:.2f}s")
            
            if result["stderr"]:
                print("Error output:")
                print(result["stderr"])
            
            if result["stdout"]:
                print("Standard output:")
                print(result["stdout"])
    
    # Performance metrics
    print_header("PERFORMANCE METRICS")
    
    avg_duration = total_duration / len(results) if results else 0
    print(f"Average test suite duration: {avg_duration:.2f}s")
    
    fastest_suite = min(results, key=lambda x: x["result"]["duration"])
    slowest_suite = max(results, key=lambda x: x["result"]["duration"])
    
    print(f"Fastest suite: {fastest_suite['suite']['name']} ({fastest_suite['result']['duration']:.2f}s)")
    print(f"Slowest suite: {slowest_suite['suite']['name']} ({slowest_suite['result']['duration']:.2f}s)")
    
    # Test coverage analysis
    print_header("TEST COVERAGE ANALYSIS")
    
    coverage_areas = {
        "Voice Processing": "TestEndToEndVoiceInteraction",
        "External Services": "TestIndianServiceIntegration", 
        "Offline Capabilities": "TestOfflineOnlineTransitions",
        "Network Performance": "TestPerformanceUnderIndianNetworkConditions",
        "System Health": "TestDeploymentHealthChecks"
    }
    
    print("Deployment test coverage:")
    for area, test_class in coverage_areas.items():
        suite_result = next((r for r in results if r["suite"]["class"] == test_class), None)
        if suite_result:
            status = "✅ COVERED" if suite_result["result"]["success"] else "❌ ISSUES"
            print(f"  {area}: {status}")
        else:
            print(f"  {area}: ❓ NOT FOUND")
    
    # Recommendations
    print_header("RECOMMENDATIONS")
    
    if len(failed_suites) == 0:
        print("🎉 All deployment integration tests passed!")
        print("✅ System is ready for deployment")
        print("✅ All integration points are functioning correctly")
        print("✅ Performance meets requirements under Indian network conditions")
    else:
        print("⚠️  Some deployment tests failed. Recommendations:")
        print("1. Review failed test output for specific issues")
        print("2. Check external service connectivity and configuration")
        print("3. Verify network simulation and offline capabilities")
        print("4. Ensure all dependencies are properly installed")
        print("5. Consider running tests individually for detailed debugging")
    
    print("\n📋 Next Steps:")
    print("1. If all tests pass: Proceed with deployment")
    print("2. If tests fail: Address issues and re-run tests")
    print("3. Monitor system performance in production")
    print("4. Set up continuous integration for these tests")
    
    return 0 if len(failed_suites) == 0 else 1


def run_specific_test_class(test_class: str):
    """Run a specific test class."""
    print_header(f"RUNNING SPECIFIC TEST CLASS: {test_class}")
    
    test_file = "tests/test_deployment_integration.py"
    result = run_test_suite(test_file, test_class, verbose=True)
    
    if result["success"]:
        print(f"✅ {test_class} - PASSED ({result['duration']:.2f}s)")
        return 0
    else:
        print(f"❌ {test_class} - FAILED ({result['duration']:.2f}s)")
        print("STDOUT:", result["stdout"])
        print("STDERR:", result["stderr"])
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test class
        test_class = sys.argv[1]
        return run_specific_test_class(test_class)
    else:
        # Run all deployment integration tests
        return run_deployment_integration_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)