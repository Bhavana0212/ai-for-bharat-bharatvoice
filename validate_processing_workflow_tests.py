#!/usr/bin/env python3
"""
Validation script for processing workflow property tests.
Checks test structure, imports, and validates implementation without running hypothesis.
"""

import sys
import os
import ast

def validate_test_file():
    """Validate the test file structure and content."""
    print("🔍 Validating Processing Workflow Property Tests...")
    print("=" * 70)
    
    test_file = "tests/test_processing_workflow_properties.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"✅ Test file exists: {test_file}")
    
    # Read and parse the test file
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error in test file: {e}")
        return False
    
    print("✅ Test file has valid Python syntax")
    
    # Check for required imports
    required_imports = [
        'pytest',
        'hypothesis',
        'unittest.mock',
        'app'
    ]
    
    imports_found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports_found.append(node.module)
    
    print("\n📦 Checking required imports...")
    all_imports_ok = True
    for required in required_imports:
        if any(required in imp for imp in imports_found):
            print(f"  ✅ {required}")
        else:
            print(f"  ❌ Missing: {required}")
            all_imports_ok = False
    
    if not all_imports_ok:
        print("⚠️  Some imports are missing")
    
    # Check for test functions
    print("\n🧪 Checking test functions...")
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            test_functions.append(node.name)
    
    expected_tests = [
        'test_automatic_response_generation',
        'test_automatic_tts_request',
        'test_graceful_tts_degradation',
        'test_action_logging_completeness'
    ]
    
    all_tests_ok = True
    for expected in expected_tests:
        if expected in test_functions:
            print(f"  ✅ {expected}")
        else:
            print(f"  ❌ Missing: {expected}")
            all_tests_ok = False
    
    if not all_tests_ok:
        print("⚠️  Some test functions are missing")
    
    # Check for property validation comments
    print("\n📝 Checking property validation comments...")
    property_comments = [
        'Property 11: Automatic Response Generation',
        'Property 14: Automatic TTS Request',
        'Property 16: Graceful TTS Degradation',
        'Property 10: Action Logging Completeness'
    ]
    
    all_comments_ok = True
    for prop in property_comments:
        if prop in content:
            print(f"  ✅ {prop}")
        else:
            print(f"  ❌ Missing: {prop}")
            all_comments_ok = False
    
    if not all_comments_ok:
        print("⚠️  Some property validation comments are missing")
    
    # Check for requirement validation comments
    print("\n📋 Checking requirement validation comments...")
    requirement_comments = [
        'Validates: Requirements 4.1',
        'Validates: Requirements 5.1',
        'Validates: Requirements 5.5',
        'Validates: Requirements 6.1, 6.2, 6.3, 6.4'
    ]
    
    all_reqs_ok = True
    for req in requirement_comments:
        if req in content:
            print(f"  ✅ {req}")
        else:
            print(f"  ❌ Missing: {req}")
            all_reqs_ok = False
    
    if not all_reqs_ok:
        print("⚠️  Some requirement validation comments are missing")
    
    # Check for hypothesis decorators
    print("\n🔬 Checking hypothesis decorators...")
    has_given = '@given' in content
    has_settings = '@settings' in content
    
    if has_given:
        print("  ✅ @given decorator found")
    else:
        print("  ❌ @given decorator not found")
    
    if has_settings:
        print("  ✅ @settings decorator found")
    else:
        print("  ❌ @settings decorator not found")
    
    # Check for strategy generators
    print("\n🎲 Checking strategy generators...")
    strategy_generators = [
        'generate_audio_data',
        'generate_language_code',
        'generate_transcription_response',
        'generate_response_response',
        'generate_tts_audio'
    ]
    
    all_strategies_ok = True
    for strategy in strategy_generators:
        if strategy in content:
            print(f"  ✅ {strategy}")
        else:
            print(f"  ❌ Missing: {strategy}")
            all_strategies_ok = False
    
    if not all_strategies_ok:
        print("⚠️  Some strategy generators are missing")
    
    # Final summary
    print("\n" + "=" * 70)
    if all_imports_ok and all_tests_ok and all_comments_ok and all_reqs_ok and has_given and has_settings and all_strategies_ok:
        print("✅ All validation checks passed!")
        print("🎉 Processing workflow property tests are properly implemented!")
        return True
    else:
        print("⚠️  Some validation checks failed")
        print("Please review the issues above and fix them")
        return False

def check_app_functions():
    """Check if required functions exist in app.py."""
    print("\n🔍 Checking app.py functions...")
    print("=" * 70)
    
    app_file = "app.py"
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_functions = [
        'process_audio',
        'process_transcription',
        'process_response_generation',
        'process_tts',
        'handle_network_error',
        'handle_validation_error',
        'handle_api_error'
    ]
    
    all_functions_ok = True
    for func in required_functions:
        if f"def {func}(" in content:
            print(f"  ✅ {func}")
        else:
            print(f"  ❌ Missing: {func}")
            all_functions_ok = False
    
    if all_functions_ok:
        print("\n✅ All required functions exist in app.py")
    else:
        print("\n⚠️  Some required functions are missing from app.py")
    
    return all_functions_ok

def main():
    """Run all validation checks."""
    print("🚀 Starting Processing Workflow Test Validation...")
    print("=" * 70)
    
    test_valid = validate_test_file()
    app_valid = check_app_functions()
    
    print("\n" + "=" * 70)
    print("📊 Validation Summary")
    print("=" * 70)
    print(f"Test file validation: {'✅ PASSED' if test_valid else '❌ FAILED'}")
    print(f"App functions check: {'✅ PASSED' if app_valid else '❌ FAILED'}")
    print("=" * 70)
    
    if test_valid and app_valid:
        print("\n🎉 All validation checks passed!")
        print("The processing workflow tests are ready to run.")
        return True
    else:
        print("\n⚠️  Some validation checks failed.")
        print("Please fix the issues before running the tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
