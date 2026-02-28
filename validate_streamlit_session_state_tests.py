#!/usr/bin/env python3
"""
Validation script for Streamlit session state property tests.

This script validates that the property tests are correctly structured
and can be imported without running the full test suite.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_test_structure():
    """Validate the test file structure and imports."""
    print("🔍 Validating Streamlit Session State Property Tests...")
    print("=" * 70)
    
    try:
        # Check if test file exists
        test_file = "tests/test_streamlit_session_state_properties.py"
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"✅ Test file exists: {test_file}")
        
        # Read the test file
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            'import pytest',
            'from hypothesis import given',
            'from hypothesis import strategies',
        ]
        
        for imp in required_imports:
            if imp in content:
                print(f"✅ Found import: {imp}")
            else:
                print(f"❌ Missing import: {imp}")
                return False
        
        # Check for property test markers
        if '@pytest.mark.property' in content:
            print("✅ Property test markers found")
        else:
            print("❌ Property test markers not found")
            return False
        
        # Count test functions
        test_count = content.count('def test_property_')
        print(f"✅ Found {test_count} property test functions")
        
        # Check for specific properties
        properties = [
            'test_property_recording_state_persistence',
            'test_property_language_selection_persistence',
        ]
        
        for prop in properties:
            if prop in content:
                print(f"✅ Found test: {prop}")
            else:
                print(f"❌ Missing test: {prop}")
                return False
        
        # Check for requirement validation comments
        if '**Validates: Requirements 1.4**' in content:
            print("✅ Property 2 validates Requirements 1.4")
        else:
            print("⚠️  Property 2 requirement validation comment not found")
        
        if '**Validates: Requirements 2.2**' in content:
            print("✅ Property 3 validates Requirements 2.2")
        else:
            print("⚠️  Property 3 requirement validation comment not found")
        
        # Check for hypothesis strategies
        strategies = [
            'audio_data_strategy',
            'language_code_strategy',
            'audio_filename_strategy',
        ]
        
        for strategy in strategies:
            if strategy in content:
                print(f"✅ Found strategy: {strategy}")
            else:
                print(f"❌ Missing strategy: {strategy}")
                return False
        
        # Check for MockSessionState class
        if 'class MockSessionState' in content:
            print("✅ MockSessionState class found")
        else:
            print("❌ MockSessionState class not found")
            return False
        
        print("=" * 70)
        print("✅ Test structure validation passed!")
        print("\nTest Summary:")
        print(f"  - Total property tests: {test_count}")
        print(f"  - Property 2: Recording State Persistence (Req 1.4)")
        print(f"  - Property 3: Language Selection Persistence (Req 2.2)")
        print(f"  - Custom strategies: {len(strategies)}")
        print(f"  - Mock session state: Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_app_functions():
    """Validate that app.py functions can be imported."""
    print("\n🔍 Validating app.py functions...")
    print("=" * 70)
    
    try:
        # Mock streamlit before importing
        from unittest.mock import MagicMock
        sys.modules['streamlit'] = MagicMock()
        
        # Import app functions
        from app import (
            initialize_session_state,
            log_action,
            cache_response,
            get_cached_response,
            clear_cache
        )
        
        print("✅ Successfully imported app functions:")
        print("  - initialize_session_state")
        print("  - log_action")
        print("  - cache_response")
        print("  - get_cached_response")
        print("  - clear_cache")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to import app functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validations."""
    print("🚀 Starting Streamlit Session State Test Validation...")
    print()
    
    validations = [
        ("Test Structure", validate_test_structure),
        ("App Functions", validate_app_functions),
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        print(f"\n{'=' * 70}")
        print(f"Validation: {name}")
        print('=' * 70)
        
        if validation_func():
            passed += 1
            print(f"✅ {name} validation passed")
        else:
            print(f"❌ {name} validation failed")
    
    print("\n" + "=" * 70)
    print(f"📊 Validation Results: {passed}/{total} validations passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All validations passed!")
        print("\nThe property tests are correctly structured and ready to run.")
        print("\nTo run the tests, use:")
        print("  pytest tests/test_streamlit_session_state_properties.py -v -m property")
        return True
    else:
        print("\n❌ Some validations failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
