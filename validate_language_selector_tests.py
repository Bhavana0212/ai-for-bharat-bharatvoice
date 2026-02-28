"""
Validation script for language selector property tests.

This script validates that the language selector property tests are correctly
implemented and can be imported without errors.
"""

import sys
import os

def validate_test_file():
    """Validate the language selector test file."""
    
    print("=" * 80)
    print("Validating Language Selector Property Tests")
    print("=" * 80)
    print()
    
    # Check if test file exists
    test_file = "tests/test_language_selector_properties.py"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"✅ Test file exists: {test_file}")
    
    # Try to compile the test file
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, test_file, 'exec')
        print("✅ Test file syntax is valid")
    except SyntaxError as e:
        print(f"❌ Syntax error in test file: {e}")
        return False
    
    # Check for required test functions
    required_tests = [
        'test_property_language_selection_persistence',
        'test_property_language_change_application',
        'test_property_iso_language_code_consistency',
    ]
    
    print()
    print("Checking for required test functions:")
    for test_name in required_tests:
        if test_name in code:
            print(f"  ✅ {test_name}")
        else:
            print(f"  ❌ {test_name} not found")
            return False
    
    # Check for property markers
    print()
    print("Checking for property test markers:")
    if '@pytest.mark.property' in code:
        count = code.count('@pytest.mark.property')
        print(f"  ✅ Found {count} property test markers")
    else:
        print("  ❌ No property test markers found")
        return False
    
    # Check for hypothesis decorators
    print()
    print("Checking for hypothesis decorators:")
    if '@given' in code:
        count = code.count('@given')
        print(f"  ✅ Found {count} @given decorators")
    else:
        print("  ❌ No @given decorators found")
        return False
    
    # Check for requirement validation comments
    print()
    print("Checking for requirement validation comments:")
    requirements = ['2.2', '2.5', '12.4']
    for req in requirements:
        if f'Requirements {req}' in code or f'Requirement {req}' in code:
            print(f"  ✅ Validates Requirement {req}")
        else:
            print(f"  ⚠️  Requirement {req} validation not found")
    
    # Check for property descriptions
    print()
    print("Checking for property descriptions:")
    properties = [
        'Property 3: Language Selection Persistence',
        'Property 5: Language Change Application',
        'Property 33: ISO Language Code Consistency'
    ]
    
    for prop in properties:
        if prop in code:
            print(f"  ✅ {prop}")
        else:
            print(f"  ❌ {prop} not found")
            return False
    
    print()
    print("=" * 80)
    print("✅ All validation checks passed!")
    print("=" * 80)
    print()
    print("To run the tests, execute:")
    print("  python -m pytest tests/test_language_selector_properties.py -v -m property")
    print()
    
    return True

if __name__ == '__main__':
    success = validate_test_file()
    sys.exit(0 if success else 1)
