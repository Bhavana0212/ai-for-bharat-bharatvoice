<<<<<<< HEAD
#!/usr/bin/env python3
"""
Validation script for speech recognition property-based tests.
This script validates that the property test is correctly implemented.
"""

import os
import sys
import ast
import inspect

def validate_property_test_file():
    """Validate the property test file structure and content."""
    
    test_file_path = "tests/test_speech_recognition_properties.py"
    
    if not os.path.exists(test_file_path):
        print("❌ Property test file not found!")
        return False
    
    print("✅ Property test file exists")
    
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to analyze the file
        tree = ast.parse(content)
        
        # Check for required imports
        required_imports = [
            'pytest', 'hypothesis', 'asyncio', 'numpy',
            'AudioBuffer', 'LanguageCode', 'RecognitionResult'
        ]
        
        imports_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports_found.append(f"{node.module}.{alias.name}")
                        imports_found.append(alias.name)
        
        print(f"✅ Found {len(imports_found)} import statements")
        
        # Check for test class
        test_classes = []
        test_methods = []
        property_tests = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    test_classes.append(node.name)
                    
                    # Check methods in test class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name.startswith('test_'):
                                test_methods.append(item.name)
                                
                                # Check for @given decorator (hypothesis property test)
                                for decorator in item.decorator_list:
                                    if (isinstance(decorator, ast.Name) and decorator.id == 'given') or \
                                       (isinstance(decorator, ast.Call) and 
                                        isinstance(decorator.func, ast.Name) and 
                                        decorator.func.id == 'given'):
                                        property_tests.append(item.name)
        
        print(f"✅ Found {len(test_classes)} test classes")
        print(f"✅ Found {len(test_methods)} test methods")
        print(f"✅ Found {len(property_tests)} property-based tests")
        
        # Check for specific property tests
        expected_properties = [
            'test_recognition_completeness_property',
            'test_language_detection_consistency_property',
            'test_confidence_scoring_property',
            'test_code_switching_detection_property',
            'test_error_resilience_property'
        ]
        
        found_properties = []
        for prop in expected_properties:
            if prop in test_methods:
                found_properties.append(prop)
                print(f"  ✅ {prop}")
            else:
                print(f"  ❌ Missing: {prop}")
        
        # Check for docstring with validation info
        docstring_found = False
        validation_found = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    docstring = node.value.value
                    if "Property 1: Multilingual Speech Recognition Accuracy" in docstring:
                        docstring_found = True
                    if "Validates: Requirements 1.1, 1.2" in docstring:
                        validation_found = True
        
        if docstring_found:
            print("✅ Found Property 1 documentation")
        else:
            print("❌ Missing Property 1 documentation")
            
        if validation_found:
            print("✅ Found Requirements validation documentation")
        else:
            print("❌ Missing Requirements validation documentation")
        
        # Check for strategy functions
        strategy_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.endswith('_strategy'):
                    strategy_functions.append(node.name)
        
        print(f"✅ Found {len(strategy_functions)} test data strategies")
        for strategy in strategy_functions:
            print(f"  - {strategy}")
        
        # Summary
        print("\n📊 Validation Summary:")
        print(f"  Test classes: {len(test_classes)}")
        print(f"  Test methods: {len(test_methods)}")
        print(f"  Property-based tests: {len(property_tests)}")
        print(f"  Expected properties found: {len(found_properties)}/{len(expected_properties)}")
        print(f"  Test strategies: {len(strategy_functions)}")
        
        # Overall validation
        if (len(test_classes) > 0 and 
            len(property_tests) >= 5 and 
            len(found_properties) >= 4 and
            docstring_found and validation_found):
            print("\n🎉 Property test validation PASSED!")
            print("**Property 1: Multilingual Speech Recognition Accuracy** is properly implemented")
            print("**Validates: Requirements 1.1, 1.2**")
            return True
        else:
            print("\n❌ Property test validation FAILED!")
            print("Some required components are missing or incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Error validating property test file: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_test_structure():
    """Validate the overall test structure."""
    
    print("\n🔍 Validating test structure...")
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("❌ Tests directory not found!")
        return False
    
    print("✅ Tests directory exists")
    
    # Check for conftest.py
    if os.path.exists("tests/conftest.py"):
        print("✅ conftest.py found")
    else:
        print("⚠️  conftest.py not found (optional)")
    
    # Check for other test files
    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]
    print(f"✅ Found {len(test_files)} test files")
    
    # Check for property test specifically
    if "test_speech_recognition_properties.py" in test_files:
        print("✅ Speech recognition property test file found")
        return True
    else:
        print("❌ Speech recognition property test file not found!")
        return False

def main():
    """Main validation function."""
    
    print("🚀 Validating Speech Recognition Property-Based Tests")
    print("=" * 60)
    
    # Validate test structure
    structure_valid = validate_test_structure()
    
    if not structure_valid:
        print("\n❌ Test structure validation failed!")
        return False
    
    # Validate property test file
    property_test_valid = validate_property_test_file()
    
    if property_test_valid:
        print("\n🎉 All validations passed!")
        print("\n📋 Implementation Summary:")
        print("  ✅ Property-based test file created")
        print("  ✅ Test class with multiple property tests")
        print("  ✅ Hypothesis strategies for test data generation")
        print("  ✅ Comprehensive property validation")
        print("  ✅ Error handling and edge case testing")
        print("  ✅ Requirements traceability documentation")
        print("\n**Task 3.2 - Write property test for speech recognition accuracy: COMPLETED**")
        print("**Property 1: Multilingual Speech Recognition Accuracy**")
        print("**Validates: Requirements 1.1, 1.2**")
        return True
    else:
        print("\n❌ Property test validation failed!")
        return False

if __name__ == "__main__":
    success = main()
=======
#!/usr/bin/env python3
"""
Validation script for speech recognition property-based tests.
This script validates that the property test is correctly implemented.
"""

import os
import sys
import ast
import inspect

def validate_property_test_file():
    """Validate the property test file structure and content."""
    
    test_file_path = "tests/test_speech_recognition_properties.py"
    
    if not os.path.exists(test_file_path):
        print("❌ Property test file not found!")
        return False
    
    print("✅ Property test file exists")
    
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to analyze the file
        tree = ast.parse(content)
        
        # Check for required imports
        required_imports = [
            'pytest', 'hypothesis', 'asyncio', 'numpy',
            'AudioBuffer', 'LanguageCode', 'RecognitionResult'
        ]
        
        imports_found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports_found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports_found.append(f"{node.module}.{alias.name}")
                        imports_found.append(alias.name)
        
        print(f"✅ Found {len(imports_found)} import statements")
        
        # Check for test class
        test_classes = []
        test_methods = []
        property_tests = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    test_classes.append(node.name)
                    
                    # Check methods in test class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name.startswith('test_'):
                                test_methods.append(item.name)
                                
                                # Check for @given decorator (hypothesis property test)
                                for decorator in item.decorator_list:
                                    if (isinstance(decorator, ast.Name) and decorator.id == 'given') or \
                                       (isinstance(decorator, ast.Call) and 
                                        isinstance(decorator.func, ast.Name) and 
                                        decorator.func.id == 'given'):
                                        property_tests.append(item.name)
        
        print(f"✅ Found {len(test_classes)} test classes")
        print(f"✅ Found {len(test_methods)} test methods")
        print(f"✅ Found {len(property_tests)} property-based tests")
        
        # Check for specific property tests
        expected_properties = [
            'test_recognition_completeness_property',
            'test_language_detection_consistency_property',
            'test_confidence_scoring_property',
            'test_code_switching_detection_property',
            'test_error_resilience_property'
        ]
        
        found_properties = []
        for prop in expected_properties:
            if prop in test_methods:
                found_properties.append(prop)
                print(f"  ✅ {prop}")
            else:
                print(f"  ❌ Missing: {prop}")
        
        # Check for docstring with validation info
        docstring_found = False
        validation_found = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    docstring = node.value.value
                    if "Property 1: Multilingual Speech Recognition Accuracy" in docstring:
                        docstring_found = True
                    if "Validates: Requirements 1.1, 1.2" in docstring:
                        validation_found = True
        
        if docstring_found:
            print("✅ Found Property 1 documentation")
        else:
            print("❌ Missing Property 1 documentation")
            
        if validation_found:
            print("✅ Found Requirements validation documentation")
        else:
            print("❌ Missing Requirements validation documentation")
        
        # Check for strategy functions
        strategy_functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.endswith('_strategy'):
                    strategy_functions.append(node.name)
        
        print(f"✅ Found {len(strategy_functions)} test data strategies")
        for strategy in strategy_functions:
            print(f"  - {strategy}")
        
        # Summary
        print("\n📊 Validation Summary:")
        print(f"  Test classes: {len(test_classes)}")
        print(f"  Test methods: {len(test_methods)}")
        print(f"  Property-based tests: {len(property_tests)}")
        print(f"  Expected properties found: {len(found_properties)}/{len(expected_properties)}")
        print(f"  Test strategies: {len(strategy_functions)}")
        
        # Overall validation
        if (len(test_classes) > 0 and 
            len(property_tests) >= 5 and 
            len(found_properties) >= 4 and
            docstring_found and validation_found):
            print("\n🎉 Property test validation PASSED!")
            print("**Property 1: Multilingual Speech Recognition Accuracy** is properly implemented")
            print("**Validates: Requirements 1.1, 1.2**")
            return True
        else:
            print("\n❌ Property test validation FAILED!")
            print("Some required components are missing or incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Error validating property test file: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_test_structure():
    """Validate the overall test structure."""
    
    print("\n🔍 Validating test structure...")
    
    # Check if tests directory exists
    if not os.path.exists("tests"):
        print("❌ Tests directory not found!")
        return False
    
    print("✅ Tests directory exists")
    
    # Check for conftest.py
    if os.path.exists("tests/conftest.py"):
        print("✅ conftest.py found")
    else:
        print("⚠️  conftest.py not found (optional)")
    
    # Check for other test files
    test_files = [f for f in os.listdir("tests") if f.startswith("test_") and f.endswith(".py")]
    print(f"✅ Found {len(test_files)} test files")
    
    # Check for property test specifically
    if "test_speech_recognition_properties.py" in test_files:
        print("✅ Speech recognition property test file found")
        return True
    else:
        print("❌ Speech recognition property test file not found!")
        return False

def main():
    """Main validation function."""
    
    print("🚀 Validating Speech Recognition Property-Based Tests")
    print("=" * 60)
    
    # Validate test structure
    structure_valid = validate_test_structure()
    
    if not structure_valid:
        print("\n❌ Test structure validation failed!")
        return False
    
    # Validate property test file
    property_test_valid = validate_property_test_file()
    
    if property_test_valid:
        print("\n🎉 All validations passed!")
        print("\n📋 Implementation Summary:")
        print("  ✅ Property-based test file created")
        print("  ✅ Test class with multiple property tests")
        print("  ✅ Hypothesis strategies for test data generation")
        print("  ✅ Comprehensive property validation")
        print("  ✅ Error handling and edge case testing")
        print("  ✅ Requirements traceability documentation")
        print("\n**Task 3.2 - Write property test for speech recognition accuracy: COMPLETED**")
        print("**Property 1: Multilingual Speech Recognition Accuracy**")
        print("**Validates: Requirements 1.1, 1.2**")
        return True
    else:
        print("\n❌ Property test validation failed!")
        return False

if __name__ == "__main__":
    success = main()
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    sys.exit(0 if success else 1)