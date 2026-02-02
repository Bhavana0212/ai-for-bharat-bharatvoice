#!/usr/bin/env python3
"""
Comprehensive Test Validation Script for BharatVoice Assistant

This script validates all property-based tests and system components
without requiring Python execution or external dependencies.
"""

import os
import sys
import ast
import glob
from typing import Dict, List, Tuple, Any

class TestValidator:
    """Validates test files and their structure."""
    
    def __init__(self):
        self.validation_results = {}
        self.property_tests = {}
        
    def validate_file_exists(self, filepath: str) -> Tuple[bool, str]:
        """Check if a file exists."""
        if os.path.exists(filepath):
            return True, f"File exists: {filepath}"
        return False, f"File missing: {filepath}"
    
    def validate_python_syntax(self, filepath: str) -> Tuple[bool, str]:
        """Validate Python syntax of a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            return True, f"Valid Python syntax: {filepath}"
        except SyntaxError as e:
            return False, f"Syntax error in {filepath}: {e}"
        except Exception as e:
            return False, f"Error reading {filepath}: {e}"
    
    def validate_property_test_structure(self, filepath: str, expected_property: str) -> Tuple[bool, str]:
        """Validate property-based test structure."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for required elements
            has_imports = False
            has_property_comment = False
            has_test_class = False
            test_methods = 0
            strategy_functions = 0
            
            # Check imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    has_imports = True
            
            # Check for property comment
            if expected_property in content:
                has_property_comment = True
            
            # Check for test class and methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if "Test" in node.name and "Properties" in node.name:
                        has_test_class = True
                
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        test_methods += 1
                    elif node.name.endswith('_strategy'):
                        strategy_functions += 1
            
            # Validate requirements
            issues = []
            if not has_imports:
                issues.append("Missing imports")
            if not has_property_comment:
                issues.append(f"Missing property comment: {expected_property}")
            if not has_test_class:
                issues.append("Missing test class")
            if test_methods < 3:
                issues.append(f"Insufficient test methods: {test_methods}")
            
            if issues:
                return False, f"Structure issues: {', '.join(issues)}"
            
            return True, f"Valid structure: {test_methods} tests, {strategy_functions} strategies"
            
        except Exception as e:
            return False, f"Error validating structure: {e}"
    
    def validate_service_implementation(self, service_path: str) -> Tuple[bool, str]:
        """Validate service implementation."""
        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for class definitions
            classes = []
            methods = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    methods.append(node.name)
            
            if not classes:
                return False, "No classes found"
            
            if len(methods) < 3:
                return False, f"Insufficient methods: {len(methods)}"
            
            return True, f"Valid implementation: {len(classes)} classes, {len(methods)} methods"
            
        except Exception as e:
            return False, f"Error validating service: {e}"

def validate_all_property_tests():
    """Validate all property-based test files."""
    validator = TestValidator()
    
    # Define property tests and their expected properties
    property_tests = {
        "tests/test_speech_recognition_properties.py": "Property 1: Multilingual Speech Recognition Accuracy",
        "tests/test_audio_processing_properties.py": "Property 3: Noise Resilience", 
        "tests/test_nlu_properties.py": "Property 5: Cultural Context Recognition",
        "tests/test_tts_synthesis_properties.py": "Property 10: Natural Speech Synthesis",
        "tests/test_indian_service_integration_properties.py": "Property 13: Indian Service Integration",
        "tests/test_offline_functionality_properties.py": "Property 14: Offline Functionality",
        "tests/test_accessibility_properties.py": "Property 18: Accessibility Support",
        "tests/test_platform_integration_properties.py": "Property 19: Indian Platform Integration",
        "tests/test_performance_properties.py": "Property 20: Performance Requirements",
        "tests/test_error_handling_properties.py": "Property 21: Localized Error Handling",
        "tests/test_adaptive_learning_properties.py": "Property 22: Adaptive Learning",
        "tests/test_system_extensibility_properties.py": "Property 23: System Extensibility"
    }
    
    results = {}
    
    print("🧪 Validating Property-Based Tests")
    print("=" * 60)
    
    for test_file, expected_property in property_tests.items():
        print(f"\n📋 {test_file}")
        
        # Check file existence
        exists, msg = validator.validate_file_exists(test_file)
        if not exists:
            results[test_file] = {"status": "MISSING", "details": msg}
            print(f"  ❌ {msg}")
            continue
        
        # Check syntax
        syntax_ok, syntax_msg = validator.validate_python_syntax(test_file)
        if not syntax_ok:
            results[test_file] = {"status": "SYNTAX_ERROR", "details": syntax_msg}
            print(f"  ❌ {syntax_msg}")
            continue
        
        # Check structure
        structure_ok, structure_msg = validator.validate_property_test_structure(test_file, expected_property)
        if structure_ok:
            results[test_file] = {"status": "VALID", "details": structure_msg}
            print(f"  ✅ {structure_msg}")
        else:
            results[test_file] = {"status": "INVALID_STRUCTURE", "details": structure_msg}
            print(f"  ⚠️  {structure_msg}")
    
    return results

def validate_service_implementations():
    """Validate core service implementations."""
    validator = TestValidator()
    
    # Define core services to validate
    services = {
        "src/bharatvoice/services/voice_processing/service.py": "Voice Processing Service",
        "src/bharatvoice/services/language_engine/service.py": "Language Engine Service", 
        "src/bharatvoice/services/context_management/service.py": "Context Management Service",
        "src/bharatvoice/services/response_generation/response_generator.py": "Response Generation Service",
        "src/bharatvoice/services/auth/auth_service.py": "Authentication Service",
        "src/bharatvoice/services/external_integrations/service_manager.py": "External Integrations Service",
        "src/bharatvoice/services/platform_integrations/platform_manager.py": "Platform Integrations Service",
        "src/bharatvoice/services/learning/adaptive_learning_service.py": "Adaptive Learning Service"
    }
    
    results = {}
    
    print("\n🔧 Validating Service Implementations")
    print("=" * 60)
    
    for service_file, service_name in services.items():
        print(f"\n📋 {service_name}")
        
        # Check file existence
        exists, msg = validator.validate_file_exists(service_file)
        if not exists:
            results[service_file] = {"status": "MISSING", "details": msg}
            print(f"  ❌ {msg}")
            continue
        
        # Check syntax
        syntax_ok, syntax_msg = validator.validate_python_syntax(service_file)
        if not syntax_ok:
            results[service_file] = {"status": "SYNTAX_ERROR", "details": syntax_msg}
            print(f"  ❌ {syntax_msg}")
            continue
        
        # Check implementation
        impl_ok, impl_msg = validator.validate_service_implementation(service_file)
        if impl_ok:
            results[service_file] = {"status": "VALID", "details": impl_msg}
            print(f"  ✅ {impl_msg}")
        else:
            results[service_file] = {"status": "INCOMPLETE", "details": impl_msg}
            print(f"  ⚠️  {impl_msg}")
    
    return results

def validate_test_runners():
    """Validate test runner scripts."""
    validator = TestValidator()
    
    # Find all test runner scripts
    runner_scripts = glob.glob("run_*_test.py") + glob.glob("validate_*.py")
    
    results = {}
    
    print("\n🏃 Validating Test Runner Scripts")
    print("=" * 60)
    
    for script in runner_scripts:
        print(f"\n📋 {script}")
        
        # Check syntax
        syntax_ok, syntax_msg = validator.validate_python_syntax(script)
        if syntax_ok:
            results[script] = {"status": "VALID", "details": syntax_msg}
            print(f"  ✅ {syntax_msg}")
        else:
            results[script] = {"status": "SYNTAX_ERROR", "details": syntax_msg}
            print(f"  ❌ {syntax_msg}")
    
    return results

def validate_project_structure():
    """Validate overall project structure."""
    print("\n🏗️  Validating Project Structure")
    print("=" * 60)
    
    # Required directories
    required_dirs = [
        "src/bharatvoice",
        "src/bharatvoice/api",
        "src/bharatvoice/services",
        "src/bharatvoice/core",
        "src/bharatvoice/config",
        "src/bharatvoice/database",
        "src/bharatvoice/cache",
        "src/bharatvoice/storage",
        "src/bharatvoice/utils",
        "tests",
        "alembic"
    ]
    
    # Required files
    required_files = [
        "pyproject.toml",
        "README.md",
        "src/bharatvoice/__init__.py",
        "src/bharatvoice/main.py",
        "tests/__init__.py",
        "alembic.ini"
    ]
    
    structure_valid = True
    
    print("\n📁 Checking directories:")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory}")
            structure_valid = False
    
    print("\n📄 Checking files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
            structure_valid = False
    
    return structure_valid

def generate_test_summary(property_results, service_results, runner_results):
    """Generate comprehensive test summary."""
    print("\n📊 COMPREHENSIVE TEST VALIDATION SUMMARY")
    print("=" * 80)
    
    # Property tests summary
    print("\n🧪 Property-Based Tests:")
    valid_properties = sum(1 for r in property_results.values() if r["status"] == "VALID")
    total_properties = len(property_results)
    print(f"  Valid: {valid_properties}/{total_properties}")
    
    for test_file, result in property_results.items():
        status_icon = "✅" if result["status"] == "VALID" else "❌" if result["status"] == "MISSING" else "⚠️"
        print(f"  {status_icon} {os.path.basename(test_file)}: {result['status']}")
    
    # Service implementations summary
    print("\n🔧 Service Implementations:")
    valid_services = sum(1 for r in service_results.values() if r["status"] == "VALID")
    total_services = len(service_results)
    print(f"  Valid: {valid_services}/{total_services}")
    
    for service_file, result in service_results.items():
        status_icon = "✅" if result["status"] == "VALID" else "❌" if result["status"] == "MISSING" else "⚠️"
        service_name = os.path.basename(service_file)
        print(f"  {status_icon} {service_name}: {result['status']}")
    
    # Test runners summary
    print("\n🏃 Test Runners:")
    valid_runners = sum(1 for r in runner_results.values() if r["status"] == "VALID")
    total_runners = len(runner_results)
    print(f"  Valid: {valid_runners}/{total_runners}")
    
    # Overall assessment
    print("\n🎯 Overall Assessment:")
    property_score = (valid_properties / total_properties) * 100 if total_properties > 0 else 0
    service_score = (valid_services / total_services) * 100 if total_services > 0 else 0
    runner_score = (valid_runners / total_runners) * 100 if total_runners > 0 else 0
    
    overall_score = (property_score + service_score + runner_score) / 3
    
    print(f"  Property Tests: {property_score:.1f}%")
    print(f"  Service Implementations: {service_score:.1f}%")
    print(f"  Test Runners: {runner_score:.1f}%")
    print(f"  Overall Score: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print("\n🎉 EXCELLENT: System is ready for comprehensive testing!")
    elif overall_score >= 75:
        print("\n✅ GOOD: System is mostly ready, minor issues to address")
    elif overall_score >= 50:
        print("\n⚠️  FAIR: System needs significant work before testing")
    else:
        print("\n❌ POOR: System requires major implementation work")
    
    return overall_score

def main():
    """Run comprehensive validation."""
    print("🚀 BharatVoice Assistant - Comprehensive Test Validation")
    print("=" * 80)
    print("This validation checks all components without requiring Python execution")
    print()
    
    # Validate project structure
    structure_valid = validate_project_structure()
    
    # Validate property-based tests
    property_results = validate_all_property_tests()
    
    # Validate service implementations
    service_results = validate_service_implementations()
    
    # Validate test runners
    runner_results = validate_test_runners()
    
    # Generate summary
    overall_score = generate_test_summary(property_results, service_results, runner_results)
    
    print("\n📝 Next Steps:")
    if overall_score >= 75:
        print("  1. Set up Python environment with required dependencies")
        print("  2. Run individual property test runners: python run_*_property_test.py")
        print("  3. Execute comprehensive test suite: pytest tests/ -v")
        print("  4. Address any failing tests")
        print("  5. Perform load testing and user journey validation")
    else:
        print("  1. Fix missing or invalid components identified above")
        print("  2. Complete service implementations")
        print("  3. Ensure all property tests are properly structured")
        print("  4. Re-run this validation script")
    
    return 0 if overall_score >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())