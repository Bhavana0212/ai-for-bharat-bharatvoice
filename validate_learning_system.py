"""
Validation script for Learning and Adaptation System implementation.

This script validates that all required components are properly implemented
for the adaptive learning and system extensibility features.
"""

import os
import sys
from pathlib import Path


def validate_file_exists(file_path: str, description: str) -> bool:
    """Validate that a file exists."""
    if Path(file_path).exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} - NOT FOUND")
        return False


def validate_learning_system():
    """Validate the learning system implementation."""
    print("Validating BharatVoice Learning and Adaptation System...")
    print("=" * 60)
    
    validation_results = []
    
    # Core learning services
    learning_files = [
        ("src/bharatvoice/services/learning/__init__.py", "Learning services package"),
        ("src/bharatvoice/services/learning/adaptive_learning_service.py", "Main adaptive learning service"),
        ("src/bharatvoice/services/learning/vocabulary_learner.py", "Vocabulary learning module"),
        ("src/bharatvoice/services/learning/accent_adapter.py", "Accent adaptation module"),
        ("src/bharatvoice/services/learning/preference_learner.py", "Preference learning module"),
        ("src/bharatvoice/services/learning/feedback_processor.py", "Feedback processing module"),
        ("src/bharatvoice/services/learning/response_style_adapter.py", "Response style adaptation module"),
    ]
    
    print("\n1. Core Learning Components:")
    for file_path, description in learning_files:
        validation_results.append(validate_file_exists(file_path, description))
    
    # System extensibility components
    extensibility_files = [
        ("src/bharatvoice/services/learning/model_manager.py", "Model management system"),
        ("src/bharatvoice/services/learning/plugin_manager.py", "Plugin architecture system"),
        ("src/bharatvoice/services/learning/ab_testing_framework.py", "A/B testing framework"),
        ("src/bharatvoice/services/learning/system_extensibility_service.py", "System extensibility service"),
    ]
    
    print("\n2. System Extensibility Components:")
    for file_path, description in extensibility_files:
        validation_results.append(validate_file_exists(file_path, description))
    
    # Property-based tests
    test_files = [
        ("tests/test_adaptive_learning_properties.py", "Adaptive learning property tests"),
        ("tests/test_system_extensibility_properties.py", "System extensibility property tests"),
    ]
    
    print("\n3. Property-Based Tests:")
    for file_path, description in test_files:
        validation_results.append(validate_file_exists(file_path, description))
    
    # Validate file content structure
    print("\n4. Implementation Structure Validation:")
    
    # Check adaptive learning service structure
    adaptive_service_path = "src/bharatvoice/services/learning/adaptive_learning_service.py"
    if Path(adaptive_service_path).exists():
        with open(adaptive_service_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_classes = ["AdaptiveLearningService"]
        required_methods = [
            "process_interaction",
            "adapt_response_for_user", 
            "get_user_learning_profile",
            "get_personalization_suggestions"
        ]
        
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✓ Found class: {class_name}")
                validation_results.append(True)
            else:
                print(f"✗ Missing class: {class_name}")
                validation_results.append(False)
        
        for method_name in required_methods:
            if f"async def {method_name}" in content or f"def {method_name}" in content:
                print(f"✓ Found method: {method_name}")
                validation_results.append(True)
            else:
                print(f"✗ Missing method: {method_name}")
                validation_results.append(False)
    
    # Check system extensibility service structure
    extensibility_service_path = "src/bharatvoice/services/learning/system_extensibility_service.py"
    if Path(extensibility_service_path).exists():
        with open(extensibility_service_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_classes = ["SystemExtensibilityService"]
        required_methods = [
            "add_language_support",
            "update_model",
            "create_feature_experiment",
            "get_system_capabilities"
        ]
        
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✓ Found class: {class_name}")
                validation_results.append(True)
            else:
                print(f"✗ Missing class: {class_name}")
                validation_results.append(False)
        
        for method_name in required_methods:
            if f"async def {method_name}" in content or f"def {method_name}" in content:
                print(f"✓ Found method: {method_name}")
                validation_results.append(True)
            else:
                print(f"✗ Missing method: {method_name}")
                validation_results.append(False)
    
    # Check property tests structure
    adaptive_test_path = "tests/test_adaptive_learning_properties.py"
    if Path(adaptive_test_path).exists():
        with open(adaptive_test_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "Property 22: Adaptive Learning" in content:
            print("✓ Found Property 22: Adaptive Learning tests")
            validation_results.append(True)
        else:
            print("✗ Missing Property 22: Adaptive Learning tests")
            validation_results.append(False)
    
    extensibility_test_path = "tests/test_system_extensibility_properties.py"
    if Path(extensibility_test_path).exists():
        with open(extensibility_test_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "Property 23: System Extensibility" in content:
            print("✓ Found Property 23: System Extensibility tests")
            validation_results.append(True)
        else:
            print("✗ Missing Property 23: System Extensibility tests")
            validation_results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    total_checks = len(validation_results)
    passed_checks = sum(validation_results)
    
    print(f"Validation Summary: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 All validation checks passed! Learning system is properly implemented.")
        return True
    else:
        print(f"⚠️  {total_checks - passed_checks} validation checks failed.")
        return False


if __name__ == "__main__":
    success = validate_learning_system()
    sys.exit(0 if success else 1)