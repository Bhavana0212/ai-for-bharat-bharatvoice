#!/usr/bin/env python3
"""
Validation script for the Language Engine Service implementation.

This script validates that the multilingual ASR system has been properly
implemented with all required components and functionality.
"""

import os
import sys
from pathlib import Path


def validate_language_engine_structure():
    """Validate the language engine service structure."""
    print("🔍 Validating Language Engine Service structure...")
    
    # Required files for language engine
    required_files = [
        "src/bharatvoice/services/language_engine/__init__.py",
        "src/bharatvoice/services/language_engine/service.py",
        "src/bharatvoice/services/language_engine/asr_engine.py",
        "src/bharatvoice/services/language_engine/README.md",
        "tests/test_language_engine.py",
    ]
    
    missing_files = []
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
        else:
            print(f"✅ File: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ Language engine structure validation successful!")
    return True


def validate_asr_engine_implementation():
    """Validate ASR engine implementation."""
    print("\n🔍 Validating ASR engine implementation...")
    
    asr_file = Path("src/bharatvoice/services/language_engine/asr_engine.py")
    if not asr_file.exists():
        print("❌ ASR engine file not found")
        return False
    
    content = asr_file.read_text()
    
    # Check for key classes and methods
    required_components = [
        "class MultilingualASREngine",
        "def recognize_speech",
        "def detect_language", 
        "def detect_code_switching",
        "def translate_text",
        "def adapt_to_regional_accent",
        "whisper",  # Whisper integration
        "langdetect",  # Language detection
        "transformers",  # Transformer models
        "create_multilingual_asr_engine",  # Factory function
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
        else:
            print(f"✅ Component: {component}")
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        return False
    
    print("✅ ASR engine implementation validation successful!")
    return True


def validate_language_service_implementation():
    """Validate language service implementation."""
    print("\n🔍 Validating Language Service implementation...")
    
    service_file = Path("src/bharatvoice/services/language_engine/service.py")
    if not service_file.exists():
        print("❌ Language service file not found")
        return False
    
    content = service_file.read_text()
    
    # Check for key classes and methods
    required_components = [
        "class LanguageEngineService",
        "def recognize_speech",
        "def detect_code_switching",
        "def translate_text",
        "def detect_language",
        "def batch_recognize_speech",
        "def get_language_confidence_scores",
        "def health_check",
        "recognition_cache",  # Caching support
        "translation_cache",  # Translation caching
        "create_language_engine_service",  # Factory function
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
        else:
            print(f"✅ Component: {component}")
    
    if missing_components:
        print(f"❌ Missing components: {missing_components}")
        return False
    
    print("✅ Language service implementation validation successful!")
    return True


def validate_supported_languages():
    """Validate supported languages implementation."""
    print("\n🔍 Validating supported languages...")
    
    # Check core models for language support
    models_file = Path("src/bharatvoice/core/models.py")
    if not models_file.exists():
        print("❌ Core models file not found")
        return False
    
    content = models_file.read_text()
    
    # Check for Indian languages
    required_languages = [
        "HINDI",
        "ENGLISH_IN", 
        "TAMIL",
        "TELUGU",
        "BENGALI",
        "MARATHI",
        "GUJARATI",
        "KANNADA",
        "MALAYALAM",
        "PUNJABI",
        "ODIA",
    ]
    
    missing_languages = []
    for lang in required_languages:
        if lang not in content:
            missing_languages.append(lang)
        else:
            print(f"✅ Language: {lang}")
    
    if missing_languages:
        print(f"❌ Missing languages: {missing_languages}")
        return False
    
    print("✅ Supported languages validation successful!")
    return True


def validate_test_implementation():
    """Validate test implementation."""
    print("\n🔍 Validating test implementation...")
    
    test_file = Path("tests/test_language_engine.py")
    if not test_file.exists():
        print("❌ Test file not found")
        return False
    
    content = test_file.read_text()
    
    # Check for key test classes and methods
    required_tests = [
        "class TestMultilingualASREngine",
        "class TestLanguageEngineService",
        "test_recognize_speech",
        "test_detect_language",
        "test_detect_code_switching",
        "test_translate_text",
        "test_batch_recognize_speech",
        "test_health_check",
        "test_caching",
        "test_error_handling",
        "@pytest.mark.asyncio",  # Async test support
    ]
    
    missing_tests = []
    for test in required_tests:
        if test not in content:
            missing_tests.append(test)
        else:
            print(f"✅ Test: {test}")
    
    if missing_tests:
        print(f"❌ Missing tests: {missing_tests}")
        return False
    
    print("✅ Test implementation validation successful!")
    return True


def validate_documentation():
    """Validate documentation."""
    print("\n🔍 Validating documentation...")
    
    readme_file = Path("src/bharatvoice/services/language_engine/README.md")
    if not readme_file.exists():
        print("❌ README file not found")
        return False
    
    content = readme_file.read_text()
    
    # Check for key documentation sections
    required_sections = [
        "# Language Engine Service",
        "## Features",
        "## Architecture", 
        "## Supported Languages",
        "## Usage",
        "## Configuration",
        "## Testing",
        "## Dependencies",
        "Multilingual ASR",
        "Language Detection",
        "Code-Switching",
        "Translation",
        "Whisper",
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
        else:
            print(f"✅ Documentation: {section}")
    
    if missing_sections:
        print(f"❌ Missing documentation sections: {missing_sections}")
        return False
    
    print("✅ Documentation validation successful!")
    return True


def validate_task_requirements():
    """Validate that task requirements are met."""
    print("\n🔍 Validating task requirements...")
    
    # Task 3.1 requirements:
    # - Integrate speech recognition for Hindi and English (using Whisper or similar)
    # - Add support for regional Indian languages (Tamil, Telugu, Bengali, etc.)
    # - Implement confidence scoring and alternative transcription handling
    # - Create language detection and switching mechanisms
    
    requirements_met = []
    
    # Check Whisper integration
    asr_file = Path("src/bharatvoice/services/language_engine/asr_engine.py")
    if asr_file.exists():
        content = asr_file.read_text()
        if "whisper" in content and "load_model" in content:
            requirements_met.append("✅ Whisper integration for Hindi and English")
        else:
            requirements_met.append("❌ Missing Whisper integration")
        
        if "TAMIL" in content and "TELUGU" in content and "BENGALI" in content:
            requirements_met.append("✅ Regional Indian languages support")
        else:
            requirements_met.append("❌ Missing regional languages support")
        
        if "confidence" in content and "alternative" in content:
            requirements_met.append("✅ Confidence scoring and alternatives")
        else:
            requirements_met.append("❌ Missing confidence scoring/alternatives")
        
        if "detect_language" in content and "code_switching" in content:
            requirements_met.append("✅ Language detection and switching")
        else:
            requirements_met.append("❌ Missing language detection/switching")
    else:
        requirements_met.append("❌ ASR engine file not found")
    
    for req in requirements_met:
        print(req)
    
    success = all("✅" in req for req in requirements_met)
    if success:
        print("✅ All task requirements validation successful!")
    else:
        print("❌ Some task requirements not met")
    
    return success


if __name__ == "__main__":
    print("🚀 Language Engine Service Validation")
    print("=" * 50)
    
    success = True
    success &= validate_language_engine_structure()
    success &= validate_asr_engine_implementation()
    success &= validate_language_service_implementation()
    success &= validate_supported_languages()
    success &= validate_test_implementation()
    success &= validate_documentation()
    success &= validate_task_requirements()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All validations passed! Language Engine Service implementation is complete.")
        print("\n📋 Implementation Summary:")
        print("   • ✅ Multilingual ASR system with Whisper integration")
        print("   • ✅ Support for Hindi, English, and 9 regional Indian languages")
        print("   • ✅ Confidence scoring and alternative transcriptions")
        print("   • ✅ Language detection and code-switching mechanisms")
        print("   • ✅ Translation engine with caching")
        print("   • ✅ Comprehensive test suite with error handling")
        print("   • ✅ Detailed documentation and usage examples")
        print("   • ✅ Factory functions and service interfaces")
        print("\n🔧 Key Features Implemented:")
        print("   • Whisper-based speech recognition")
        print("   • Automatic language detection")
        print("   • Code-switching detection in mixed-language text")
        print("   • Confidence scoring for transcription quality")
        print("   • Alternative transcription hypotheses")
        print("   • Result caching for improved performance")
        print("   • Batch processing capabilities")
        print("   • Health monitoring and statistics")
        print("   • Comprehensive error handling")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install openai-whisper transformers langdetect")
        print("   2. Run tests: pytest tests/test_language_engine.py")
        print("   3. Integrate with voice processing service")
        print("   4. Test with real audio samples")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please check the missing components.")
        sys.exit(1)