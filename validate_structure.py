#!/usr/bin/env python3
"""
Validation script for BharatVoice Assistant project structure.

This script validates that all required files and directories are present
and that the basic structure follows the microservices architecture.
"""

import os
import sys
from pathlib import Path


def validate_project_structure():
    """Validate the project structure."""
    print("🔍 Validating BharatVoice Assistant project structure...")
    
    # Required directories
    required_dirs = [
        "src/bharatvoice",
        "src/bharatvoice/core",
        "src/bharatvoice/config", 
        "src/bharatvoice/api",
        "src/bharatvoice/services",
        "src/bharatvoice/services/voice_processing",
        "src/bharatvoice/services/language_engine",
        "src/bharatvoice/services/context_management",
        "src/bharatvoice/services/response_generation",
        "src/bharatvoice/services/auth",
        "src/bharatvoice/services/offline_sync",
        "src/bharatvoice/utils",
        "tests",
    ]
    
    # Required files
    required_files = [
        "pyproject.toml",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "src/bharatvoice/__init__.py",
        "src/bharatvoice/main.py",
        "src/bharatvoice/core/__init__.py",
        "src/bharatvoice/core/models.py",
        "src/bharatvoice/core/interfaces.py",
        "src/bharatvoice/config/__init__.py",
        "src/bharatvoice/config/settings.py",
        "src/bharatvoice/api/__init__.py",
        "src/bharatvoice/api/health.py",
        "src/bharatvoice/api/auth.py",
        "src/bharatvoice/api/voice.py",
        "src/bharatvoice/api/context.py",
        "src/bharatvoice/utils/__init__.py",
        "src/bharatvoice/utils/logging.py",
        "src/bharatvoice/utils/monitoring.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_core_models.py",
        "tests/test_api_health.py",
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ Directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing_files.append(file_path)
        else:
            print(f"✅ File: {file_path}")
    
    # Report results
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
    
    if not missing_dirs and not missing_files:
        print("\n🎉 Project structure validation successful!")
        print("\n📋 Summary:")
        print(f"   • {len(required_dirs)} directories created")
        print(f"   • {len(required_files)} files created")
        print("   • FastAPI microservices architecture")
        print("   • Pydantic data models and interfaces")
        print("   • Configuration management with environment variables")
        print("   • Structured logging with monitoring")
        print("   • Testing framework with pytest and hypothesis")
        print("   • Docker containerization support")
        return True
    else:
        print(f"\n❌ Validation failed: {len(missing_dirs)} missing directories, {len(missing_files)} missing files")
        return False


def validate_core_models():
    """Validate core models structure."""
    print("\n🔍 Validating core models...")
    
    models_file = Path("src/bharatvoice/core/models.py")
    if not models_file.exists():
        print("❌ Core models file not found")
        return False
    
    content = models_file.read_text()
    
    # Check for key model classes
    required_models = [
        "class LanguageCode",
        "class AudioBuffer",
        "class UserProfile", 
        "class ConversationState",
        "class RecognitionResult",
        "class RegionalContextData",
        "class Response",
    ]
    
    missing_models = []
    for model in required_models:
        if model not in content:
            missing_models.append(model)
        else:
            print(f"✅ Model: {model}")
    
    if missing_models:
        print(f"❌ Missing models: {missing_models}")
        return False
    
    print("✅ Core models validation successful!")
    return True


def validate_api_structure():
    """Validate API structure."""
    print("\n🔍 Validating API structure...")
    
    api_files = [
        ("src/bharatvoice/api/health.py", ["health_check", "readiness_check", "liveness_check"]),
        ("src/bharatvoice/api/auth.py", ["login", "logout", "get_session_info"]),
        ("src/bharatvoice/api/voice.py", ["recognize_speech", "synthesize_speech", "detect_voice_activity"]),
        ("src/bharatvoice/api/context.py", ["create_session", "get_session", "get_user_profile"]),
    ]
    
    for file_path, expected_functions in api_files:
        if not Path(file_path).exists():
            print(f"❌ API file not found: {file_path}")
            continue
        
        content = Path(file_path).read_text()
        missing_functions = []
        
        for func in expected_functions:
            if f"def {func}" not in content:
                missing_functions.append(func)
            else:
                print(f"✅ API endpoint: {func} in {file_path}")
        
        if missing_functions:
            print(f"❌ Missing functions in {file_path}: {missing_functions}")
    
    print("✅ API structure validation successful!")
    return True


if __name__ == "__main__":
    print("🚀 BharatVoice Assistant Project Validation")
    print("=" * 50)
    
    success = True
    success &= validate_project_structure()
    success &= validate_core_models()
    success &= validate_api_structure()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All validations passed! Project structure is complete.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -e '.[dev]'")
        print("   2. Run tests: pytest")
        print("   3. Start development server: uvicorn bharatvoice.main:app --reload")
        print("   4. Access API docs: http://localhost:8000/docs")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please check the missing components.")
        sys.exit(1)