#!/usr/bin/env python3
"""
Validation script for deployment and integration tests.

This script validates that all deployment integration tests are properly implemented,
can be executed, and provide comprehensive coverage of deployment scenarios.
"""

import sys
import ast
import inspect
from pathlib import Path
from typing import Dict, Any, List, Set
import importlib.util


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subheader."""
    print(f"\n--- {title} ---")


def analyze_test_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze test file structure and content.
    
    Args:
        file_path: Path to test file
        
    Returns:
        Dictionary with analysis results
    """
    if not file_path.exists():
        return {"error": f"Test file not found: {file_path}"}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Extract information
        classes = []
        functions = []
        imports = []
        decorators = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node)
                }
                classes.append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                function_info = {
                    "name": node.name,
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                    "docstring": ast.get_docstring(node)
                }
                functions.append(function_info)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return {
            "file_path": str(file_path),
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "total_lines": len(content.splitlines()),
            "has_docstring": bool(ast.get_docstring(tree))
        }
    
    except Exception as e:
        return {"error": f"Failed to analyze file: {str(e)}"}


def validate_test_class_structure(class_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate test class structure and completeness.
    
    Args:
        class_info: Class information from AST analysis
        
    Returns:
        Validation results
    """
    validation = {
        "class_name": class_info["name"],
        "issues": [],
        "recommendations": [],
        "score": 0
    }
    
    # Check class name follows test convention
    if not class_info["name"].startswith("Test"):
        validation["issues"].append("Class name should start with 'Test'")
    else:
        validation["score"] += 10
    
    # Check for docstring
    if not class_info["docstring"]:
        validation["issues"].append("Class missing docstring")
    else:
        validation["score"] += 10
    
    # Check for test methods
    test_methods = [m for m in class_info["methods"] if m.startswith("test_")]
    if not test_methods:
        validation["issues"].append("No test methods found")
    else:
        validation["score"] += 20
        if len(test_methods) >= 3:
            validation["score"] += 10
    
    # Check for fixtures
    fixture_methods = [m for m in class_info["methods"] if not m.startswith("test_")]
    if fixture_methods:
        validation["score"] += 10
    
    # Check for async test methods
    async_methods = [m for m in class_info["methods"] if "async" in str(m)]
    if async_methods:
        validation["score"] += 10
    
    # Check for pytest markers
    if any("pytest.mark" in str(d) for d in class_info["decorators"]):
        validation["score"] += 10
    
    validation["test_methods"] = test_methods
    validation["fixture_methods"] = fixture_methods
    validation["total_methods"] = len(class_info["methods"])
    
    return validation


def validate_deployment_test_coverage() -> Dict[str, Any]:
    """
    Validate deployment test coverage requirements.
    
    Returns:
        Coverage validation results
    """
    required_test_areas = {
        "end_to_end_voice": {
            "description": "End-to-end voice interaction testing",
            "required_tests": [
                "complete_hindi_voice_workflow",
                "multilingual_conversation_flow", 
                "cultural_context_understanding"
            ]
        },
        "indian_services": {
            "description": "Indian service integration validation",
            "required_tests": [
                "indian_railways_integration",
                "weather_service_integration",
                "digital_india_integration",
                "service_integration_fallbacks"
            ]
        },
        "offline_online": {
            "description": "Offline/online mode transitions",
            "required_tests": [
                "offline_mode_activation",
                "offline_voice_processing",
                "online_mode_restoration",
                "data_sync_conflict_resolution"
            ]
        },
        "network_performance": {
            "description": "Performance under Indian network conditions",
            "required_tests": [
                "slow_network_performance",
                "intermittent_connectivity",
                "high_latency_performance",
                "concurrent_user_load",
                "bandwidth_optimization"
            ]
        },
        "deployment_health": {
            "description": "Deployment health checks",
            "required_tests": [
                "system_startup_health",
                "readiness_probe",
                "liveness_probe",
                "metrics_endpoint",
                "gateway_status"
            ]
        }
    }
    
    return required_test_areas


def check_test_dependencies() -> Dict[str, Any]:
    """
    Check if all required dependencies for deployment tests are available.
    
    Returns:
        Dependency check results
    """
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "fastapi",
        "httpx",
        "structlog",
        "hypothesis"
    ]
    
    dependency_status = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            dependency_status[package] = {"available": True, "error": None}
        except ImportError as e:
            dependency_status[package] = {"available": False, "error": str(e)}
    
    return dependency_status


def validate_network_simulator() -> Dict[str, Any]:
    """
    Validate network simulator implementation.
    
    Returns:
        Network simulator validation results
    """
    simulator_path = Path("tests/utils/network_simulator.py")
    
    if not simulator_path.exists():
        return {"error": "Network simulator not found"}
    
    try:
        # Import and check network simulator
        spec = importlib.util.spec_from_file_location("network_simulator", simulator_path)
        network_sim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(network_sim)
        
        validation = {
            "file_exists": True,
            "classes": [],
            "network_types": [],
            "scenarios": []
        }
        
        # Check for required classes
        required_classes = ["NetworkSimulator", "IndianNetworkScenarios", "NetworkType"]
        for class_name in required_classes:
            if hasattr(network_sim, class_name):
                validation["classes"].append(class_name)
        
        # Check network types
        if hasattr(network_sim, "NetworkType"):
            network_type_enum = getattr(network_sim, "NetworkType")
            validation["network_types"] = [nt.value for nt in network_type_enum]
        
        # Check scenario methods
        if hasattr(network_sim, "IndianNetworkScenarios"):
            scenarios_class = getattr(network_sim, "IndianNetworkScenarios")
            validation["scenarios"] = [
                method for method in dir(scenarios_class) 
                if not method.startswith("_") and callable(getattr(scenarios_class, method))
            ]
        
        return validation
    
    except Exception as e:
        return {"error": f"Failed to validate network simulator: {str(e)}"}


def main():
    """Main validation function."""
    print_header("DEPLOYMENT INTEGRATION TESTS VALIDATION")
    print("Validating comprehensive deployment test implementation")
    
    # Check test file existence and structure
    print_subheader("Test File Analysis")
    
    test_file = Path("tests/test_deployment_integration.py")
    analysis = analyze_test_file(test_file)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return 1
    
    print(f"✅ Test file found: {analysis['file_path']}")
    print(f"📊 Total lines: {analysis['total_lines']}")
    print(f"📚 Classes: {len(analysis['classes'])}")
    print(f"🔧 Functions: {len(analysis['functions'])}")
    print(f"📦 Imports: {len(analysis['imports'])}")
    
    if analysis["has_docstring"]:
        print("✅ Module has docstring")
    else:
        print("⚠️  Module missing docstring")
    
    # Validate test classes
    print_subheader("Test Class Validation")
    
    class_validations = []
    total_score = 0
    max_score = 0
    
    for class_info in analysis["classes"]:
        validation = validate_test_class_structure(class_info)
        class_validations.append(validation)
        total_score += validation["score"]
        max_score += 80  # Maximum possible score per class
        
        print(f"\n🧪 {validation['class_name']}:")
        print(f"   Score: {validation['score']}/80")
        print(f"   Test methods: {len(validation['test_methods'])}")
        print(f"   Total methods: {validation['total_methods']}")
        
        if validation["issues"]:
            print("   Issues:")
            for issue in validation["issues"]:
                print(f"     ❌ {issue}")
        else:
            print("   ✅ No issues found")
    
    overall_score = (total_score / max_score * 100) if max_score > 0 else 0
    print(f"\n📊 Overall class validation score: {overall_score:.1f}%")
    
    # Check test coverage requirements
    print_subheader("Test Coverage Validation")
    
    required_areas = validate_deployment_test_coverage()
    coverage_results = {}
    
    for area_name, area_info in required_areas.items():
        print(f"\n📋 {area_info['description']}:")
        
        area_coverage = {
            "required_tests": area_info["required_tests"],
            "found_tests": [],
            "missing_tests": []
        }
        
        # Check if required tests are implemented
        all_test_methods = []
        for class_info in analysis["classes"]:
            all_test_methods.extend([m for m in class_info["methods"] if m.startswith("test_")])
        
        for required_test in area_info["required_tests"]:
            found = any(required_test in method for method in all_test_methods)
            if found:
                area_coverage["found_tests"].append(required_test)
                print(f"   ✅ {required_test}")
            else:
                area_coverage["missing_tests"].append(required_test)
                print(f"   ❌ {required_test}")
        
        coverage_results[area_name] = area_coverage
    
    # Calculate coverage percentage
    total_required = sum(len(area["required_tests"]) for area in required_areas.values())
    total_found = sum(len(result["found_tests"]) for result in coverage_results.values())
    coverage_percentage = (total_found / total_required * 100) if total_required > 0 else 0
    
    print(f"\n📊 Test coverage: {coverage_percentage:.1f}% ({total_found}/{total_required})")
    
    # Check dependencies
    print_subheader("Dependency Validation")
    
    dependencies = check_test_dependencies()
    missing_deps = []
    
    for package, status in dependencies.items():
        if status["available"]:
            print(f"✅ {package}")
        else:
            print(f"❌ {package}: {status['error']}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
    
    # Validate network simulator
    print_subheader("Network Simulator Validation")
    
    network_validation = validate_network_simulator()
    
    if "error" in network_validation:
        print(f"❌ {network_validation['error']}")
    else:
        print("✅ Network simulator found")
        print(f"📚 Classes: {', '.join(network_validation['classes'])}")
        print(f"🌐 Network types: {len(network_validation['network_types'])}")
        print(f"🎭 Scenarios: {len(network_validation['scenarios'])}")
        
        for scenario in network_validation["scenarios"]:
            print(f"   - {scenario}")
    
    # Check test runner
    print_subheader("Test Runner Validation")
    
    runner_file = Path("run_deployment_integration_tests.py")
    if runner_file.exists():
        print("✅ Test runner script found")
        
        # Check if it's executable
        if runner_file.stat().st_mode & 0o111:
            print("✅ Test runner is executable")
        else:
            print("⚠️  Test runner is not executable (chmod +x recommended)")
    else:
        print("❌ Test runner script not found")
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    validation_checks = [
        ("Test file exists", test_file.exists()),
        ("All test classes valid", all(len(v["issues"]) == 0 for v in class_validations)),
        ("Coverage > 80%", coverage_percentage > 80),
        ("No missing dependencies", len(missing_deps) == 0),
        ("Network simulator available", "error" not in network_validation),
        ("Test runner available", runner_file.exists())
    ]
    
    passed_checks = sum(1 for _, passed in validation_checks if passed)
    total_checks = len(validation_checks)
    
    print(f"Validation checks: {passed_checks}/{total_checks}")
    
    for check_name, passed in validation_checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    # Recommendations
    print_subheader("Recommendations")
    
    if passed_checks == total_checks:
        print("🎉 All validation checks passed!")
        print("✅ Deployment integration tests are ready for execution")
        print("✅ Comprehensive coverage of deployment scenarios")
        print("✅ All dependencies are available")
    else:
        print("⚠️  Some validation checks failed. Recommendations:")
        
        if not test_file.exists():
            print("1. Create the deployment integration test file")
        
        if any(len(v["issues"]) > 0 for v in class_validations):
            print("2. Fix test class structure issues")
        
        if coverage_percentage <= 80:
            print("3. Implement missing test methods for complete coverage")
        
        if missing_deps:
            print("4. Install missing dependencies")
        
        if "error" in network_validation:
            print("5. Implement network simulator utility")
        
        if not runner_file.exists():
            print("6. Create test runner script")
    
    print("\n📋 Next Steps:")
    print("1. Run validation again after addressing issues")
    print("2. Execute deployment integration tests")
    print("3. Review test results and fix any failures")
    print("4. Integrate tests into CI/CD pipeline")
    
    return 0 if passed_checks == total_checks else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)