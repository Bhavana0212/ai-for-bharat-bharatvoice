<<<<<<< HEAD
"""
Property-Based Tests for System Extensibility Framework.

**Property 23: System Extensibility**
Tests that the system can be extended with new languages, models, and features
while maintaining backward compatibility.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from bharatvoice.core.models import LanguageCode, UserInteraction
from bharatvoice.services.learning import (
    SystemExtensibilityService, ModelManager, PluginManager, ABTestingFramework
)
from bharatvoice.services.learning.model_manager import ModelType, ModelStatus, ModelVersion
from bharatvoice.services.learning.plugin_manager import PluginType, PluginStatus
from bharatvoice.services.learning.ab_testing_framework import MetricType, VariantType, ExperimentStatus


# Test data generators
@st.composite
def model_metadata(draw):
    """Generate model metadata."""
    return {
        "author": draw(st.text(min_size=3, max_size=50)),
        "description": draw(st.text(min_size=10, max_size=200)),
        "training_data": draw(st.text(min_size=5, max_size=100)),
        "accuracy": draw(st.floats(min_value=0.5, max_value=1.0)),
        "model_size_mb": draw(st.integers(min_value=1, max_value=1000))
    }


@st.composite
def plugin_config(draw):
    """Generate plugin configuration."""
    return {
        "enabled": draw(st.booleans()),
        "priority": draw(st.integers(min_value=1, max_value=10)),
        "cache_size": draw(st.integers(min_value=100, max_value=10000)),
        "timeout": draw(st.floats(min_value=1.0, max_value=30.0))
    }


class TestSystemExtensibilityProperties:
    """Property-based tests for system extensibility."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def extensibility_service(self, temp_dir):
        """Create extensibility service for testing."""
        models_dir = Path(temp_dir) / "models"
        plugins_dir = Path(temp_dir) / "plugins"
        models_dir.mkdir()
        plugins_dir.mkdir()
        
        return SystemExtensibilityService(
            models_directory=str(models_dir),
            plugins_directory=str(plugins_dir),
            enable_ab_testing=True,
            auto_load_plugins=False
        )
    
    @given(
        languages=st.lists(st.sampled_from(list(LanguageCode)), min_size=1, max_size=5, unique=True),
        model_types=st.lists(st.sampled_from(list(ModelType)), min_size=1, max_size=3, unique=True),
        metadata=model_metadata()
    )
    @settings(max_examples=20, deadline=10000)
    async def test_language_support_extensibility(self, extensibility_service, temp_dir, languages, model_types, metadata):
        """
        **Property 23: System Extensibility**
        **Validates: Requirements 1.1, 1.2, 2.1**
        
        Test that new languages can be added with proper model support.
        """
        await extensibility_service.initialize()
        
        for language in languages:
            # Create dummy model files
            model_files = {}
            for model_type in model_types:
                model_file = Path(temp_dir) / f"{language.value}_{model_type.value}_model.bin"
                model_file.write_text(f"dummy model data for {language.value} {model_type.value}")
                model_files[model_type] = str(model_file)
            
            # Add language support
            result = await extensibility_service.add_language_support(
                language=language,
                model_files=model_files,
                metadata=metadata
            )
            
            # Verify language was added successfully
            assert result["success"], f"Failed to add language {language.value}: {result.get('errors', [])}"
            assert result["language"] == language.value, "Language should match"
            assert len(result["models_registered"]) == len(model_types), \
                f"Should register {len(model_types)} models, got {len(result['models_registered'])}"
        
        # Verify all languages are supported
        supported_languages = await extensibility_service.get_supported_languages()
        
        for language in languages:
            for model_type in model_types:
                assert language in supported_languages.get(model_type, []), \
                    f"Language {language.value} should be supported for {model_type.value}"
    
    @given(
        model_versions=st.lists(st.text(min_size=5, max_size=20), min_size=2, max_size=5, unique=True),
        model_type=st.sampled_from(list(ModelType)),
        language=st.sampled_from(list(LanguageCode)),
        metadata=model_metadata()
    )
    @settings(max_examples=15, deadline=8000)
    async def test_model_update_backward_compatibility(self, extensibility_service, temp_dir, model_versions, model_type, language, metadata):
        """
        Test that model updates maintain backward compatibility.
        """
        await extensibility_service.initialize()
        
        model_id = f"{model_type.value}_{language.value}"
        
        # Register initial model version
        initial_version = model_versions[0]
        initial_model_file = Path(temp_dir) / f"{model_id}_{initial_version}.bin"
        initial_model_file.write_text(f"model data v{initial_version}")
        
        await extensibility_service.model_manager.register_model(
            model_id=model_id,
            version=initial_version,
            model_type=model_type,
            language=language,
            model_path=str(initial_model_file),
            metadata=metadata,
            auto_activate=True
        )
        
        # Update to newer versions
        for new_version in model_versions[1:]:
            new_model_file = Path(temp_dir) / f"{model_id}_{new_version}.bin"
            new_model_file.write_text(f"model data v{new_version}")
            
            result = await extensibility_service.update_model(
                model_id=model_id,
                new_version=new_version,
                model_path=str(new_model_file)
            )
            
            # Update should succeed
            assert result["update_successful"], f"Model update should succeed: {result.get('errors', [])}"
            
            # Verify new version is active
            active_model = await extensibility_service.model_manager.get_active_model_by_id(model_id)
            assert active_model is not None, "Should have active model"
            assert active_model.version == new_version, f"Active version should be {new_version}"
            
            # Verify old versions are still available (backward compatibility)
            for old_version in model_versions[:model_versions.index(new_version) + 1]:
                old_model = await extensibility_service.model_manager.get_model_version(model_id, old_version)
                assert old_model is not None, f"Old version {old_version} should still be available"
    
    @given(
        experiment_name=st.text(min_size=5, max_size=50),
        hypothesis=st.text(min_size=10, max_size=200),
        primary_metric=st.sampled_from(list(MetricType)),
        duration_days=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=20, deadline=5000)
    async def test_ab_testing_framework_extensibility(self, extensibility_service, experiment_name, hypothesis, primary_metric, duration_days):
        """
        Test that A/B testing framework can be extended with new experiments.
        """
        await extensibility_service.initialize()
        
        # Create feature experiment
        control_config = {"feature_enabled": False, "algorithm": "v1"}
        treatment_config = {"feature_enabled": True, "algorithm": "v2"}
        
        experiment_id = await extensibility_service.create_feature_experiment(
            name=experiment_name,
            description=f"Testing {experiment_name}",
            hypothesis=hypothesis,
            feature_config_control=control_config,
            feature_config_treatment=treatment_config,
            primary_metric=primary_metric,
            duration_days=duration_days
        )
        
        # Experiment should be created
        assert experiment_id is not None, "Experiment should be created successfully"
        
        # Start experiment
        started = await extensibility_service.ab_testing_framework.start_experiment(experiment_id)
        assert started, "Experiment should start successfully"
        
        # Verify experiment is active
        experiments = await extensibility_service.ab_testing_framework.list_experiments(
            status=ExperimentStatus.ACTIVE
        )
        
        active_experiment_ids = [exp["experiment_id"] for exp in experiments]
        assert experiment_id in active_experiment_ids, "Experiment should be in active list"
        
        # Test user assignment
        user_id = uuid4()
        assignment = await extensibility_service.ab_testing_framework.assign_user_to_experiment(
            experiment_id, user_id
        )
        
        assert assignment is not None, "User should be assigned to experiment"
        assert assignment["experiment_id"] == experiment_id, "Assignment should match experiment"
        assert assignment["variant_type"] in ["control", "treatment"], "Should assign to valid variant"
    
    @given(
        interactions=st.lists(
            st.builds(
                UserInteraction,
                interaction_id=st.uuids(),
                user_id=st.uuids(),
                input_text=st.text(min_size=5, max_size=100),
                input_language=st.sampled_from(list(LanguageCode)),
                response_text=st.text(min_size=10, max_size=200),
                response_language=st.sampled_from(list(LanguageCode)),
                timestamp=st.datetimes(min_value=datetime(2024, 1, 1), max_value=datetime(2024, 12, 31)),
                intent=st.one_of(st.none(), st.text(min_size=3, max_size=30)),
                entities=st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20), max_size=3),
                confidence_score=st.floats(min_value=0.0, max_value=1.0),
                processing_time=st.floats(min_value=0.1, max_value=5.0)
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=15, deadline=8000)
    async def test_system_handles_multiple_experiments(self, extensibility_service, interactions):
        """
        Test that system can handle multiple concurrent experiments.
        """
        await extensibility_service.initialize()
        
        # Create multiple experiments
        experiment_ids = []
        for i in range(min(3, len(interactions))):
            experiment_id = await extensibility_service.create_feature_experiment(
                name=f"Test Experiment {i}",
                description=f"Testing feature {i}",
                hypothesis=f"Feature {i} improves user experience",
                feature_config_control={"feature": f"control_{i}"},
                feature_config_treatment={"feature": f"treatment_{i}"},
                primary_metric=MetricType.USER_SATISFACTION
            )
            
            if experiment_id:
                experiment_ids.append(experiment_id)
                await extensibility_service.ab_testing_framework.start_experiment(experiment_id)
        
        assume(len(experiment_ids) > 0)
        
        # Process interactions for experiments
        for interaction in interactions[:len(experiment_ids)]:
            result = await extensibility_service.process_user_interaction_for_experiments(
                user_id=interaction.user_id,
                interaction=interaction,
                response_time=interaction.processing_time,
                user_satisfaction=4.0,  # Good satisfaction
                accuracy_score=0.85
            )
            
            # Should process without errors
            assert len(result.get("errors", [])) == 0, f"Should process without errors: {result.get('errors', [])}"
            assert len(result.get("experiments_processed", [])) >= 0, "Should process experiments"
    
    async def test_system_capabilities_comprehensive(self, extensibility_service):
        """
        Test that system capabilities are reported comprehensively.
        """
        await extensibility_service.initialize()
        
        capabilities = await extensibility_service.get_system_capabilities()
        
        # Verify capability structure
        assert "model_management" in capabilities, "Should report model management capabilities"
        assert "plugin_system" in capabilities, "Should report plugin system capabilities"
        assert "ab_testing" in capabilities, "Should report A/B testing capabilities"
        assert "language_support" in capabilities, "Should report language support"
        assert "extensibility_features" in capabilities, "Should report extensibility features"
        
        # Verify model management capabilities
        model_mgmt = capabilities["model_management"]
        assert "supported_model_types" in model_mgmt, "Should list supported model types"
        assert "active_models" in model_mgmt, "Should list active models"
        assert "total_models" in model_mgmt, "Should report total models"
        
        # Verify extensibility features
        ext_features = capabilities["extensibility_features"]
        expected_features = ["model_updates", "plugin_architecture", "language_expansion", "backward_compatibility"]
        for feature in expected_features:
            assert feature in ext_features, f"Should support {feature}"
    
    async def test_health_check_comprehensive(self, extensibility_service):
        """
        Test that system health check is comprehensive.
        """
        await extensibility_service.initialize()
        
        health_check = await extensibility_service.perform_system_health_check()
        
        # Verify health check structure
        assert "overall_status" in health_check, "Should report overall status"
        assert "components" in health_check, "Should report component status"
        assert "recommendations" in health_check, "Should provide recommendations"
        assert "timestamp" in health_check, "Should include timestamp"
        
        # Verify component checks
        components = health_check["components"]
        expected_components = ["model_manager", "plugin_manager", "ab_testing"]
        for component in expected_components:
            assert component in components, f"Should check {component}"
            assert "status" in components[component], f"Should report {component} status"
            assert "issues" in components[component], f"Should report {component} issues"
        
        # Overall status should be valid
        valid_statuses = ["healthy", "warning", "unhealthy", "error"]
        assert health_check["overall_status"] in valid_statuses, \
            f"Overall status should be one of {valid_statuses}"


class SystemExtensibilityStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for system extensibility.
    """
    
    def __init__(self):
        super().__init__()
        self.extensibility_service = None
        self.temp_dir = None
        self.registered_models = {}
        self.active_experiments = []
        self.supported_languages = set()
    
    @initialize()
    def setup(self):
        """Initialize the extensibility service."""
        self.temp_dir = tempfile.mkdtemp()
        models_dir = Path(self.temp_dir) / "models"
        plugins_dir = Path(self.temp_dir) / "plugins"
        models_dir.mkdir()
        plugins_dir.mkdir()
        
        self.extensibility_service = SystemExtensibilityService(
            models_directory=str(models_dir),
            plugins_directory=str(plugins_dir),
            enable_ab_testing=True,
            auto_load_plugins=False
        )
    
    @rule(
        language=st.sampled_from(list(LanguageCode)),
        model_type=st.sampled_from(list(ModelType))
    )
    async def add_language_model(self, language, model_type):
        """Add a language model."""
        model_file = Path(self.temp_dir) / f"{language.value}_{model_type.value}.bin"
        model_file.write_text(f"dummy model for {language.value} {model_type.value}")
        
        model_files = {model_type: str(model_file)}
        
        result = await self.extensibility_service.add_language_support(
            language=language,
            model_files=model_files
        )
        
        if result["success"]:
            self.registered_models[(language, model_type)] = result["models_registered"]
            self.supported_languages.add(language)
    
    @rule(
        experiment_name=st.text(min_size=5, max_size=30)
    )
    async def create_experiment(self, experiment_name):
        """Create an A/B experiment."""
        if len(self.active_experiments) < 5:  # Limit concurrent experiments
            experiment_id = await self.extensibility_service.create_feature_experiment(
                name=experiment_name,
                description=f"Test experiment {experiment_name}",
                hypothesis="Testing hypothesis",
                feature_config_control={"enabled": False},
                feature_config_treatment={"enabled": True},
                primary_metric=MetricType.USER_SATISFACTION
            )
            
            if experiment_id:
                self.active_experiments.append(experiment_id)
    
    @rule()
    async def check_system_capabilities(self):
        """Check system capabilities."""
        capabilities = await self.extensibility_service.get_system_capabilities()
        assert capabilities is not None, "Should return capabilities"
        assert "extensibility_features" in capabilities, "Should have extensibility features"
    
    @invariant()
    def system_consistency(self):
        """Check system consistency."""
        assert self.extensibility_service is not None, "Service should exist"
        assert len(self.active_experiments) >= 0, "Experiment count should be non-negative"
        assert len(self.registered_models) >= 0, "Model count should be non-negative"
    
    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


# Test runner for stateful tests
TestSystemExtensibilityStateful = SystemExtensibilityStateMachine.TestCase


@pytest.mark.asyncio
class TestSystemExtensibilityIntegration:
    """Integration tests for system extensibility."""
    
    async def test_complete_extensibility_workflow(self):
        """
        **Property 23: System Extensibility**
        Test complete extensibility workflow with multiple components.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize service
            extensibility_service = SystemExtensibilityService(
                models_directory=str(Path(temp_dir) / "models"),
                plugins_directory=str(Path(temp_dir) / "plugins"),
                enable_ab_testing=True
            )
            
            await extensibility_service.initialize()
            
            # Add new language support
            language = LanguageCode.HINDI
            model_files = {}
            
            for model_type in [ModelType.SPEECH_RECOGNITION, ModelType.TEXT_TO_SPEECH]:
                model_file = Path(temp_dir) / f"{language.value}_{model_type.value}.bin"
                model_file.write_text(f"dummy model data for {language.value}")
                model_files[model_type] = str(model_file)
            
            lang_result = await extensibility_service.add_language_support(
                language=language,
                model_files=model_files,
                metadata={"version": "1.0", "accuracy": 0.95}
            )
            
            assert lang_result["success"], "Language addition should succeed"
            
            # Create A/B experiment for new feature
            experiment_id = await extensibility_service.create_feature_experiment(
                name="Hindi Language Feature Test",
                description="Testing Hindi language support",
                hypothesis="Hindi support improves user satisfaction",
                feature_config_control={"hindi_enabled": False},
                feature_config_treatment={"hindi_enabled": True},
                primary_metric=MetricType.USER_SATISFACTION,
                target_languages=[language]
            )
            
            assert experiment_id is not None, "Experiment should be created"
            
            # Test user interaction processing
            user_id = uuid4()
            interaction = UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="नमस्ते, आप कैसे हैं?",
                input_language=language,
                response_text="मैं ठीक हूँ, धन्यवाद!",
                response_language=language,
                timestamp=datetime.utcnow(),
                intent="greeting",
                entities={"greeting": "namaste"},
                confidence_score=0.9,
                processing_time=1.5
            )
            
            exp_result = await extensibility_service.process_user_interaction_for_experiments(
                user_id=user_id,
                interaction=interaction,
                response_time=1.5,
                user_satisfaction=4.5,
                accuracy_score=0.9
            )
            
            assert len(exp_result.get("errors", [])) == 0, "Should process without errors"
            
            # Verify system capabilities
            capabilities = await extensibility_service.get_system_capabilities()
            
            # Should support the new language
            lang_support = capabilities["language_support"]
            assert any(language in langs for langs in lang_support.values()), \
                "Should support the new language"
            
            # Perform health check
            health = await extensibility_service.perform_system_health_check()
            assert health["overall_status"] in ["healthy", "warning"], \
                "System should be healthy or have warnings only"
            
            # Get experiment results
            results = await extensibility_service.get_experiment_results(experiment_id)
            assert results is not None, "Should return experiment results"
            assert results["experiment_id"] == experiment_id, "Results should match experiment"


if __name__ == "__main__":
    # Run property-based tests
=======
"""
Property-Based Tests for System Extensibility Framework.

**Property 23: System Extensibility**
Tests that the system can be extended with new languages, models, and features
while maintaining backward compatibility.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from bharatvoice.core.models import LanguageCode, UserInteraction
from bharatvoice.services.learning import (
    SystemExtensibilityService, ModelManager, PluginManager, ABTestingFramework
)
from bharatvoice.services.learning.model_manager import ModelType, ModelStatus, ModelVersion
from bharatvoice.services.learning.plugin_manager import PluginType, PluginStatus
from bharatvoice.services.learning.ab_testing_framework import MetricType, VariantType, ExperimentStatus


# Test data generators
@st.composite
def model_metadata(draw):
    """Generate model metadata."""
    return {
        "author": draw(st.text(min_size=3, max_size=50)),
        "description": draw(st.text(min_size=10, max_size=200)),
        "training_data": draw(st.text(min_size=5, max_size=100)),
        "accuracy": draw(st.floats(min_value=0.5, max_value=1.0)),
        "model_size_mb": draw(st.integers(min_value=1, max_value=1000))
    }


@st.composite
def plugin_config(draw):
    """Generate plugin configuration."""
    return {
        "enabled": draw(st.booleans()),
        "priority": draw(st.integers(min_value=1, max_value=10)),
        "cache_size": draw(st.integers(min_value=100, max_value=10000)),
        "timeout": draw(st.floats(min_value=1.0, max_value=30.0))
    }


class TestSystemExtensibilityProperties:
    """Property-based tests for system extensibility."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def extensibility_service(self, temp_dir):
        """Create extensibility service for testing."""
        models_dir = Path(temp_dir) / "models"
        plugins_dir = Path(temp_dir) / "plugins"
        models_dir.mkdir()
        plugins_dir.mkdir()
        
        return SystemExtensibilityService(
            models_directory=str(models_dir),
            plugins_directory=str(plugins_dir),
            enable_ab_testing=True,
            auto_load_plugins=False
        )
    
    @given(
        languages=st.lists(st.sampled_from(list(LanguageCode)), min_size=1, max_size=5, unique=True),
        model_types=st.lists(st.sampled_from(list(ModelType)), min_size=1, max_size=3, unique=True),
        metadata=model_metadata()
    )
    @settings(max_examples=20, deadline=10000)
    async def test_language_support_extensibility(self, extensibility_service, temp_dir, languages, model_types, metadata):
        """
        **Property 23: System Extensibility**
        **Validates: Requirements 1.1, 1.2, 2.1**
        
        Test that new languages can be added with proper model support.
        """
        await extensibility_service.initialize()
        
        for language in languages:
            # Create dummy model files
            model_files = {}
            for model_type in model_types:
                model_file = Path(temp_dir) / f"{language.value}_{model_type.value}_model.bin"
                model_file.write_text(f"dummy model data for {language.value} {model_type.value}")
                model_files[model_type] = str(model_file)
            
            # Add language support
            result = await extensibility_service.add_language_support(
                language=language,
                model_files=model_files,
                metadata=metadata
            )
            
            # Verify language was added successfully
            assert result["success"], f"Failed to add language {language.value}: {result.get('errors', [])}"
            assert result["language"] == language.value, "Language should match"
            assert len(result["models_registered"]) == len(model_types), \
                f"Should register {len(model_types)} models, got {len(result['models_registered'])}"
        
        # Verify all languages are supported
        supported_languages = await extensibility_service.get_supported_languages()
        
        for language in languages:
            for model_type in model_types:
                assert language in supported_languages.get(model_type, []), \
                    f"Language {language.value} should be supported for {model_type.value}"
    
    @given(
        model_versions=st.lists(st.text(min_size=5, max_size=20), min_size=2, max_size=5, unique=True),
        model_type=st.sampled_from(list(ModelType)),
        language=st.sampled_from(list(LanguageCode)),
        metadata=model_metadata()
    )
    @settings(max_examples=15, deadline=8000)
    async def test_model_update_backward_compatibility(self, extensibility_service, temp_dir, model_versions, model_type, language, metadata):
        """
        Test that model updates maintain backward compatibility.
        """
        await extensibility_service.initialize()
        
        model_id = f"{model_type.value}_{language.value}"
        
        # Register initial model version
        initial_version = model_versions[0]
        initial_model_file = Path(temp_dir) / f"{model_id}_{initial_version}.bin"
        initial_model_file.write_text(f"model data v{initial_version}")
        
        await extensibility_service.model_manager.register_model(
            model_id=model_id,
            version=initial_version,
            model_type=model_type,
            language=language,
            model_path=str(initial_model_file),
            metadata=metadata,
            auto_activate=True
        )
        
        # Update to newer versions
        for new_version in model_versions[1:]:
            new_model_file = Path(temp_dir) / f"{model_id}_{new_version}.bin"
            new_model_file.write_text(f"model data v{new_version}")
            
            result = await extensibility_service.update_model(
                model_id=model_id,
                new_version=new_version,
                model_path=str(new_model_file)
            )
            
            # Update should succeed
            assert result["update_successful"], f"Model update should succeed: {result.get('errors', [])}"
            
            # Verify new version is active
            active_model = await extensibility_service.model_manager.get_active_model_by_id(model_id)
            assert active_model is not None, "Should have active model"
            assert active_model.version == new_version, f"Active version should be {new_version}"
            
            # Verify old versions are still available (backward compatibility)
            for old_version in model_versions[:model_versions.index(new_version) + 1]:
                old_model = await extensibility_service.model_manager.get_model_version(model_id, old_version)
                assert old_model is not None, f"Old version {old_version} should still be available"
    
    @given(
        experiment_name=st.text(min_size=5, max_size=50),
        hypothesis=st.text(min_size=10, max_size=200),
        primary_metric=st.sampled_from(list(MetricType)),
        duration_days=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=20, deadline=5000)
    async def test_ab_testing_framework_extensibility(self, extensibility_service, experiment_name, hypothesis, primary_metric, duration_days):
        """
        Test that A/B testing framework can be extended with new experiments.
        """
        await extensibility_service.initialize()
        
        # Create feature experiment
        control_config = {"feature_enabled": False, "algorithm": "v1"}
        treatment_config = {"feature_enabled": True, "algorithm": "v2"}
        
        experiment_id = await extensibility_service.create_feature_experiment(
            name=experiment_name,
            description=f"Testing {experiment_name}",
            hypothesis=hypothesis,
            feature_config_control=control_config,
            feature_config_treatment=treatment_config,
            primary_metric=primary_metric,
            duration_days=duration_days
        )
        
        # Experiment should be created
        assert experiment_id is not None, "Experiment should be created successfully"
        
        # Start experiment
        started = await extensibility_service.ab_testing_framework.start_experiment(experiment_id)
        assert started, "Experiment should start successfully"
        
        # Verify experiment is active
        experiments = await extensibility_service.ab_testing_framework.list_experiments(
            status=ExperimentStatus.ACTIVE
        )
        
        active_experiment_ids = [exp["experiment_id"] for exp in experiments]
        assert experiment_id in active_experiment_ids, "Experiment should be in active list"
        
        # Test user assignment
        user_id = uuid4()
        assignment = await extensibility_service.ab_testing_framework.assign_user_to_experiment(
            experiment_id, user_id
        )
        
        assert assignment is not None, "User should be assigned to experiment"
        assert assignment["experiment_id"] == experiment_id, "Assignment should match experiment"
        assert assignment["variant_type"] in ["control", "treatment"], "Should assign to valid variant"
    
    @given(
        interactions=st.lists(
            st.builds(
                UserInteraction,
                interaction_id=st.uuids(),
                user_id=st.uuids(),
                input_text=st.text(min_size=5, max_size=100),
                input_language=st.sampled_from(list(LanguageCode)),
                response_text=st.text(min_size=10, max_size=200),
                response_language=st.sampled_from(list(LanguageCode)),
                timestamp=st.datetimes(min_value=datetime(2024, 1, 1), max_value=datetime(2024, 12, 31)),
                intent=st.one_of(st.none(), st.text(min_size=3, max_size=30)),
                entities=st.dictionaries(st.text(min_size=1, max_size=10), st.text(min_size=1, max_size=20), max_size=3),
                confidence_score=st.floats(min_value=0.0, max_value=1.0),
                processing_time=st.floats(min_value=0.1, max_value=5.0)
            ),
            min_size=1, max_size=10
        )
    )
    @settings(max_examples=15, deadline=8000)
    async def test_system_handles_multiple_experiments(self, extensibility_service, interactions):
        """
        Test that system can handle multiple concurrent experiments.
        """
        await extensibility_service.initialize()
        
        # Create multiple experiments
        experiment_ids = []
        for i in range(min(3, len(interactions))):
            experiment_id = await extensibility_service.create_feature_experiment(
                name=f"Test Experiment {i}",
                description=f"Testing feature {i}",
                hypothesis=f"Feature {i} improves user experience",
                feature_config_control={"feature": f"control_{i}"},
                feature_config_treatment={"feature": f"treatment_{i}"},
                primary_metric=MetricType.USER_SATISFACTION
            )
            
            if experiment_id:
                experiment_ids.append(experiment_id)
                await extensibility_service.ab_testing_framework.start_experiment(experiment_id)
        
        assume(len(experiment_ids) > 0)
        
        # Process interactions for experiments
        for interaction in interactions[:len(experiment_ids)]:
            result = await extensibility_service.process_user_interaction_for_experiments(
                user_id=interaction.user_id,
                interaction=interaction,
                response_time=interaction.processing_time,
                user_satisfaction=4.0,  # Good satisfaction
                accuracy_score=0.85
            )
            
            # Should process without errors
            assert len(result.get("errors", [])) == 0, f"Should process without errors: {result.get('errors', [])}"
            assert len(result.get("experiments_processed", [])) >= 0, "Should process experiments"
    
    async def test_system_capabilities_comprehensive(self, extensibility_service):
        """
        Test that system capabilities are reported comprehensively.
        """
        await extensibility_service.initialize()
        
        capabilities = await extensibility_service.get_system_capabilities()
        
        # Verify capability structure
        assert "model_management" in capabilities, "Should report model management capabilities"
        assert "plugin_system" in capabilities, "Should report plugin system capabilities"
        assert "ab_testing" in capabilities, "Should report A/B testing capabilities"
        assert "language_support" in capabilities, "Should report language support"
        assert "extensibility_features" in capabilities, "Should report extensibility features"
        
        # Verify model management capabilities
        model_mgmt = capabilities["model_management"]
        assert "supported_model_types" in model_mgmt, "Should list supported model types"
        assert "active_models" in model_mgmt, "Should list active models"
        assert "total_models" in model_mgmt, "Should report total models"
        
        # Verify extensibility features
        ext_features = capabilities["extensibility_features"]
        expected_features = ["model_updates", "plugin_architecture", "language_expansion", "backward_compatibility"]
        for feature in expected_features:
            assert feature in ext_features, f"Should support {feature}"
    
    async def test_health_check_comprehensive(self, extensibility_service):
        """
        Test that system health check is comprehensive.
        """
        await extensibility_service.initialize()
        
        health_check = await extensibility_service.perform_system_health_check()
        
        # Verify health check structure
        assert "overall_status" in health_check, "Should report overall status"
        assert "components" in health_check, "Should report component status"
        assert "recommendations" in health_check, "Should provide recommendations"
        assert "timestamp" in health_check, "Should include timestamp"
        
        # Verify component checks
        components = health_check["components"]
        expected_components = ["model_manager", "plugin_manager", "ab_testing"]
        for component in expected_components:
            assert component in components, f"Should check {component}"
            assert "status" in components[component], f"Should report {component} status"
            assert "issues" in components[component], f"Should report {component} issues"
        
        # Overall status should be valid
        valid_statuses = ["healthy", "warning", "unhealthy", "error"]
        assert health_check["overall_status"] in valid_statuses, \
            f"Overall status should be one of {valid_statuses}"


class SystemExtensibilityStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for system extensibility.
    """
    
    def __init__(self):
        super().__init__()
        self.extensibility_service = None
        self.temp_dir = None
        self.registered_models = {}
        self.active_experiments = []
        self.supported_languages = set()
    
    @initialize()
    def setup(self):
        """Initialize the extensibility service."""
        self.temp_dir = tempfile.mkdtemp()
        models_dir = Path(self.temp_dir) / "models"
        plugins_dir = Path(self.temp_dir) / "plugins"
        models_dir.mkdir()
        plugins_dir.mkdir()
        
        self.extensibility_service = SystemExtensibilityService(
            models_directory=str(models_dir),
            plugins_directory=str(plugins_dir),
            enable_ab_testing=True,
            auto_load_plugins=False
        )
    
    @rule(
        language=st.sampled_from(list(LanguageCode)),
        model_type=st.sampled_from(list(ModelType))
    )
    async def add_language_model(self, language, model_type):
        """Add a language model."""
        model_file = Path(self.temp_dir) / f"{language.value}_{model_type.value}.bin"
        model_file.write_text(f"dummy model for {language.value} {model_type.value}")
        
        model_files = {model_type: str(model_file)}
        
        result = await self.extensibility_service.add_language_support(
            language=language,
            model_files=model_files
        )
        
        if result["success"]:
            self.registered_models[(language, model_type)] = result["models_registered"]
            self.supported_languages.add(language)
    
    @rule(
        experiment_name=st.text(min_size=5, max_size=30)
    )
    async def create_experiment(self, experiment_name):
        """Create an A/B experiment."""
        if len(self.active_experiments) < 5:  # Limit concurrent experiments
            experiment_id = await self.extensibility_service.create_feature_experiment(
                name=experiment_name,
                description=f"Test experiment {experiment_name}",
                hypothesis="Testing hypothesis",
                feature_config_control={"enabled": False},
                feature_config_treatment={"enabled": True},
                primary_metric=MetricType.USER_SATISFACTION
            )
            
            if experiment_id:
                self.active_experiments.append(experiment_id)
    
    @rule()
    async def check_system_capabilities(self):
        """Check system capabilities."""
        capabilities = await self.extensibility_service.get_system_capabilities()
        assert capabilities is not None, "Should return capabilities"
        assert "extensibility_features" in capabilities, "Should have extensibility features"
    
    @invariant()
    def system_consistency(self):
        """Check system consistency."""
        assert self.extensibility_service is not None, "Service should exist"
        assert len(self.active_experiments) >= 0, "Experiment count should be non-negative"
        assert len(self.registered_models) >= 0, "Model count should be non-negative"
    
    def teardown(self):
        """Clean up temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


# Test runner for stateful tests
TestSystemExtensibilityStateful = SystemExtensibilityStateMachine.TestCase


@pytest.mark.asyncio
class TestSystemExtensibilityIntegration:
    """Integration tests for system extensibility."""
    
    async def test_complete_extensibility_workflow(self):
        """
        **Property 23: System Extensibility**
        Test complete extensibility workflow with multiple components.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize service
            extensibility_service = SystemExtensibilityService(
                models_directory=str(Path(temp_dir) / "models"),
                plugins_directory=str(Path(temp_dir) / "plugins"),
                enable_ab_testing=True
            )
            
            await extensibility_service.initialize()
            
            # Add new language support
            language = LanguageCode.HINDI
            model_files = {}
            
            for model_type in [ModelType.SPEECH_RECOGNITION, ModelType.TEXT_TO_SPEECH]:
                model_file = Path(temp_dir) / f"{language.value}_{model_type.value}.bin"
                model_file.write_text(f"dummy model data for {language.value}")
                model_files[model_type] = str(model_file)
            
            lang_result = await extensibility_service.add_language_support(
                language=language,
                model_files=model_files,
                metadata={"version": "1.0", "accuracy": 0.95}
            )
            
            assert lang_result["success"], "Language addition should succeed"
            
            # Create A/B experiment for new feature
            experiment_id = await extensibility_service.create_feature_experiment(
                name="Hindi Language Feature Test",
                description="Testing Hindi language support",
                hypothesis="Hindi support improves user satisfaction",
                feature_config_control={"hindi_enabled": False},
                feature_config_treatment={"hindi_enabled": True},
                primary_metric=MetricType.USER_SATISFACTION,
                target_languages=[language]
            )
            
            assert experiment_id is not None, "Experiment should be created"
            
            # Test user interaction processing
            user_id = uuid4()
            interaction = UserInteraction(
                interaction_id=uuid4(),
                user_id=user_id,
                input_text="नमस्ते, आप कैसे हैं?",
                input_language=language,
                response_text="मैं ठीक हूँ, धन्यवाद!",
                response_language=language,
                timestamp=datetime.utcnow(),
                intent="greeting",
                entities={"greeting": "namaste"},
                confidence_score=0.9,
                processing_time=1.5
            )
            
            exp_result = await extensibility_service.process_user_interaction_for_experiments(
                user_id=user_id,
                interaction=interaction,
                response_time=1.5,
                user_satisfaction=4.5,
                accuracy_score=0.9
            )
            
            assert len(exp_result.get("errors", [])) == 0, "Should process without errors"
            
            # Verify system capabilities
            capabilities = await extensibility_service.get_system_capabilities()
            
            # Should support the new language
            lang_support = capabilities["language_support"]
            assert any(language in langs for langs in lang_support.values()), \
                "Should support the new language"
            
            # Perform health check
            health = await extensibility_service.perform_system_health_check()
            assert health["overall_status"] in ["healthy", "warning"], \
                "System should be healthy or have warnings only"
            
            # Get experiment results
            results = await extensibility_service.get_experiment_results(experiment_id)
            assert results is not None, "Should return experiment results"
            assert results["experiment_id"] == experiment_id, "Results should match experiment"


if __name__ == "__main__":
    # Run property-based tests
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__, "-v", "--tb=short"])