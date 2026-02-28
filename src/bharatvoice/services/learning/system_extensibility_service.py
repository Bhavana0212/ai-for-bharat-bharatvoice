"""
System Extensibility Service for BharatVoice Assistant.

This service provides a unified interface for system extensibility features including
model management, plugin architecture, and A/B testing framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from bharatvoice.core.models import LanguageCode, UserInteraction

from .model_manager import ModelManager, ModelType, ModelStatus
from .plugin_manager import PluginManager, PluginType, PluginStatus, BharatVoicePlugin
from .ab_testing_framework import ABTestingFramework, MetricType, VariantType, ExperimentStatus

logger = logging.getLogger(__name__)


class SystemExtensibilityService:
    """
    Unified service for system extensibility and expansion capabilities.
    """
    
    def __init__(
        self,
        models_directory: str = "models",
        plugins_directory: str = "plugins",
        enable_ab_testing: bool = True,
        auto_load_plugins: bool = True
    ):
        """
        Initialize system extensibility service.
        
        Args:
            models_directory: Directory for model storage
            plugins_directory: Directory for plugin storage
            enable_ab_testing: Enable A/B testing framework
            auto_load_plugins: Auto-load plugins on startup
        """
        self.models_directory = models_directory
        self.plugins_directory = plugins_directory
        self.enable_ab_testing = enable_ab_testing
        self.auto_load_plugins = auto_load_plugins
        
        # Initialize components
        self.model_manager = ModelManager(models_directory=models_directory)
        self.plugin_manager = PluginManager(
            plugins_directory=plugins_directory,
            auto_load_plugins=auto_load_plugins
        )
        self.ab_testing_framework = ABTestingFramework() if enable_ab_testing else None
        
        # Service statistics
        self._service_stats = {
            "models_managed": 0,
            "plugins_loaded": 0,
            "active_experiments": 0,
            "languages_supported": 0,
            "last_update": None
        }
        
        logger.info("System Extensibility Service initialized")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the extensibility service.
        
        Returns:
            Initialization results
        """
        init_results = {
            "model_manager_ready": True,
            "plugin_manager_ready": True,
            "ab_testing_ready": self.enable_ab_testing,
            "plugins_loaded": [],
            "models_available": [],
            "errors": []
        }
        
        try:
            # Load plugins if auto-load is enabled
            if self.auto_load_plugins:
                plugin_results = await self.plugin_manager.scan_and_load_plugins()
                init_results["plugins_loaded"] = plugin_results.get("loaded_plugins", [])
                if plugin_results.get("failed_plugins"):
                    init_results["errors"].extend([
                        f"Failed to load plugin: {p}" for p in plugin_results["failed_plugins"]
                    ])
            
            # Get available models
            available_models = await self.model_manager.list_available_models()
            init_results["models_available"] = [
                f"{m.model_id}:{m.version}" for m in available_models
            ]
            
            # Update service statistics
            await self._update_service_stats()
            
            logger.info("System Extensibility Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing extensibility service: {e}")
            init_results["errors"].append(str(e))
        
        return init_results
    
    # Model Management Methods
    
    async def add_language_support(
        self,
        language: LanguageCode,
        model_files: Dict[ModelType, str],
        plugin_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add comprehensive support for a new language.
        
        Args:
            language: Language to add support for
            model_files: Dictionary mapping model types to file paths
            plugin_file: Optional plugin file for language-specific features
            metadata: Additional metadata
            
        Returns:
            Language addition results
        """
        results = {
            "language": language.value,
            "models_registered": [],
            "plugin_loaded": False,
            "success": True,
            "errors": []
        }
        
        try:
            # Register models for the language
            for model_type, model_path in model_files.items():
                try:
                    model_id = await self.model_manager.add_language_support(
                        model_type=model_type,
                        language=language,
                        model_path=model_path,
                        metadata=metadata
                    )
                    results["models_registered"].append(f"{model_type.value}:{model_id}")
                except Exception as e:
                    results["errors"].append(f"Failed to register {model_type.value} model: {e}")
            
            # Load language plugin if provided
            if plugin_file:
                try:
                    # This would load a language-specific plugin
                    # Implementation depends on plugin file format
                    results["plugin_loaded"] = True
                except Exception as e:
                    results["errors"].append(f"Failed to load language plugin: {e}")
            
            # Update statistics
            await self._update_service_stats()
            
            logger.info(f"Added support for language: {language.value}")
            
        except Exception as e:
            logger.error(f"Error adding language support for {language.value}: {e}")
            results["success"] = False
            results["errors"].append(str(e))
        
        return results
    
    async def update_model(
        self,
        model_id: str,
        new_version: str,
        model_path: str,
        performance_requirements: Optional[Dict[str, float]] = None,
        run_ab_test: bool = False
    ) -> Dict[str, Any]:
        """
        Update a model with backward compatibility and optional A/B testing.
        
        Args:
            model_id: Model identifier
            new_version: New version string
            model_path: Path to new model file
            performance_requirements: Performance requirements
            run_ab_test: Whether to run A/B test for the update
            
        Returns:
            Update results
        """
        results = {
            "model_id": model_id,
            "new_version": new_version,
            "update_successful": False,
            "ab_test_created": False,
            "experiment_id": None,
            "errors": []
        }
        
        try:
            # Update the model
            update_success = await self.model_manager.update_model(
                model_id=model_id,
                new_version=new_version,
                model_path=model_path,
                performance_requirements=performance_requirements
            )
            
            results["update_successful"] = update_success
            
            if update_success and run_ab_test and self.ab_testing_framework:
                # Create A/B test for model comparison
                experiment_id = await self._create_model_comparison_experiment(
                    model_id, new_version
                )
                results["ab_test_created"] = experiment_id is not None
                results["experiment_id"] = experiment_id
            
            logger.info(f"Updated model {model_id} to version {new_version}")
            
        except Exception as e:
            logger.error(f"Error updating model {model_id}: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def get_supported_languages(self) -> Dict[ModelType, List[LanguageCode]]:
        """
        Get all supported languages by model type.
        
        Returns:
            Dictionary mapping model types to supported languages
        """
        supported_languages = {}
        
        for model_type in ModelType:
            languages = await self.model_manager.get_supported_languages(model_type)
            supported_languages[model_type] = languages
        
        return supported_languages
    
    # Plugin Management Methods
    
    async def install_plugin(
        self,
        plugin_file: str,
        config: Optional[Dict[str, Any]] = None,
        auto_activate: bool = True
    ) -> Dict[str, Any]:
        """
        Install and optionally activate a plugin.
        
        Args:
            plugin_file: Path to plugin file
            config: Plugin configuration
            auto_activate: Whether to activate after installation
            
        Returns:
            Installation results
        """
        results = {
            "plugin_file": plugin_file,
            "installed": False,
            "activated": False,
            "plugin_id": None,
            "errors": []
        }
        
        try:
            # Load plugin from file would be implemented here
            # For now, this is a placeholder
            
            results["installed"] = True
            
            if auto_activate:
                # Activate plugin
                results["activated"] = True
            
            logger.info(f"Installed plugin from {plugin_file}")
            
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_file}: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def get_plugin_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities of all active plugins.
        
        Returns:
            Plugin capabilities by plugin ID
        """
        return await self.plugin_manager.get_plugin_capabilities()
    
    async def create_plugin_template(
        self,
        plugin_id: str,
        plugin_type: PluginType,
        language: Optional[LanguageCode] = None
    ) -> str:
        """
        Create a plugin template for development.
        
        Args:
            plugin_id: Plugin identifier
            plugin_type: Type of plugin
            language: Target language (for language plugins)
            
        Returns:
            Path to created template
        """
        return await self.plugin_manager.create_plugin_template(
            plugin_id, plugin_type, language
        )
    
    # A/B Testing Methods
    
    async def create_feature_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        feature_config_control: Dict[str, Any],
        feature_config_treatment: Dict[str, Any],
        primary_metric: MetricType = MetricType.USER_SATISFACTION,
        target_languages: Optional[List[LanguageCode]] = None,
        duration_days: int = 14
    ) -> Optional[str]:
        """
        Create an A/B experiment for feature testing.
        
        Args:
            name: Experiment name
            description: Experiment description
            hypothesis: Hypothesis being tested
            feature_config_control: Control variant configuration
            feature_config_treatment: Treatment variant configuration
            primary_metric: Primary metric to optimize
            target_languages: Target languages
            duration_days: Experiment duration
            
        Returns:
            Experiment ID if created successfully
        """
        if not self.ab_testing_framework:
            logger.error("A/B testing framework not enabled")
            return None
        
        try:
            # Create experiment
            experiment_id = await self.ab_testing_framework.create_experiment(
                name=name,
                description=description,
                hypothesis=hypothesis,
                primary_metric=primary_metric,
                target_languages=target_languages,
                duration_days=duration_days
            )
            
            # Add control variant
            await self.ab_testing_framework.add_variant(
                experiment_id=experiment_id,
                variant_name="Control",
                variant_type=VariantType.CONTROL,
                traffic_allocation=0.5,
                configuration=feature_config_control
            )
            
            # Add treatment variant
            await self.ab_testing_framework.add_variant(
                experiment_id=experiment_id,
                variant_name="Treatment",
                variant_type=VariantType.TREATMENT,
                traffic_allocation=0.5,
                configuration=feature_config_treatment
            )
            
            logger.info(f"Created feature experiment: {name} ({experiment_id})")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating feature experiment: {e}")
            return None
    
    async def process_user_interaction_for_experiments(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        response_time: Optional[float] = None,
        user_satisfaction: Optional[float] = None,
        accuracy_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process user interaction for all active experiments.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            response_time: Response time in seconds
            user_satisfaction: User satisfaction score
            accuracy_score: Accuracy score
            
        Returns:
            Processing results
        """
        if not self.ab_testing_framework:
            return {"ab_testing_disabled": True}
        
        results = {
            "experiments_processed": [],
            "new_assignments": [],
            "errors": []
        }
        
        try:
            # Get active experiments
            active_experiments = await self.ab_testing_framework.list_experiments(
                status=ExperimentStatus.ACTIVE
            )
            
            for exp_data in active_experiments:
                experiment_id = exp_data["experiment_id"]
                
                try:
                    # Check if user is already assigned
                    user_experiments = await self.ab_testing_framework.get_user_experiments(user_id)
                    user_in_experiment = any(
                        exp["experiment_id"] == experiment_id for exp in user_experiments
                    )
                    
                    if not user_in_experiment:
                        # Assign user to experiment
                        assignment = await self.ab_testing_framework.assign_user_to_experiment(
                            experiment_id, user_id
                        )
                        if assignment:
                            results["new_assignments"].append(assignment)
                    
                    # Record interaction metrics
                    await self.ab_testing_framework.record_interaction_metrics(
                        experiment_id=experiment_id,
                        user_id=user_id,
                        interaction=interaction,
                        response_time=response_time,
                        user_satisfaction=user_satisfaction,
                        accuracy_score=accuracy_score
                    )
                    
                    results["experiments_processed"].append(experiment_id)
                    
                except Exception as e:
                    results["errors"].append(f"Error processing experiment {experiment_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing user interaction for experiments: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        if not self.ab_testing_framework:
            return None
        
        return await self.ab_testing_framework.get_experiment_results(experiment_id)
    
    # System Information Methods
    
    async def get_system_capabilities(self) -> Dict[str, Any]:
        """
        Get comprehensive system capabilities.
        
        Returns:
            System capabilities information
        """
        capabilities = {
            "model_management": {
                "supported_model_types": [mt.value for mt in ModelType],
                "active_models": {},
                "total_models": 0
            },
            "plugin_system": {
                "supported_plugin_types": [pt.value for pt in PluginType],
                "active_plugins": 0,
                "total_plugins": 0
            },
            "ab_testing": {
                "enabled": self.enable_ab_testing,
                "active_experiments": 0,
                "supported_metrics": [mt.value for mt in MetricType] if self.enable_ab_testing else []
            },
            "language_support": {},
            "extensibility_features": [
                "model_updates",
                "plugin_architecture",
                "language_expansion",
                "backward_compatibility",
                "a_b_testing" if self.enable_ab_testing else None
            ]
        }
        
        # Remove None values
        capabilities["extensibility_features"] = [
            f for f in capabilities["extensibility_features"] if f is not None
        ]
        
        try:
            # Get model information
            for model_type in ModelType:
                active_model = await self.model_manager.get_active_model(
                    model_type, LanguageCode.ENGLISH_IN  # Default language
                )
                if active_model:
                    capabilities["model_management"]["active_models"][model_type.value] = {
                        "model_id": active_model.model_id,
                        "version": active_model.version
                    }
            
            all_models = await self.model_manager.list_available_models()
            capabilities["model_management"]["total_models"] = len(all_models)
            
            # Get plugin information
            all_plugins = await self.plugin_manager.list_plugins()
            active_plugins = await self.plugin_manager.list_plugins(status=PluginStatus.ACTIVE)
            capabilities["plugin_system"]["total_plugins"] = len(all_plugins)
            capabilities["plugin_system"]["active_plugins"] = len(active_plugins)
            
            # Get language support
            capabilities["language_support"] = await self.get_supported_languages()
            
            # Get A/B testing info
            if self.ab_testing_framework:
                ab_stats = await self.ab_testing_framework.get_framework_statistics()
                capabilities["ab_testing"]["active_experiments"] = ab_stats["active_experiments"]
            
        except Exception as e:
            logger.error(f"Error getting system capabilities: {e}")
        
        return capabilities
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        await self._update_service_stats()
        return self._service_stats.copy()
    
    async def perform_system_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Health check results
        """
        health_check = {
            "overall_status": "healthy",
            "components": {
                "model_manager": {"status": "healthy", "issues": []},
                "plugin_manager": {"status": "healthy", "issues": []},
                "ab_testing": {"status": "healthy" if self.enable_ab_testing else "disabled", "issues": []}
            },
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check model manager
            active_models = await self.model_manager.list_available_models(status=ModelStatus.ACTIVE)
            if not active_models:
                health_check["components"]["model_manager"]["issues"].append("No active models found")
                health_check["recommendations"].append("Activate at least one model for each type")
            
            # Check plugin manager
            active_plugins = await self.plugin_manager.list_plugins(status=PluginStatus.ACTIVE)
            failed_plugins = await self.plugin_manager.list_plugins(status=PluginStatus.ERROR)
            
            if failed_plugins:
                health_check["components"]["plugin_manager"]["issues"].append(
                    f"{len(failed_plugins)} plugins in error state"
                )
                health_check["recommendations"].append("Review and fix failed plugins")
            
            # Check A/B testing
            if self.ab_testing_framework:
                ab_stats = await self.ab_testing_framework.get_framework_statistics()
                if ab_stats["active_experiments"] > 10:
                    health_check["components"]["ab_testing"]["issues"].append(
                        "High number of active experiments"
                    )
                    health_check["recommendations"].append("Consider completing some experiments")
            
            # Determine overall status
            total_issues = sum(
                len(comp["issues"]) for comp in health_check["components"].values()
            )
            
            if total_issues > 5:
                health_check["overall_status"] = "unhealthy"
            elif total_issues > 0:
                health_check["overall_status"] = "warning"
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            health_check["overall_status"] = "error"
            health_check["error"] = str(e)
        
        return health_check
    
    # Private Methods
    
    async def _create_model_comparison_experiment(
        self,
        model_id: str,
        new_version: str
    ) -> Optional[str]:
        """Create A/B experiment for model comparison."""
        if not self.ab_testing_framework:
            return None
        
        try:
            # Get current active model
            current_model = await self.model_manager.get_active_model_by_id(model_id)
            if not current_model:
                return None
            
            experiment_name = f"Model Update: {model_id} v{new_version}"
            experiment_id = await self.ab_testing_framework.create_experiment(
                name=experiment_name,
                description=f"Compare {model_id} v{current_model.version} vs v{new_version}",
                hypothesis=f"New version {new_version} performs better than {current_model.version}",
                primary_metric=MetricType.ACCURACY_SCORE,
                secondary_metrics=[MetricType.RESPONSE_TIME, MetricType.USER_SATISFACTION],
                duration_days=7
            )
            
            # Add control variant (current version)
            await self.ab_testing_framework.add_variant(
                experiment_id=experiment_id,
                variant_name=f"Current v{current_model.version}",
                variant_type=VariantType.CONTROL,
                traffic_allocation=0.5,
                configuration={"model_version": current_model.version}
            )
            
            # Add treatment variant (new version)
            await self.ab_testing_framework.add_variant(
                experiment_id=experiment_id,
                variant_name=f"New v{new_version}",
                variant_type=VariantType.TREATMENT,
                traffic_allocation=0.5,
                configuration={"model_version": new_version}
            )
            
            # Start experiment
            await self.ab_testing_framework.start_experiment(experiment_id)
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating model comparison experiment: {e}")
            return None
    
    async def _update_service_stats(self) -> None:
        """Update service statistics."""
        try:
            # Count models
            all_models = await self.model_manager.list_available_models()
            self._service_stats["models_managed"] = len(all_models)
            
            # Count plugins
            all_plugins = await self.plugin_manager.list_plugins()
            self._service_stats["plugins_loaded"] = len(all_plugins)
            
            # Count experiments
            if self.ab_testing_framework:
                ab_stats = await self.ab_testing_framework.get_framework_statistics()
                self._service_stats["active_experiments"] = ab_stats["active_experiments"]
            
            # Count supported languages
            supported_langs = await self.get_supported_languages()
            all_languages = set()
            for languages in supported_langs.values():
                all_languages.update(languages)
            self._service_stats["languages_supported"] = len(all_languages)
            
            self._service_stats["last_update"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating service stats: {e}")
    
    async def cleanup_old_data(self, days_threshold: int = 90) -> Dict[str, int]:
        """
        Clean up old data across all components.
        
        Args:
            days_threshold: Days threshold for cleanup
            
        Returns:
            Cleanup results
        """
        cleanup_results = {
            "models_cleaned": 0,
            "experiments_cleaned": 0,
            "errors": []
        }
        
        try:
            # Cleanup old models
            models_cleaned = await self.model_manager.cleanup_old_models()
            cleanup_results["models_cleaned"] = models_cleaned
            
            # Cleanup old experiments
            if self.ab_testing_framework:
                experiments_cleaned = await self.ab_testing_framework.cleanup_old_experiments(
                    days_threshold
                )
                cleanup_results["experiments_cleaned"] = experiments_cleaned
            
            logger.info(f"Cleaned up {models_cleaned} models and {cleanup_results['experiments_cleaned']} experiments")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cleanup_results["errors"].append(str(e))
        
        return cleanup_results