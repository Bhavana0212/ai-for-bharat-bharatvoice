<<<<<<< HEAD
"""
Model Management Module for BharatVoice Assistant.

This module handles model updates, expansion capabilities, and version management
for the extensible learning system.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from pathlib import Path
from enum import Enum

from bharatvoice.core.models import LanguageCode

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of models that can be managed."""
    
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    LANGUAGE_MODEL = "language_model"
    TRANSLATION = "translation"
    ACCENT_ADAPTATION = "accent_adaptation"
    VOCABULARY_EXPANSION = "vocabulary_expansion"
    STYLE_ADAPTATION = "style_adaptation"


class ModelStatus(str, Enum):
    """Model status states."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    UPDATING = "updating"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelVersion:
    """Represents a model version with metadata."""
    
    def __init__(
        self,
        model_id: str,
        version: str,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.version = version
        self.model_type = model_type
        self.language = language
        self.model_path = model_path
        self.metadata = metadata or {}
        self.status = ModelStatus.INACTIVE
        self.created_at = datetime.utcnow()
        self.activated_at: Optional[datetime] = None
        self.performance_metrics: Dict[str, float] = {}
        self.compatibility_info: Dict[str, Any] = {}
        self.checksum = ""
        
    def calculate_checksum(self) -> str:
        """Calculate model file checksum."""
        if Path(self.model_path).exists():
            with open(self.model_path, 'rb') as f:
                content = f.read()
                self.checksum = hashlib.sha256(content).hexdigest()
        return self.checksum
    
    def is_compatible_with(self, other_version: 'ModelVersion') -> bool:
        """Check compatibility with another model version."""
        if self.model_type != other_version.model_type:
            return False
        
        if self.language != other_version.language:
            return False
        
        # Check version compatibility
        self_major = int(self.version.split('.')[0])
        other_major = int(other_version.version.split('.')[0])
        
        return self_major == other_major
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type.value,
            "language": self.language.value,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "performance_metrics": self.performance_metrics,
            "compatibility_info": self.compatibility_info,
            "checksum": self.checksum
        }


class ModelManager:
    """
    Manages model updates, expansion, and version control for the learning system.
    """
    
    def __init__(
        self,
        models_directory: str = "models",
        max_versions_per_model: int = 5,
        auto_cleanup_days: int = 30
    ):
        """
        Initialize model manager.
        
        Args:
            models_directory: Directory to store models
            max_versions_per_model: Maximum versions to keep per model
            auto_cleanup_days: Days after which to cleanup old versions
        """
        self.models_directory = Path(models_directory)
        self.max_versions_per_model = max_versions_per_model
        self.auto_cleanup_days = auto_cleanup_days
        
        # Ensure models directory exists
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self._model_registry: Dict[str, List[ModelVersion]] = {}
        self._active_models: Dict[Tuple[ModelType, LanguageCode], ModelVersion] = {}
        
        # Model update queue
        self._update_queue: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Model Manager initialized")
    
    async def register_model(
        self,
        model_id: str,
        version: str,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_activate: bool = False
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_id: Unique model identifier
            version: Model version string
            model_type: Type of model
            language: Model language
            model_path: Path to model file
            metadata: Additional metadata
            auto_activate: Whether to activate immediately
            
        Returns:
            Registered model version
        """
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            language=language,
            model_path=model_path,
            metadata=metadata
        )
        
        # Calculate checksum
        model_version.calculate_checksum()
        
        # Add to registry
        if model_id not in self._model_registry:
            self._model_registry[model_id] = []
        
        self._model_registry[model_id].append(model_version)
        
        # Sort versions by creation date
        self._model_registry[model_id].sort(key=lambda x: x.created_at, reverse=True)
        
        # Cleanup old versions if needed
        if len(self._model_registry[model_id]) > self.max_versions_per_model:
            old_versions = self._model_registry[model_id][self.max_versions_per_model:]
            for old_version in old_versions:
                await self._cleanup_model_version(old_version)
            self._model_registry[model_id] = self._model_registry[model_id][:self.max_versions_per_model]
        
        # Auto-activate if requested
        if auto_activate:
            await self.activate_model(model_id, version)
        
        logger.info(f"Registered model {model_id} version {version}")
        return model_version
    
    async def activate_model(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """
        Activate a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version to activate
            
        Returns:
            True if activation successful
        """
        model_version = await self.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model version {model_id}:{version} not found")
            return False
        
        # Check if model file exists
        if not Path(model_version.model_path).exists():
            logger.error(f"Model file not found: {model_version.model_path}")
            return False
        
        # Verify checksum
        current_checksum = model_version.calculate_checksum()
        if model_version.checksum and current_checksum != model_version.checksum:
            logger.error(f"Model checksum mismatch for {model_id}:{version}")
            return False
        
        # Deactivate current active model of same type/language
        model_key = (model_version.model_type, model_version.language)
        if model_key in self._active_models:
            old_model = self._active_models[model_key]
            old_model.status = ModelStatus.INACTIVE
            logger.info(f"Deactivated model {old_model.model_id}:{old_model.version}")
        
        # Activate new model
        model_version.status = ModelStatus.ACTIVE
        model_version.activated_at = datetime.utcnow()
        self._active_models[model_key] = model_version
        
        logger.info(f"Activated model {model_id}:{version}")
        return True
    
    async def update_model(
        self,
        model_id: str,
        new_version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Update a model to a new version with backward compatibility checks.
        
        Args:
            model_id: Model identifier
            new_version: New version string
            model_path: Path to new model file
            metadata: Updated metadata
            performance_requirements: Minimum performance requirements
            
        Returns:
            True if update successful
        """
        # Get current active model
        current_model = await self.get_active_model_by_id(model_id)
        if not current_model:
            logger.error(f"No active model found for {model_id}")
            return False
        
        # Register new version
        new_model = await self.register_model(
            model_id=model_id,
            version=new_version,
            model_type=current_model.model_type,
            language=current_model.language,
            model_path=model_path,
            metadata=metadata
        )
        
        # Check backward compatibility
        if not new_model.is_compatible_with(current_model):
            logger.warning(f"New model version {new_version} may not be backward compatible")
        
        # Test performance if requirements specified
        if performance_requirements:
            performance_ok = await self._test_model_performance(new_model, performance_requirements)
            if not performance_ok:
                logger.error(f"New model version {new_version} does not meet performance requirements")
                return False
        
        # Activate new version
        success = await self.activate_model(model_id, new_version)
        if success:
            logger.info(f"Successfully updated model {model_id} to version {new_version}")
        
        return success
    
    async def add_language_support(
        self,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        base_model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add support for a new language.
        
        Args:
            model_type: Type of model
            language: New language to support
            model_path: Path to language model
            base_model_id: Base model to extend from
            metadata: Language-specific metadata
            
        Returns:
            New model ID for the language
        """
        # Generate new model ID for language
        model_id = f"{model_type.value}_{language.value}"
        
        # Copy metadata from base model if specified
        if base_model_id and metadata is None:
            base_model = await self.get_active_model_by_id(base_model_id)
            if base_model:
                metadata = base_model.metadata.copy()
                metadata["base_model"] = base_model_id
        
        # Register new language model
        model_version = await self.register_model(
            model_id=model_id,
            version="1.0.0",
            model_type=model_type,
            language=language,
            model_path=model_path,
            metadata=metadata,
            auto_activate=True
        )
        
        logger.info(f"Added {language.value} support for {model_type.value}")
        return model_id
    
    async def get_active_model(
        self,
        model_type: ModelType,
        language: LanguageCode
    ) -> Optional[ModelVersion]:
        """
        Get active model for type and language.
        
        Args:
            model_type: Model type
            language: Language
            
        Returns:
            Active model version if available
        """
        model_key = (model_type, language)
        return self._active_models.get(model_key)
    
    async def get_active_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Get active model by ID."""
        for model_version in self._active_models.values():
            if model_version.model_id == model_id:
                return model_version
        return None
    
    async def get_model_version(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersion]:
        """
        Get specific model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            
        Returns:
            Model version if found
        """
        if model_id not in self._model_registry:
            return None
        
        for model_version in self._model_registry[model_id]:
            if model_version.version == version:
                return model_version
        
        return None
    
    async def list_available_models(
        self,
        model_type: Optional[ModelType] = None,
        language: Optional[LanguageCode] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelVersion]:
        """
        List available models with optional filters.
        
        Args:
            model_type: Filter by model type
            language: Filter by language
            status: Filter by status
            
        Returns:
            List of matching model versions
        """
        models = []
        
        for model_versions in self._model_registry.values():
            for model_version in model_versions:
                # Apply filters
                if model_type and model_version.model_type != model_type:
                    continue
                if language and model_version.language != language:
                    continue
                if status and model_version.status != status:
                    continue
                
                models.append(model_version)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    async def get_supported_languages(
        self,
        model_type: ModelType
    ) -> List[LanguageCode]:
        """
        Get list of supported languages for a model type.
        
        Args:
            model_type: Model type
            
        Returns:
            List of supported languages
        """
        languages = set()
        
        for model_key, model_version in self._active_models.items():
            if model_key[0] == model_type:
                languages.add(model_key[1])
        
        return list(languages)
    
    async def record_performance_metrics(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Record performance metrics for a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            metrics: Performance metrics
        """
        model_version = await self.get_model_version(model_id, version)
        if model_version:
            model_version.performance_metrics.update(metrics)
            
            # Add to performance history
            history_key = f"{model_id}:{version}"
            if history_key not in self._performance_history:
                self._performance_history[history_key] = []
            
            self._performance_history[history_key].append({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics.copy()
            })
            
            logger.info(f"Recorded performance metrics for {model_id}:{version}")
    
    async def get_model_performance_history(
        self,
        model_id: str,
        version: str
    ) -> List[Dict[str, Any]]:
        """Get performance history for a model version."""
        history_key = f"{model_id}:{version}"
        return self._performance_history.get(history_key, [])
    
    async def rollback_model(
        self,
        model_id: str,
        target_version: Optional[str] = None
    ) -> bool:
        """
        Rollback model to previous or specified version.
        
        Args:
            model_id: Model identifier
            target_version: Specific version to rollback to (optional)
            
        Returns:
            True if rollback successful
        """
        if model_id not in self._model_registry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        versions = self._model_registry[model_id]
        
        if target_version:
            # Rollback to specific version
            target_model = await self.get_model_version(model_id, target_version)
            if not target_model:
                logger.error(f"Target version {target_version} not found")
                return False
        else:
            # Rollback to previous version
            if len(versions) < 2:
                logger.error(f"No previous version available for {model_id}")
                return False
            target_model = versions[1]  # Second most recent
        
        # Activate target version
        success = await self.activate_model(model_id, target_model.version)
        if success:
            logger.info(f"Rolled back model {model_id} to version {target_model.version}")
        
        return success
    
    async def export_model_registry(self) -> Dict[str, Any]:
        """Export model registry for backup or migration."""
        registry_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "models": {},
            "active_models": {}
        }
        
        # Export model registry
        for model_id, versions in self._model_registry.items():
            registry_data["models"][model_id] = [
                version.to_dict() for version in versions
            ]
        
        # Export active models
        for (model_type, language), model_version in self._active_models.items():
            key = f"{model_type.value}_{language.value}"
            registry_data["active_models"][key] = {
                "model_id": model_version.model_id,
                "version": model_version.version
            }
        
        return registry_data
    
    async def import_model_registry(self, registry_data: Dict[str, Any]) -> bool:
        """Import model registry from backup or migration."""
        try:
            # Import models
            for model_id, versions_data in registry_data.get("models", {}).items():
                self._model_registry[model_id] = []
                for version_data in versions_data:
                    model_version = ModelVersion(
                        model_id=version_data["model_id"],
                        version=version_data["version"],
                        model_type=ModelType(version_data["model_type"]),
                        language=LanguageCode(version_data["language"]),
                        model_path=version_data["model_path"],
                        metadata=version_data.get("metadata", {})
                    )
                    model_version.status = ModelStatus(version_data["status"])
                    model_version.created_at = datetime.fromisoformat(version_data["created_at"])
                    if version_data.get("activated_at"):
                        model_version.activated_at = datetime.fromisoformat(version_data["activated_at"])
                    model_version.performance_metrics = version_data.get("performance_metrics", {})
                    model_version.compatibility_info = version_data.get("compatibility_info", {})
                    model_version.checksum = version_data.get("checksum", "")
                    
                    self._model_registry[model_id].append(model_version)
            
            # Import active models
            for key, active_data in registry_data.get("active_models", {}).items():
                model_type_str, language_str = key.split("_", 1)
                model_type = ModelType(model_type_str)
                language = LanguageCode(language_str)
                
                model_version = await self.get_model_version(
                    active_data["model_id"],
                    active_data["version"]
                )
                if model_version:
                    self._active_models[(model_type, language)] = model_version
            
            logger.info("Successfully imported model registry")
            return True
            
        except Exception as e:
            logger.error(f"Error importing model registry: {e}")
            return False
    
    async def _test_model_performance(
        self,
        model_version: ModelVersion,
        requirements: Dict[str, float]
    ) -> bool:
        """Test model performance against requirements."""
        # This is a placeholder for actual performance testing
        # In production, this would run the model on test data
        
        # Simulate performance testing
        await asyncio.sleep(0.1)  # Simulate testing time
        
        # For now, assume all models meet requirements
        return True
    
    async def _cleanup_model_version(self, model_version: ModelVersion) -> None:
        """Clean up old model version files."""
        try:
            model_path = Path(model_version.model_path)
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Cleaned up model file: {model_version.model_path}")
        except Exception as e:
            logger.error(f"Error cleaning up model file {model_version.model_path}: {e}")
    
    async def cleanup_old_models(self) -> int:
        """Clean up old, unused model versions."""
        cleanup_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=self.auto_cleanup_days)
        
        for model_id, versions in self._model_registry.items():
            versions_to_remove = []
            
            for version in versions:
                if (version.status == ModelStatus.INACTIVE and 
                    version.created_at < cutoff_date and
                    len(versions) > 1):  # Keep at least one version
                    versions_to_remove.append(version)
            
            for version in versions_to_remove:
                await self._cleanup_model_version(version)
                versions.remove(version)
                cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} old model versions")
=======
"""
Model Management Module for BharatVoice Assistant.

This module handles model updates, expansion capabilities, and version management
for the extensible learning system.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from pathlib import Path
from enum import Enum

from bharatvoice.core.models import LanguageCode

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of models that can be managed."""
    
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    LANGUAGE_MODEL = "language_model"
    TRANSLATION = "translation"
    ACCENT_ADAPTATION = "accent_adaptation"
    VOCABULARY_EXPANSION = "vocabulary_expansion"
    STYLE_ADAPTATION = "style_adaptation"


class ModelStatus(str, Enum):
    """Model status states."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    UPDATING = "updating"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelVersion:
    """Represents a model version with metadata."""
    
    def __init__(
        self,
        model_id: str,
        version: str,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.version = version
        self.model_type = model_type
        self.language = language
        self.model_path = model_path
        self.metadata = metadata or {}
        self.status = ModelStatus.INACTIVE
        self.created_at = datetime.utcnow()
        self.activated_at: Optional[datetime] = None
        self.performance_metrics: Dict[str, float] = {}
        self.compatibility_info: Dict[str, Any] = {}
        self.checksum = ""
        
    def calculate_checksum(self) -> str:
        """Calculate model file checksum."""
        if Path(self.model_path).exists():
            with open(self.model_path, 'rb') as f:
                content = f.read()
                self.checksum = hashlib.sha256(content).hexdigest()
        return self.checksum
    
    def is_compatible_with(self, other_version: 'ModelVersion') -> bool:
        """Check compatibility with another model version."""
        if self.model_type != other_version.model_type:
            return False
        
        if self.language != other_version.language:
            return False
        
        # Check version compatibility
        self_major = int(self.version.split('.')[0])
        other_major = int(other_version.version.split('.')[0])
        
        return self_major == other_major
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type.value,
            "language": self.language.value,
            "model_path": self.model_path,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "performance_metrics": self.performance_metrics,
            "compatibility_info": self.compatibility_info,
            "checksum": self.checksum
        }


class ModelManager:
    """
    Manages model updates, expansion, and version control for the learning system.
    """
    
    def __init__(
        self,
        models_directory: str = "models",
        max_versions_per_model: int = 5,
        auto_cleanup_days: int = 30
    ):
        """
        Initialize model manager.
        
        Args:
            models_directory: Directory to store models
            max_versions_per_model: Maximum versions to keep per model
            auto_cleanup_days: Days after which to cleanup old versions
        """
        self.models_directory = Path(models_directory)
        self.max_versions_per_model = max_versions_per_model
        self.auto_cleanup_days = auto_cleanup_days
        
        # Ensure models directory exists
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self._model_registry: Dict[str, List[ModelVersion]] = {}
        self._active_models: Dict[Tuple[ModelType, LanguageCode], ModelVersion] = {}
        
        # Model update queue
        self._update_queue: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Model Manager initialized")
    
    async def register_model(
        self,
        model_id: str,
        version: str,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        auto_activate: bool = False
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_id: Unique model identifier
            version: Model version string
            model_type: Type of model
            language: Model language
            model_path: Path to model file
            metadata: Additional metadata
            auto_activate: Whether to activate immediately
            
        Returns:
            Registered model version
        """
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            language=language,
            model_path=model_path,
            metadata=metadata
        )
        
        # Calculate checksum
        model_version.calculate_checksum()
        
        # Add to registry
        if model_id not in self._model_registry:
            self._model_registry[model_id] = []
        
        self._model_registry[model_id].append(model_version)
        
        # Sort versions by creation date
        self._model_registry[model_id].sort(key=lambda x: x.created_at, reverse=True)
        
        # Cleanup old versions if needed
        if len(self._model_registry[model_id]) > self.max_versions_per_model:
            old_versions = self._model_registry[model_id][self.max_versions_per_model:]
            for old_version in old_versions:
                await self._cleanup_model_version(old_version)
            self._model_registry[model_id] = self._model_registry[model_id][:self.max_versions_per_model]
        
        # Auto-activate if requested
        if auto_activate:
            await self.activate_model(model_id, version)
        
        logger.info(f"Registered model {model_id} version {version}")
        return model_version
    
    async def activate_model(
        self,
        model_id: str,
        version: str
    ) -> bool:
        """
        Activate a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version to activate
            
        Returns:
            True if activation successful
        """
        model_version = await self.get_model_version(model_id, version)
        if not model_version:
            logger.error(f"Model version {model_id}:{version} not found")
            return False
        
        # Check if model file exists
        if not Path(model_version.model_path).exists():
            logger.error(f"Model file not found: {model_version.model_path}")
            return False
        
        # Verify checksum
        current_checksum = model_version.calculate_checksum()
        if model_version.checksum and current_checksum != model_version.checksum:
            logger.error(f"Model checksum mismatch for {model_id}:{version}")
            return False
        
        # Deactivate current active model of same type/language
        model_key = (model_version.model_type, model_version.language)
        if model_key in self._active_models:
            old_model = self._active_models[model_key]
            old_model.status = ModelStatus.INACTIVE
            logger.info(f"Deactivated model {old_model.model_id}:{old_model.version}")
        
        # Activate new model
        model_version.status = ModelStatus.ACTIVE
        model_version.activated_at = datetime.utcnow()
        self._active_models[model_key] = model_version
        
        logger.info(f"Activated model {model_id}:{version}")
        return True
    
    async def update_model(
        self,
        model_id: str,
        new_version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Update a model to a new version with backward compatibility checks.
        
        Args:
            model_id: Model identifier
            new_version: New version string
            model_path: Path to new model file
            metadata: Updated metadata
            performance_requirements: Minimum performance requirements
            
        Returns:
            True if update successful
        """
        # Get current active model
        current_model = await self.get_active_model_by_id(model_id)
        if not current_model:
            logger.error(f"No active model found for {model_id}")
            return False
        
        # Register new version
        new_model = await self.register_model(
            model_id=model_id,
            version=new_version,
            model_type=current_model.model_type,
            language=current_model.language,
            model_path=model_path,
            metadata=metadata
        )
        
        # Check backward compatibility
        if not new_model.is_compatible_with(current_model):
            logger.warning(f"New model version {new_version} may not be backward compatible")
        
        # Test performance if requirements specified
        if performance_requirements:
            performance_ok = await self._test_model_performance(new_model, performance_requirements)
            if not performance_ok:
                logger.error(f"New model version {new_version} does not meet performance requirements")
                return False
        
        # Activate new version
        success = await self.activate_model(model_id, new_version)
        if success:
            logger.info(f"Successfully updated model {model_id} to version {new_version}")
        
        return success
    
    async def add_language_support(
        self,
        model_type: ModelType,
        language: LanguageCode,
        model_path: str,
        base_model_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add support for a new language.
        
        Args:
            model_type: Type of model
            language: New language to support
            model_path: Path to language model
            base_model_id: Base model to extend from
            metadata: Language-specific metadata
            
        Returns:
            New model ID for the language
        """
        # Generate new model ID for language
        model_id = f"{model_type.value}_{language.value}"
        
        # Copy metadata from base model if specified
        if base_model_id and metadata is None:
            base_model = await self.get_active_model_by_id(base_model_id)
            if base_model:
                metadata = base_model.metadata.copy()
                metadata["base_model"] = base_model_id
        
        # Register new language model
        model_version = await self.register_model(
            model_id=model_id,
            version="1.0.0",
            model_type=model_type,
            language=language,
            model_path=model_path,
            metadata=metadata,
            auto_activate=True
        )
        
        logger.info(f"Added {language.value} support for {model_type.value}")
        return model_id
    
    async def get_active_model(
        self,
        model_type: ModelType,
        language: LanguageCode
    ) -> Optional[ModelVersion]:
        """
        Get active model for type and language.
        
        Args:
            model_type: Model type
            language: Language
            
        Returns:
            Active model version if available
        """
        model_key = (model_type, language)
        return self._active_models.get(model_key)
    
    async def get_active_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Get active model by ID."""
        for model_version in self._active_models.values():
            if model_version.model_id == model_id:
                return model_version
        return None
    
    async def get_model_version(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersion]:
        """
        Get specific model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            
        Returns:
            Model version if found
        """
        if model_id not in self._model_registry:
            return None
        
        for model_version in self._model_registry[model_id]:
            if model_version.version == version:
                return model_version
        
        return None
    
    async def list_available_models(
        self,
        model_type: Optional[ModelType] = None,
        language: Optional[LanguageCode] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelVersion]:
        """
        List available models with optional filters.
        
        Args:
            model_type: Filter by model type
            language: Filter by language
            status: Filter by status
            
        Returns:
            List of matching model versions
        """
        models = []
        
        for model_versions in self._model_registry.values():
            for model_version in model_versions:
                # Apply filters
                if model_type and model_version.model_type != model_type:
                    continue
                if language and model_version.language != language:
                    continue
                if status and model_version.status != status:
                    continue
                
                models.append(model_version)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    async def get_supported_languages(
        self,
        model_type: ModelType
    ) -> List[LanguageCode]:
        """
        Get list of supported languages for a model type.
        
        Args:
            model_type: Model type
            
        Returns:
            List of supported languages
        """
        languages = set()
        
        for model_key, model_version in self._active_models.items():
            if model_key[0] == model_type:
                languages.add(model_key[1])
        
        return list(languages)
    
    async def record_performance_metrics(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Record performance metrics for a model version.
        
        Args:
            model_id: Model identifier
            version: Model version
            metrics: Performance metrics
        """
        model_version = await self.get_model_version(model_id, version)
        if model_version:
            model_version.performance_metrics.update(metrics)
            
            # Add to performance history
            history_key = f"{model_id}:{version}"
            if history_key not in self._performance_history:
                self._performance_history[history_key] = []
            
            self._performance_history[history_key].append({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics.copy()
            })
            
            logger.info(f"Recorded performance metrics for {model_id}:{version}")
    
    async def get_model_performance_history(
        self,
        model_id: str,
        version: str
    ) -> List[Dict[str, Any]]:
        """Get performance history for a model version."""
        history_key = f"{model_id}:{version}"
        return self._performance_history.get(history_key, [])
    
    async def rollback_model(
        self,
        model_id: str,
        target_version: Optional[str] = None
    ) -> bool:
        """
        Rollback model to previous or specified version.
        
        Args:
            model_id: Model identifier
            target_version: Specific version to rollback to (optional)
            
        Returns:
            True if rollback successful
        """
        if model_id not in self._model_registry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        versions = self._model_registry[model_id]
        
        if target_version:
            # Rollback to specific version
            target_model = await self.get_model_version(model_id, target_version)
            if not target_model:
                logger.error(f"Target version {target_version} not found")
                return False
        else:
            # Rollback to previous version
            if len(versions) < 2:
                logger.error(f"No previous version available for {model_id}")
                return False
            target_model = versions[1]  # Second most recent
        
        # Activate target version
        success = await self.activate_model(model_id, target_model.version)
        if success:
            logger.info(f"Rolled back model {model_id} to version {target_model.version}")
        
        return success
    
    async def export_model_registry(self) -> Dict[str, Any]:
        """Export model registry for backup or migration."""
        registry_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "models": {},
            "active_models": {}
        }
        
        # Export model registry
        for model_id, versions in self._model_registry.items():
            registry_data["models"][model_id] = [
                version.to_dict() for version in versions
            ]
        
        # Export active models
        for (model_type, language), model_version in self._active_models.items():
            key = f"{model_type.value}_{language.value}"
            registry_data["active_models"][key] = {
                "model_id": model_version.model_id,
                "version": model_version.version
            }
        
        return registry_data
    
    async def import_model_registry(self, registry_data: Dict[str, Any]) -> bool:
        """Import model registry from backup or migration."""
        try:
            # Import models
            for model_id, versions_data in registry_data.get("models", {}).items():
                self._model_registry[model_id] = []
                for version_data in versions_data:
                    model_version = ModelVersion(
                        model_id=version_data["model_id"],
                        version=version_data["version"],
                        model_type=ModelType(version_data["model_type"]),
                        language=LanguageCode(version_data["language"]),
                        model_path=version_data["model_path"],
                        metadata=version_data.get("metadata", {})
                    )
                    model_version.status = ModelStatus(version_data["status"])
                    model_version.created_at = datetime.fromisoformat(version_data["created_at"])
                    if version_data.get("activated_at"):
                        model_version.activated_at = datetime.fromisoformat(version_data["activated_at"])
                    model_version.performance_metrics = version_data.get("performance_metrics", {})
                    model_version.compatibility_info = version_data.get("compatibility_info", {})
                    model_version.checksum = version_data.get("checksum", "")
                    
                    self._model_registry[model_id].append(model_version)
            
            # Import active models
            for key, active_data in registry_data.get("active_models", {}).items():
                model_type_str, language_str = key.split("_", 1)
                model_type = ModelType(model_type_str)
                language = LanguageCode(language_str)
                
                model_version = await self.get_model_version(
                    active_data["model_id"],
                    active_data["version"]
                )
                if model_version:
                    self._active_models[(model_type, language)] = model_version
            
            logger.info("Successfully imported model registry")
            return True
            
        except Exception as e:
            logger.error(f"Error importing model registry: {e}")
            return False
    
    async def _test_model_performance(
        self,
        model_version: ModelVersion,
        requirements: Dict[str, float]
    ) -> bool:
        """Test model performance against requirements."""
        # This is a placeholder for actual performance testing
        # In production, this would run the model on test data
        
        # Simulate performance testing
        await asyncio.sleep(0.1)  # Simulate testing time
        
        # For now, assume all models meet requirements
        return True
    
    async def _cleanup_model_version(self, model_version: ModelVersion) -> None:
        """Clean up old model version files."""
        try:
            model_path = Path(model_version.model_path)
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Cleaned up model file: {model_version.model_path}")
        except Exception as e:
            logger.error(f"Error cleaning up model file {model_version.model_path}: {e}")
    
    async def cleanup_old_models(self) -> int:
        """Clean up old, unused model versions."""
        cleanup_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=self.auto_cleanup_days)
        
        for model_id, versions in self._model_registry.items():
            versions_to_remove = []
            
            for version in versions:
                if (version.status == ModelStatus.INACTIVE and 
                    version.created_at < cutoff_date and
                    len(versions) > 1):  # Keep at least one version
                    versions_to_remove.append(version)
            
            for version in versions_to_remove:
                await self._cleanup_model_version(version)
                versions.remove(version)
                cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} old model versions")
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        return cleanup_count