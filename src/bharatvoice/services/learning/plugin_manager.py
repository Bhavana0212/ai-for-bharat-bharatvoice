"""
Plugin Management Module for BharatVoice Assistant.

This module provides a plugin architecture for extending the system with
new Indian languages, dialects, and custom functionality.
"""

import asyncio
import logging
import importlib
import inspect
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Callable
from uuid import UUID, uuid4
from pathlib import Path
from enum import Enum

from bharatvoice.core.models import LanguageCode

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported."""
    
    LANGUAGE_SUPPORT = "language_support"
    DIALECT_SUPPORT = "dialect_support"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    SERVICE_INTEGRATION = "service_integration"
    VOICE_PROCESSING = "voice_processing"
    LEARNING_ALGORITHM = "learning_algorithm"


class PluginStatus(str, Enum):
    """Plugin status states."""
    
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class BharatVoicePlugin(ABC):
    """
    Base class for all BharatVoice plugins.
    """
    
    def __init__(self, plugin_id: str, version: str):
        self.plugin_id = plugin_id
        self.version = version
        self.name = ""
        self.description = ""
        self.author = ""
        self.plugin_type = PluginType.LANGUAGE_SUPPORT
        self.supported_languages: List[LanguageCode] = []
        self.dependencies: List[str] = []
        self.configuration: Dict[str, Any] = {}
        self.status = PluginStatus.LOADED
        self.loaded_at = datetime.utcnow()
        self.error_message = ""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """
        Activate the plugin.
        
        Returns:
            True if activation successful
        """
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """
        Deactivate the plugin.
        
        Returns:
            True if deactivation successful
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get plugin capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        pass
    
    async def validate_dependencies(self) -> bool:
        """Validate plugin dependencies."""
        # Basic dependency validation
        for dep in self.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.error(f"Plugin {self.plugin_id} missing dependency: {dep}")
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plugin info to dictionary."""
        return {
            "plugin_id": self.plugin_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "supported_languages": [lang.value for lang in self.supported_languages],
            "dependencies": self.dependencies,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat(),
            "error_message": self.error_message
        }


class LanguageSupportPlugin(BharatVoicePlugin):
    """
    Base class for language support plugins.
    """
    
    def __init__(self, plugin_id: str, version: str, language: LanguageCode):
        super().__init__(plugin_id, version)
        self.plugin_type = PluginType.LANGUAGE_SUPPORT
        self.supported_languages = [language]
        self.language = language
    
    @abstractmethod
    async def get_language_models(self) -> Dict[str, str]:
        """
        Get language-specific model paths.
        
        Returns:
            Dictionary mapping model types to file paths
        """
        pass
    
    @abstractmethod
    async def get_phoneme_mappings(self) -> Dict[str, List[str]]:
        """
        Get phoneme mappings for the language.
        
        Returns:
            Dictionary mapping phonemes to variations
        """
        pass
    
    @abstractmethod
    async def get_cultural_patterns(self) -> Dict[str, Any]:
        """
        Get cultural patterns and expressions.
        
        Returns:
            Dictionary of cultural patterns
        """
        pass
    
    @abstractmethod
    async def process_text(self, text: str) -> str:
        """
        Process text for language-specific requirements.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        pass


class DialectSupportPlugin(BharatVoicePlugin):
    """
    Base class for dialect support plugins.
    """
    
    def __init__(self, plugin_id: str, version: str, base_language: LanguageCode, dialect: str):
        super().__init__(plugin_id, version)
        self.plugin_type = PluginType.DIALECT_SUPPORT
        self.base_language = base_language
        self.dialect = dialect
        self.supported_languages = [base_language]
    
    @abstractmethod
    async def get_dialect_variations(self) -> Dict[str, str]:
        """
        Get dialect-specific word variations.
        
        Returns:
            Dictionary mapping standard words to dialect variations
        """
        pass
    
    @abstractmethod
    async def adapt_pronunciation(self, text: str) -> str:
        """
        Adapt text for dialect-specific pronunciation.
        
        Args:
            text: Input text
            
        Returns:
            Dialect-adapted text
        """
        pass


class PluginManager:
    """
    Manages plugins for system extensibility.
    """
    
    def __init__(
        self,
        plugins_directory: str = "plugins",
        auto_load_plugins: bool = True
    ):
        """
        Initialize plugin manager.
        
        Args:
            plugins_directory: Directory containing plugins
            auto_load_plugins: Whether to auto-load plugins on startup
        """
        self.plugins_directory = Path(plugins_directory)
        self.auto_load_plugins = auto_load_plugins
        
        # Plugin registry
        self._plugins: Dict[str, BharatVoicePlugin] = {}
        self._plugin_hooks: Dict[str, List[Callable]] = {}
        
        # Plugin loading statistics
        self._loading_stats = {
            "total_plugins_loaded": 0,
            "active_plugins": 0,
            "failed_plugins": 0,
            "last_scan": None
        }
        
        # Ensure plugins directory exists
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Plugin Manager initialized")
    
    async def scan_and_load_plugins(self) -> Dict[str, Any]:
        """
        Scan plugins directory and load available plugins.
        
        Returns:
            Loading results
        """
        loading_results = {
            "loaded_plugins": [],
            "failed_plugins": [],
            "total_scanned": 0
        }
        
        # Scan for Python plugin files
        plugin_files = list(self.plugins_directory.glob("*.py"))
        loading_results["total_scanned"] = len(plugin_files)
        
        for plugin_file in plugin_files:
            if plugin_file.name.startswith("__"):
                continue  # Skip __init__.py and similar files
            
            try:
                plugin = await self._load_plugin_from_file(plugin_file)
                if plugin:
                    loading_results["loaded_plugins"].append(plugin.plugin_id)
                    self._loading_stats["total_plugins_loaded"] += 1
                else:
                    loading_results["failed_plugins"].append(plugin_file.name)
                    self._loading_stats["failed_plugins"] += 1
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_file.name}: {e}")
                loading_results["failed_plugins"].append(plugin_file.name)
                self._loading_stats["failed_plugins"] += 1
        
        self._loading_stats["last_scan"] = datetime.utcnow().isoformat()
        
        logger.info(f"Loaded {len(loading_results['loaded_plugins'])} plugins, "
                   f"{len(loading_results['failed_plugins'])} failed")
        
        return loading_results
    
    async def register_plugin(
        self,
        plugin: BharatVoicePlugin,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance
            config: Plugin configuration
            
        Returns:
            True if registration successful
        """
        try:
            # Validate dependencies
            if not await plugin.validate_dependencies():
                plugin.status = PluginStatus.ERROR
                plugin.error_message = "Dependency validation failed"
                return False
            
            # Initialize plugin
            init_success = await plugin.initialize(config or {})
            if not init_success:
                plugin.status = PluginStatus.ERROR
                plugin.error_message = "Initialization failed"
                return False
            
            # Register plugin
            self._plugins[plugin.plugin_id] = plugin
            
            logger.info(f"Registered plugin: {plugin.plugin_id} v{plugin.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin {plugin.plugin_id}: {e}")
            plugin.status = PluginStatus.ERROR
            plugin.error_message = str(e)
            return False
    
    async def activate_plugin(self, plugin_id: str) -> bool:
        """
        Activate a registered plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if activation successful
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin {plugin_id} not found")
            return False
        
        plugin = self._plugins[plugin_id]
        
        try:
            success = await plugin.activate()
            if success:
                plugin.status = PluginStatus.ACTIVE
                self._loading_stats["active_plugins"] += 1
                logger.info(f"Activated plugin: {plugin_id}")
            else:
                plugin.status = PluginStatus.ERROR
                plugin.error_message = "Activation failed"
            
            return success
            
        except Exception as e:
            logger.error(f"Error activating plugin {plugin_id}: {e}")
            plugin.status = PluginStatus.ERROR
            plugin.error_message = str(e)
            return False
    
    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """
        Deactivate an active plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if deactivation successful
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin {plugin_id} not found")
            return False
        
        plugin = self._plugins[plugin_id]
        
        try:
            success = await plugin.deactivate()
            if success:
                plugin.status = PluginStatus.INACTIVE
                if self._loading_stats["active_plugins"] > 0:
                    self._loading_stats["active_plugins"] -= 1
                logger.info(f"Deactivated plugin: {plugin_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deactivating plugin {plugin_id}: {e}")
            plugin.status = PluginStatus.ERROR
            plugin.error_message = str(e)
            return False
    
    async def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if unregistration successful
        """
        if plugin_id not in self._plugins:
            return True  # Already unregistered
        
        plugin = self._plugins[plugin_id]
        
        # Deactivate if active
        if plugin.status == PluginStatus.ACTIVE:
            await self.deactivate_plugin(plugin_id)
        
        # Remove from registry
        del self._plugins[plugin_id]
        
        logger.info(f"Unregistered plugin: {plugin_id}")
        return True
    
    async def get_plugin(self, plugin_id: str) -> Optional[BharatVoicePlugin]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)
    
    async def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        status: Optional[PluginStatus] = None,
        language: Optional[LanguageCode] = None
    ) -> List[BharatVoicePlugin]:
        """
        List plugins with optional filters.
        
        Args:
            plugin_type: Filter by plugin type
            status: Filter by status
            language: Filter by supported language
            
        Returns:
            List of matching plugins
        """
        plugins = []
        
        for plugin in self._plugins.values():
            # Apply filters
            if plugin_type and plugin.plugin_type != plugin_type:
                continue
            if status and plugin.status != status:
                continue
            if language and language not in plugin.supported_languages:
                continue
            
            plugins.append(plugin)
        
        return plugins
    
    async def get_language_support_plugins(
        self,
        language: LanguageCode
    ) -> List[LanguageSupportPlugin]:
        """
        Get language support plugins for a specific language.
        
        Args:
            language: Target language
            
        Returns:
            List of language support plugins
        """
        plugins = await self.list_plugins(
            plugin_type=PluginType.LANGUAGE_SUPPORT,
            status=PluginStatus.ACTIVE,
            language=language
        )
        
        return [p for p in plugins if isinstance(p, LanguageSupportPlugin)]
    
    async def get_dialect_support_plugins(
        self,
        base_language: LanguageCode
    ) -> List[DialectSupportPlugin]:
        """
        Get dialect support plugins for a base language.
        
        Args:
            base_language: Base language
            
        Returns:
            List of dialect support plugins
        """
        plugins = await self.list_plugins(
            plugin_type=PluginType.DIALECT_SUPPORT,
            status=PluginStatus.ACTIVE,
            language=base_language
        )
        
        return [p for p in plugins if isinstance(p, DialectSupportPlugin)]
    
    async def register_hook(
        self,
        hook_name: str,
        callback: Callable
    ) -> None:
        """
        Register a plugin hook callback.
        
        Args:
            hook_name: Name of the hook
            callback: Callback function
        """
        if hook_name not in self._plugin_hooks:
            self._plugin_hooks[hook_name] = []
        
        self._plugin_hooks[hook_name].append(callback)
        logger.info(f"Registered hook: {hook_name}")
    
    async def trigger_hook(
        self,
        hook_name: str,
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Trigger all callbacks for a hook.
        
        Args:
            hook_name: Name of the hook
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
            
        Returns:
            List of callback results
        """
        if hook_name not in self._plugin_hooks:
            return []
        
        results = []
        for callback in self._plugin_hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name} callback: {e}")
                results.append(None)
        
        return results
    
    async def get_plugin_capabilities(
        self,
        plugin_type: Optional[PluginType] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get capabilities of all plugins or plugins of specific type.
        
        Args:
            plugin_type: Filter by plugin type
            
        Returns:
            Dictionary mapping plugin IDs to capabilities
        """
        capabilities = {}
        
        plugins = await self.list_plugins(plugin_type=plugin_type, status=PluginStatus.ACTIVE)
        
        for plugin in plugins:
            try:
                plugin_capabilities = await plugin.get_capabilities()
                capabilities[plugin.plugin_id] = plugin_capabilities
            except Exception as e:
                logger.error(f"Error getting capabilities for plugin {plugin.plugin_id}: {e}")
                capabilities[plugin.plugin_id] = {"error": str(e)}
        
        return capabilities
    
    async def get_loading_statistics(self) -> Dict[str, Any]:
        """Get plugin loading statistics."""
        return self._loading_stats.copy()
    
    async def _load_plugin_from_file(self, plugin_file: Path) -> Optional[BharatVoicePlugin]:
        """
        Load a plugin from a Python file.
        
        Args:
            plugin_file: Path to plugin file
            
        Returns:
            Loaded plugin instance or None
        """
        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location(
                plugin_file.stem, plugin_file
            )
            if not spec or not spec.loader:
                logger.error(f"Could not load spec for {plugin_file}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class in module
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BharatVoicePlugin) and 
                    obj != BharatVoicePlugin and
                    obj != LanguageSupportPlugin and
                    obj != DialectSupportPlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {plugin_file}")
                return None
            
            # Instantiate plugin
            plugin = plugin_class()
            
            # Register the plugin
            success = await self.register_plugin(plugin)
            if success:
                return plugin
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_file}: {e}")
            return None
    
    async def create_plugin_template(
        self,
        plugin_id: str,
        plugin_type: PluginType,
        language: Optional[LanguageCode] = None
    ) -> str:
        """
        Create a plugin template file.
        
        Args:
            plugin_id: Plugin identifier
            plugin_type: Type of plugin
            language: Target language (for language plugins)
            
        Returns:
            Path to created template file
        """
        template_content = self._generate_plugin_template(plugin_id, plugin_type, language)
        
        template_file = self.plugins_directory / f"{plugin_id}_plugin.py"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"Created plugin template: {template_file}")
        return str(template_file)
    
    def _generate_plugin_template(
        self,
        plugin_id: str,
        plugin_type: PluginType,
        language: Optional[LanguageCode]
    ) -> str:
        """Generate plugin template code."""
        if plugin_type == PluginType.LANGUAGE_SUPPORT and language:
            return f'''"""
{plugin_id.title()} Language Support Plugin for BharatVoice Assistant.

This plugin adds support for {language.value} language.
"""

from typing import Dict, List, Any
from bharatvoice.core.models import LanguageCode
from bharatvoice.services.learning.plugin_manager import LanguageSupportPlugin


class {plugin_id.title()}LanguagePlugin(LanguageSupportPlugin):
    """
    Language support plugin for {language.value}.
    """
    
    def __init__(self):
        super().__init__(
            plugin_id="{plugin_id}",
            version="1.0.0",
            language=LanguageCode.{language.name}
        )
        self.name = "{language.value} Language Support"
        self.description = "Adds {language.value} language support to BharatVoice"
        self.author = "Your Name"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.configuration = config
        # Add initialization logic here
        return True
    
    async def activate(self) -> bool:
        """Activate the plugin."""
        # Add activation logic here
        return True
    
    async def deactivate(self) -> bool:
        """Deactivate the plugin."""
        # Add deactivation logic here
        return True
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities."""
        return {{
            "language": "{language.value}",
            "features": ["speech_recognition", "text_to_speech", "translation"],
            "models_available": True
        }}
    
    async def get_language_models(self) -> Dict[str, str]:
        """Get language-specific model paths."""
        return {{
            "speech_recognition": "models/{language.value}/asr_model.bin",
            "text_to_speech": "models/{language.value}/tts_model.bin",
            "translation": "models/{language.value}/translation_model.bin"
        }}
    
    async def get_phoneme_mappings(self) -> Dict[str, List[str]]:
        """Get phoneme mappings for the language."""
        return {{
            # Add phoneme mappings here
            "a": ["a", "aa"],
            "i": ["i", "ii"],
            "u": ["u", "uu"]
        }}
    
    async def get_cultural_patterns(self) -> Dict[str, Any]:
        """Get cultural patterns and expressions."""
        return {{
            "greetings": ["Hello", "Hi"],
            "polite_expressions": ["Please", "Thank you"],
            "cultural_references": []
        }}
    
    async def process_text(self, text: str) -> str:
        """Process text for language-specific requirements."""
        # Add text processing logic here
        return text
'''
        else:
            return f'''"""
{plugin_id.title()} Plugin for BharatVoice Assistant.

This plugin extends BharatVoice functionality.
"""

from typing import Dict, Any
from bharatvoice.services.learning.plugin_manager import BharatVoicePlugin, PluginType


class {plugin_id.title()}Plugin(BharatVoicePlugin):
    """
    Custom plugin for BharatVoice.
    """
    
    def __init__(self):
        super().__init__(
            plugin_id="{plugin_id}",
            version="1.0.0"
        )
        self.name = "{plugin_id.title()} Plugin"
        self.description = "Custom plugin for BharatVoice"
        self.author = "Your Name"
        self.plugin_type = PluginType.{plugin_type.name}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.configuration = config
        # Add initialization logic here
        return True
    
    async def activate(self) -> bool:
        """Activate the plugin."""
        # Add activation logic here
        return True
    
    async def deactivate(self) -> bool:
        """Deactivate the plugin."""
        # Add deactivation logic here
        return True
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities."""
        return {{
            "plugin_type": "{plugin_type.value}",
            "features": [],
            "version": self.version
        }}
'''