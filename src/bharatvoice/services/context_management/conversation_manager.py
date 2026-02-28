"""
Conversation State Management for BharatVoice Assistant.

This module implements conversation state management with session handling,
multi-turn dialog support, context preservation, timeout mechanisms,
and conversation history storage and retrieval.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from bharatvoice.core.models import (
    ConversationState,
    LanguageCode,
    UserInteraction,
    UserProfile,
)
from bharatvoice.core.interfaces import ContextManager


logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation state, session handling, and multi-turn dialog support.
    
    Features:
    - Session-based conversation state management
    - Multi-turn dialog context preservation
    - Automatic conversation timeout and cleanup
    - Conversation history storage and retrieval
    - Context variable management
    """
    
    def __init__(
        self,
        session_timeout_minutes: int = 30,
        max_history_length: int = 50,
        cleanup_interval_minutes: int = 5
    ):
        """
        Initialize conversation manager.
        
        Args:
            session_timeout_minutes: Minutes before inactive session expires
            max_history_length: Maximum number of interactions to keep in history
            cleanup_interval_minutes: Minutes between cleanup runs
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_history_length = max_history_length
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        # In-memory storage for active conversations
        # In production, this would be backed by Redis or similar
        self._active_sessions: Dict[UUID, ConversationState] = {}
        self._session_locks: Dict[UUID, asyncio.Lock] = {}
        
        # Conversation history storage
        # In production, this would be backed by a database
        self._conversation_history: Dict[UUID, List[UserInteraction]] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def create_session(
        self,
        user_id: UUID,
        initial_language: LanguageCode = LanguageCode.HINDI
    ) -> ConversationState:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            initial_language: Initial conversation language
            
        Returns:
            New conversation state
        """
        session_id = uuid4()
        
        conversation_state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            current_language=initial_language,
            conversation_history=[],
            context_variables={},
            last_interaction_time=datetime.utcnow(),
            is_active=True
        )
        
        # Store session with lock
        self._active_sessions[session_id] = conversation_state
        self._session_locks[session_id] = asyncio.Lock()
        
        logger.info(f"Created new conversation session {session_id} for user {user_id}")
        return conversation_state
    
    async def get_session(self, session_id: UUID) -> Optional[ConversationState]:
        """
        Retrieve conversation state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation state if exists and active, None otherwise
        """
        if session_id not in self._active_sessions:
            return None
        
        conversation_state = self._active_sessions[session_id]
        
        # Check if session has expired
        if self._is_session_expired(conversation_state):
            await self._cleanup_session(session_id)
            return None
        
        return conversation_state
    
    async def update_session(
        self,
        session_id: UUID,
        interaction: UserInteraction
    ) -> Optional[ConversationState]:
        """
        Update conversation state with new interaction.
        
        Args:
            session_id: Session identifier
            interaction: New user interaction
            
        Returns:
            Updated conversation state, None if session doesn't exist
        """
        if session_id not in self._active_sessions:
            logger.warning(f"Attempted to update non-existent session {session_id}")
            return None
        
        async with self._session_locks[session_id]:
            conversation_state = self._active_sessions[session_id]
            
            # Check if session has expired
            if self._is_session_expired(conversation_state):
                await self._cleanup_session(session_id)
                return None
            
            # Add interaction to history
            conversation_state.conversation_history.append(interaction)
            
            # Trim history if it exceeds maximum length
            if len(conversation_state.conversation_history) > self.max_history_length:
                # Keep recent interactions and move older ones to persistent storage
                old_interactions = conversation_state.conversation_history[:-self.max_history_length]
                conversation_state.conversation_history = conversation_state.conversation_history[-self.max_history_length:]
                
                # Store old interactions in conversation history
                if session_id not in self._conversation_history:
                    self._conversation_history[session_id] = []
                self._conversation_history[session_id].extend(old_interactions)
            
            # Update session metadata
            conversation_state.last_interaction_time = datetime.utcnow()
            
            # Update current language if it changed
            if interaction.input_language != conversation_state.current_language:
                conversation_state.current_language = interaction.input_language
                logger.info(f"Session {session_id} language switched to {interaction.input_language}")
            
            logger.debug(f"Updated session {session_id} with new interaction")
            return conversation_state
    
    async def set_context_variable(
        self,
        session_id: UUID,
        key: str,
        value: any
    ) -> bool:
        """
        Set a context variable for the session.
        
        Args:
            session_id: Session identifier
            key: Context variable key
            value: Context variable value
            
        Returns:
            True if set successfully, False if session doesn't exist
        """
        if session_id not in self._active_sessions:
            return False
        
        async with self._session_locks[session_id]:
            conversation_state = self._active_sessions[session_id]
            
            if self._is_session_expired(conversation_state):
                await self._cleanup_session(session_id)
                return False
            
            conversation_state.context_variables[key] = value
            conversation_state.last_interaction_time = datetime.utcnow()
            
            logger.debug(f"Set context variable '{key}' for session {session_id}")
            return True
    
    async def get_context_variable(
        self,
        session_id: UUID,
        key: str,
        default: any = None
    ) -> any:
        """
        Get a context variable from the session.
        
        Args:
            session_id: Session identifier
            key: Context variable key
            default: Default value if key doesn't exist
            
        Returns:
            Context variable value or default
        """
        if session_id not in self._active_sessions:
            return default
        
        conversation_state = self._active_sessions[session_id]
        
        if self._is_session_expired(conversation_state):
            await self._cleanup_session(session_id)
            return default
        
        return conversation_state.context_variables.get(key, default)
    
    async def get_conversation_history(
        self,
        session_id: UUID,
        limit: Optional[int] = None
    ) -> List[UserInteraction]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of user interactions (most recent first)
        """
        history = []
        
        # Get current session history
        if session_id in self._active_sessions:
            conversation_state = self._active_sessions[session_id]
            if not self._is_session_expired(conversation_state):
                history.extend(reversed(conversation_state.conversation_history))
        
        # Get stored history
        if session_id in self._conversation_history:
            stored_history = list(reversed(self._conversation_history[session_id]))
            history.extend(stored_history)
        
        # Apply limit if specified
        if limit is not None:
            history = history[:limit]
        
        return history
    
    async def end_session(self, session_id: UUID) -> bool:
        """
        Explicitly end a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was ended, False if it didn't exist
        """
        if session_id not in self._active_sessions:
            return False
        
        await self._cleanup_session(session_id)
        logger.info(f"Explicitly ended session {session_id}")
        return True
    
    async def get_active_sessions(self, user_id: Optional[UUID] = None) -> List[ConversationState]:
        """
        Get list of active sessions, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of active conversation states
        """
        active_sessions = []
        
        for session_id, conversation_state in self._active_sessions.items():
            if self._is_session_expired(conversation_state):
                # Schedule cleanup but don't block
                asyncio.create_task(self._cleanup_session(session_id))
                continue
            
            if user_id is None or conversation_state.user_id == user_id:
                active_sessions.append(conversation_state)
        
        return active_sessions
    
    async def get_session_statistics(self) -> Dict[str, any]:
        """
        Get statistics about active sessions.
        
        Returns:
            Dictionary with session statistics
        """
        total_sessions = len(self._active_sessions)
        expired_sessions = 0
        
        for conversation_state in self._active_sessions.values():
            if self._is_session_expired(conversation_state):
                expired_sessions += 1
        
        active_sessions = total_sessions - expired_sessions
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "expired_sessions": expired_sessions,
            "total_stored_history": len(self._conversation_history)
        }
    
    def _is_session_expired(self, conversation_state: ConversationState) -> bool:
        """
        Check if a session has expired.
        
        Args:
            conversation_state: Conversation state to check
            
        Returns:
            True if session has expired
        """
        if not conversation_state.is_active:
            return True
        
        time_since_last_interaction = datetime.utcnow() - conversation_state.last_interaction_time
        return time_since_last_interaction > self.session_timeout
    
    async def _cleanup_session(self, session_id: UUID) -> None:
        """
        Clean up an expired or ended session.
        
        Args:
            session_id: Session identifier to clean up
        """
        if session_id in self._active_sessions:
            conversation_state = self._active_sessions[session_id]
            
            # Move remaining history to persistent storage
            if conversation_state.conversation_history:
                if session_id not in self._conversation_history:
                    self._conversation_history[session_id] = []
                self._conversation_history[session_id].extend(conversation_state.conversation_history)
            
            # Remove from active sessions
            del self._active_sessions[session_id]
            
            # Remove lock
            if session_id in self._session_locks:
                del self._session_locks[session_id]
            
            logger.debug(f"Cleaned up session {session_id}")
    
    def _start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        async def cleanup_expired_sessions():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval.total_seconds())
                    
                    expired_sessions = []
                    for session_id, conversation_state in self._active_sessions.items():
                        if self._is_session_expired(conversation_state):
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        await self._cleanup_session(session_id)
                    
                    if expired_sessions:
                        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    async def shutdown(self) -> None:
        """Shutdown the conversation manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all active sessions
        session_ids = list(self._active_sessions.keys())
        for session_id in session_ids:
            await self._cleanup_session(session_id)
        
        logger.info("Conversation manager shutdown complete")


class ConversationContextManager:
    """
    High-level context manager that integrates conversation state management
    with user profiles and regional context.
    """
    
    def __init__(self, conversation_manager: ConversationManager):
        """
        Initialize context manager.
        
        Args:
            conversation_manager: Conversation manager instance
        """
        self.conversation_manager = conversation_manager
        self._user_profiles: Dict[UUID, UserProfile] = {}  # In production: database
    
    async def start_conversation(
        self,
        user_id: UUID,
        initial_language: Optional[LanguageCode] = None
    ) -> ConversationState:
        """
        Start a new conversation with context initialization.
        
        Args:
            user_id: User identifier
            initial_language: Initial conversation language
            
        Returns:
            New conversation state
        """
        # Get user profile to determine initial language
        user_profile = await self.get_user_profile(user_id)
        
        if initial_language is None:
            initial_language = user_profile.primary_language if user_profile else LanguageCode.HINDI
        
        # Create conversation session
        conversation_state = await self.conversation_manager.create_session(
            user_id=user_id,
            initial_language=initial_language
        )
        
        # Set initial context variables from user profile
        if user_profile:
            await self.conversation_manager.set_context_variable(
                conversation_state.session_id,
                "user_preferred_languages",
                [lang.value for lang in user_profile.preferred_languages]
            )
            
            if user_profile.location:
                await self.conversation_manager.set_context_variable(
                    conversation_state.session_id,
                    "user_location",
                    {
                        "city": user_profile.location.city,
                        "state": user_profile.location.state,
                        "timezone": user_profile.location.timezone
                    }
                )
        
        return conversation_state
    
    async def process_interaction(
        self,
        session_id: UUID,
        interaction: UserInteraction
    ) -> Optional[ConversationState]:
        """
        Process a user interaction with context preservation.
        
        Args:
            session_id: Session identifier
            interaction: User interaction to process
            
        Returns:
            Updated conversation state
        """
        # Update conversation state
        conversation_state = await self.conversation_manager.update_session(
            session_id, interaction
        )
        
        if conversation_state is None:
            return None
        
        # Update context variables based on interaction
        await self._update_context_from_interaction(session_id, interaction)
        
        return conversation_state
    
    async def get_user_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """
        Get user profile (placeholder implementation).
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile if exists
        """
        # In production, this would query a database
        return self._user_profiles.get(user_id)
    
    async def _update_context_from_interaction(
        self,
        session_id: UUID,
        interaction: UserInteraction
    ) -> None:
        """
        Update context variables based on user interaction.
        
        Args:
            session_id: Session identifier
            interaction: User interaction
        """
        # Update last interaction language
        await self.conversation_manager.set_context_variable(
            session_id,
            "last_input_language",
            interaction.input_language.value
        )
        
        # Track entities mentioned in conversation
        if interaction.entities:
            current_entities = await self.conversation_manager.get_context_variable(
                session_id,
                "mentioned_entities",
                {}
            )
            
            for entity_name, entity_value in interaction.entities.items():
                current_entities[entity_name] = entity_value
            
            await self.conversation_manager.set_context_variable(
                session_id,
                "mentioned_entities",
                current_entities
            )
        
        # Update intent history
        if interaction.intent:
            intent_history = await self.conversation_manager.get_context_variable(
                session_id,
                "intent_history",
                []
            )
            
            intent_history.append({
                "intent": interaction.intent,
                "timestamp": interaction.timestamp.isoformat(),
                "confidence": interaction.confidence
            })
            
            # Keep only last 10 intents
            if len(intent_history) > 10:
                intent_history = intent_history[-10:]
            
            await self.conversation_manager.set_context_variable(
                session_id,
                "intent_history",
                intent_history
            )