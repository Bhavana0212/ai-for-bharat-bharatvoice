"""
Vocabulary Learning Module for BharatVoice Assistant.

This module implements vocabulary learning from user interactions,
including new word detection, usage pattern analysis, and vocabulary expansion.
"""

import asyncio
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
import re

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    UserProfile
)

logger = logging.getLogger(__name__)


class VocabularyEntry:
    """Represents a learned vocabulary entry."""
    
    def __init__(
        self,
        word: str,
        language: LanguageCode,
        frequency: int = 1,
        contexts: Optional[List[str]] = None,
        first_seen: Optional[datetime] = None
    ):
        self.word = word.lower()
        self.language = language
        self.frequency = frequency
        self.contexts = contexts or []
        self.first_seen = first_seen or datetime.utcnow()
        self.last_seen = datetime.utcnow()
        self.confidence = 0.5  # Initial confidence
        
    def update_usage(self, context: str) -> None:
        """Update usage statistics for this vocabulary entry."""
        self.frequency += 1
        self.last_seen = datetime.utcnow()
        
        # Add context if not already present
        if context not in self.contexts:
            self.contexts.append(context)
            # Keep only recent contexts (max 10)
            if len(self.contexts) > 10:
                self.contexts = self.contexts[-10:]
        
        # Update confidence based on frequency and recency
        days_since_first = (datetime.utcnow() - self.first_seen).days
        recency_factor = min(1.0, self.frequency / max(1, days_since_first))
        self.confidence = min(0.95, 0.5 + (self.frequency * 0.1) + (recency_factor * 0.2))


class VocabularyLearner:
    """
    Learns vocabulary from user interactions and adapts to user's language patterns.
    """
    
    def __init__(
        self,
        min_frequency_threshold: int = 3,
        context_window_size: int = 5,
        learning_rate: float = 0.1
    ):
        """
        Initialize vocabulary learner.
        
        Args:
            min_frequency_threshold: Minimum frequency to consider a word learned
            context_window_size: Number of words around target word for context
            learning_rate: Rate of vocabulary adaptation
        """
        self.min_frequency_threshold = min_frequency_threshold
        self.context_window_size = context_window_size
        self.learning_rate = learning_rate
        
        # User vocabulary storage
        self._user_vocabularies: Dict[UUID, Dict[str, VocabularyEntry]] = defaultdict(dict)
        
        # Language-specific word patterns
        self._language_patterns = {
            LanguageCode.HINDI: re.compile(r'[\u0900-\u097F]+'),
            LanguageCode.TAMIL: re.compile(r'[\u0B80-\u0BFF]+'),
            LanguageCode.TELUGU: re.compile(r'[\u0C00-\u0C7F]+'),
            LanguageCode.BENGALI: re.compile(r'[\u0980-\u09FF]+'),
            LanguageCode.MARATHI: re.compile(r'[\u0900-\u097F]+'),
            LanguageCode.GUJARATI: re.compile(r'[\u0A80-\u0AFF]+'),
            LanguageCode.KANNADA: re.compile(r'[\u0C80-\u0CFF]+'),
            LanguageCode.MALAYALAM: re.compile(r'[\u0D00-\u0D7F]+'),
            LanguageCode.PUNJABI: re.compile(r'[\u0A00-\u0A7F]+'),
            LanguageCode.ODIA: re.compile(r'[\u0B00-\u0B7F]+'),
            LanguageCode.ENGLISH_IN: re.compile(r'[a-zA-Z]+')
        }
        
        # Common stop words to ignore (basic set)
        self._stop_words = {
            LanguageCode.ENGLISH_IN: {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            },
            LanguageCode.HINDI: {
                'और', 'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'है', 'हैं', 'था',
                'थे', 'थी', 'होगा', 'होगी', 'होंगे', 'यह', 'वह', 'ये', 'वे', 'मैं',
                'तुम', 'आप', 'हम', 'उन', 'इन', 'उस', 'इस'
            }
        }
        
        logger.info("Vocabulary Learner initialized")
    
    async def learn_from_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction
    ) -> Dict[str, Any]:
        """
        Learn vocabulary from user interaction.
        
        Args:
            user_id: User identifier
            interaction: User interaction to learn from
            
        Returns:
            Learning results with new vocabulary discovered
        """
        learning_result = {
            "new_words": [],
            "updated_words": [],
            "total_vocabulary_size": 0,
            "language_distribution": {}
        }
        
        # Extract words from input text
        words = await self._extract_words(interaction.input_text, interaction.input_language)
        
        # Process each word
        for word, context in words:
            if await self._should_learn_word(word, interaction.input_language):
                vocab_key = f"{word}_{interaction.input_language.value}"
                
                if vocab_key in self._user_vocabularies[user_id]:
                    # Update existing vocabulary entry
                    entry = self._user_vocabularies[user_id][vocab_key]
                    entry.update_usage(context)
                    learning_result["updated_words"].append({
                        "word": word,
                        "language": interaction.input_language.value,
                        "frequency": entry.frequency,
                        "confidence": entry.confidence
                    })
                else:
                    # Create new vocabulary entry
                    entry = VocabularyEntry(
                        word=word,
                        language=interaction.input_language,
                        contexts=[context]
                    )
                    self._user_vocabularies[user_id][vocab_key] = entry
                    learning_result["new_words"].append({
                        "word": word,
                        "language": interaction.input_language.value,
                        "context": context
                    })
        
        # Update learning statistics
        learning_result["total_vocabulary_size"] = len(self._user_vocabularies[user_id])
        learning_result["language_distribution"] = await self._get_language_distribution(user_id)
        
        logger.info(f"Learned {len(learning_result['new_words'])} new words for user {user_id}")
        return learning_result
    
    async def get_user_vocabulary(
        self,
        user_id: UUID,
        language: Optional[LanguageCode] = None,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Get user's learned vocabulary.
        
        Args:
            user_id: User identifier
            language: Filter by specific language (optional)
            min_confidence: Minimum confidence threshold
            
        Returns:
            User vocabulary data
        """
        user_vocab = self._user_vocabularies.get(user_id, {})
        
        filtered_vocab = {}
        for vocab_key, entry in user_vocab.items():
            if entry.confidence >= min_confidence:
                if language is None or entry.language == language:
                    filtered_vocab[vocab_key] = {
                        "word": entry.word,
                        "language": entry.language.value,
                        "frequency": entry.frequency,
                        "confidence": entry.confidence,
                        "contexts": entry.contexts[-3:],  # Recent contexts
                        "first_seen": entry.first_seen.isoformat(),
                        "last_seen": entry.last_seen.isoformat()
                    }
        
        return {
            "vocabulary": filtered_vocab,
            "total_words": len(filtered_vocab),
            "languages": list(set(entry["language"] for entry in filtered_vocab.values()))
        }
    
    async def suggest_vocabulary_expansion(
        self,
        user_id: UUID,
        target_language: LanguageCode
    ) -> List[Dict[str, Any]]:
        """
        Suggest vocabulary expansion based on user's learning patterns.
        
        Args:
            user_id: User identifier
            target_language: Language to suggest expansions for
            
        Returns:
            List of vocabulary expansion suggestions
        """
        user_vocab = self._user_vocabularies.get(user_id, {})
        suggestions = []
        
        # Find frequently used words that could have related terms
        frequent_words = []
        for entry in user_vocab.values():
            if (entry.language == target_language and 
                entry.frequency >= self.min_frequency_threshold):
                frequent_words.append(entry)
        
        # Sort by frequency and confidence
        frequent_words.sort(key=lambda x: x.frequency * x.confidence, reverse=True)
        
        # Generate suggestions based on word categories
        for entry in frequent_words[:10]:  # Top 10 frequent words
            word_suggestions = await self._generate_related_words(entry.word, target_language)
            for suggestion in word_suggestions:
                suggestions.append({
                    "suggested_word": suggestion,
                    "related_to": entry.word,
                    "language": target_language.value,
                    "reason": "related_vocabulary",
                    "confidence": 0.8
                })
        
        return suggestions[:20]  # Return top 20 suggestions
    
    async def analyze_vocabulary_growth(
        self,
        user_id: UUID,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze user's vocabulary growth over time.
        
        Args:
            user_id: User identifier
            days_back: Number of days to analyze
            
        Returns:
            Vocabulary growth analysis
        """
        user_vocab = self._user_vocabularies.get(user_id, {})
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Analyze growth by time periods
        weekly_growth = defaultdict(int)
        language_growth = defaultdict(int)
        
        for entry in user_vocab.values():
            if entry.first_seen >= cutoff_date:
                # Calculate week number
                week_num = (entry.first_seen - cutoff_date).days // 7
                weekly_growth[week_num] += 1
                language_growth[entry.language.value] += 1
        
        # Calculate growth rate
        total_new_words = sum(weekly_growth.values())
        avg_weekly_growth = total_new_words / max(1, days_back // 7)
        
        return {
            "total_vocabulary": len(user_vocab),
            "new_words_period": total_new_words,
            "average_weekly_growth": avg_weekly_growth,
            "weekly_breakdown": dict(weekly_growth),
            "language_breakdown": dict(language_growth),
            "growth_trend": "increasing" if avg_weekly_growth > 2 else "stable"
        }
    
    async def _extract_words(
        self,
        text: str,
        language: LanguageCode
    ) -> List[Tuple[str, str]]:
        """
        Extract words with context from text.
        
        Args:
            text: Input text
            language: Text language
            
        Returns:
            List of (word, context) tuples
        """
        words_with_context = []
        
        # Get language-specific pattern
        pattern = self._language_patterns.get(language, re.compile(r'\w+'))
        
        # Find all words
        words = pattern.findall(text.lower())
        text_words = text.lower().split()
        
        for word in words:
            if len(word) > 2:  # Ignore very short words
                # Find context around the word
                try:
                    word_index = text_words.index(word)
                    start_idx = max(0, word_index - self.context_window_size)
                    end_idx = min(len(text_words), word_index + self.context_window_size + 1)
                    context = " ".join(text_words[start_idx:end_idx])
                    words_with_context.append((word, context))
                except ValueError:
                    # Word not found in split text, use the word itself as context
                    words_with_context.append((word, word))
        
        return words_with_context
    
    async def _should_learn_word(self, word: str, language: LanguageCode) -> bool:
        """
        Determine if a word should be learned.
        
        Args:
            word: Word to evaluate
            language: Word language
            
        Returns:
            True if word should be learned
        """
        # Skip very short or very long words
        if len(word) < 3 or len(word) > 20:
            return False
        
        # Skip stop words
        stop_words = self._stop_words.get(language, set())
        if word.lower() in stop_words:
            return False
        
        # Skip numbers and special characters
        if re.match(r'^[\d\W]+$', word):
            return False
        
        return True
    
    async def _get_language_distribution(self, user_id: UUID) -> Dict[str, int]:
        """Get distribution of vocabulary by language."""
        user_vocab = self._user_vocabularies.get(user_id, {})
        distribution = defaultdict(int)
        
        for entry in user_vocab.values():
            distribution[entry.language.value] += 1
        
        return dict(distribution)
    
    async def _generate_related_words(
        self,
        word: str,
        language: LanguageCode
    ) -> List[str]:
        """
        Generate related words for vocabulary expansion.
        
        Args:
            word: Base word
            language: Target language
            
        Returns:
            List of related words
        """
        # This is a simplified implementation
        # In production, this would use word embeddings or language models
        related_words = []
        
        # Basic word relationship patterns
        if language == LanguageCode.ENGLISH_IN:
            # Add common suffixes/prefixes
            if word.endswith('ing'):
                base = word[:-3]
                related_words.extend([base, base + 'ed', base + 'er'])
            elif word.endswith('ed'):
                base = word[:-2]
                related_words.extend([base, base + 'ing', base + 'er'])
        
        # For Indian languages, this would include morphological variations
        # This is a placeholder for more sophisticated word relationship detection
        
        return related_words[:5]  # Return top 5 related words
    
    async def cleanup_old_vocabulary(
        self,
        user_id: UUID,
        days_threshold: int = 90
    ) -> int:
        """
        Clean up old, unused vocabulary entries.
        
        Args:
            user_id: User identifier
            days_threshold: Days after which to consider vocabulary old
            
        Returns:
            Number of entries cleaned up
        """
        user_vocab = self._user_vocabularies.get(user_id, {})
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        
        entries_to_remove = []
        for vocab_key, entry in user_vocab.items():
            if (entry.last_seen < cutoff_date and 
                entry.frequency < self.min_frequency_threshold):
                entries_to_remove.append(vocab_key)
        
        # Remove old entries
        for vocab_key in entries_to_remove:
            del user_vocab[vocab_key]
        
        logger.info(f"Cleaned up {len(entries_to_remove)} old vocabulary entries for user {user_id}")
        return len(entries_to_remove)