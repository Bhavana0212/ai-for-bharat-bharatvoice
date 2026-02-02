"""
Feedback Processing Module for BharatVoice Assistant.

This module processes user feedback to improve response quality and system performance.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
from enum import Enum

from bharatvoice.core.models import (
    UserInteraction,
    LanguageCode,
    Response
)

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback that can be collected."""
    
    RATING = "rating"  # Numerical rating (1-5)
    THUMBS = "thumbs"  # Thumbs up/down
    CORRECTION = "correction"  # User correction of response
    COMPLAINT = "complaint"  # User complaint about response
    SUGGESTION = "suggestion"  # User suggestion for improvement
    IMPLICIT = "implicit"  # Implicit feedback from behavior


class FeedbackSeverity(str, Enum):
    """Severity levels for feedback."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackEntry:
    """Represents a single feedback entry."""
    
    def __init__(
        self,
        feedback_id: UUID,
        user_id: UUID,
        interaction_id: UUID,
        feedback_type: FeedbackType,
        content: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.feedback_id = feedback_id
        self.user_id = user_id
        self.interaction_id = interaction_id
        self.feedback_type = feedback_type
        self.content = content
        self.timestamp = timestamp or datetime.utcnow()
        self.processed = False
        self.severity = FeedbackSeverity.LOW
        self.improvement_actions: List[str] = []
        
    def calculate_severity(self) -> FeedbackSeverity:
        """Calculate feedback severity based on content."""
        if self.feedback_type == FeedbackType.RATING:
            rating = self.content.get("rating", 3)
            if rating <= 1:
                return FeedbackSeverity.CRITICAL
            elif rating <= 2:
                return FeedbackSeverity.HIGH
            elif rating <= 3:
                return FeedbackSeverity.MEDIUM
            else:
                return FeedbackSeverity.LOW
        
        elif self.feedback_type == FeedbackType.THUMBS:
            return FeedbackSeverity.HIGH if not self.content.get("positive", True) else FeedbackSeverity.LOW
        
        elif self.feedback_type == FeedbackType.COMPLAINT:
            return FeedbackSeverity.HIGH
        
        elif self.feedback_type == FeedbackType.CORRECTION:
            return FeedbackSeverity.MEDIUM
        
        return FeedbackSeverity.LOW


class FeedbackProcessor:
    """
    Processes user feedback to improve system responses and performance.
    """
    
    def __init__(
        self,
        feedback_retention_days: int = 180,
        min_feedback_for_action: int = 3
    ):
        """
        Initialize feedback processor.
        
        Args:
            feedback_retention_days: Days to retain feedback data
            min_feedback_for_action: Minimum feedback count before taking action
        """
        self.feedback_retention_days = feedback_retention_days
        self.min_feedback_for_action = min_feedback_for_action
        
        # Feedback storage
        self._feedback_entries: Dict[UUID, FeedbackEntry] = {}
        self._user_feedback: Dict[UUID, List[UUID]] = defaultdict(list)
        self._interaction_feedback: Dict[UUID, List[UUID]] = defaultdict(list)
        
        # Feedback analysis results
        self._feedback_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._improvement_suggestions: List[Dict[str, Any]] = []
        
        logger.info("Feedback Processor initialized")
    
    async def collect_feedback(
        self,
        user_id: UUID,
        interaction_id: UUID,
        feedback_type: FeedbackType,
        feedback_content: Dict[str, Any]
    ) -> UUID:
        """
        Collect feedback from user.
        
        Args:
            user_id: User identifier
            interaction_id: Interaction identifier
            feedback_type: Type of feedback
            feedback_content: Feedback content
            
        Returns:
            Feedback entry ID
        """
        feedback_id = uuid4()
        
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            user_id=user_id,
            interaction_id=interaction_id,
            feedback_type=feedback_type,
            content=feedback_content
        )
        
        feedback_entry.severity = feedback_entry.calculate_severity()
        
        # Store feedback
        self._feedback_entries[feedback_id] = feedback_entry
        self._user_feedback[user_id].append(feedback_id)
        self._interaction_feedback[interaction_id].append(feedback_id)
        
        # Process feedback immediately if critical
        if feedback_entry.severity == FeedbackSeverity.CRITICAL:
            await self._process_critical_feedback(feedback_entry)
        
        logger.info(f"Collected {feedback_type.value} feedback from user {user_id}")
        return feedback_id
    
    async def process_feedback_batch(
        self,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Process a batch of unprocessed feedback.
        
        Args:
            batch_size: Number of feedback entries to process
            
        Returns:
            Processing results
        """
        unprocessed_feedback = [
            entry for entry in self._feedback_entries.values()
            if not entry.processed
        ]
        
        # Sort by severity and timestamp
        unprocessed_feedback.sort(
            key=lambda x: (x.severity.value, x.timestamp),
            reverse=True
        )
        
        batch = unprocessed_feedback[:batch_size]
        processing_results = {
            "processed_count": 0,
            "improvements_identified": [],
            "patterns_detected": [],
            "actions_recommended": []
        }
        
        for feedback_entry in batch:
            result = await self._process_single_feedback(feedback_entry)
            processing_results["processed_count"] += 1
            
            if result.get("improvements"):
                processing_results["improvements_identified"].extend(result["improvements"])
            
            if result.get("patterns"):
                processing_results["patterns_detected"].extend(result["patterns"])
            
            if result.get("actions"):
                processing_results["actions_recommended"].extend(result["actions"])
            
            feedback_entry.processed = True
        
        logger.info(f"Processed {processing_results['processed_count']} feedback entries")
        return processing_results
    
    async def analyze_feedback_trends(
        self,
        user_id: Optional[UUID] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends over time.
        
        Args:
            user_id: Specific user to analyze (optional)
            days_back: Number of days to analyze
            
        Returns:
            Feedback trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Filter feedback by date and user
        relevant_feedback = []
        for entry in self._feedback_entries.values():
            if entry.timestamp >= cutoff_date:
                if user_id is None or entry.user_id == user_id:
                    relevant_feedback.append(entry)
        
        if not relevant_feedback:
            return {"message": "No feedback data available for analysis"}
        
        # Analyze trends
        trends = {
            "total_feedback": len(relevant_feedback),
            "feedback_by_type": defaultdict(int),
            "feedback_by_severity": defaultdict(int),
            "average_rating": 0.0,
            "satisfaction_trend": "stable",
            "common_issues": [],
            "improvement_areas": []
        }
        
        ratings = []
        daily_feedback = defaultdict(int)
        
        for entry in relevant_feedback:
            trends["feedback_by_type"][entry.feedback_type.value] += 1
            trends["feedback_by_severity"][entry.severity.value] += 1
            
            # Collect ratings
            if entry.feedback_type == FeedbackType.RATING:
                rating = entry.content.get("rating", 3)
                ratings.append(rating)
            
            # Daily feedback count
            day_key = entry.timestamp.date().isoformat()
            daily_feedback[day_key] += 1
        
        # Calculate average rating
        if ratings:
            trends["average_rating"] = sum(ratings) / len(ratings)
        
        # Determine satisfaction trend
        if len(ratings) >= 10:
            recent_ratings = ratings[-5:]
            older_ratings = ratings[-10:-5] if len(ratings) >= 10 else ratings[:-5]
            
            if recent_ratings and older_ratings:
                recent_avg = sum(recent_ratings) / len(recent_ratings)
                older_avg = sum(older_ratings) / len(older_ratings)
                
                if recent_avg > older_avg + 0.3:
                    trends["satisfaction_trend"] = "improving"
                elif recent_avg < older_avg - 0.3:
                    trends["satisfaction_trend"] = "declining"
        
        # Identify common issues
        trends["common_issues"] = await self._identify_common_issues(relevant_feedback)
        trends["improvement_areas"] = await self._identify_improvement_areas(relevant_feedback)
        
        return trends
    
    async def get_improvement_suggestions(
        self,
        category: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get improvement suggestions based on feedback analysis.
        
        Args:
            category: Filter by category (optional)
            priority: Filter by priority (optional)
            
        Returns:
            List of improvement suggestions
        """
        suggestions = self._improvement_suggestions.copy()
        
        # Apply filters
        if category:
            suggestions = [s for s in suggestions if s.get("category") == category]
        
        if priority:
            suggestions = [s for s in suggestions if s.get("priority") == priority]
        
        # Sort by priority and confidence
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        suggestions.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 1),
                x.get("confidence", 0.0)
            ),
            reverse=True
        )
        
        return suggestions
    
    async def incorporate_feedback_for_response_improvement(
        self,
        interaction_id: UUID,
        original_response: str,
        feedback_content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Incorporate feedback to improve response quality.
        
        Args:
            interaction_id: Interaction identifier
            original_response: Original response text
            feedback_content: User feedback
            
        Returns:
            Improved response if applicable
        """
        if "correction" not in feedback_content:
            return None
        
        correction = feedback_content["correction"]
        improved_response = original_response
        
        # Apply simple correction rules
        if "wrong_information" in correction:
            # Mark for fact-checking
            improved_response = f"Let me verify that information. {original_response}"
        
        elif "too_long" in correction:
            # Make response more concise
            sentences = original_response.split('. ')
            if len(sentences) > 2:
                improved_response = '. '.join(sentences[:2]) + '.'
        
        elif "too_short" in correction:
            # Add more detail
            improved_response = f"{original_response} Would you like me to provide more details about this?"
        
        elif "wrong_language" in correction:
            # Note language preference issue
            improved_response = original_response  # Would trigger language model retraining
        
        return improved_response if improved_response != original_response else None
    
    async def _process_single_feedback(
        self,
        feedback_entry: FeedbackEntry
    ) -> Dict[str, Any]:
        """Process a single feedback entry."""
        result = {
            "improvements": [],
            "patterns": [],
            "actions": []
        }
        
        # Process based on feedback type
        if feedback_entry.feedback_type == FeedbackType.RATING:
            rating = feedback_entry.content.get("rating", 3)
            
            if rating <= 2:
                result["improvements"].append({
                    "type": "low_rating",
                    "interaction_id": str(feedback_entry.interaction_id),
                    "rating": rating,
                    "suggestion": "Review interaction for quality issues"
                })
        
        elif feedback_entry.feedback_type == FeedbackType.CORRECTION:
            correction_type = feedback_entry.content.get("correction_type", "general")
            result["improvements"].append({
                "type": "user_correction",
                "correction_type": correction_type,
                "original": feedback_entry.content.get("original", ""),
                "corrected": feedback_entry.content.get("corrected", ""),
                "suggestion": f"Update {correction_type} handling"
            })
        
        elif feedback_entry.feedback_type == FeedbackType.COMPLAINT:
            complaint_category = feedback_entry.content.get("category", "general")
            result["actions"].append({
                "type": "investigate_complaint",
                "category": complaint_category,
                "priority": feedback_entry.severity.value,
                "description": feedback_entry.content.get("description", "")
            })
        
        # Store improvement actions in feedback entry
        feedback_entry.improvement_actions = [
            action.get("suggestion", "") for action in 
            result["improvements"] + result["actions"]
        ]
        
        return result
    
    async def _process_critical_feedback(
        self,
        feedback_entry: FeedbackEntry
    ) -> None:
        """Process critical feedback immediately."""
        logger.warning(f"Critical feedback received: {feedback_entry.feedback_id}")
        
        # Add to high-priority improvement suggestions
        suggestion = {
            "id": str(feedback_entry.feedback_id),
            "category": "critical_issue",
            "priority": "critical",
            "description": f"Critical feedback from user {feedback_entry.user_id}",
            "feedback_type": feedback_entry.feedback_type.value,
            "content": feedback_entry.content,
            "timestamp": feedback_entry.timestamp.isoformat(),
            "confidence": 1.0
        }
        
        self._improvement_suggestions.append(suggestion)
    
    async def _identify_common_issues(
        self,
        feedback_entries: List[FeedbackEntry]
    ) -> List[Dict[str, Any]]:
        """Identify common issues from feedback."""
        issue_counts = defaultdict(int)
        issue_examples = defaultdict(list)
        
        for entry in feedback_entries:
            if entry.feedback_type in [FeedbackType.COMPLAINT, FeedbackType.CORRECTION]:
                issue_type = entry.content.get("category", "general")
                issue_counts[issue_type] += 1
                issue_examples[issue_type].append({
                    "description": entry.content.get("description", ""),
                    "timestamp": entry.timestamp.isoformat()
                })
        
        # Sort by frequency
        common_issues = []
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= self.min_feedback_for_action:
                common_issues.append({
                    "issue_type": issue_type,
                    "frequency": count,
                    "examples": issue_examples[issue_type][:3]  # Top 3 examples
                })
        
        return common_issues
    
    async def _identify_improvement_areas(
        self,
        feedback_entries: List[FeedbackEntry]
    ) -> List[Dict[str, Any]]:
        """Identify areas for improvement based on feedback patterns."""
        improvement_areas = []
        
        # Analyze rating patterns
        low_ratings = [
            entry for entry in feedback_entries
            if entry.feedback_type == FeedbackType.RATING and entry.content.get("rating", 3) <= 2
        ]
        
        if len(low_ratings) >= self.min_feedback_for_action:
            improvement_areas.append({
                "area": "response_quality",
                "priority": "high",
                "description": f"{len(low_ratings)} low ratings received",
                "suggested_action": "Review and improve response generation"
            })
        
        # Analyze correction patterns
        corrections = [
            entry for entry in feedback_entries
            if entry.feedback_type == FeedbackType.CORRECTION
        ]
        
        if corrections:
            correction_types = defaultdict(int)
            for entry in corrections:
                correction_type = entry.content.get("correction_type", "general")
                correction_types[correction_type] += 1
            
            for correction_type, count in correction_types.items():
                if count >= self.min_feedback_for_action:
                    improvement_areas.append({
                        "area": f"{correction_type}_accuracy",
                        "priority": "medium",
                        "description": f"{count} corrections for {correction_type}",
                        "suggested_action": f"Improve {correction_type} processing"
                    })
        
        return improvement_areas
    
    async def cleanup_old_feedback(self) -> int:
        """Clean up old feedback entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.feedback_retention_days)
        
        entries_to_remove = []
        for feedback_id, entry in self._feedback_entries.items():
            if entry.timestamp < cutoff_date:
                entries_to_remove.append(feedback_id)
        
        # Remove old entries
        for feedback_id in entries_to_remove:
            entry = self._feedback_entries[feedback_id]
            
            # Remove from user and interaction mappings
            if feedback_id in self._user_feedback[entry.user_id]:
                self._user_feedback[entry.user_id].remove(feedback_id)
            
            if feedback_id in self._interaction_feedback[entry.interaction_id]:
                self._interaction_feedback[entry.interaction_id].remove(feedback_id)
            
            # Remove main entry
            del self._feedback_entries[feedback_id]
        
        logger.info(f"Cleaned up {len(entries_to_remove)} old feedback entries")
        return len(entries_to_remove)
    
    async def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback processing statistics."""
        total_feedback = len(self._feedback_entries)
        processed_feedback = sum(1 for entry in self._feedback_entries.values() if entry.processed)
        
        feedback_by_type = defaultdict(int)
        feedback_by_severity = defaultdict(int)
        
        for entry in self._feedback_entries.values():
            feedback_by_type[entry.feedback_type.value] += 1
            feedback_by_severity[entry.severity.value] += 1
        
        return {
            "total_feedback": total_feedback,
            "processed_feedback": processed_feedback,
            "pending_feedback": total_feedback - processed_feedback,
            "feedback_by_type": dict(feedback_by_type),
            "feedback_by_severity": dict(feedback_by_severity),
            "improvement_suggestions": len(self._improvement_suggestions),
            "active_users_with_feedback": len(self._user_feedback)
        }