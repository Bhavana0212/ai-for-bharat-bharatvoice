"""
A/B Testing Framework for BharatVoice Assistant.

This module provides A/B testing capabilities for feature improvements,
model comparisons, and user experience optimization.
"""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from enum import Enum
import statistics

from bharatvoice.core.models import UserInteraction, LanguageCode

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """A/B experiment status states."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(str, Enum):
    """Types of experiment variants."""
    
    CONTROL = "control"
    TREATMENT = "treatment"


class MetricType(str, Enum):
    """Types of metrics to track."""
    
    CONVERSION_RATE = "conversion_rate"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    ACCURACY_SCORE = "accuracy_score"
    ENGAGEMENT_RATE = "engagement_rate"
    ERROR_RATE = "error_rate"


class ExperimentVariant:
    """Represents a variant in an A/B experiment."""
    
    def __init__(
        self,
        variant_id: str,
        name: str,
        variant_type: VariantType,
        traffic_allocation: float,
        configuration: Dict[str, Any]
    ):
        self.variant_id = variant_id
        self.name = name
        self.variant_type = variant_type
        self.traffic_allocation = traffic_allocation  # 0.0 to 1.0
        self.configuration = configuration
        
        # Metrics tracking
        self.metrics: Dict[MetricType, List[float]] = {metric: [] for metric in MetricType}
        self.user_assignments: List[UUID] = []
        self.interaction_count = 0
        self.created_at = datetime.utcnow()
    
    def add_metric_value(self, metric_type: MetricType, value: float) -> None:
        """Add a metric value for this variant."""
        self.metrics[metric_type].append(value)
    
    def get_metric_stats(self, metric_type: MetricType) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        values = self.metrics[metric_type]
        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "variant_type": self.variant_type.value,
            "traffic_allocation": self.traffic_allocation,
            "configuration": self.configuration,
            "user_count": len(self.user_assignments),
            "interaction_count": self.interaction_count,
            "created_at": self.created_at.isoformat(),
            "metric_stats": {
                metric.value: self.get_metric_stats(metric)
                for metric in MetricType
            }
        }


class ABExperiment:
    """Represents an A/B experiment."""
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        hypothesis: str,
        primary_metric: MetricType,
        secondary_metrics: Optional[List[MetricType]] = None
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.hypothesis = hypothesis
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or []
        
        self.status = ExperimentStatus.DRAFT
        self.variants: Dict[str, ExperimentVariant] = {}
        self.user_assignments: Dict[UUID, str] = {}  # user_id -> variant_id
        
        # Experiment configuration
        self.target_languages: List[LanguageCode] = []
        self.target_user_segments: List[str] = []
        self.minimum_sample_size = 100
        self.confidence_level = 0.95
        self.statistical_power = 0.8
        
        # Timing
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self.duration_days: Optional[int] = None
        
        # Results
        self.results: Dict[str, Any] = {}
        self.winner_variant_id: Optional[str] = None
        self.statistical_significance: Optional[float] = None
    
    def add_variant(self, variant: ExperimentVariant) -> None:
        """Add a variant to the experiment."""
        self.variants[variant.variant_id] = variant
    
    def assign_user_to_variant(self, user_id: UUID) -> Optional[str]:
        """
        Assign user to a variant based on traffic allocation.
        
        Args:
            user_id: User identifier
            
        Returns:
            Assigned variant ID or None if experiment not active
        """
        if self.status != ExperimentStatus.ACTIVE:
            return None
        
        # Check if user already assigned
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]
        
        # Assign based on traffic allocation
        rand_value = random.random()
        cumulative_allocation = 0.0
        
        for variant_id, variant in self.variants.items():
            cumulative_allocation += variant.traffic_allocation
            if rand_value <= cumulative_allocation:
                self.user_assignments[user_id] = variant_id
                variant.user_assignments.append(user_id)
                return variant_id
        
        # Fallback to control variant
        control_variants = [v for v in self.variants.values() if v.variant_type == VariantType.CONTROL]
        if control_variants:
            variant_id = control_variants[0].variant_id
            self.user_assignments[user_id] = variant_id
            control_variants[0].user_assignments.append(user_id)
            return variant_id
        
        return None
    
    def record_interaction(
        self,
        user_id: UUID,
        interaction: UserInteraction,
        metrics: Dict[MetricType, float]
    ) -> None:
        """
        Record interaction metrics for assigned variant.
        
        Args:
            user_id: User identifier
            interaction: User interaction
            metrics: Metrics to record
        """
        variant_id = self.user_assignments.get(user_id)
        if not variant_id or variant_id not in self.variants:
            return
        
        variant = self.variants[variant_id]
        variant.interaction_count += 1
        
        # Record metrics
        for metric_type, value in metrics.items():
            variant.add_metric_value(metric_type, value)
    
    def calculate_results(self) -> Dict[str, Any]:
        """Calculate experiment results and statistical significance."""
        if len(self.variants) < 2:
            return {"error": "Need at least 2 variants for comparison"}
        
        results = {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "variants": {},
            "comparisons": {},
            "winner": None,
            "statistical_significance": None,
            "confidence_level": self.confidence_level
        }
        
        # Calculate variant statistics
        for variant_id, variant in self.variants.items():
            results["variants"][variant_id] = variant.to_dict()
        
        # Perform statistical comparisons
        control_variants = [v for v in self.variants.values() if v.variant_type == VariantType.CONTROL]
        treatment_variants = [v for v in self.variants.values() if v.variant_type == VariantType.TREATMENT]
        
        if control_variants and treatment_variants:
            control = control_variants[0]
            
            for treatment in treatment_variants:
                comparison_key = f"{control.variant_id}_vs_{treatment.variant_id}"
                comparison = self._compare_variants(control, treatment)
                results["comparisons"][comparison_key] = comparison
                
                # Determine winner for primary metric
                if comparison.get("primary_metric_significant"):
                    primary_comparison = comparison["metrics"].get(self.primary_metric.value, {})
                    if primary_comparison.get("treatment_better", False):
                        results["winner"] = treatment.variant_id
                        results["statistical_significance"] = primary_comparison.get("p_value", 1.0)
        
        self.results = results
        return results
    
    def is_ready_for_analysis(self) -> bool:
        """Check if experiment has enough data for analysis."""
        for variant in self.variants.values():
            if len(variant.user_assignments) < self.minimum_sample_size:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "primary_metric": self.primary_metric.value,
            "secondary_metrics": [m.value for m in self.secondary_metrics],
            "status": self.status.value,
            "variants": {vid: v.to_dict() for vid, v in self.variants.items()},
            "total_users": len(self.user_assignments),
            "target_languages": [lang.value for lang in self.target_languages],
            "target_user_segments": self.target_user_segments,
            "minimum_sample_size": self.minimum_sample_size,
            "confidence_level": self.confidence_level,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_days": self.duration_days,
            "results": self.results,
            "winner_variant_id": self.winner_variant_id,
            "statistical_significance": self.statistical_significance
        }
    
    def _compare_variants(
        self,
        control: ExperimentVariant,
        treatment: ExperimentVariant
    ) -> Dict[str, Any]:
        """Compare two variants statistically."""
        comparison = {
            "control_variant": control.variant_id,
            "treatment_variant": treatment.variant_id,
            "metrics": {},
            "primary_metric_significant": False
        }
        
        # Compare each metric
        all_metrics = [self.primary_metric] + self.secondary_metrics
        
        for metric in all_metrics:
            control_values = control.metrics[metric]
            treatment_values = treatment.metrics[metric]
            
            if len(control_values) < 10 or len(treatment_values) < 10:
                # Not enough data for statistical test
                comparison["metrics"][metric.value] = {
                    "insufficient_data": True,
                    "control_mean": statistics.mean(control_values) if control_values else 0,
                    "treatment_mean": statistics.mean(treatment_values) if treatment_values else 0
                }
                continue
            
            # Perform t-test (simplified)
            control_mean = statistics.mean(control_values)
            treatment_mean = statistics.mean(treatment_values)
            
            # Calculate effect size (Cohen's d)
            pooled_std = self._calculate_pooled_std(control_values, treatment_values)
            cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Simplified p-value calculation (in production, use proper statistical tests)
            p_value = self._calculate_p_value(control_values, treatment_values)
            
            is_significant = p_value < (1 - self.confidence_level)
            treatment_better = treatment_mean > control_mean
            
            comparison["metrics"][metric.value] = {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "difference": treatment_mean - control_mean,
                "relative_improvement": ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0,
                "cohens_d": cohens_d,
                "p_value": p_value,
                "is_significant": is_significant,
                "treatment_better": treatment_better
            }
            
            # Check if primary metric is significant
            if metric == self.primary_metric and is_significant:
                comparison["primary_metric_significant"] = True
        
        return comparison
    
    def _calculate_pooled_std(self, values1: List[float], values2: List[float]) -> float:
        """Calculate pooled standard deviation."""
        if len(values1) <= 1 or len(values2) <= 1:
            return 0.0
        
        n1, n2 = len(values1), len(values2)
        var1 = statistics.variance(values1)
        var2 = statistics.variance(values2)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return pooled_var ** 0.5
    
    def _calculate_p_value(self, values1: List[float], values2: List[float]) -> float:
        """Simplified p-value calculation (placeholder for proper statistical test)."""
        # This is a simplified placeholder
        # In production, use scipy.stats.ttest_ind or similar
        
        if len(values1) < 10 or len(values2) < 10:
            return 1.0
        
        mean1, mean2 = statistics.mean(values1), statistics.mean(values2)
        std1, std2 = statistics.stdev(values1), statistics.stdev(values2)
        
        # Simplified calculation
        if std1 == 0 and std2 == 0:
            return 0.0 if mean1 != mean2 else 1.0
        
        # Return a placeholder p-value based on difference magnitude
        difference = abs(mean1 - mean2)
        avg_std = (std1 + std2) / 2
        
        if avg_std == 0:
            return 0.0 if difference > 0 else 1.0
        
        z_score = difference / avg_std
        
        # Rough p-value approximation
        if z_score > 2.58:  # 99% confidence
            return 0.01
        elif z_score > 1.96:  # 95% confidence
            return 0.05
        elif z_score > 1.64:  # 90% confidence
            return 0.10
        else:
            return 0.20


class ABTestingFramework:
    """
    A/B Testing Framework for feature improvements and optimization.
    """
    
    def __init__(
        self,
        default_confidence_level: float = 0.95,
        default_statistical_power: float = 0.8,
        default_minimum_sample_size: int = 100
    ):
        """
        Initialize A/B testing framework.
        
        Args:
            default_confidence_level: Default confidence level for experiments
            default_statistical_power: Default statistical power
            default_minimum_sample_size: Default minimum sample size
        """
        self.default_confidence_level = default_confidence_level
        self.default_statistical_power = default_statistical_power
        self.default_minimum_sample_size = default_minimum_sample_size
        
        # Experiment registry
        self._experiments: Dict[str, ABExperiment] = {}
        
        # Framework statistics
        self._framework_stats = {
            "total_experiments": 0,
            "active_experiments": 0,
            "completed_experiments": 0,
            "total_users_in_experiments": 0
        }
        
        logger.info("A/B Testing Framework initialized")
    
    async def create_experiment(
        self,
        name: str,
        description: str,
        hypothesis: str,
        primary_metric: MetricType,
        secondary_metrics: Optional[List[MetricType]] = None,
        target_languages: Optional[List[LanguageCode]] = None,
        duration_days: Optional[int] = None
    ) -> str:
        """
        Create a new A/B experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            hypothesis: Hypothesis being tested
            primary_metric: Primary metric to optimize
            secondary_metrics: Additional metrics to track
            target_languages: Target languages for experiment
            duration_days: Experiment duration in days
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid4())
        
        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics
        )
        
        experiment.target_languages = target_languages or []
        experiment.duration_days = duration_days
        experiment.confidence_level = self.default_confidence_level
        experiment.statistical_power = self.default_statistical_power
        experiment.minimum_sample_size = self.default_minimum_sample_size
        
        self._experiments[experiment_id] = experiment
        self._framework_stats["total_experiments"] += 1
        
        logger.info(f"Created experiment: {name} ({experiment_id})")
        return experiment_id
    
    async def add_variant(
        self,
        experiment_id: str,
        variant_name: str,
        variant_type: VariantType,
        traffic_allocation: float,
        configuration: Dict[str, Any]
    ) -> str:
        """
        Add a variant to an experiment.
        
        Args:
            experiment_id: Experiment identifier
            variant_name: Variant name
            variant_type: Variant type (control/treatment)
            traffic_allocation: Traffic allocation (0.0 to 1.0)
            configuration: Variant configuration
            
        Returns:
            Variant ID
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self._experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot add variants to {experiment.status.value} experiment")
        
        variant_id = f"{experiment_id}_{variant_type.value}_{len(experiment.variants)}"
        
        variant = ExperimentVariant(
            variant_id=variant_id,
            name=variant_name,
            variant_type=variant_type,
            traffic_allocation=traffic_allocation,
            configuration=configuration
        )
        
        experiment.add_variant(variant)
        
        logger.info(f"Added variant {variant_name} to experiment {experiment_id}")
        return variant_id
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if started successfully
        """
        if experiment_id not in self._experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self._experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Cannot start experiment in {experiment.status.value} status")
            return False
        
        # Validate experiment setup
        if len(experiment.variants) < 2:
            logger.error("Experiment needs at least 2 variants")
            return False
        
        # Check traffic allocation sums to 1.0
        total_allocation = sum(v.traffic_allocation for v in experiment.variants.values())
        if abs(total_allocation - 1.0) > 0.01:
            logger.error(f"Traffic allocation sums to {total_allocation}, should be 1.0")
            return False
        
        # Start experiment
        experiment.status = ExperimentStatus.ACTIVE
        experiment.started_at = datetime.utcnow()
        self._framework_stats["active_experiments"] += 1
        
        logger.info(f"Started experiment: {experiment.name} ({experiment_id})")
        return True
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """
        Stop an active experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            True if stopped successfully
        """
        if experiment_id not in self._experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self._experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            logger.error(f"Cannot stop experiment in {experiment.status.value} status")
            return False
        
        # Stop experiment
        experiment.status = ExperimentStatus.COMPLETED
        experiment.ended_at = datetime.utcnow()
        self._framework_stats["active_experiments"] -= 1
        self._framework_stats["completed_experiments"] += 1
        
        # Calculate final results
        experiment.calculate_results()
        
        logger.info(f"Stopped experiment: {experiment.name} ({experiment_id})")
        return True
    
    async def assign_user_to_experiment(
        self,
        experiment_id: str,
        user_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Assign user to experiment variant.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            
        Returns:
            Assignment information or None
        """
        if experiment_id not in self._experiments:
            return None
        
        experiment = self._experiments[experiment_id]
        variant_id = experiment.assign_user_to_variant(user_id)
        
        if variant_id:
            variant = experiment.variants[variant_id]
            return {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "variant_name": variant.name,
                "variant_type": variant.variant_type.value,
                "configuration": variant.configuration
            }
        
        return None
    
    async def record_interaction_metrics(
        self,
        experiment_id: str,
        user_id: UUID,
        interaction: UserInteraction,
        response_time: Optional[float] = None,
        user_satisfaction: Optional[float] = None,
        accuracy_score: Optional[float] = None,
        conversion: bool = False,
        error_occurred: bool = False
    ) -> None:
        """
        Record interaction metrics for experiment analysis.
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            interaction: User interaction
            response_time: Response time in seconds
            user_satisfaction: User satisfaction score (0-5)
            accuracy_score: Accuracy score (0-1)
            conversion: Whether conversion occurred
            error_occurred: Whether error occurred
        """
        if experiment_id not in self._experiments:
            return
        
        experiment = self._experiments[experiment_id]
        
        # Calculate metrics
        metrics = {}
        
        if conversion:
            metrics[MetricType.CONVERSION_RATE] = 1.0
        else:
            metrics[MetricType.CONVERSION_RATE] = 0.0
        
        if response_time is not None:
            metrics[MetricType.RESPONSE_TIME] = response_time
        
        if user_satisfaction is not None:
            metrics[MetricType.USER_SATISFACTION] = user_satisfaction
        
        if accuracy_score is not None:
            metrics[MetricType.ACCURACY_SCORE] = accuracy_score
        
        # Engagement rate (simplified)
        metrics[MetricType.ENGAGEMENT_RATE] = 1.0 if len(interaction.input_text) > 10 else 0.5
        
        # Error rate
        metrics[MetricType.ERROR_RATE] = 1.0 if error_occurred else 0.0
        
        # Record metrics
        experiment.record_interaction(user_id, interaction, metrics)
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment results and analysis.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment results or None
        """
        if experiment_id not in self._experiments:
            return None
        
        experiment = self._experiments[experiment_id]
        
        # Calculate fresh results
        results = experiment.calculate_results()
        
        # Add readiness status
        results["ready_for_analysis"] = experiment.is_ready_for_analysis()
        results["sample_size_met"] = all(
            len(v.user_assignments) >= experiment.minimum_sample_size
            for v in experiment.variants.values()
        )
        
        return results
    
    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        language: Optional[LanguageCode] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filters.
        
        Args:
            status: Filter by status
            language: Filter by target language
            
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for experiment in self._experiments.values():
            # Apply filters
            if status and experiment.status != status:
                continue
            if language and language not in experiment.target_languages:
                continue
            
            experiments.append(experiment.to_dict())
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        return experiments
    
    async def get_user_experiments(self, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get experiments that a user is participating in.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of user's experiment assignments
        """
        user_experiments = []
        
        for experiment in self._experiments.values():
            if user_id in experiment.user_assignments:
                variant_id = experiment.user_assignments[user_id]
                variant = experiment.variants[variant_id]
                
                user_experiments.append({
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.name,
                    "variant_id": variant_id,
                    "variant_name": variant.name,
                    "variant_type": variant.variant_type.value,
                    "configuration": variant.configuration
                })
        
        return user_experiments
    
    async def get_framework_statistics(self) -> Dict[str, Any]:
        """Get A/B testing framework statistics."""
        stats = self._framework_stats.copy()
        
        # Calculate additional stats
        total_users = set()
        for experiment in self._experiments.values():
            total_users.update(experiment.user_assignments.keys())
        
        stats["total_users_in_experiments"] = len(total_users)
        
        # Active experiment details
        active_experiments = [
            exp for exp in self._experiments.values()
            if exp.status == ExperimentStatus.ACTIVE
        ]
        
        stats["active_experiment_details"] = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "users_assigned": len(exp.user_assignments),
                "variants": len(exp.variants),
                "ready_for_analysis": exp.is_ready_for_analysis()
            }
            for exp in active_experiments
        ]
        
        return stats
    
    async def cleanup_old_experiments(self, days_threshold: int = 90) -> int:
        """
        Clean up old completed experiments.
        
        Args:
            days_threshold: Days threshold for cleanup
            
        Returns:
            Number of experiments cleaned up
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
        experiments_to_remove = []
        
        for experiment_id, experiment in self._experiments.items():
            if (experiment.status == ExperimentStatus.COMPLETED and
                experiment.ended_at and
                experiment.ended_at < cutoff_date):
                experiments_to_remove.append(experiment_id)
        
        for experiment_id in experiments_to_remove:
            del self._experiments[experiment_id]
        
        logger.info(f"Cleaned up {len(experiments_to_remove)} old experiments")
        return len(experiments_to_remove)