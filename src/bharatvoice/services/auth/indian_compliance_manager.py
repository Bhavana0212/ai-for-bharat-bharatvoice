"""
Indian Privacy Law Compliance Manager for BharatVoice Assistant.

This module provides comprehensive compliance with Indian data protection laws
including PDPB (Personal Data Protection Bill), IT Act 2000, and related regulations.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings
from .privacy_manager import PrivacyManager, DataCategory, ConsentType


logger = structlog.get_logger(__name__)


class IndianDataClassification(str, Enum):
    """Indian data classification as per PDPB."""
    
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"
    CRITICAL_PERSONAL_DATA = "critical_personal_data"
    NON_PERSONAL_DATA = "non_personal_data"


class DataLocalizationRequirement(str, Enum):
    """Data localization requirements."""
    
    MUST_STORE_IN_INDIA = "must_store_in_india"
    COPY_IN_INDIA = "copy_in_india"
    NO_RESTRICTION = "no_restriction"


class LegalBasisIndia(str, Enum):
    """Legal basis for processing under Indian law."""
    
    CONSENT = "consent"
    LEGITIMATE_INTEREST = "legitimate_interest"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_TASK = "public_task"
    CONTRACTUAL_NECESSITY = "contractual_necessity"


class IndianDataMapping(BaseModel):
    """Mapping of data categories to Indian classifications."""
    
    category: DataCategory
    indian_classification: IndianDataClassification
    localization_requirement: DataLocalizationRequirement
    retention_period_days: int
    requires_explicit_consent: bool
    cross_border_transfer_allowed: bool
    anonymization_required: bool


class ComplianceAuditLog(BaseModel):
    """Compliance audit log entry."""
    
    log_id: str
    user_id: str
    action: str
    data_classification: IndianDataClassification
    legal_basis: LegalBasisIndia
    timestamp: datetime
    location: str  # Data processing location
    compliance_status: str
    details: Dict[str, Any]


class DataLocalizationStatus(BaseModel):
    """Data localization compliance status."""
    
    user_id: str
    data_category: DataCategory
    storage_location: str
    is_compliant: bool
    last_checked: datetime
    issues: List[str] = []


class ConsentRecord(BaseModel):
    """Enhanced consent record for Indian compliance."""
    
    consent_id: str
    user_id: str
    data_categories: List[DataCategory]
    purpose: str
    legal_basis: LegalBasisIndia
    consent_language: str  # Language in which consent was obtained
    consent_method: str  # "explicit", "implied", "opt_in", "opt_out"
    granular_consent: Dict[str, bool]  # Granular consent for different purposes
    withdrawal_method: Optional[str] = None
    is_minor: bool = False  # Special handling for minors
    guardian_consent: Optional[str] = None  # Guardian consent for minors
    created_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None


class IndianComplianceManager:
    """Indian privacy law compliance manager."""
    
    def __init__(
        self,
        settings: Settings,
        privacy_manager: PrivacyManager,
        redis_client=None,
        database=None
    ):
        """
        Initialize Indian compliance manager.
        
        Args:
            settings: Application settings
            privacy_manager: Privacy manager instance
            redis_client: Redis client for caching
            database: Database connection
        """
        self.settings = settings
        self.privacy_manager = privacy_manager
        self.redis_client = redis_client
        self.database = database
        
        # Indian compliance configuration
        self.data_mappings = self._get_indian_data_mappings()
        self.default_retention_periods = self._get_indian_retention_periods()
        self.localization_requirements = self._get_localization_requirements()
        
        # Compliance settings
        self.consent_expiry_period = timedelta(days=365)  # 1 year
        self.minor_age_threshold = 18
        self.data_breach_notification_period = timedelta(hours=72)
        self.user_response_period = timedelta(days=30)
        
        logger.info("Indian compliance manager initialized")
    
    async def validate_data_localization(self, user_id: str) -> Dict[str, DataLocalizationStatus]:
        """
        Validate data localization compliance for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of data category -> localization status
        """
        try:
            statuses = {}
            
            for category in DataCategory:
                mapping = self.data_mappings.get(category)
                if not mapping:
                    continue
                
                # Check current storage location
                storage_location = await self._get_data_storage_location(user_id, category)
                
                # Validate against requirements
                is_compliant = self._check_localization_compliance(
                    storage_location, mapping.localization_requirement
                )
                
                issues = []
                if not is_compliant:
                    if mapping.localization_requirement == DataLocalizationRequirement.MUST_STORE_IN_INDIA:
                        issues.append("Data must be stored within India")
                    elif mapping.localization_requirement == DataLocalizationRequirement.COPY_IN_INDIA:
                        issues.append("A copy of data must be maintained in India")
                
                status = DataLocalizationStatus(
                    user_id=user_id,
                    data_category=category,
                    storage_location=storage_location,
                    is_compliant=is_compliant,
                    last_checked=datetime.utcnow(),
                    issues=issues
                )
                
                statuses[category.value] = status
            
            logger.info("Data localization validation completed", user_id=user_id)
            return statuses
            
        except Exception as e:
            logger.error("Data localization validation failed", user_id=user_id, exc_info=e)
            raise
    
    async def obtain_granular_consent(
        self,
        user_id: str,
        purposes: Dict[str, str],  # purpose_id -> description
        data_categories: List[DataCategory],
        language: str = "en-IN",
        is_minor: bool = False,
        guardian_id: Optional[str] = None
    ) -> ConsentRecord:
        """
        Obtain granular consent as per Indian requirements.
        
        Args:
            user_id: User identifier
            purposes: Dictionary of purposes for data processing
            data_categories: Data categories requiring consent
            language: Language for consent (default: English-India)
            is_minor: Whether user is a minor
            guardian_id: Guardian ID if user is minor
            
        Returns:
            Consent record
        """
        try:
            consent_id = str(uuid4())
            
            # Validate minor consent requirements
            if is_minor and not guardian_id:
                raise ValueError("Guardian consent required for minors")
            
            # Create granular consent mapping
            granular_consent = {}
            for purpose_id in purposes.keys():
                granular_consent[purpose_id] = True  # Default to granted
            
            consent_record = ConsentRecord(
                consent_id=consent_id,
                user_id=user_id,
                data_categories=data_categories,
                purpose="; ".join(purposes.values()),
                legal_basis=LegalBasisIndia.CONSENT,
                consent_language=language,
                consent_method="explicit",
                granular_consent=granular_consent,
                is_minor=is_minor,
                guardian_consent=guardian_id,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + self.consent_expiry_period
            )
            
            # Store consent record
            await self._store_indian_consent(consent_record)
            
            # Log compliance audit
            await self._log_compliance_action(
                user_id=user_id,
                action="granular_consent_obtained",
                data_classification=IndianDataClassification.PERSONAL_DATA,
                legal_basis=LegalBasisIndia.CONSENT,
                details={
                    "purposes": purposes,
                    "data_categories": [cat.value for cat in data_categories],
                    "language": language,
                    "is_minor": is_minor
                }
            )
            
            logger.info(
                "Granular consent obtained",
                user_id=user_id,
                consent_id=consent_id,
                is_minor=is_minor
            )
            
            return consent_record
            
        except Exception as e:
            logger.error("Failed to obtain granular consent", user_id=user_id, exc_info=e)
            raise
    
    async def handle_data_breach(
        self,
        breach_id: str,
        affected_users: List[str],
        data_categories: List[DataCategory],
        breach_description: str,
        severity: str = "high"
    ) -> Dict[str, Any]:
        """
        Handle data breach as per Indian compliance requirements.
        
        Args:
            breach_id: Breach identifier
            affected_users: List of affected user IDs
            data_categories: Affected data categories
            breach_description: Description of the breach
            severity: Breach severity level
            
        Returns:
            Breach handling report
        """
        try:
            breach_time = datetime.utcnow()
            
            # Classify data types affected
            classifications = set()
            for category in data_categories:
                mapping = self.data_mappings.get(category)
                if mapping:
                    classifications.add(mapping.indian_classification)
            
            # Determine notification requirements
            requires_authority_notification = any(
                cls in [IndianDataClassification.SENSITIVE_PERSONAL_DATA, 
                       IndianDataClassification.CRITICAL_PERSONAL_DATA]
                for cls in classifications
            )
            
            requires_user_notification = severity in ["high", "critical"]
            
            # Log breach
            breach_log = {
                "breach_id": breach_id,
                "breach_time": breach_time.isoformat(),
                "affected_users_count": len(affected_users),
                "data_categories": [cat.value for cat in data_categories],
                "data_classifications": list(classifications),
                "severity": severity,
                "description": breach_description,
                "requires_authority_notification": requires_authority_notification,
                "requires_user_notification": requires_user_notification,
                "notification_deadline": (breach_time + self.data_breach_notification_period).isoformat()
            }
            
            # Store breach record
            await self._store_breach_record(breach_log)
            
            # Schedule notifications
            notifications_scheduled = []
            
            if requires_authority_notification:
                # Schedule authority notification
                await self._schedule_authority_notification(breach_id, breach_time)
                notifications_scheduled.append("data_protection_authority")
            
            if requires_user_notification:
                # Schedule user notifications
                await self._schedule_user_notifications(breach_id, affected_users, breach_time)
                notifications_scheduled.append("affected_users")
            
            # Log compliance action
            for user_id in affected_users[:10]:  # Log first 10 users
                await self._log_compliance_action(
                    user_id=user_id,
                    action="data_breach_handled",
                    data_classification=IndianDataClassification.PERSONAL_DATA,
                    legal_basis=LegalBasisIndia.LEGAL_OBLIGATION,
                    details=breach_log
                )
            
            logger.critical(
                "Data breach handled",
                breach_id=breach_id,
                affected_users=len(affected_users),
                severity=severity
            )
            
            return {
                "breach_id": breach_id,
                "status": "handled",
                "notifications_scheduled": notifications_scheduled,
                "compliance_deadline": (breach_time + self.data_breach_notification_period).isoformat(),
                "breach_log": breach_log
            }
            
        except Exception as e:
            logger.error("Data breach handling failed", breach_id=breach_id, exc_info=e)
            raise
    
    async def process_user_rights_request(
        self,
        user_id: str,
        request_type: str,  # "access", "rectification", "erasure", "portability"
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process user rights request as per Indian law.
        
        Args:
            user_id: User identifier
            request_type: Type of rights request
            details: Request details
            
        Returns:
            Request processing result
        """
        try:
            request_id = str(uuid4())
            request_time = datetime.utcnow()
            response_deadline = request_time + self.user_response_period
            
            result = {
                "request_id": request_id,
                "user_id": user_id,
                "request_type": request_type,
                "status": "processing",
                "requested_at": request_time.isoformat(),
                "response_deadline": response_deadline.isoformat()
            }
            
            if request_type == "access":
                # Right to access personal data
                user_data = await self._compile_user_data_report(user_id)
                result["data_report"] = user_data
                result["status"] = "completed"
                
            elif request_type == "rectification":
                # Right to rectify inaccurate data
                await self._schedule_data_rectification(user_id, details)
                result["status"] = "scheduled"
                
            elif request_type == "erasure":
                # Right to erasure (right to be forgotten)
                categories = details.get("categories", list(DataCategory))
                deletion_request = await self.privacy_manager.schedule_data_deletion(
                    user_id=user_id,
                    categories=categories,
                    reason="user_rights_request"
                )
                result["deletion_request_id"] = deletion_request.request_id
                result["status"] = "scheduled"
                
            elif request_type == "portability":
                # Right to data portability
                portable_data = await self._export_portable_data(user_id)
                result["portable_data"] = portable_data
                result["status"] = "completed"
            
            # Store request record
            await self._store_rights_request(result)
            
            # Log compliance action
            await self._log_compliance_action(
                user_id=user_id,
                action=f"user_rights_{request_type}",
                data_classification=IndianDataClassification.PERSONAL_DATA,
                legal_basis=LegalBasisIndia.LEGAL_OBLIGATION,
                details=details
            )
            
            logger.info(
                "User rights request processed",
                user_id=user_id,
                request_type=request_type,
                request_id=request_id
            )
            
            return result
            
        except Exception as e:
            logger.error("User rights request processing failed", user_id=user_id, exc_info=e)
            raise
    
    async def generate_compliance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            period_days: Reporting period in days
            
        Returns:
            Compliance report
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            # Get compliance metrics
            consent_metrics = await self._get_consent_metrics(start_date, end_date)
            localization_metrics = await self._get_localization_metrics()
            breach_metrics = await self._get_breach_metrics(start_date, end_date)
            rights_request_metrics = await self._get_rights_request_metrics(start_date, end_date)
            audit_summary = await self._get_audit_summary(start_date, end_date)
            
            report = {
                "report_id": str(uuid4()),
                "generated_at": end_date.isoformat(),
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": period_days
                },
                "compliance_framework": "Indian Data Protection Laws",
                "consent_management": consent_metrics,
                "data_localization": localization_metrics,
                "data_breaches": breach_metrics,
                "user_rights_requests": rights_request_metrics,
                "audit_summary": audit_summary,
                "compliance_score": self._calculate_compliance_score(
                    consent_metrics, localization_metrics, breach_metrics
                ),
                "recommendations": self._generate_compliance_recommendations(
                    consent_metrics, localization_metrics, breach_metrics
                )
            }
            
            logger.info("Compliance report generated", period_days=period_days)
            return report
            
        except Exception as e:
            logger.error("Compliance report generation failed", exc_info=e)
            raise
    
    def _get_indian_data_mappings(self) -> Dict[DataCategory, IndianDataMapping]:
        """Get Indian data classification mappings."""
        return {
            DataCategory.VOICE_DATA: IndianDataMapping(
                category=DataCategory.VOICE_DATA,
                indian_classification=IndianDataClassification.SENSITIVE_PERSONAL_DATA,
                localization_requirement=DataLocalizationRequirement.COPY_IN_INDIA,
                retention_period_days=90,
                requires_explicit_consent=True,
                cross_border_transfer_allowed=False,
                anonymization_required=True
            ),
            DataCategory.PROFILE_DATA: IndianDataMapping(
                category=DataCategory.PROFILE_DATA,
                indian_classification=IndianDataClassification.PERSONAL_DATA,
                localization_requirement=DataLocalizationRequirement.COPY_IN_INDIA,
                retention_period_days=1095,  # 3 years
                requires_explicit_consent=True,
                cross_border_transfer_allowed=True,
                anonymization_required=False
            ),
            DataCategory.BIOMETRIC_DATA: IndianDataMapping(
                category=DataCategory.BIOMETRIC_DATA,
                indian_classification=IndianDataClassification.CRITICAL_PERSONAL_DATA,
                localization_requirement=DataLocalizationRequirement.MUST_STORE_IN_INDIA,
                retention_period_days=180,  # 6 months
                requires_explicit_consent=True,
                cross_border_transfer_allowed=False,
                anonymization_required=True
            ),
            DataCategory.LOCATION_DATA: IndianDataMapping(
                category=DataCategory.LOCATION_DATA,
                indian_classification=IndianDataClassification.SENSITIVE_PERSONAL_DATA,
                localization_requirement=DataLocalizationRequirement.COPY_IN_INDIA,
                retention_period_days=30,
                requires_explicit_consent=True,
                cross_border_transfer_allowed=False,
                anonymization_required=True
            ),
            DataCategory.USAGE_DATA: IndianDataMapping(
                category=DataCategory.USAGE_DATA,
                indian_classification=IndianDataClassification.PERSONAL_DATA,
                localization_requirement=DataLocalizationRequirement.NO_RESTRICTION,
                retention_period_days=365,
                requires_explicit_consent=False,
                cross_border_transfer_allowed=True,
                anonymization_required=True
            )
        }
    
    def _get_indian_retention_periods(self) -> Dict[IndianDataClassification, int]:
        """Get retention periods by Indian data classification."""
        return {
            IndianDataClassification.PERSONAL_DATA: 1095,  # 3 years
            IndianDataClassification.SENSITIVE_PERSONAL_DATA: 180,  # 6 months
            IndianDataClassification.CRITICAL_PERSONAL_DATA: 90,  # 3 months
            IndianDataClassification.NON_PERSONAL_DATA: 1825  # 5 years
        }
    
    def _get_localization_requirements(self) -> Dict[IndianDataClassification, DataLocalizationRequirement]:
        """Get localization requirements by data classification."""
        return {
            IndianDataClassification.PERSONAL_DATA: DataLocalizationRequirement.NO_RESTRICTION,
            IndianDataClassification.SENSITIVE_PERSONAL_DATA: DataLocalizationRequirement.COPY_IN_INDIA,
            IndianDataClassification.CRITICAL_PERSONAL_DATA: DataLocalizationRequirement.MUST_STORE_IN_INDIA,
            IndianDataClassification.NON_PERSONAL_DATA: DataLocalizationRequirement.NO_RESTRICTION
        }
    
    def _check_localization_compliance(
        self,
        storage_location: str,
        requirement: DataLocalizationRequirement
    ) -> bool:
        """Check if storage location meets localization requirement."""
        if requirement == DataLocalizationRequirement.NO_RESTRICTION:
            return True
        elif requirement == DataLocalizationRequirement.COPY_IN_INDIA:
            # Check if there's a copy in India (simplified check)
            return "india" in storage_location.lower() or "in" in storage_location.lower()
        elif requirement == DataLocalizationRequirement.MUST_STORE_IN_INDIA:
            # Check if data is stored only in India
            return storage_location.lower() in ["india", "in", "mumbai", "bangalore", "delhi"]
        return False
    
    def _calculate_compliance_score(
        self,
        consent_metrics: Dict[str, Any],
        localization_metrics: Dict[str, Any],
        breach_metrics: Dict[str, Any]
    ) -> int:
        """Calculate overall compliance score (0-100)."""
        score = 100
        
        # Deduct points for compliance issues
        if consent_metrics.get("expired_consents", 0) > 0:
            score -= 10
        
        if localization_metrics.get("non_compliant_count", 0) > 0:
            score -= 20
        
        if breach_metrics.get("total_breaches", 0) > 0:
            score -= 30
        
        if breach_metrics.get("unnotified_breaches", 0) > 0:
            score -= 20
        
        return max(0, score)
    
    def _generate_compliance_recommendations(
        self,
        consent_metrics: Dict[str, Any],
        localization_metrics: Dict[str, Any],
        breach_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if consent_metrics.get("expired_consents", 0) > 0:
            recommendations.append("Renew expired user consents")
        
        if localization_metrics.get("non_compliant_count", 0) > 0:
            recommendations.append("Address data localization compliance issues")
        
        if breach_metrics.get("total_breaches", 0) > 0:
            recommendations.append("Review and strengthen data security measures")
        
        if breach_metrics.get("unnotified_breaches", 0) > 0:
            recommendations.append("Complete pending breach notifications")
        
        if not recommendations:
            recommendations.append("Maintain current compliance standards")
        
        return recommendations
    
    # Mock database operations (replace with actual database implementation)
    
    async def _get_data_storage_location(self, user_id: str, category: DataCategory) -> str:
        """Get data storage location (mock implementation)."""
        return "india"  # Mock: assume data is stored in India
    
    async def _store_indian_consent(self, consent_record: ConsentRecord) -> None:
        """Store Indian consent record (mock implementation)."""
        logger.info("Indian consent stored (mock)", consent_id=consent_record.consent_id)
    
    async def _log_compliance_action(
        self,
        user_id: str,
        action: str,
        data_classification: IndianDataClassification,
        legal_basis: LegalBasisIndia,
        details: Dict[str, Any]
    ) -> None:
        """Log compliance action (mock implementation)."""
        log_entry = ComplianceAuditLog(
            log_id=str(uuid4()),
            user_id=user_id,
            action=action,
            data_classification=data_classification,
            legal_basis=legal_basis,
            timestamp=datetime.utcnow(),
            location="india",
            compliance_status="compliant",
            details=details
        )
        logger.info("Compliance action logged (mock)", action=action, user_id=user_id)
    
    async def _store_breach_record(self, breach_log: Dict[str, Any]) -> None:
        """Store breach record (mock implementation)."""
        logger.info("Breach record stored (mock)", breach_id=breach_log["breach_id"])
    
    async def _schedule_authority_notification(self, breach_id: str, breach_time: datetime) -> None:
        """Schedule authority notification (mock implementation)."""
        logger.info("Authority notification scheduled (mock)", breach_id=breach_id)
    
    async def _schedule_user_notifications(self, breach_id: str, user_ids: List[str], breach_time: datetime) -> None:
        """Schedule user notifications (mock implementation)."""
        logger.info("User notifications scheduled (mock)", breach_id=breach_id, user_count=len(user_ids))
    
    async def _compile_user_data_report(self, user_id: str) -> Dict[str, Any]:
        """Compile user data report (mock implementation)."""
        return {"user_id": user_id, "data": "mock_user_data"}
    
    async def _schedule_data_rectification(self, user_id: str, details: Dict[str, Any]) -> None:
        """Schedule data rectification (mock implementation)."""
        logger.info("Data rectification scheduled (mock)", user_id=user_id)
    
    async def _export_portable_data(self, user_id: str) -> Dict[str, Any]:
        """Export portable data (mock implementation)."""
        return {"user_id": user_id, "portable_data": "mock_portable_data"}
    
    async def _store_rights_request(self, request_data: Dict[str, Any]) -> None:
        """Store rights request (mock implementation)."""
        logger.info("Rights request stored (mock)", request_id=request_data["request_id"])
    
    async def _get_consent_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get consent metrics (mock implementation)."""
        return {
            "total_consents": 100,
            "active_consents": 95,
            "expired_consents": 5,
            "withdrawn_consents": 2
        }
    
    async def _get_localization_metrics(self) -> Dict[str, Any]:
        """Get localization metrics (mock implementation)."""
        return {
            "total_data_categories": 5,
            "compliant_count": 5,
            "non_compliant_count": 0,
            "compliance_percentage": 100.0
        }
    
    async def _get_breach_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get breach metrics (mock implementation)."""
        return {
            "total_breaches": 0,
            "notified_breaches": 0,
            "unnotified_breaches": 0,
            "average_notification_time_hours": 0
        }
    
    async def _get_rights_request_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get rights request metrics (mock implementation)."""
        return {
            "total_requests": 10,
            "completed_requests": 8,
            "pending_requests": 2,
            "average_response_time_days": 15
        }
    
    async def _get_audit_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get audit summary (mock implementation)."""
        return {
            "total_audit_logs": 500,
            "compliance_actions": 450,
            "non_compliance_actions": 0,
            "most_common_actions": ["consent_granted", "data_accessed", "data_processed"]
        }