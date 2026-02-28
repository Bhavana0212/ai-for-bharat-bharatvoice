"""
Privacy manager for BharatVoice Assistant.

This module provides comprehensive privacy management including data anonymization,
automated deletion, consent management, and compliance with Indian data protection laws.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel

from bharatvoice.config import Settings
from .encryption_manager import EncryptionManager


logger = structlog.get_logger(__name__)


class ConsentType(str, Enum):
    """Types of user consent."""
    
    DATA_COLLECTION = "data_collection"
    VOICE_PROCESSING = "voice_processing"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    THIRD_PARTY_SHARING = "third_party_sharing"
    LOCATION_TRACKING = "location_tracking"
    BIOMETRIC_DATA = "biometric_data"


class ConsentStatus(str, Enum):
    """Consent status values."""
    
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class DataCategory(str, Enum):
    """Categories of personal data."""
    
    VOICE_DATA = "voice_data"
    PROFILE_DATA = "profile_data"
    USAGE_DATA = "usage_data"
    LOCATION_DATA = "location_data"
    BIOMETRIC_DATA = "biometric_data"
    CONVERSATION_DATA = "conversation_data"
    PREFERENCE_DATA = "preference_data"


class UserConsent(BaseModel):
    """User consent record."""
    
    consent_id: str
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    purpose: str
    legal_basis: str
    version: str = "1.0"
    metadata: Dict[str, Any] = {}


class DataRetentionPolicy(BaseModel):
    """Data retention policy configuration."""
    
    category: DataCategory
    retention_period: timedelta
    auto_delete: bool = True
    anonymize_after: Optional[timedelta] = None
    legal_hold_exempt: bool = False


class PrivacyAuditLog(BaseModel):
    """Privacy audit log entry."""
    
    log_id: str
    user_id: str
    action: str
    data_category: DataCategory
    timestamp: datetime
    details: Dict[str, Any]
    compliance_framework: str  # "GDPR", "PDPB", "CCPA", etc.


class DataDeletionRequest(BaseModel):
    """Data deletion request."""
    
    request_id: str
    user_id: str
    requested_at: datetime
    scheduled_for: datetime
    categories: List[DataCategory]
    status: str  # "pending", "in_progress", "completed", "failed"
    reason: str
    completed_at: Optional[datetime] = None


class PrivacyManager:
    """Privacy manager for data protection compliance."""
    
    def __init__(
        self,
        settings: Settings,
        encryption_manager: EncryptionManager,
        redis_client=None,
        database=None
    ):
        """
        Initialize privacy manager.
        
        Args:
            settings: Application settings
            encryption_manager: Encryption manager instance
            redis_client: Redis client for caching
            database: Database connection
        """
        self.settings = settings
        self.encryption_manager = encryption_manager
        self.redis_client = redis_client
        self.database = database
        
        # Privacy configuration
        self.default_retention_policies = self._get_default_retention_policies()
        self.consent_expiry_period = timedelta(days=365)  # 1 year
        self.deletion_grace_period = timedelta(days=30)  # 30 days as per Indian law
        
        # Compliance frameworks
        self.supported_frameworks = ["GDPR", "PDPB", "CCPA"]
        
        logger.info("Privacy manager initialized")
    
    async def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        status: ConsentStatus,
        purpose: str,
        legal_basis: str = "consent",
        expires_in: Optional[timedelta] = None
    ) -> UserConsent:
        """
        Record user consent.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent
            status: Consent status
            purpose: Purpose of data processing
            legal_basis: Legal basis for processing
            expires_in: Consent expiration period
            
        Returns:
            Consent record
        """
        try:
            consent_id = str(uuid4())
            now = datetime.utcnow()
            
            expires_at = None
            if expires_in:
                expires_at = now + expires_in
            elif status == ConsentStatus.GRANTED:
                expires_at = now + self.consent_expiry_period
            
            consent = UserConsent(
                consent_id=consent_id,
                user_id=user_id,
                consent_type=consent_type,
                status=status,
                granted_at=now if status == ConsentStatus.GRANTED else None,
                withdrawn_at=now if status == ConsentStatus.WITHDRAWN else None,
                expires_at=expires_at,
                purpose=purpose,
                legal_basis=legal_basis
            )
            
            # Store consent record
            await self._store_consent(consent)
            
            # Log audit entry
            await self._log_privacy_action(
                user_id=user_id,
                action=f"consent_{status.value}",
                data_category=DataCategory.PROFILE_DATA,
                details={
                    "consent_type": consent_type.value,
                    "purpose": purpose,
                    "legal_basis": legal_basis
                }
            )
            
            logger.info(
                "Consent recorded",
                user_id=user_id,
                consent_type=consent_type.value,
                status=status.value
            )
            
            return consent
            
        except Exception as e:
            logger.error("Failed to record consent", user_id=user_id, exc_info=e)
            raise
    
    async def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """
        Check if user has valid consent for data processing.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to check
            
        Returns:
            True if consent is valid
        """
        try:
            consent = await self._get_latest_consent(user_id, consent_type)
            
            if not consent:
                return False
            
            # Check if consent is granted and not expired
            if consent.status != ConsentStatus.GRANTED:
                return False
            
            if consent.expires_at and datetime.utcnow() > consent.expires_at:
                # Mark as expired
                await self._update_consent_status(consent.consent_id, ConsentStatus.EXPIRED)
                return False
            
            return True
            
        except Exception as e:
            logger.error("Failed to check consent", user_id=user_id, exc_info=e)
            return False
    
    async def withdraw_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """
        Withdraw user consent.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to withdraw
            
        Returns:
            True if consent was withdrawn successfully
        """
        try:
            consent = await self._get_latest_consent(user_id, consent_type)
            
            if not consent or consent.status != ConsentStatus.GRANTED:
                logger.warning("No active consent to withdraw", user_id=user_id, consent_type=consent_type.value)
                return False
            
            # Update consent status
            await self._update_consent_status(consent.consent_id, ConsentStatus.WITHDRAWN)
            
            # Log audit entry
            await self._log_privacy_action(
                user_id=user_id,
                action="consent_withdrawn",
                data_category=DataCategory.PROFILE_DATA,
                details={"consent_type": consent_type.value}
            )
            
            # Trigger data processing stop for this consent type
            await self._handle_consent_withdrawal(user_id, consent_type)
            
            logger.info("Consent withdrawn", user_id=user_id, consent_type=consent_type.value)
            return True
            
        except Exception as e:
            logger.error("Failed to withdraw consent", user_id=user_id, exc_info=e)
            return False
    
    async def anonymize_user_data(
        self,
        user_id: str,
        categories: List[DataCategory],
        preserve_analytics: bool = True
    ) -> Dict[str, bool]:
        """
        Anonymize user data for compliance.
        
        Args:
            user_id: User identifier
            categories: Data categories to anonymize
            preserve_analytics: Whether to preserve anonymized data for analytics
            
        Returns:
            Dictionary of category -> success status
        """
        try:
            results = {}
            
            for category in categories:
                try:
                    # Get data for category
                    data = await self._get_user_data_by_category(user_id, category)
                    
                    if not data:
                        results[category.value] = True
                        continue
                    
                    # Determine fields to anonymize based on category
                    fields_to_anonymize = self._get_anonymization_fields(category)
                    
                    # Anonymize data
                    anonymized_data = self.encryption_manager.anonymize_data(
                        data, fields_to_anonymize
                    )
                    
                    if preserve_analytics:
                        # Store anonymized data for analytics
                        await self._store_anonymized_data(user_id, category, anonymized_data)
                    
                    # Remove original data
                    await self._delete_user_data_by_category(user_id, category)
                    
                    results[category.value] = True
                    
                    # Log audit entry
                    await self._log_privacy_action(
                        user_id=user_id,
                        action="data_anonymized",
                        data_category=category,
                        details={"preserve_analytics": preserve_analytics}
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to anonymize {category.value}", user_id=user_id, exc_info=e)
                    results[category.value] = False
            
            logger.info("Data anonymization completed", user_id=user_id, results=results)
            return results
            
        except Exception as e:
            logger.error("Data anonymization failed", user_id=user_id, exc_info=e)
            raise
    
    async def schedule_data_deletion(
        self,
        user_id: str,
        categories: List[DataCategory],
        reason: str = "user_request"
    ) -> DataDeletionRequest:
        """
        Schedule data deletion for compliance.
        
        Args:
            user_id: User identifier
            categories: Data categories to delete
            reason: Reason for deletion
            
        Returns:
            Data deletion request
        """
        try:
            request_id = str(uuid4())
            now = datetime.utcnow()
            scheduled_for = now + self.deletion_grace_period
            
            deletion_request = DataDeletionRequest(
                request_id=request_id,
                user_id=user_id,
                requested_at=now,
                scheduled_for=scheduled_for,
                categories=categories,
                status="pending",
                reason=reason
            )
            
            # Store deletion request
            await self._store_deletion_request(deletion_request)
            
            # Schedule deletion in Redis
            self.encryption_manager.schedule_data_deletion(user_id, scheduled_for)
            
            # Log audit entry
            await self._log_privacy_action(
                user_id=user_id,
                action="deletion_scheduled",
                data_category=DataCategory.PROFILE_DATA,
                details={
                    "categories": [cat.value for cat in categories],
                    "scheduled_for": scheduled_for.isoformat(),
                    "reason": reason
                }
            )
            
            logger.info(
                "Data deletion scheduled",
                user_id=user_id,
                request_id=request_id,
                scheduled_for=scheduled_for.isoformat()
            )
            
            return deletion_request
            
        except Exception as e:
            logger.error("Failed to schedule data deletion", user_id=user_id, exc_info=e)
            raise
    
    async def execute_data_deletion(self, request_id: str) -> bool:
        """
        Execute scheduled data deletion.
        
        Args:
            request_id: Deletion request identifier
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get deletion request
            deletion_request = await self._get_deletion_request(request_id)
            if not deletion_request:
                logger.error("Deletion request not found", request_id=request_id)
                return False
            
            # Update status to in_progress
            await self._update_deletion_request_status(request_id, "in_progress")
            
            # Delete data by categories
            success = True
            for category in deletion_request.categories:
                try:
                    await self._delete_user_data_by_category(deletion_request.user_id, category)
                    
                    # Log audit entry
                    await self._log_privacy_action(
                        user_id=deletion_request.user_id,
                        action="data_deleted",
                        data_category=category,
                        details={"request_id": request_id}
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to delete {category.value}", request_id=request_id, exc_info=e)
                    success = False
            
            # Update deletion request status
            status = "completed" if success else "failed"
            await self._update_deletion_request_status(
                request_id, status, datetime.utcnow() if success else None
            )
            
            logger.info(
                "Data deletion executed",
                request_id=request_id,
                user_id=deletion_request.user_id,
                success=success
            )
            
            return success
            
        except Exception as e:
            logger.error("Data deletion execution failed", request_id=request_id, exc_info=e)
            await self._update_deletion_request_status(request_id, "failed")
            return False
    
    async def get_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate privacy report for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Privacy report dictionary
        """
        try:
            # Get user consents
            consents = await self._get_user_consents(user_id)
            
            # Get data categories and retention info
            data_categories = await self._get_user_data_categories(user_id)
            
            # Get audit logs
            audit_logs = await self._get_user_audit_logs(user_id, limit=50)
            
            # Get deletion requests
            deletion_requests = await self._get_user_deletion_requests(user_id)
            
            report = {
                "user_id": user_id,
                "generated_at": datetime.utcnow().isoformat(),
                "consents": [consent.dict() for consent in consents],
                "data_categories": data_categories,
                "audit_logs": [log.dict() for log in audit_logs],
                "deletion_requests": [req.dict() for req in deletion_requests],
                "retention_policies": {
                    cat.value: policy.dict() 
                    for cat, policy in self.default_retention_policies.items()
                },
                "compliance_frameworks": self.supported_frameworks
            }
            
            logger.info("Privacy report generated", user_id=user_id)
            return report
            
        except Exception as e:
            logger.error("Failed to generate privacy report", user_id=user_id, exc_info=e)
            raise
    
    def _get_default_retention_policies(self) -> Dict[DataCategory, DataRetentionPolicy]:
        """Get default data retention policies."""
        return {
            DataCategory.VOICE_DATA: DataRetentionPolicy(
                category=DataCategory.VOICE_DATA,
                retention_period=timedelta(days=90),
                auto_delete=True,
                anonymize_after=timedelta(days=30)
            ),
            DataCategory.PROFILE_DATA: DataRetentionPolicy(
                category=DataCategory.PROFILE_DATA,
                retention_period=timedelta(days=1095),  # 3 years
                auto_delete=False
            ),
            DataCategory.USAGE_DATA: DataRetentionPolicy(
                category=DataCategory.USAGE_DATA,
                retention_period=timedelta(days=365),  # 1 year
                auto_delete=True,
                anonymize_after=timedelta(days=90)
            ),
            DataCategory.LOCATION_DATA: DataRetentionPolicy(
                category=DataCategory.LOCATION_DATA,
                retention_period=timedelta(days=30),
                auto_delete=True
            ),
            DataCategory.CONVERSATION_DATA: DataRetentionPolicy(
                category=DataCategory.CONVERSATION_DATA,
                retention_period=timedelta(days=180),  # 6 months
                auto_delete=True,
                anonymize_after=timedelta(days=30)
            )
        }
    
    def _get_anonymization_fields(self, category: DataCategory) -> List[str]:
        """Get fields to anonymize for data category."""
        field_mapping = {
            DataCategory.VOICE_DATA: ["user_id", "device_id", "ip_address"],
            DataCategory.PROFILE_DATA: ["name", "email", "phone_number", "address"],
            DataCategory.USAGE_DATA: ["user_id", "device_id", "ip_address"],
            DataCategory.LOCATION_DATA: ["latitude", "longitude", "address"],
            DataCategory.CONVERSATION_DATA: ["user_id", "session_id", "device_id"]
        }
        return field_mapping.get(category, [])
    
    async def _handle_consent_withdrawal(self, user_id: str, consent_type: ConsentType) -> None:
        """Handle consent withdrawal by stopping related processing."""
        # In production, implement specific actions for each consent type
        logger.info("Handling consent withdrawal", user_id=user_id, consent_type=consent_type.value)
    
    # Mock database operations (replace with actual database implementation)
    
    async def _store_consent(self, consent: UserConsent) -> None:
        """Store consent record (mock implementation)."""
        logger.info("Consent stored (mock)", consent_id=consent.consent_id)
    
    async def _get_latest_consent(self, user_id: str, consent_type: ConsentType) -> Optional[UserConsent]:
        """Get latest consent record (mock implementation)."""
        # Return mock consent for demo
        return UserConsent(
            consent_id=str(uuid4()),
            user_id=user_id,
            consent_type=consent_type,
            status=ConsentStatus.GRANTED,
            granted_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.consent_expiry_period,
            purpose="Service functionality",
            legal_basis="consent"
        )
    
    async def _update_consent_status(self, consent_id: str, status: ConsentStatus) -> None:
        """Update consent status (mock implementation)."""
        logger.info("Consent status updated (mock)", consent_id=consent_id, status=status.value)
    
    async def _log_privacy_action(
        self,
        user_id: str,
        action: str,
        data_category: DataCategory,
        details: Dict[str, Any]
    ) -> None:
        """Log privacy action (mock implementation)."""
        log_entry = PrivacyAuditLog(
            log_id=str(uuid4()),
            user_id=user_id,
            action=action,
            data_category=data_category,
            timestamp=datetime.utcnow(),
            details=details,
            compliance_framework="PDPB"
        )
        logger.info("Privacy action logged (mock)", action=action, user_id=user_id)
    
    async def _get_user_data_by_category(self, user_id: str, category: DataCategory) -> Optional[Dict[str, Any]]:
        """Get user data by category (mock implementation)."""
        return {"mock_data": "value", "user_id": user_id, "category": category.value}
    
    async def _store_anonymized_data(self, user_id: str, category: DataCategory, data: Dict[str, Any]) -> None:
        """Store anonymized data (mock implementation)."""
        logger.info("Anonymized data stored (mock)", user_id=user_id, category=category.value)
    
    async def _delete_user_data_by_category(self, user_id: str, category: DataCategory) -> None:
        """Delete user data by category (mock implementation)."""
        logger.info("User data deleted (mock)", user_id=user_id, category=category.value)
    
    async def _store_deletion_request(self, deletion_request: DataDeletionRequest) -> None:
        """Store deletion request (mock implementation)."""
        logger.info("Deletion request stored (mock)", request_id=deletion_request.request_id)
    
    async def _get_deletion_request(self, request_id: str) -> Optional[DataDeletionRequest]:
        """Get deletion request (mock implementation)."""
        return None
    
    async def _update_deletion_request_status(
        self,
        request_id: str,
        status: str,
        completed_at: Optional[datetime] = None
    ) -> None:
        """Update deletion request status (mock implementation)."""
        logger.info("Deletion request status updated (mock)", request_id=request_id, status=status)
    
    async def _get_user_consents(self, user_id: str) -> List[UserConsent]:
        """Get user consents (mock implementation)."""
        return []
    
    async def _get_user_data_categories(self, user_id: str) -> List[str]:
        """Get user data categories (mock implementation)."""
        return [cat.value for cat in DataCategory]
    
    async def _get_user_audit_logs(self, user_id: str, limit: int = 50) -> List[PrivacyAuditLog]:
        """Get user audit logs (mock implementation)."""
        return []
    
    async def _get_user_deletion_requests(self, user_id: str) -> List[DataDeletionRequest]:
        """Get user deletion requests (mock implementation)."""
        return []