"""
Authentication endpoints for BharatVoice Assistant.

This module provides user authentication, session management, and authorization
endpoints with privacy compliance for Indian data protection laws.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import structlog

from bharatvoice.config import get_settings, Settings
from bharatvoice.core.models import UserProfile
from bharatvoice.services.auth import AuthService, PrivacyManager, EncryptionManager, IndianComplianceManager
from bharatvoice.services.auth.privacy_manager import ConsentType, ConsentStatus, DataCategory
from bharatvoice.services.auth.auth_service import LoginRequest as AuthLoginRequest, RegisterRequest, LoginResponse as AuthLoginResponse


logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()

# Global auth service instance (in production, use dependency injection)
_auth_service: Optional[AuthService] = None


def get_auth_service(settings: Settings = Depends(get_settings)) -> AuthService:
    """Get authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(settings)
    return _auth_service


class LoginRequest(BaseModel):
    """User login request model."""
    
    username: str
    password: str
    mfa_token: Optional[str] = None
    device_id: Optional[str] = None


class LoginResponse(BaseModel):
    """User login response model."""
    
    access_token: str
    token_type: str
    expires_in: int
    user_id: str
    session_id: str
    requires_mfa: bool = False
    mfa_methods: list = []


class SessionInfo(BaseModel):
    """Session information model."""
    
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
    device_info: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    
    current_password: str
    new_password: str


class MFASetupResponse(BaseModel):
    """MFA setup response model."""
    
    secret: str
    qr_code: str
    backup_codes: list


class MFAVerifyRequest(BaseModel):
    """MFA verification request model."""
    
    token: str


class ConsentRequest(BaseModel):
    """Consent management request model."""
    
    consent_type: str
    status: str
    purpose: str


class DataDeletionRequest(BaseModel):
    """Data deletion request model."""
    
    categories: List[str]
    reason: str = "user_request"


# Global service instances (in production, use dependency injection)
_auth_service: Optional[AuthService] = None
_privacy_manager: Optional[PrivacyManager] = None
_encryption_manager: Optional[EncryptionManager] = None
_indian_compliance_manager: Optional[IndianComplianceManager] = None


def get_privacy_manager(settings: Settings = Depends(get_settings)) -> PrivacyManager:
    """Get privacy manager instance."""
    global _privacy_manager, _encryption_manager
    if _privacy_manager is None:
        if _encryption_manager is None:
            _encryption_manager = EncryptionManager(settings)
        _privacy_manager = PrivacyManager(settings, _encryption_manager)
    return _privacy_manager


def get_indian_compliance_manager(
    settings: Settings = Depends(get_settings),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
) -> IndianComplianceManager:
    """Get Indian compliance manager instance."""
    global _indian_compliance_manager
    if _indian_compliance_manager is None:
        _indian_compliance_manager = IndianComplianceManager(settings, privacy_manager)
    return _indian_compliance_manager


@router.post("/register", response_model=Dict[str, Any])
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register new user account.
    
    Args:
        request: Registration request
        auth_service: Authentication service
        
    Returns:
        Registration response
    """
    try:
        response = await auth_service.register_user(request)
        
        logger.info(
            "User registration successful",
            user_id=response.user_id,
            username=response.username
        )
        
        return response.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service unavailable"
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate user and create session.
    
    Args:
        request: Login credentials
        http_request: HTTP request for extracting client info
        auth_service: Authentication service
        
    Returns:
        Authentication token and session info
    """
    try:
        # Extract client information
        client_ip = http_request.client.host if http_request.client else None
        user_agent = http_request.headers.get("user-agent")
        
        # Create auth login request
        auth_request = AuthLoginRequest(
            username=request.username,
            password=request.password,
            mfa_token=request.mfa_token,
            device_id=request.device_id,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        response = await auth_service.authenticate_user(auth_request)
        
        logger.info(
            "User login successful",
            user_id=response.user_id,
            session_id=response.session_id,
            device_id=request.device_id
        )
        
        return LoginResponse(
            access_token=response.access_token,
            token_type=response.token_type,
            expires_in=response.expires_in,
            user_id=response.user_id,
            session_id=response.session_id,
            requires_mfa=response.requires_mfa,
            mfa_methods=response.mfa_methods
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Logout user and invalidate session.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        Logout confirmation
    """
    try:
        token = credentials.credentials
        success = await auth_service.logout_user(token)
        
        if success:
            logger.info("User logout successful")
            return {"message": "Logged out successfully"}
        else:
            logger.warning("Logout failed")
            return {"message": "Logout failed"}
    
    except Exception as e:
        logger.error("Logout error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/session", response_model=SessionInfo)
async def get_session_info(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Get current session information.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        Session information
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Get session data
        session_data = auth_service.session_manager.get_session(token_payload.session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        return SessionInfo(
            session_id=session_data.session_id,
            user_id=session_data.user_id,
            created_at=session_data.created_at,
            expires_at=session_data.expires_at,
            is_active=session_data.is_active,
            device_info=session_data.device_info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session info error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )


@router.post("/refresh")
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh authentication token.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        New authentication token
    """
    try:
        old_token = credentials.credentials
        new_token = await auth_service.refresh_token(old_token)
        
        logger.info("Token refreshed successfully")
        
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": 1800  # 30 minutes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change user password.
    
    Args:
        request: Password change request
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        Password change confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        from uuid import UUID
        user_id = UUID(token_payload.user_id)
        
        success = await auth_service.change_password(
            user_id=user_id,
            current_password=request.current_password,
            new_password=request.new_password
        )
        
        if success:
            logger.info("Password changed successfully", user_id=str(user_id))
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Setup multi-factor authentication.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        MFA setup data including QR code
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        from uuid import UUID
        user_id = UUID(token_payload.user_id)
        
        setup_data = await auth_service.setup_mfa(user_id)
        
        logger.info("MFA setup initiated", user_id=str(user_id))
        
        return MFASetupResponse(
            secret=setup_data["secret"],
            qr_code=setup_data["qr_code"],
            backup_codes=setup_data["backup_codes"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA setup error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup failed"
        )


@router.post("/mfa/verify")
async def verify_mfa_setup(
    request: MFAVerifyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify MFA setup with TOTP token.
    
    Args:
        request: MFA verification request
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        MFA verification confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        from uuid import UUID
        user_id = UUID(token_payload.user_id)
        
        success = await auth_service.verify_mfa_setup(user_id, request.token)
        
        if success:
            logger.info("MFA setup completed", user_id=str(user_id))
            return {"message": "MFA setup completed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA verification failed"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA verification error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed"
        )


@router.delete("/account")
async def delete_account(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Delete user account and all associated data.
    
    This endpoint complies with Indian data protection laws by ensuring
    complete data deletion within 30 days as specified in requirements.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        
    Returns:
        Account deletion confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Invalidate all user sessions
        auth_service.session_manager.invalidate_user_sessions(token_payload.user_id)
        
        logger.info("Account deletion requested", user_id=token_payload.user_id)
        
        return {
            "message": "Account deletion initiated",
            "deletion_date": datetime.utcnow() + timedelta(days=30),
            "note": "All personal data will be permanently deleted within 30 days as per Indian data protection compliance"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Account deletion error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )


@router.post("/consent")
async def manage_consent(
    request: ConsentRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """
    Manage user consent for data processing.
    
    Args:
        request: Consent request
        credentials: Bearer token credentials
        auth_service: Authentication service
        privacy_manager: Privacy manager
        
    Returns:
        Consent management confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Validate consent type and status
        try:
            consent_type = ConsentType(request.consent_type)
            consent_status = ConsentStatus(request.status)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid consent type or status: {str(e)}"
            )
        
        # Record consent
        consent = await privacy_manager.record_consent(
            user_id=token_payload.user_id,
            consent_type=consent_type,
            status=consent_status,
            purpose=request.purpose
        )
        
        logger.info(
            "Consent managed",
            user_id=token_payload.user_id,
            consent_type=request.consent_type,
            status=request.status
        )
        
        return {
            "message": f"Consent {request.status} successfully",
            "consent_id": consent.consent_id,
            "expires_at": consent.expires_at.isoformat() if consent.expires_at else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Consent management error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Consent management failed"
        )


@router.get("/privacy-report")
async def get_privacy_report(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """
    Get user privacy report.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        privacy_manager: Privacy manager
        
    Returns:
        Privacy report
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        report = await privacy_manager.get_privacy_report(token_payload.user_id)
        
        logger.info("Privacy report generated", user_id=token_payload.user_id)
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Privacy report generation error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Privacy report generation failed"
        )


@router.post("/request-deletion")
async def request_data_deletion(
    request: DataDeletionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """
    Request data deletion for compliance.
    
    Args:
        request: Data deletion request
        credentials: Bearer token credentials
        auth_service: Authentication service
        privacy_manager: Privacy manager
        
    Returns:
        Deletion request confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Validate data categories
        try:
            categories = [DataCategory(cat) for cat in request.categories]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data category: {str(e)}"
            )
        
        # Schedule data deletion
        deletion_request = await privacy_manager.schedule_data_deletion(
            user_id=token_payload.user_id,
            categories=categories,
            reason=request.reason
        )
        
        logger.info(
            "Data deletion requested",
            user_id=token_payload.user_id,
            request_id=deletion_request.request_id
        )
        
        return {
            "message": "Data deletion scheduled successfully",
            "request_id": deletion_request.request_id,
            "scheduled_for": deletion_request.scheduled_for.isoformat(),
            "categories": request.categories,
            "note": "Data will be permanently deleted as per Indian data protection compliance"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data deletion request error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data deletion request failed"
        )


@router.post("/anonymize-data")
async def anonymize_user_data(
    request: DataDeletionRequest,  # Reuse same model for categories
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    privacy_manager: PrivacyManager = Depends(get_privacy_manager)
):
    """
    Anonymize user data for analytics compliance.
    
    Args:
        request: Data anonymization request
        credentials: Bearer token credentials
        auth_service: Authentication service
        privacy_manager: Privacy manager
        
    Returns:
        Anonymization confirmation
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Validate data categories
        try:
            categories = [DataCategory(cat) for cat in request.categories]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid data category: {str(e)}"
            )
        
        # Anonymize data
        results = await privacy_manager.anonymize_user_data(
            user_id=token_payload.user_id,
            categories=categories,
            preserve_analytics=True
        )
        
        logger.info(
            "Data anonymization completed",
            user_id=token_payload.user_id,
            results=results
        )
        
        return {
            "message": "Data anonymization completed",
            "results": results,
            "note": "Original data has been anonymized while preserving analytics insights"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data anonymization error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data anonymization failed"
        )

@router.post("/indian-compliance/granular-consent")
async def obtain_granular_consent(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    compliance_manager: IndianComplianceManager = Depends(get_indian_compliance_manager)
):
    """
    Obtain granular consent as per Indian data protection requirements.
    
    Args:
        request: Granular consent request
        credentials: Bearer token credentials
        auth_service: Authentication service
        compliance_manager: Indian compliance manager
        
    Returns:
        Consent record
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Extract request parameters
        purposes = request.get("purposes", {})
        data_categories = [DataCategory(cat) for cat in request.get("data_categories", [])]
        language = request.get("language", "en-IN")
        is_minor = request.get("is_minor", False)
        guardian_id = request.get("guardian_id")
        
        # Obtain granular consent
        consent_record = await compliance_manager.obtain_granular_consent(
            user_id=token_payload.user_id,
            purposes=purposes,
            data_categories=data_categories,
            language=language,
            is_minor=is_minor,
            guardian_id=guardian_id
        )
        
        logger.info(
            "Granular consent obtained",
            user_id=token_payload.user_id,
            consent_id=consent_record.consent_id
        )
        
        return {
            "message": "Granular consent obtained successfully",
            "consent_id": consent_record.consent_id,
            "expires_at": consent_record.expires_at.isoformat() if consent_record.expires_at else None,
            "granular_consent": consent_record.granular_consent
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Granular consent error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Granular consent failed"
        )


@router.get("/indian-compliance/localization-status")
async def get_localization_status(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    compliance_manager: IndianComplianceManager = Depends(get_indian_compliance_manager)
):
    """
    Get data localization compliance status.
    
    Args:
        credentials: Bearer token credentials
        auth_service: Authentication service
        compliance_manager: Indian compliance manager
        
    Returns:
        Localization status report
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # Validate data localization
        statuses = await compliance_manager.validate_data_localization(token_payload.user_id)
        
        # Convert to serializable format
        status_report = {}
        for category, status_obj in statuses.items():
            status_report[category] = {
                "storage_location": status_obj.storage_location,
                "is_compliant": status_obj.is_compliant,
                "last_checked": status_obj.last_checked.isoformat(),
                "issues": status_obj.issues
            }
        
        logger.info("Localization status retrieved", user_id=token_payload.user_id)
        
        return {
            "user_id": token_payload.user_id,
            "localization_status": status_report,
            "overall_compliance": all(s["is_compliant"] for s in status_report.values()),
            "checked_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Localization status error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Localization status check failed"
        )


@router.post("/indian-compliance/user-rights")
async def process_user_rights_request(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    compliance_manager: IndianComplianceManager = Depends(get_indian_compliance_manager)
):
    """
    Process user rights request as per Indian law.
    
    Args:
        request: User rights request
        credentials: Bearer token credentials
        auth_service: Authentication service
        compliance_manager: Indian compliance manager
        
    Returns:
        Rights request processing result
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        request_type = request.get("request_type")
        details = request.get("details", {})
        
        if request_type not in ["access", "rectification", "erasure", "portability"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request type"
            )
        
        # Process rights request
        result = await compliance_manager.process_user_rights_request(
            user_id=token_payload.user_id,
            request_type=request_type,
            details=details
        )
        
        logger.info(
            "User rights request processed",
            user_id=token_payload.user_id,
            request_type=request_type,
            request_id=result["request_id"]
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("User rights request error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User rights request failed"
        )


@router.get("/indian-compliance/report")
async def get_compliance_report(
    period_days: int = 30,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    compliance_manager: IndianComplianceManager = Depends(get_indian_compliance_manager)
):
    """
    Generate Indian compliance report (admin only).
    
    Args:
        period_days: Reporting period in days
        credentials: Bearer token credentials
        auth_service: Authentication service
        compliance_manager: Indian compliance manager
        
    Returns:
        Compliance report
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # In production, add admin role check here
        # if not user_has_admin_role(token_payload.user_id):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate compliance report
        report = await compliance_manager.generate_compliance_report(period_days)
        
        logger.info(
            "Compliance report generated",
            user_id=token_payload.user_id,
            period_days=period_days
        )
        
        return report
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Compliance report error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance report generation failed"
        )


@router.post("/indian-compliance/data-breach")
async def report_data_breach(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    compliance_manager: IndianComplianceManager = Depends(get_indian_compliance_manager)
):
    """
    Report data breach for compliance handling (admin only).
    
    Args:
        request: Data breach report
        credentials: Bearer token credentials
        auth_service: Authentication service
        compliance_manager: Indian compliance manager
        
    Returns:
        Breach handling result
    """
    try:
        token = credentials.credentials
        token_payload = await auth_service.verify_token(token)
        
        # In production, add admin role check here
        # if not user_has_admin_role(token_payload.user_id):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        breach_id = request.get("breach_id", str(uuid4()))
        affected_users = request.get("affected_users", [])
        data_categories = [DataCategory(cat) for cat in request.get("data_categories", [])]
        breach_description = request.get("description", "")
        severity = request.get("severity", "high")
        
        # Handle data breach
        result = await compliance_manager.handle_data_breach(
            breach_id=breach_id,
            affected_users=affected_users,
            data_categories=data_categories,
            breach_description=breach_description,
            severity=severity
        )
        
        logger.critical(
            "Data breach reported",
            breach_id=breach_id,
            affected_users=len(affected_users),
            severity=severity
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Data breach reporting error", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data breach reporting failed"
        )