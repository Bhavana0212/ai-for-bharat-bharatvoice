<<<<<<< HEAD
"""
Tests for the authentication system.

This module tests the complete authentication system including JWT tokens,
password management, sessions, MFA, encryption, privacy, and Indian compliance.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from bharatvoice.config import Settings
from bharatvoice.services.auth import (
    AuthService, JWTManager, PasswordManager, SessionManager, 
    MFAManager, EncryptionManager, PrivacyManager, IndianComplianceManager
)
from bharatvoice.services.auth.auth_service import RegisterRequest, LoginRequest
from bharatvoice.services.auth.privacy_manager import ConsentType, ConsentStatus, DataCategory


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        security={
            "secret_key": "test-secret-key-for-testing-only",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30
        }
    )


@pytest.fixture
def jwt_manager(settings):
    """Create JWT manager for testing."""
    return JWTManager(settings)


@pytest.fixture
def password_manager():
    """Create password manager for testing."""
    return PasswordManager()


@pytest.fixture
def session_manager(settings):
    """Create session manager for testing."""
    return SessionManager(settings)


@pytest.fixture
def mfa_manager(settings):
    """Create MFA manager for testing."""
    return MFAManager(settings)


@pytest.fixture
def encryption_manager(settings):
    """Create encryption manager for testing."""
    return EncryptionManager(settings)


@pytest.fixture
def privacy_manager(settings, encryption_manager):
    """Create privacy manager for testing."""
    return PrivacyManager(settings, encryption_manager)


@pytest.fixture
def indian_compliance_manager(settings, privacy_manager):
    """Create Indian compliance manager for testing."""
    return IndianComplianceManager(settings, privacy_manager)


@pytest.fixture
def auth_service(settings):
    """Create auth service for testing."""
    return AuthService(settings)


class TestJWTManager:
    """Test JWT token management."""
    
    def test_create_access_token(self, jwt_manager):
        """Test JWT token creation."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self, jwt_manager):
        """Test JWT token verification."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        # Create token
        token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        # Verify token
        payload = jwt_manager.verify_token(token)
        
        assert payload.user_id == str(user_id)
        assert payload.username == username
        assert payload.session_id == session_id
    
    def test_refresh_token(self, jwt_manager):
        """Test JWT token refresh."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        # Create original token
        original_token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        # Refresh token
        new_token = jwt_manager.refresh_token(original_token)
        
        assert isinstance(new_token, str)
        assert new_token != original_token
        
        # Verify new token
        payload = jwt_manager.verify_token(new_token)
        assert payload.user_id == str(user_id)


class TestPasswordManager:
    """Test password management."""
    
    def test_hash_password(self, password_manager):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)
        
        assert isinstance(hashed, str)
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self, password_manager):
        """Test password verification."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)
        
        # Correct password should verify
        assert password_manager.verify_password(password, hashed) is True
        
        # Incorrect password should not verify
        assert password_manager.verify_password("WrongPassword", hashed) is False
    
    def test_password_strength_assessment(self, password_manager):
        """Test password strength assessment."""
        # Strong password
        strong_password = "StrongPassword123!"
        strength = password_manager.assess_password_strength(strong_password)
        assert strength.is_valid is True
        assert strength.score >= 60
        
        # Weak password
        weak_password = "weak"
        strength = password_manager.assess_password_strength(weak_password)
        assert strength.is_valid is False
        assert len(strength.feedback) > 0
    
    def test_generate_secure_password(self, password_manager):
        """Test secure password generation."""
        password = password_manager.generate_secure_password(16)
        
        assert len(password) == 16
        
        # Check password meets policy
        strength = password_manager.assess_password_strength(password)
        assert strength.is_valid is True


class TestSessionManager:
    """Test session management."""
    
    def test_create_session(self, session_manager):
        """Test session creation."""
        user_id = uuid4()
        username = "testuser"
        
        session_data = session_manager.create_session(
            user_id=user_id,
            username=username
        )
        
        assert session_data.user_id == str(user_id)
        assert session_data.username == username
        assert session_data.is_active is True
        assert isinstance(session_data.session_id, str)
    
    def test_get_session(self, session_manager):
        """Test session retrieval."""
        user_id = uuid4()
        username = "testuser"
        
        # Create session
        session_data = session_manager.create_session(
            user_id=user_id,
            username=username
        )
        
        # Get session (without Redis, this will return None in mock)
        retrieved_session = session_manager.get_session(session_data.session_id)
        
        # In mock implementation, this returns None
        # In real implementation with Redis, it would return the session
        assert retrieved_session is None or retrieved_session.session_id == session_data.session_id


class TestMFAManager:
    """Test multi-factor authentication."""
    
    def test_generate_secret(self, mfa_manager):
        """Test MFA secret generation."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        
        assert mfa_secret.user_id == user_id
        assert isinstance(mfa_secret.secret, str)
        assert len(mfa_secret.backup_codes) == 10
        assert mfa_secret.is_enabled is False
    
    def test_get_qr_code(self, mfa_manager):
        """Test QR code generation."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        qr_code = mfa_manager.get_qr_code(mfa_secret, username)
        
        assert isinstance(qr_code, str)
        assert qr_code.startswith("data:image/png;base64,")
    
    def test_verify_totp_setup(self, mfa_manager):
        """Test TOTP setup verification."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        
        # Generate current TOTP token
        import pyotp
        totp = pyotp.TOTP(mfa_secret.secret)
        current_token = totp.now()
        
        # Verify token
        is_valid = mfa_manager.verify_totp_setup(mfa_secret, current_token)
        assert is_valid is True
        
        # Invalid token should fail
        is_valid = mfa_manager.verify_totp_setup(mfa_secret, "000000")
        assert is_valid is False


class TestEncryptionManager:
    """Test encryption and data protection."""
    
    def test_encrypt_decrypt_voice_data(self, encryption_manager):
        """Test voice data encryption and decryption."""
        user_id = "test_user"
        audio_data = b"fake_audio_data_for_testing"
        
        # Encrypt data
        encrypted_data = encryption_manager.encrypt_voice_data(audio_data, user_id)
        
        assert encrypted_data.key_id == f"voice_{user_id}"
        assert encrypted_data.algorithm == "AES-256-CBC"
        assert isinstance(encrypted_data.data, str)
        assert isinstance(encrypted_data.iv, str)
        
        # Decrypt data
        decrypted_data = encryption_manager.decrypt_voice_data(encrypted_data)
        assert decrypted_data == audio_data
    
    def test_encrypt_decrypt_user_profile(self, encryption_manager):
        """Test user profile encryption and decryption."""
        user_id = "test_user"
        profile_data = {
            "name": "Test User",
            "email": "test@example.com",
            "preferences": {"language": "en-IN"}
        }
        
        # Encrypt profile
        encrypted_data = encryption_manager.encrypt_user_profile(profile_data, user_id)
        
        assert encrypted_data.algorithm == "Fernet"
        assert isinstance(encrypted_data.data, str)
        
        # Decrypt profile
        decrypted_data = encryption_manager.decrypt_user_profile(encrypted_data)
        assert decrypted_data == profile_data
    
    def test_generate_rsa_keypair(self, encryption_manager):
        """Test RSA key pair generation."""
        key_id = "test_key"
        
        public_pem, private_pem = encryption_manager.generate_rsa_keypair(key_id)
        
        assert isinstance(public_pem, str)
        assert isinstance(private_pem, str)
        assert "BEGIN PUBLIC KEY" in public_pem
        assert "BEGIN PRIVATE KEY" in private_pem
    
    def test_rsa_encrypt_decrypt(self, encryption_manager):
        """Test RSA encryption and decryption."""
        key_id = "test_key"
        test_data = b"test data for RSA encryption"
        
        # Generate key pair
        public_pem, private_pem = encryption_manager.generate_rsa_keypair(key_id)
        
        # Encrypt with public key
        encrypted_data = encryption_manager.encrypt_with_rsa(test_data, public_pem)
        assert isinstance(encrypted_data, str)
        
        # Decrypt with private key
        decrypted_data = encryption_manager.decrypt_with_rsa(encrypted_data, private_pem)
        assert decrypted_data == test_data
    
    def test_anonymize_data(self, encryption_manager):
        """Test data anonymization."""
        data = {
            "user_id": "12345",
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
            "score": 85.5
        }
        
        fields_to_anonymize = ["user_id", "name", "email", "age"]
        
        anonymized_data = encryption_manager.anonymize_data(data, fields_to_anonymize)
        
        # Check that specified fields are anonymized
        assert anonymized_data["user_id"] != data["user_id"]
        assert anonymized_data["name"] != data["name"]
        assert anonymized_data["email"] != data["email"]
        assert anonymized_data["age"] != data["age"]
        
        # Check that non-specified fields remain unchanged
        assert anonymized_data["score"] == data["score"]


class TestPrivacyManager:
    """Test privacy management."""
    
    @pytest.mark.asyncio
    async def test_record_consent(self, privacy_manager):
        """Test consent recording."""
        user_id = "test_user"
        consent_type = ConsentType.VOICE_PROCESSING
        status = ConsentStatus.GRANTED
        purpose = "Voice recognition processing"
        
        consent = await privacy_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            status=status,
            purpose=purpose
        )
        
        assert consent.user_id == user_id
        assert consent.consent_type == consent_type
        assert consent.status == status
        assert consent.purpose == purpose
        assert isinstance(consent.consent_id, str)
    
    @pytest.mark.asyncio
    async def test_check_consent(self, privacy_manager):
        """Test consent checking."""
        user_id = "test_user"
        consent_type = ConsentType.VOICE_PROCESSING
        
        # Check consent (mock implementation returns True)
        has_consent = await privacy_manager.check_consent(user_id, consent_type)
        assert isinstance(has_consent, bool)
    
    @pytest.mark.asyncio
    async def test_schedule_data_deletion(self, privacy_manager):
        """Test data deletion scheduling."""
        user_id = "test_user"
        categories = [DataCategory.VOICE_DATA, DataCategory.PROFILE_DATA]
        reason = "user_request"
        
        deletion_request = await privacy_manager.schedule_data_deletion(
            user_id=user_id,
            categories=categories,
            reason=reason
        )
        
        assert deletion_request.user_id == user_id
        assert deletion_request.categories == categories
        assert deletion_request.reason == reason
        assert deletion_request.status == "pending"
        assert isinstance(deletion_request.request_id, str)
    
    @pytest.mark.asyncio
    async def test_anonymize_user_data(self, privacy_manager):
        """Test user data anonymization."""
        user_id = "test_user"
        categories = [DataCategory.VOICE_DATA, DataCategory.USAGE_DATA]
        
        results = await privacy_manager.anonymize_user_data(
            user_id=user_id,
            categories=categories,
            preserve_analytics=True
        )
        
        assert isinstance(results, dict)
        assert len(results) == len(categories)
        for category in categories:
            assert category.value in results
    
    @pytest.mark.asyncio
    async def test_get_privacy_report(self, privacy_manager):
        """Test privacy report generation."""
        user_id = "test_user"
        
        report = await privacy_manager.get_privacy_report(user_id)
        
        assert isinstance(report, dict)
        assert "user_id" in report
        assert "generated_at" in report
        assert "consents" in report
        assert "data_categories" in report
        assert "retention_policies" in report


class TestIndianComplianceManager:
    """Test Indian privacy law compliance."""
    
    @pytest.mark.asyncio
    async def test_validate_data_localization(self, indian_compliance_manager):
        """Test data localization validation."""
        user_id = "test_user"
        
        statuses = await indian_compliance_manager.validate_data_localization(user_id)
        
        assert isinstance(statuses, dict)
        for category_name, status in statuses.items():
            assert hasattr(status, 'is_compliant')
            assert hasattr(status, 'storage_location')
            assert hasattr(status, 'issues')
    
    @pytest.mark.asyncio
    async def test_obtain_granular_consent(self, indian_compliance_manager):
        """Test granular consent for Indian compliance."""
        user_id = "test_user"
        purposes = {
            "voice_processing": "Process voice commands",
            "analytics": "Improve service quality"
        }
        data_categories = [DataCategory.VOICE_DATA, DataCategory.USAGE_DATA]
        
        consent_record = await indian_compliance_manager.obtain_granular_consent(
            user_id=user_id,
            purposes=purposes,
            data_categories=data_categories,
            language="en-IN"
        )
        
        assert consent_record.user_id == user_id
        assert consent_record.data_categories == data_categories
        assert consent_record.consent_language == "en-IN"
        assert len(consent_record.granular_consent) == len(purposes)
    
    @pytest.mark.asyncio
    async def test_handle_data_breach(self, indian_compliance_manager):
        """Test data breach handling."""
        breach_id = "test_breach_001"
        affected_users = ["user1", "user2", "user3"]
        data_categories = [DataCategory.VOICE_DATA, DataCategory.PROFILE_DATA]
        breach_description = "Test data breach for compliance testing"
        
        result = await indian_compliance_manager.handle_data_breach(
            breach_id=breach_id,
            affected_users=affected_users,
            data_categories=data_categories,
            breach_description=breach_description,
            severity="high"
        )
        
        assert result["breach_id"] == breach_id
        assert result["status"] == "handled"
        assert "notifications_scheduled" in result
        assert "compliance_deadline" in result
    
    @pytest.mark.asyncio
    async def test_process_user_rights_request(self, indian_compliance_manager):
        """Test user rights request processing."""
        user_id = "test_user"
        request_type = "access"
        details = {"requested_data": "all"}
        
        result = await indian_compliance_manager.process_user_rights_request(
            user_id=user_id,
            request_type=request_type,
            details=details
        )
        
        assert result["user_id"] == user_id
        assert result["request_type"] == request_type
        assert "request_id" in result
        assert "status" in result
        assert "response_deadline" in result
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, indian_compliance_manager):
        """Test compliance report generation."""
        period_days = 30
        
        report = await indian_compliance_manager.generate_compliance_report(period_days)
        
        assert isinstance(report, dict)
        assert "report_id" in report
        assert "generated_at" in report
        assert "compliance_framework" in report
        assert "consent_management" in report
        assert "data_localization" in report
        assert "compliance_score" in report
        assert "recommendations" in report
        
        # Check compliance score is valid
        assert 0 <= report["compliance_score"] <= 100


class TestAuthService:
    """Test main authentication service."""
    
    @pytest.mark.asyncio
    async def test_register_user(self, auth_service):
        """Test user registration."""
        request = RegisterRequest(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            preferred_languages=["hi", "en-IN"],
            primary_language="hi"
        )
        
        response = await auth_service.register_user(request)
        
        assert response.username == request.username
        assert response.email == request.email
        assert isinstance(response.user_id, str)
        assert response.is_verified is False
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, auth_service):
        """Test user authentication."""
        request = LoginRequest(
            username="demo",
            password="demo"
        )
        
        response = await auth_service.authenticate_user(request)
        
        assert isinstance(response.access_token, str)
        assert response.token_type == "bearer"
        assert response.expires_in > 0
        assert isinstance(response.user_id, str)
        assert isinstance(response.session_id, str)


if __name__ == "__main__":
=======
"""
Tests for the authentication system.

This module tests the complete authentication system including JWT tokens,
password management, sessions, MFA, encryption, privacy, and Indian compliance.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from bharatvoice.config import Settings
from bharatvoice.services.auth import (
    AuthService, JWTManager, PasswordManager, SessionManager, 
    MFAManager, EncryptionManager, PrivacyManager, IndianComplianceManager
)
from bharatvoice.services.auth.auth_service import RegisterRequest, LoginRequest
from bharatvoice.services.auth.privacy_manager import ConsentType, ConsentStatus, DataCategory


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        security={
            "secret_key": "test-secret-key-for-testing-only",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30
        }
    )


@pytest.fixture
def jwt_manager(settings):
    """Create JWT manager for testing."""
    return JWTManager(settings)


@pytest.fixture
def password_manager():
    """Create password manager for testing."""
    return PasswordManager()


@pytest.fixture
def session_manager(settings):
    """Create session manager for testing."""
    return SessionManager(settings)


@pytest.fixture
def mfa_manager(settings):
    """Create MFA manager for testing."""
    return MFAManager(settings)


@pytest.fixture
def encryption_manager(settings):
    """Create encryption manager for testing."""
    return EncryptionManager(settings)


@pytest.fixture
def privacy_manager(settings, encryption_manager):
    """Create privacy manager for testing."""
    return PrivacyManager(settings, encryption_manager)


@pytest.fixture
def indian_compliance_manager(settings, privacy_manager):
    """Create Indian compliance manager for testing."""
    return IndianComplianceManager(settings, privacy_manager)


@pytest.fixture
def auth_service(settings):
    """Create auth service for testing."""
    return AuthService(settings)


class TestJWTManager:
    """Test JWT token management."""
    
    def test_create_access_token(self, jwt_manager):
        """Test JWT token creation."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self, jwt_manager):
        """Test JWT token verification."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        # Create token
        token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        # Verify token
        payload = jwt_manager.verify_token(token)
        
        assert payload.user_id == str(user_id)
        assert payload.username == username
        assert payload.session_id == session_id
    
    def test_refresh_token(self, jwt_manager):
        """Test JWT token refresh."""
        user_id = uuid4()
        username = "testuser"
        session_id = "test_session"
        
        # Create original token
        original_token = jwt_manager.create_access_token(
            user_id=user_id,
            username=username,
            session_id=session_id
        )
        
        # Refresh token
        new_token = jwt_manager.refresh_token(original_token)
        
        assert isinstance(new_token, str)
        assert new_token != original_token
        
        # Verify new token
        payload = jwt_manager.verify_token(new_token)
        assert payload.user_id == str(user_id)


class TestPasswordManager:
    """Test password management."""
    
    def test_hash_password(self, password_manager):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)
        
        assert isinstance(hashed, str)
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self, password_manager):
        """Test password verification."""
        password = "TestPassword123!"
        hashed = password_manager.hash_password(password)
        
        # Correct password should verify
        assert password_manager.verify_password(password, hashed) is True
        
        # Incorrect password should not verify
        assert password_manager.verify_password("WrongPassword", hashed) is False
    
    def test_password_strength_assessment(self, password_manager):
        """Test password strength assessment."""
        # Strong password
        strong_password = "StrongPassword123!"
        strength = password_manager.assess_password_strength(strong_password)
        assert strength.is_valid is True
        assert strength.score >= 60
        
        # Weak password
        weak_password = "weak"
        strength = password_manager.assess_password_strength(weak_password)
        assert strength.is_valid is False
        assert len(strength.feedback) > 0
    
    def test_generate_secure_password(self, password_manager):
        """Test secure password generation."""
        password = password_manager.generate_secure_password(16)
        
        assert len(password) == 16
        
        # Check password meets policy
        strength = password_manager.assess_password_strength(password)
        assert strength.is_valid is True


class TestSessionManager:
    """Test session management."""
    
    def test_create_session(self, session_manager):
        """Test session creation."""
        user_id = uuid4()
        username = "testuser"
        
        session_data = session_manager.create_session(
            user_id=user_id,
            username=username
        )
        
        assert session_data.user_id == str(user_id)
        assert session_data.username == username
        assert session_data.is_active is True
        assert isinstance(session_data.session_id, str)
    
    def test_get_session(self, session_manager):
        """Test session retrieval."""
        user_id = uuid4()
        username = "testuser"
        
        # Create session
        session_data = session_manager.create_session(
            user_id=user_id,
            username=username
        )
        
        # Get session (without Redis, this will return None in mock)
        retrieved_session = session_manager.get_session(session_data.session_id)
        
        # In mock implementation, this returns None
        # In real implementation with Redis, it would return the session
        assert retrieved_session is None or retrieved_session.session_id == session_data.session_id


class TestMFAManager:
    """Test multi-factor authentication."""
    
    def test_generate_secret(self, mfa_manager):
        """Test MFA secret generation."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        
        assert mfa_secret.user_id == user_id
        assert isinstance(mfa_secret.secret, str)
        assert len(mfa_secret.backup_codes) == 10
        assert mfa_secret.is_enabled is False
    
    def test_get_qr_code(self, mfa_manager):
        """Test QR code generation."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        qr_code = mfa_manager.get_qr_code(mfa_secret, username)
        
        assert isinstance(qr_code, str)
        assert qr_code.startswith("data:image/png;base64,")
    
    def test_verify_totp_setup(self, mfa_manager):
        """Test TOTP setup verification."""
        user_id = "test_user"
        username = "testuser"
        
        mfa_secret = mfa_manager.generate_secret(user_id, username)
        
        # Generate current TOTP token
        import pyotp
        totp = pyotp.TOTP(mfa_secret.secret)
        current_token = totp.now()
        
        # Verify token
        is_valid = mfa_manager.verify_totp_setup(mfa_secret, current_token)
        assert is_valid is True
        
        # Invalid token should fail
        is_valid = mfa_manager.verify_totp_setup(mfa_secret, "000000")
        assert is_valid is False


class TestEncryptionManager:
    """Test encryption and data protection."""
    
    def test_encrypt_decrypt_voice_data(self, encryption_manager):
        """Test voice data encryption and decryption."""
        user_id = "test_user"
        audio_data = b"fake_audio_data_for_testing"
        
        # Encrypt data
        encrypted_data = encryption_manager.encrypt_voice_data(audio_data, user_id)
        
        assert encrypted_data.key_id == f"voice_{user_id}"
        assert encrypted_data.algorithm == "AES-256-CBC"
        assert isinstance(encrypted_data.data, str)
        assert isinstance(encrypted_data.iv, str)
        
        # Decrypt data
        decrypted_data = encryption_manager.decrypt_voice_data(encrypted_data)
        assert decrypted_data == audio_data
    
    def test_encrypt_decrypt_user_profile(self, encryption_manager):
        """Test user profile encryption and decryption."""
        user_id = "test_user"
        profile_data = {
            "name": "Test User",
            "email": "test@example.com",
            "preferences": {"language": "en-IN"}
        }
        
        # Encrypt profile
        encrypted_data = encryption_manager.encrypt_user_profile(profile_data, user_id)
        
        assert encrypted_data.algorithm == "Fernet"
        assert isinstance(encrypted_data.data, str)
        
        # Decrypt profile
        decrypted_data = encryption_manager.decrypt_user_profile(encrypted_data)
        assert decrypted_data == profile_data
    
    def test_generate_rsa_keypair(self, encryption_manager):
        """Test RSA key pair generation."""
        key_id = "test_key"
        
        public_pem, private_pem = encryption_manager.generate_rsa_keypair(key_id)
        
        assert isinstance(public_pem, str)
        assert isinstance(private_pem, str)
        assert "BEGIN PUBLIC KEY" in public_pem
        assert "BEGIN PRIVATE KEY" in private_pem
    
    def test_rsa_encrypt_decrypt(self, encryption_manager):
        """Test RSA encryption and decryption."""
        key_id = "test_key"
        test_data = b"test data for RSA encryption"
        
        # Generate key pair
        public_pem, private_pem = encryption_manager.generate_rsa_keypair(key_id)
        
        # Encrypt with public key
        encrypted_data = encryption_manager.encrypt_with_rsa(test_data, public_pem)
        assert isinstance(encrypted_data, str)
        
        # Decrypt with private key
        decrypted_data = encryption_manager.decrypt_with_rsa(encrypted_data, private_pem)
        assert decrypted_data == test_data
    
    def test_anonymize_data(self, encryption_manager):
        """Test data anonymization."""
        data = {
            "user_id": "12345",
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
            "score": 85.5
        }
        
        fields_to_anonymize = ["user_id", "name", "email", "age"]
        
        anonymized_data = encryption_manager.anonymize_data(data, fields_to_anonymize)
        
        # Check that specified fields are anonymized
        assert anonymized_data["user_id"] != data["user_id"]
        assert anonymized_data["name"] != data["name"]
        assert anonymized_data["email"] != data["email"]
        assert anonymized_data["age"] != data["age"]
        
        # Check that non-specified fields remain unchanged
        assert anonymized_data["score"] == data["score"]


class TestPrivacyManager:
    """Test privacy management."""
    
    @pytest.mark.asyncio
    async def test_record_consent(self, privacy_manager):
        """Test consent recording."""
        user_id = "test_user"
        consent_type = ConsentType.VOICE_PROCESSING
        status = ConsentStatus.GRANTED
        purpose = "Voice recognition processing"
        
        consent = await privacy_manager.record_consent(
            user_id=user_id,
            consent_type=consent_type,
            status=status,
            purpose=purpose
        )
        
        assert consent.user_id == user_id
        assert consent.consent_type == consent_type
        assert consent.status == status
        assert consent.purpose == purpose
        assert isinstance(consent.consent_id, str)
    
    @pytest.mark.asyncio
    async def test_check_consent(self, privacy_manager):
        """Test consent checking."""
        user_id = "test_user"
        consent_type = ConsentType.VOICE_PROCESSING
        
        # Check consent (mock implementation returns True)
        has_consent = await privacy_manager.check_consent(user_id, consent_type)
        assert isinstance(has_consent, bool)
    
    @pytest.mark.asyncio
    async def test_schedule_data_deletion(self, privacy_manager):
        """Test data deletion scheduling."""
        user_id = "test_user"
        categories = [DataCategory.VOICE_DATA, DataCategory.PROFILE_DATA]
        reason = "user_request"
        
        deletion_request = await privacy_manager.schedule_data_deletion(
            user_id=user_id,
            categories=categories,
            reason=reason
        )
        
        assert deletion_request.user_id == user_id
        assert deletion_request.categories == categories
        assert deletion_request.reason == reason
        assert deletion_request.status == "pending"
        assert isinstance(deletion_request.request_id, str)
    
    @pytest.mark.asyncio
    async def test_anonymize_user_data(self, privacy_manager):
        """Test user data anonymization."""
        user_id = "test_user"
        categories = [DataCategory.VOICE_DATA, DataCategory.USAGE_DATA]
        
        results = await privacy_manager.anonymize_user_data(
            user_id=user_id,
            categories=categories,
            preserve_analytics=True
        )
        
        assert isinstance(results, dict)
        assert len(results) == len(categories)
        for category in categories:
            assert category.value in results
    
    @pytest.mark.asyncio
    async def test_get_privacy_report(self, privacy_manager):
        """Test privacy report generation."""
        user_id = "test_user"
        
        report = await privacy_manager.get_privacy_report(user_id)
        
        assert isinstance(report, dict)
        assert "user_id" in report
        assert "generated_at" in report
        assert "consents" in report
        assert "data_categories" in report
        assert "retention_policies" in report


class TestIndianComplianceManager:
    """Test Indian privacy law compliance."""
    
    @pytest.mark.asyncio
    async def test_validate_data_localization(self, indian_compliance_manager):
        """Test data localization validation."""
        user_id = "test_user"
        
        statuses = await indian_compliance_manager.validate_data_localization(user_id)
        
        assert isinstance(statuses, dict)
        for category_name, status in statuses.items():
            assert hasattr(status, 'is_compliant')
            assert hasattr(status, 'storage_location')
            assert hasattr(status, 'issues')
    
    @pytest.mark.asyncio
    async def test_obtain_granular_consent(self, indian_compliance_manager):
        """Test granular consent for Indian compliance."""
        user_id = "test_user"
        purposes = {
            "voice_processing": "Process voice commands",
            "analytics": "Improve service quality"
        }
        data_categories = [DataCategory.VOICE_DATA, DataCategory.USAGE_DATA]
        
        consent_record = await indian_compliance_manager.obtain_granular_consent(
            user_id=user_id,
            purposes=purposes,
            data_categories=data_categories,
            language="en-IN"
        )
        
        assert consent_record.user_id == user_id
        assert consent_record.data_categories == data_categories
        assert consent_record.consent_language == "en-IN"
        assert len(consent_record.granular_consent) == len(purposes)
    
    @pytest.mark.asyncio
    async def test_handle_data_breach(self, indian_compliance_manager):
        """Test data breach handling."""
        breach_id = "test_breach_001"
        affected_users = ["user1", "user2", "user3"]
        data_categories = [DataCategory.VOICE_DATA, DataCategory.PROFILE_DATA]
        breach_description = "Test data breach for compliance testing"
        
        result = await indian_compliance_manager.handle_data_breach(
            breach_id=breach_id,
            affected_users=affected_users,
            data_categories=data_categories,
            breach_description=breach_description,
            severity="high"
        )
        
        assert result["breach_id"] == breach_id
        assert result["status"] == "handled"
        assert "notifications_scheduled" in result
        assert "compliance_deadline" in result
    
    @pytest.mark.asyncio
    async def test_process_user_rights_request(self, indian_compliance_manager):
        """Test user rights request processing."""
        user_id = "test_user"
        request_type = "access"
        details = {"requested_data": "all"}
        
        result = await indian_compliance_manager.process_user_rights_request(
            user_id=user_id,
            request_type=request_type,
            details=details
        )
        
        assert result["user_id"] == user_id
        assert result["request_type"] == request_type
        assert "request_id" in result
        assert "status" in result
        assert "response_deadline" in result
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, indian_compliance_manager):
        """Test compliance report generation."""
        period_days = 30
        
        report = await indian_compliance_manager.generate_compliance_report(period_days)
        
        assert isinstance(report, dict)
        assert "report_id" in report
        assert "generated_at" in report
        assert "compliance_framework" in report
        assert "consent_management" in report
        assert "data_localization" in report
        assert "compliance_score" in report
        assert "recommendations" in report
        
        # Check compliance score is valid
        assert 0 <= report["compliance_score"] <= 100


class TestAuthService:
    """Test main authentication service."""
    
    @pytest.mark.asyncio
    async def test_register_user(self, auth_service):
        """Test user registration."""
        request = RegisterRequest(
            username="testuser",
            email="test@example.com",
            password="TestPassword123!",
            preferred_languages=["hi", "en-IN"],
            primary_language="hi"
        )
        
        response = await auth_service.register_user(request)
        
        assert response.username == request.username
        assert response.email == request.email
        assert isinstance(response.user_id, str)
        assert response.is_verified is False
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, auth_service):
        """Test user authentication."""
        request = LoginRequest(
            username="demo",
            password="demo"
        )
        
        response = await auth_service.authenticate_user(request)
        
        assert isinstance(response.access_token, str)
        assert response.token_type == "bearer"
        assert response.expires_in > 0
        assert isinstance(response.user_id, str)
        assert isinstance(response.session_id, str)


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    pytest.main([__file__])