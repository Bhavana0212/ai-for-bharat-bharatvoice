"""
Payment service data models for BharatVoice Assistant.

This module defines data structures for payment processing, UPI transactions,
and payment security features.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class PaymentMethod(str, Enum):
    """Supported payment methods."""
    
    UPI = "upi"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    NET_BANKING = "net_banking"
    WALLET = "wallet"


class TransactionStatus(str, Enum):
    """Transaction status values."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    TIMEOUT = "timeout"


class UPIProvider(str, Enum):
    """UPI service providers."""
    
    GOOGLE_PAY = "googlepay"
    PHONEPE = "phonepe"
    PAYTM = "paytm"
    BHIM = "bhim"
    AMAZON_PAY = "amazonpay"
    GENERIC = "generic"


class PaymentRequest(BaseModel):
    """Payment request data structure."""
    
    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    user_id: UUID = Field(..., description="User identifier")
    amount: Decimal = Field(..., gt=0, description="Payment amount")
    currency: str = Field(default="INR", description="Currency code")
    payment_method: PaymentMethod = Field(..., description="Payment method")
    recipient_id: str = Field(..., description="Recipient identifier (UPI ID, account number, etc.)")
    description: str = Field(..., description="Payment description")
    reference_id: Optional[str] = Field(None, description="External reference ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        if v > Decimal('100000'):  # 1 lakh INR limit
            raise ValueError('Amount exceeds maximum limit of ₹1,00,000')
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        if v not in ['INR', 'USD', 'EUR']:
            raise ValueError('Unsupported currency')
        return v


class PaymentResponse(BaseModel):
    """Payment response data structure."""
    
    response_id: UUID = Field(default_factory=uuid4, description="Unique response ID")
    request_id: UUID = Field(..., description="Original request ID")
    transaction_id: Optional[str] = Field(None, description="Payment gateway transaction ID")
    status: TransactionStatus = Field(..., description="Transaction status")
    amount: Decimal = Field(..., description="Transaction amount")
    currency: str = Field(..., description="Currency code")
    gateway_response: Dict[str, Any] = Field(default_factory=dict, description="Gateway response data")
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class UPITransaction(BaseModel):
    """UPI-specific transaction data."""
    
    transaction_id: UUID = Field(default_factory=uuid4, description="Unique transaction ID")
    upi_id: str = Field(..., description="UPI ID")
    provider: UPIProvider = Field(..., description="UPI provider")
    virtual_payment_address: str = Field(..., description="Virtual Payment Address (VPA)")
    amount: Decimal = Field(..., description="Transaction amount")
    reference_number: str = Field(..., description="UPI reference number")
    merchant_transaction_id: Optional[str] = Field(None, description="Merchant transaction ID")
    status: TransactionStatus = Field(..., description="Transaction status")
    initiated_at: datetime = Field(default_factory=datetime.utcnow, description="Transaction initiation time")
    completed_at: Optional[datetime] = Field(None, description="Transaction completion time")
    
    @validator('upi_id')
    def validate_upi_id(cls, v):
        if '@' not in v:
            raise ValueError('Invalid UPI ID format')
        return v.lower()


class PaymentHistory(BaseModel):
    """User payment history record."""
    
    user_id: UUID = Field(..., description="User identifier")
    transactions: List[UPITransaction] = Field(default_factory=list, description="Transaction history")
    total_amount: Decimal = Field(default=Decimal('0'), description="Total transaction amount")
    successful_transactions: int = Field(default=0, description="Number of successful transactions")
    failed_transactions: int = Field(default=0, description="Number of failed transactions")
    last_transaction_date: Optional[datetime] = Field(None, description="Last transaction date")
    
    def add_transaction(self, transaction: UPITransaction) -> None:
        """Add a transaction to history."""
        self.transactions.append(transaction)
        if transaction.status == TransactionStatus.SUCCESS:
            self.total_amount += transaction.amount
            self.successful_transactions += 1
        elif transaction.status == TransactionStatus.FAILED:
            self.failed_transactions += 1
        self.last_transaction_date = transaction.initiated_at


class MFAChallenge(BaseModel):
    """Multi-factor authentication challenge for payments."""
    
    challenge_id: UUID = Field(default_factory=uuid4, description="Challenge ID")
    user_id: UUID = Field(..., description="User identifier")
    challenge_type: str = Field(..., description="Challenge type (otp, biometric, etc.)")
    challenge_data: Dict[str, Any] = Field(default_factory=dict, description="Challenge data")
    expires_at: datetime = Field(..., description="Challenge expiration time")
    attempts: int = Field(default=0, description="Number of attempts")
    max_attempts: int = Field(default=3, description="Maximum allowed attempts")
    is_verified: bool = Field(default=False, description="Whether challenge is verified")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Challenge creation time")


class PaymentSecurityContext(BaseModel):
    """Security context for payment processing."""
    
    user_id: UUID = Field(..., description="User identifier")
    device_fingerprint: str = Field(..., description="Device fingerprint")
    ip_address: str = Field(..., description="User IP address")
    location_data: Optional[Dict[str, Any]] = Field(None, description="Location data")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk assessment score")
    requires_mfa: bool = Field(default=False, description="Whether MFA is required")
    mfa_challenge: Optional[MFAChallenge] = Field(None, description="MFA challenge if required")
    session_token: str = Field(..., description="Secure session token")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Context creation time")


class VoicePaymentCommand(BaseModel):
    """Voice command for payment processing."""
    
    command_id: UUID = Field(default_factory=uuid4, description="Command ID")
    user_id: UUID = Field(..., description="User identifier")
    voice_input: str = Field(..., description="Transcribed voice input")
    parsed_intent: str = Field(..., description="Parsed payment intent")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Command confidence")
    requires_confirmation: bool = Field(default=True, description="Whether confirmation is required")
    confirmation_text: str = Field(..., description="Confirmation text to read back")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Command timestamp")