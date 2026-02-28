"""
UPI payment service implementation for BharatVoice Assistant.

This module provides UPI payment processing capabilities including transaction
initiation, status checking, and payment security features.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

import aiohttp
from pydantic import ValidationError

from ...core.interfaces import BaseService
from ...core.models import ServiceResult, ServiceType
from .models import (
    MFAChallenge,
    PaymentHistory,
    PaymentRequest,
    PaymentResponse,
    PaymentSecurityContext,
    TransactionStatus,
    UPIProvider,
    UPITransaction,
    VoicePaymentCommand
)

logger = logging.getLogger(__name__)


class UPIService(BaseService):
    """
    UPI payment service for processing payments through various UPI providers.
    
    This service handles UPI payment initiation, transaction status checking,
    payment security, and multi-factor authentication for voice-guided payments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize UPI service with configuration.
        
        Args:
            config: Service configuration including API keys and endpoints
        """
        super().__init__(config)
        self.api_base_url = config.get('upi_api_base_url', 'https://api.upi-gateway.in/v1')
        self.merchant_id = config.get('merchant_id', 'BHARATVOICE_MERCHANT')
        self.api_key = config.get('upi_api_key', '')
        self.secret_key = config.get('upi_secret_key', '')
        self.timeout = config.get('timeout', 30.0)
        self.max_retries = config.get('max_retries', 3)
        
        # Payment limits and security settings
        self.min_amount = Decimal(config.get('min_amount', '1.00'))
        self.max_amount = Decimal(config.get('max_amount', '100000.00'))
        self.mfa_threshold = Decimal(config.get('mfa_threshold', '5000.00'))
        
        # In-memory storage for demo (replace with database in production)
        self._payment_history: Dict[UUID, PaymentHistory] = {}
        self._active_challenges: Dict[UUID, MFAChallenge] = {}
        self._transaction_cache: Dict[str, UPITransaction] = {}
        
        logger.info("UPI service initialized")
    
    async def process_voice_payment_command(
        self, 
        command: VoicePaymentCommand
    ) -> Dict[str, Any]:
        """
        Process a voice payment command and extract payment details.
        
        Args:
            command: Voice payment command with parsed intent and entities
            
        Returns:
            Dict containing processed payment details and next steps
        """
        try:
            logger.info(f"Processing voice payment command: {command.command_id}")
            
            # Extract payment details from voice command
            amount = self._extract_amount(command.extracted_entities)
            recipient = self._extract_recipient(command.extracted_entities)
            description = self._extract_description(command.voice_input)
            
            if not amount or not recipient:
                return {
                    'success': False,
                    'error': 'Could not extract payment amount or recipient from voice command',
                    'requires_clarification': True,
                    'clarification_prompt': 'Please specify the amount and recipient for the payment'
                }
            
            # Create payment request
            payment_request = PaymentRequest(
                user_id=command.user_id,
                amount=amount,
                payment_method='upi',
                recipient_id=recipient,
                description=description or f"Voice payment to {recipient}"
            )
            
            # Generate confirmation text
            confirmation_text = self._generate_confirmation_text(payment_request)
            
            return {
                'success': True,
                'payment_request': payment_request.dict(),
                'confirmation_text': confirmation_text,
                'requires_confirmation': True,
                'estimated_processing_time': '30 seconds'
            }
            
        except Exception as e:
            logger.error(f"Error processing voice payment command: {e}")
            return {
                'success': False,
                'error': f'Failed to process payment command: {str(e)}',
                'requires_retry': True
            }
    
    async def initiate_payment(
        self, 
        payment_request: PaymentRequest,
        security_context: PaymentSecurityContext
    ) -> PaymentResponse:
        """
        Initiate a UPI payment transaction.
        
        Args:
            payment_request: Payment request details
            security_context: Security context for the payment
            
        Returns:
            PaymentResponse with transaction details
        """
        try:
            logger.info(f"Initiating UPI payment: {payment_request.request_id}")
            
            # Validate payment request
            if not self._validate_payment_request(payment_request):
                return PaymentResponse(
                    request_id=payment_request.request_id,
                    status=TransactionStatus.FAILED,
                    amount=payment_request.amount,
                    currency=payment_request.currency,
                    error_code='VALIDATION_ERROR',
                    error_message='Invalid payment request',
                    processing_time=0.1
                )
            
            # Check if MFA is required
            if payment_request.amount >= self.mfa_threshold or security_context.requires_mfa:
                mfa_challenge = await self._create_mfa_challenge(
                    payment_request.user_id, 
                    security_context
                )
                
                return PaymentResponse(
                    request_id=payment_request.request_id,
                    status=TransactionStatus.PENDING,
                    amount=payment_request.amount,
                    currency=payment_request.currency,
                    gateway_response={'mfa_challenge_id': str(mfa_challenge.challenge_id)},
                    processing_time=0.5
                )
            
            # Process payment through UPI gateway
            transaction = await self._process_upi_transaction(payment_request, security_context)
            
            # Update payment history
            await self._update_payment_history(payment_request.user_id, transaction)
            
            return PaymentResponse(
                request_id=payment_request.request_id,
                transaction_id=str(transaction.transaction_id),
                status=transaction.status,
                amount=transaction.amount,
                currency=payment_request.currency,
                gateway_response={'upi_reference': transaction.reference_number},
                processing_time=2.5
            )
            
        except Exception as e:
            logger.error(f"Error initiating payment: {e}")
            return PaymentResponse(
                request_id=payment_request.request_id,
                status=TransactionStatus.FAILED,
                amount=payment_request.amount,
                currency=payment_request.currency,
                error_code='PROCESSING_ERROR',
                error_message=str(e),
                processing_time=1.0
            )
    
    async def verify_mfa_challenge(
        self, 
        challenge_id: UUID, 
        verification_code: str
    ) -> Dict[str, Any]:
        """
        Verify MFA challenge for payment authorization.
        
        Args:
            challenge_id: MFA challenge ID
            verification_code: User-provided verification code
            
        Returns:
            Dict containing verification result
        """
        try:
            challenge = self._active_challenges.get(challenge_id)
            if not challenge:
                return {
                    'success': False,
                    'error': 'Invalid or expired challenge',
                    'error_code': 'CHALLENGE_NOT_FOUND'
                }
            
            # Check if challenge has expired
            if datetime.utcnow() > challenge.expires_at:
                del self._active_challenges[challenge_id]
                return {
                    'success': False,
                    'error': 'Challenge has expired',
                    'error_code': 'CHALLENGE_EXPIRED'
                }
            
            # Check attempt limits
            if challenge.attempts >= challenge.max_attempts:
                del self._active_challenges[challenge_id]
                return {
                    'success': False,
                    'error': 'Maximum attempts exceeded',
                    'error_code': 'MAX_ATTEMPTS_EXCEEDED'
                }
            
            # Verify the code (simplified verification for demo)
            challenge.attempts += 1
            expected_code = challenge.challenge_data.get('expected_code', '123456')
            
            if verification_code == expected_code:
                challenge.is_verified = True
                return {
                    'success': True,
                    'message': 'MFA verification successful',
                    'challenge_verified': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Invalid verification code',
                    'error_code': 'INVALID_CODE',
                    'attempts_remaining': challenge.max_attempts - challenge.attempts
                }
                
        except Exception as e:
            logger.error(f"Error verifying MFA challenge: {e}")
            return {
                'success': False,
                'error': f'MFA verification failed: {str(e)}',
                'error_code': 'VERIFICATION_ERROR'
            }
    
    async def check_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """
        Check the status of a UPI transaction.
        
        Args:
            transaction_id: Transaction ID to check
            
        Returns:
            Dict containing transaction status and details
        """
        try:
            # Check cache first
            transaction = self._transaction_cache.get(transaction_id)
            if transaction:
                return {
                    'success': True,
                    'transaction_id': transaction_id,
                    'status': transaction.status.value,
                    'amount': float(transaction.amount),
                    'reference_number': transaction.reference_number,
                    'completed_at': transaction.completed_at.isoformat() if transaction.completed_at else None
                }
            
            # Query UPI gateway for status (mock implementation)
            status_data = await self._query_gateway_status(transaction_id)
            
            return {
                'success': True,
                'transaction_id': transaction_id,
                'status': status_data.get('status', 'unknown'),
                'gateway_response': status_data
            }
            
        except Exception as e:
            logger.error(f"Error checking transaction status: {e}")
            return {
                'success': False,
                'error': f'Failed to check transaction status: {str(e)}'
            }
    
    async def get_payment_history(
        self, 
        user_id: UUID, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get payment history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of transactions to return
            
        Returns:
            Dict containing payment history
        """
        try:
            history = self._payment_history.get(user_id, PaymentHistory(user_id=user_id))
            
            # Sort transactions by date (most recent first)
            sorted_transactions = sorted(
                history.transactions, 
                key=lambda t: t.initiated_at, 
                reverse=True
            )[:limit]
            
            return {
                'success': True,
                'user_id': str(user_id),
                'total_transactions': len(history.transactions),
                'successful_transactions': history.successful_transactions,
                'failed_transactions': history.failed_transactions,
                'total_amount': float(history.total_amount),
                'last_transaction_date': history.last_transaction_date.isoformat() if history.last_transaction_date else None,
                'transactions': [
                    {
                        'transaction_id': str(t.transaction_id),
                        'amount': float(t.amount),
                        'status': t.status.value,
                        'upi_id': t.upi_id,
                        'reference_number': t.reference_number,
                        'initiated_at': t.initiated_at.isoformat(),
                        'completed_at': t.completed_at.isoformat() if t.completed_at else None
                    }
                    for t in sorted_transactions
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting payment history: {e}")
            return {
                'success': False,
                'error': f'Failed to get payment history: {str(e)}'
            }
    
    def _extract_amount(self, entities: Dict[str, Any]) -> Optional[Decimal]:
        """Extract payment amount from voice command entities."""
        try:
            # Look for amount in various formats
            if 'amount' in entities:
                return Decimal(str(entities['amount']))
            
            if 'money' in entities:
                return Decimal(str(entities['money']))
            
            # Look for numeric values with currency indicators
            for key, value in entities.items():
                if 'rupee' in key.lower() or 'inr' in key.lower():
                    return Decimal(str(value))
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _extract_recipient(self, entities: Dict[str, Any]) -> Optional[str]:
        """Extract payment recipient from voice command entities."""
        # Look for UPI ID, phone number, or contact name
        if 'upi_id' in entities:
            return entities['upi_id']
        
        if 'phone_number' in entities:
            return entities['phone_number']
        
        if 'contact_name' in entities:
            return entities['contact_name']
        
        if 'recipient' in entities:
            return entities['recipient']
        
        return None
    
    def _extract_description(self, voice_input: str) -> Optional[str]:
        """Extract payment description from voice input."""
        # Simple extraction - in production, use NLP to extract context
        if 'for' in voice_input.lower():
            parts = voice_input.lower().split('for', 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return None
    
    def _generate_confirmation_text(self, payment_request: PaymentRequest) -> str:
        """Generate confirmation text for voice payment."""
        return (
            f"Please confirm: Send ₹{payment_request.amount} to {payment_request.recipient_id} "
            f"for {payment_request.description}. Say 'confirm' to proceed or 'cancel' to abort."
        )
    
    def _validate_payment_request(self, payment_request: PaymentRequest) -> bool:
        """Validate payment request parameters."""
        try:
            # Check amount limits
            if payment_request.amount < self.min_amount or payment_request.amount > self.max_amount:
                return False
            
            # Validate recipient format (simplified)
            if not payment_request.recipient_id or len(payment_request.recipient_id) < 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _create_mfa_challenge(
        self, 
        user_id: UUID, 
        security_context: PaymentSecurityContext
    ) -> MFAChallenge:
        """Create MFA challenge for payment authorization."""
        # Generate OTP (simplified for demo)
        otp = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        
        challenge = MFAChallenge(
            user_id=user_id,
            challenge_type='otp',
            challenge_data={'expected_code': otp, 'delivery_method': 'sms'},
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        self._active_challenges[challenge.challenge_id] = challenge
        
        # In production, send OTP via SMS/email
        logger.info(f"MFA challenge created: {challenge.challenge_id} (OTP: {otp})")
        
        return challenge
    
    async def _process_upi_transaction(
        self, 
        payment_request: PaymentRequest,
        security_context: PaymentSecurityContext
    ) -> UPITransaction:
        """Process UPI transaction through payment gateway."""
        # Create transaction record
        transaction = UPITransaction(
            upi_id=payment_request.recipient_id,
            provider=UPIProvider.GENERIC,
            virtual_payment_address=payment_request.recipient_id,
            amount=payment_request.amount,
            reference_number=f"UPI{secrets.randbelow(1000000000):09d}",
            merchant_transaction_id=str(payment_request.request_id),
            status=TransactionStatus.PROCESSING
        )
        
        # Simulate payment processing (replace with actual UPI gateway integration)
        await asyncio.sleep(2)  # Simulate processing time
        
        # Mock success/failure based on amount (for demo)
        if payment_request.amount <= Decimal('50000'):
            transaction.status = TransactionStatus.SUCCESS
            transaction.completed_at = datetime.utcnow()
        else:
            transaction.status = TransactionStatus.FAILED
        
        # Cache transaction
        self._transaction_cache[str(transaction.transaction_id)] = transaction
        
        logger.info(f"UPI transaction processed: {transaction.transaction_id} - {transaction.status}")
        
        return transaction
    
    async def _update_payment_history(self, user_id: UUID, transaction: UPITransaction) -> None:
        """Update user payment history with new transaction."""
        if user_id not in self._payment_history:
            self._payment_history[user_id] = PaymentHistory(user_id=user_id)
        
        self._payment_history[user_id].add_transaction(transaction)
    
    async def _query_gateway_status(self, transaction_id: str) -> Dict[str, Any]:
        """Query UPI gateway for transaction status."""
        # Mock implementation - replace with actual gateway API calls
        return {
            'status': 'success',
            'gateway_transaction_id': f"GTW{transaction_id}",
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def health_check(self) -> ServiceResult:
        """Check UPI service health."""
        try:
            # Test basic connectivity and configuration
            if not self.api_key or not self.merchant_id:
                return ServiceResult(
                    service_type=ServiceType.UPI_PAYMENT,
                    success=False,
                    error_message="Missing API configuration",
                    response_time=0.1
                )
            
            return ServiceResult(
                service_type=ServiceType.UPI_PAYMENT,
                success=True,
                data={'status': 'healthy', 'merchant_id': self.merchant_id},
                response_time=0.1
            )
            
        except Exception as e:
            return ServiceResult(
                service_type=ServiceType.UPI_PAYMENT,
                success=False,
                error_message=str(e),
                response_time=0.1
            )