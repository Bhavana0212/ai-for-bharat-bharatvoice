"""
Payment manager for BharatVoice Assistant.

This module provides high-level payment management capabilities including
payment orchestration, security validation, and voice-guided payment flows.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...core.interfaces import BaseService
from ...core.models import ServiceResult, ServiceType
from .models import (
    PaymentHistory,
    PaymentRequest,
    PaymentResponse,
    PaymentSecurityContext,
    TransactionStatus,
    VoicePaymentCommand
)
from .upi_service import UPIService

logger = logging.getLogger(__name__)


class PaymentManager(BaseService):
    """
    High-level payment manager that orchestrates payment processing across
    different payment methods and provides voice-guided payment capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize payment manager with configuration.
        
        Args:
            config: Service configuration
        """
        super().__init__(config)
        
        # Initialize payment services
        self.upi_service = UPIService(config.get('upi', {}))
        
        # Security settings
        self.max_daily_amount = config.get('max_daily_amount', 50000.0)
        self.max_transaction_amount = config.get('max_transaction_amount', 100000.0)
        self.fraud_detection_enabled = config.get('fraud_detection_enabled', True)
        
        # Voice payment settings
        self.voice_confirmation_required = config.get('voice_confirmation_required', True)
        self.voice_timeout_seconds = config.get('voice_timeout_seconds', 30)
        
        logger.info("Payment manager initialized")
    
    async def process_voice_payment(
        self, 
        voice_command: VoicePaymentCommand,
        security_context: PaymentSecurityContext
    ) -> Dict[str, Any]:
        """
        Process a complete voice-guided payment flow.
        
        Args:
            voice_command: Voice payment command
            security_context: Security context for the payment
            
        Returns:
            Dict containing payment processing result and next steps
        """
        try:
            logger.info(f"Processing voice payment: {voice_command.command_id}")
            
            # Step 1: Parse voice command and extract payment details
            command_result = await self.upi_service.process_voice_payment_command(voice_command)
            
            if not command_result['success']:
                return {
                    'success': False,
                    'step': 'command_parsing',
                    'error': command_result['error'],
                    'voice_response': self._generate_error_response(command_result['error']),
                    'requires_clarification': command_result.get('requires_clarification', False),
                    'clarification_prompt': command_result.get('clarification_prompt')
                }
            
            # Step 2: Security validation
            security_result = await self._validate_payment_security(
                command_result['payment_request'], 
                security_context
            )
            
            if not security_result['allowed']:
                return {
                    'success': False,
                    'step': 'security_validation',
                    'error': security_result['reason'],
                    'voice_response': self._generate_security_error_response(security_result['reason']),
                    'requires_action': security_result.get('requires_action')
                }
            
            # Step 3: Generate confirmation and wait for user response
            if voice_command.requires_confirmation:
                return {
                    'success': True,
                    'step': 'confirmation_required',
                    'payment_request': command_result['payment_request'],
                    'confirmation_text': command_result['confirmation_text'],
                    'voice_response': command_result['confirmation_text'],
                    'timeout_seconds': self.voice_timeout_seconds,
                    'next_action': 'await_confirmation'
                }
            
            # Step 4: Process payment directly (if no confirmation required)
            payment_request = PaymentRequest(**command_result['payment_request'])
            payment_result = await self.upi_service.initiate_payment(payment_request, security_context)
            
            return await self._format_payment_result(payment_result, 'direct_processing')
            
        except Exception as e:
            logger.error(f"Error processing voice payment: {e}")
            return {
                'success': False,
                'step': 'processing_error',
                'error': str(e),
                'voice_response': "I'm sorry, there was an error processing your payment. Please try again."
            }
    
    async def confirm_voice_payment(
        self,
        payment_request: PaymentRequest,
        security_context: PaymentSecurityContext,
        confirmation_response: str
    ) -> Dict[str, Any]:
        """
        Process payment confirmation from voice input.
        
        Args:
            payment_request: Payment request to confirm
            security_context: Security context
            confirmation_response: User's confirmation response
            
        Returns:
            Dict containing payment result
        """
        try:
            # Parse confirmation response
            if not self._is_confirmation_positive(confirmation_response):
                return {
                    'success': False,
                    'step': 'confirmation_declined',
                    'message': 'Payment cancelled by user',
                    'voice_response': 'Payment cancelled. Is there anything else I can help you with?'
                }
            
            # Process the payment
            payment_result = await self.upi_service.initiate_payment(payment_request, security_context)
            
            return await self._format_payment_result(payment_result, 'confirmed_processing')
            
        except Exception as e:
            logger.error(f"Error confirming voice payment: {e}")
            return {
                'success': False,
                'step': 'confirmation_error',
                'error': str(e),
                'voice_response': "There was an error confirming your payment. Please try again."
            }
    
    async def handle_mfa_verification(
        self,
        challenge_id: UUID,
        verification_input: str,
        is_voice_input: bool = True
    ) -> Dict[str, Any]:
        """
        Handle MFA verification for payment authorization.
        
        Args:
            challenge_id: MFA challenge ID
            verification_input: User's verification input (OTP, etc.)
            is_voice_input: Whether input came from voice
            
        Returns:
            Dict containing verification result
        """
        try:
            # Extract verification code from voice input if needed
            if is_voice_input:
                verification_code = self._extract_verification_code(verification_input)
            else:
                verification_code = verification_input.strip()
            
            # Verify with UPI service
            verification_result = await self.upi_service.verify_mfa_challenge(
                challenge_id, 
                verification_code
            )
            
            if verification_result['success']:
                return {
                    'success': True,
                    'step': 'mfa_verified',
                    'message': 'Verification successful',
                    'voice_response': 'Verification successful. Processing your payment now.',
                    'next_action': 'process_payment'
                }
            else:
                error_response = self._generate_mfa_error_response(verification_result)
                return {
                    'success': False,
                    'step': 'mfa_verification_failed',
                    'error': verification_result['error'],
                    'voice_response': error_response,
                    'attempts_remaining': verification_result.get('attempts_remaining'),
                    'next_action': 'retry_verification' if verification_result.get('attempts_remaining', 0) > 0 else 'abort_payment'
                }
                
        except Exception as e:
            logger.error(f"Error handling MFA verification: {e}")
            return {
                'success': False,
                'step': 'mfa_error',
                'error': str(e),
                'voice_response': "There was an error verifying your code. Please try again."
            }
    
    async def get_payment_status_voice_response(
        self, 
        user_id: UUID, 
        transaction_reference: str
    ) -> Dict[str, Any]:
        """
        Get payment status and generate voice response.
        
        Args:
            user_id: User identifier
            transaction_reference: Transaction reference to check
            
        Returns:
            Dict containing status and voice response
        """
        try:
            # Check transaction status
            status_result = await self.upi_service.check_transaction_status(transaction_reference)
            
            if not status_result['success']:
                return {
                    'success': False,
                    'error': status_result['error'],
                    'voice_response': "I couldn't check the payment status right now. Please try again later."
                }
            
            # Generate voice response based on status
            status = status_result['status']
            voice_response = self._generate_status_voice_response(status, status_result)
            
            return {
                'success': True,
                'transaction_status': status,
                'transaction_details': status_result,
                'voice_response': voice_response
            }
            
        except Exception as e:
            logger.error(f"Error getting payment status: {e}")
            return {
                'success': False,
                'error': str(e),
                'voice_response': "There was an error checking the payment status."
            }
    
    async def get_payment_history_summary(
        self, 
        user_id: UUID, 
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get payment history summary with voice response.
        
        Args:
            user_id: User identifier
            period_days: Number of days to include in summary
            
        Returns:
            Dict containing history summary and voice response
        """
        try:
            # Get payment history
            history_result = await self.upi_service.get_payment_history(user_id, limit=100)
            
            if not history_result['success']:
                return {
                    'success': False,
                    'error': history_result['error'],
                    'voice_response': "I couldn't retrieve your payment history right now."
                }
            
            # Generate summary
            summary = self._generate_payment_summary(history_result, period_days)
            voice_response = self._generate_history_voice_response(summary)
            
            return {
                'success': True,
                'summary': summary,
                'voice_response': voice_response,
                'detailed_history': history_result
            }
            
        except Exception as e:
            logger.error(f"Error getting payment history: {e}")
            return {
                'success': False,
                'error': str(e),
                'voice_response': "There was an error retrieving your payment history."
            }
    
    def _is_confirmation_positive(self, response: str) -> bool:
        """Check if confirmation response is positive."""
        positive_words = ['yes', 'confirm', 'proceed', 'ok', 'okay', 'sure', 'go ahead', 'haan', 'theek hai']
        negative_words = ['no', 'cancel', 'abort', 'stop', 'nahi', 'mat karo']
        
        response_lower = response.lower().strip()
        
        # Check for negative words first
        if any(word in response_lower for word in negative_words):
            return False
        
        # Check for positive words
        return any(word in response_lower for word in positive_words)
    
    def _extract_verification_code(self, voice_input: str) -> str:
        """Extract verification code from voice input."""
        # Simple extraction - in production, use speech-to-text with number recognition
        import re
        
        # Look for 6-digit numbers
        numbers = re.findall(r'\b\d{6}\b', voice_input)
        if numbers:
            return numbers[0]
        
        # Look for individual digits
        digits = re.findall(r'\b\d\b', voice_input)
        if len(digits) >= 6:
            return ''.join(digits[:6])
        
        return voice_input.strip()
    
    async def _validate_payment_security(
        self, 
        payment_request: Dict[str, Any], 
        security_context: PaymentSecurityContext
    ) -> Dict[str, Any]:
        """Validate payment security and fraud detection."""
        try:
            amount = float(payment_request['amount'])
            
            # Check transaction limits
            if amount > self.max_transaction_amount:
                return {
                    'allowed': False,
                    'reason': f'Amount exceeds maximum limit of ₹{self.max_transaction_amount:,.2f}',
                    'requires_action': 'reduce_amount'
                }
            
            # Check daily limits (simplified - in production, check actual daily spending)
            if amount > self.max_daily_amount:
                return {
                    'allowed': False,
                    'reason': f'Amount exceeds daily limit of ₹{self.max_daily_amount:,.2f}',
                    'requires_action': 'wait_or_reduce'
                }
            
            # Fraud detection (simplified)
            if self.fraud_detection_enabled and security_context.risk_score > 0.7:
                return {
                    'allowed': False,
                    'reason': 'Transaction flagged for security review',
                    'requires_action': 'contact_support'
                }
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Error validating payment security: {e}")
            return {
                'allowed': False,
                'reason': 'Security validation failed',
                'requires_action': 'retry'
            }
    
    async def _format_payment_result(
        self, 
        payment_result: PaymentResponse, 
        step: str
    ) -> Dict[str, Any]:
        """Format payment result for voice response."""
        if payment_result.status == TransactionStatus.SUCCESS:
            voice_response = (
                f"Payment successful! ₹{payment_result.amount} has been sent. "
                f"Your transaction reference is {payment_result.transaction_id}."
            )
            return {
                'success': True,
                'step': step,
                'payment_status': 'success',
                'transaction_id': payment_result.transaction_id,
                'amount': float(payment_result.amount),
                'voice_response': voice_response
            }
        
        elif payment_result.status == TransactionStatus.PENDING:
            if 'mfa_challenge_id' in payment_result.gateway_response:
                voice_response = (
                    "For security, please provide the verification code sent to your registered mobile number."
                )
                return {
                    'success': True,
                    'step': 'mfa_required',
                    'payment_status': 'pending_mfa',
                    'mfa_challenge_id': payment_result.gateway_response['mfa_challenge_id'],
                    'voice_response': voice_response,
                    'next_action': 'await_mfa'
                }
            else:
                voice_response = "Your payment is being processed. I'll update you once it's complete."
                return {
                    'success': True,
                    'step': step,
                    'payment_status': 'processing',
                    'transaction_id': payment_result.transaction_id,
                    'voice_response': voice_response
                }
        
        else:  # Failed
            error_msg = payment_result.error_message or "Payment failed"
            voice_response = f"Payment failed: {error_msg}. Please try again or contact support."
            return {
                'success': False,
                'step': step,
                'payment_status': 'failed',
                'error': error_msg,
                'voice_response': voice_response
            }
    
    def _generate_error_response(self, error: str) -> str:
        """Generate voice response for errors."""
        error_responses = {
            'amount': "I couldn't understand the payment amount. Please specify how much you want to send.",
            'recipient': "I couldn't identify the recipient. Please specify who you want to send money to.",
            'invalid': "The payment details seem invalid. Please try again with correct information."
        }
        
        for key, response in error_responses.items():
            if key in error.lower():
                return response
        
        return "I couldn't process your payment request. Please try again with clear details."
    
    def _generate_security_error_response(self, reason: str) -> str:
        """Generate voice response for security errors."""
        if 'limit' in reason.lower():
            return f"Sorry, {reason.lower()}. Please try with a smaller amount."
        elif 'security' in reason.lower():
            return "This transaction requires additional security verification. Please contact support."
        else:
            return f"Payment not allowed: {reason}"
    
    def _generate_mfa_error_response(self, verification_result: Dict[str, Any]) -> str:
        """Generate voice response for MFA errors."""
        error_code = verification_result.get('error_code', '')
        
        if error_code == 'INVALID_CODE':
            attempts = verification_result.get('attempts_remaining', 0)
            if attempts > 0:
                return f"Invalid code. You have {attempts} attempts remaining. Please try again."
            else:
                return "Maximum attempts exceeded. Payment cancelled for security."
        elif error_code == 'CHALLENGE_EXPIRED':
            return "Verification code has expired. Please start the payment process again."
        else:
            return "Verification failed. Please try again or contact support."
    
    def _generate_status_voice_response(self, status: str, details: Dict[str, Any]) -> str:
        """Generate voice response for payment status."""
        status_responses = {
            'success': f"Your payment was successful. Amount: ₹{details.get('amount', 'N/A')}",
            'pending': "Your payment is still being processed. Please check again in a few minutes.",
            'failed': "Your payment failed. Please try again or contact support.",
            'cancelled': "Your payment was cancelled.",
            'refunded': "Your payment has been refunded."
        }
        
        return status_responses.get(status, f"Payment status: {status}")
    
    def _generate_payment_summary(self, history_result: Dict[str, Any], period_days: int) -> Dict[str, Any]:
        """Generate payment summary from history."""
        transactions = history_result.get('transactions', [])
        
        # Filter transactions by period
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        recent_transactions = [
            t for t in transactions 
            if datetime.fromisoformat(t['initiated_at'].replace('Z', '+00:00')) > cutoff_date
        ]
        
        successful_amount = sum(
            t['amount'] for t in recent_transactions 
            if t['status'] == 'success'
        )
        
        return {
            'period_days': period_days,
            'total_transactions': len(recent_transactions),
            'successful_transactions': len([t for t in recent_transactions if t['status'] == 'success']),
            'total_amount': successful_amount,
            'recent_transactions': recent_transactions[:5]  # Last 5 transactions
        }
    
    def _generate_history_voice_response(self, summary: Dict[str, Any]) -> str:
        """Generate voice response for payment history."""
        period = summary['period_days']
        total_txns = summary['total_transactions']
        successful_txns = summary['successful_transactions']
        total_amount = summary['total_amount']
        
        if total_txns == 0:
            return f"You haven't made any payments in the last {period} days."
        
        response = (
            f"In the last {period} days, you made {total_txns} payment{'s' if total_txns != 1 else ''}, "
            f"with {successful_txns} successful transaction{'s' if successful_txns != 1 else ''} "
            f"totaling ₹{total_amount:,.2f}."
        )
        
        return response
    
    async def health_check(self) -> ServiceResult:
        """Check payment manager health."""
        try:
            # Check UPI service health
            upi_health = await self.upi_service.health_check()
            
            if not upi_health.success:
                return ServiceResult(
                    service_type=ServiceType.UPI_PAYMENT,
                    success=False,
                    error_message=f"UPI service unhealthy: {upi_health.error_message}",
                    response_time=upi_health.response_time
                )
            
            return ServiceResult(
                service_type=ServiceType.UPI_PAYMENT,
                success=True,
                data={'status': 'healthy', 'services': ['upi']},
                response_time=0.1
            )
            
        except Exception as e:
            return ServiceResult(
                service_type=ServiceType.UPI_PAYMENT,
                success=False,
                error_message=str(e),
                response_time=0.1
            )