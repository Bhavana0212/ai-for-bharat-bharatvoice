<<<<<<< HEAD
"""
Digital India Platform Integration Service.

This module provides integration with Digital India government services,
including service directories, application assistance, and document requirements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from bharatvoice.core.models import ServiceResult, ServiceType, GovernmentService


class ServiceCategory(str, Enum):
    """Government service categories."""
    IDENTITY_DOCUMENTS = "identity_documents"
    SOCIAL_WELFARE = "social_welfare"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    EMPLOYMENT = "employment"
    AGRICULTURE = "agriculture"
    BUSINESS_REGISTRATION = "business_registration"
    TAX_SERVICES = "tax_services"
    PENSION = "pension"
    CERTIFICATES = "certificates"


class ApplicationStatus(str, Enum):
    """Application status types."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass
class DocumentRequirement:
    """Document requirement information."""
    document_name: str
    document_type: str  # Original, Copy, Self-attested
    is_mandatory: bool
    description: str
    sample_format: Optional[str] = None
    where_to_obtain: Optional[str] = None


@dataclass
class ServiceStep:
    """Service application step."""
    step_number: int
    title: str
    description: str
    estimated_time: str
    required_documents: List[str]
    online_available: bool
    offline_location: Optional[str] = None


@dataclass
class GovernmentServiceInfo:
    """Comprehensive government service information."""
    service_id: str
    service_name: str
    department: str
    category: ServiceCategory
    description: str
    eligibility_criteria: List[str]
    required_documents: List[DocumentRequirement]
    application_steps: List[ServiceStep]
    processing_time: str
    fees: Optional[str]
    online_portal: Optional[str]
    helpline_number: Optional[str]
    office_locations: List[Dict[str, str]]


@dataclass
class ApplicationTracker:
    """Application tracking information."""
    application_id: str
    service_name: str
    applicant_name: str
    submission_date: str
    current_status: ApplicationStatus
    expected_completion: str
    last_updated: str
    next_action_required: Optional[str]
    tracking_url: Optional[str]


class DigitalIndiaService:
    """
    Service for integrating with Digital India government platforms.
    
    Provides access to government service directories, application assistance,
    document requirements, and service tracking capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.base_url = "https://digitalindia.gov.in/api"  # Mock API endpoint
        
        # Initialize service data
        self._government_services = self._initialize_government_services()
        self._document_templates = self._initialize_document_templates()
        self._state_portals = self._initialize_state_portals()
    
    def _initialize_government_services(self) -> Dict[str, GovernmentServiceInfo]:
        """Initialize government service information."""
        return {
            "aadhaar_card": GovernmentServiceInfo(
                service_id="aadhaar_card",
                service_name="Aadhaar Card Application/Update",
                department="Unique Identification Authority of India (UIDAI)",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for new Aadhaar card or update existing information",
                eligibility_criteria=[
                    "Indian resident for 182+ days in 12 months preceding application",
                    "All age groups eligible",
                    "Valid proof of identity and address required"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Proof of Identity",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Valid identity document like passport, driving license, voter ID",
                        where_to_obtain="Issuing authority"
                    ),
                    DocumentRequirement(
                        document_name="Proof of Address",
                        document_type="Original + Copy", 
                        is_mandatory=True,
                        description="Valid address proof like utility bill, bank statement, rent agreement",
                        where_to_obtain="Service provider/Bank/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Date of Birth Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Visit Aadhaar Center",
                        description="Locate nearest Aadhaar enrollment center",
                        estimated_time="30 minutes",
                        required_documents=["All POI, POA, DOB documents"],
                        online_available=False,
                        offline_location="Aadhaar Enrollment Center"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Fill Application Form",
                        description="Complete enrollment form with personal details",
                        estimated_time="15 minutes",
                        required_documents=[],
                        online_available=False
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Biometric Capture",
                        description="Fingerprint and iris scan capture",
                        estimated_time="10 minutes",
                        required_documents=[],
                        online_available=False
                    )
                ],
                processing_time="90 days",
                fees="Free",
                online_portal="https://uidai.gov.in",
                helpline_number="1947",
                office_locations=[
                    {"type": "Enrollment Center", "address": "Various locations", "timings": "9 AM - 5 PM"}
                ]
            ),
            "pan_card": GovernmentServiceInfo(
                service_id="pan_card",
                service_name="PAN Card Application",
                department="Income Tax Department",
                category=ServiceCategory.TAX_SERVICES,
                description="Apply for Permanent Account Number (PAN) card",
                eligibility_criteria=[
                    "Indian citizens and foreign nationals",
                    "Required for income tax purposes",
                    "Mandatory for financial transactions above specified limits"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Identity Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Aadhaar card, passport, driving license, voter ID",
                        where_to_obtain="Respective issuing authority"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, bank statement",
                        where_to_obtain="Service provider/Bank"
                    ),
                    DocumentRequirement(
                        document_name="Date of Birth Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Application",
                        description="Fill Form 49A online at NSDL/UTIITSL website",
                        estimated_time="20 minutes",
                        required_documents=["Scanned copies of all documents"],
                        online_available=True,
                        offline_location=None
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Document Submission",
                        description="Submit physical documents to PAN center or by post",
                        estimated_time="30 minutes",
                        required_documents=["Original documents + copies"],
                        online_available=False,
                        offline_location="PAN Service Center"
                    )
                ],
                processing_time="15-20 working days",
                fees="₹110 (online), ₹115 (offline)",
                online_portal="https://www.onlineservices.nsdl.com/paam/endUserRegisterContact.html",
                helpline_number="020-27218080",
                office_locations=[
                    {"type": "PAN Service Center", "address": "Various locations", "timings": "9 AM - 6 PM"}
                ]
            ),
            "passport": GovernmentServiceInfo(
                service_id="passport",
                service_name="Passport Application",
                department="Ministry of External Affairs",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for fresh passport or renewal",
                eligibility_criteria=[
                    "Indian citizen",
                    "Valid proof of citizenship",
                    "No criminal background (for certain categories)"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Birth Certificate",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate issued by municipal authority",
                        where_to_obtain="Municipal Corporation/Registrar of Births"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, rent agreement",
                        where_to_obtain="Service provider/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Identity Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, voter ID, driving license",
                        where_to_obtain="Respective issuing authority"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Registration",
                        description="Register on Passport Seva portal and fill application",
                        estimated_time="45 minutes",
                        required_documents=["Digital copies for upload"],
                        online_available=True
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Appointment Booking",
                        description="Book appointment at Passport Seva Kendra",
                        estimated_time="10 minutes",
                        required_documents=[],
                        online_available=True
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Document Verification",
                        description="Visit PSK for document verification and biometrics",
                        estimated_time="2-3 hours",
                        required_documents=["All original documents"],
                        online_available=False,
                        offline_location="Passport Seva Kendra"
                    )
                ],
                processing_time="30-45 days (normal), 7-10 days (tatkal)",
                fees="₹1,500 (36 pages), ₹2,000 (60 pages), ₹3,500 (tatkal)",
                online_portal="https://passportindia.gov.in",
                helpline_number="1800-258-1800",
                office_locations=[
                    {"type": "Passport Seva Kendra", "address": "Various cities", "timings": "9 AM - 5 PM"}
                ]
            ),
            "driving_license": GovernmentServiceInfo(
                service_id="driving_license",
                service_name="Driving License Application",
                department="Ministry of Road Transport and Highways",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for learner's license or permanent driving license",
                eligibility_criteria=[
                    "Minimum age 18 years (16 for two-wheeler without gear)",
                    "Physically and mentally fit to drive",
                    "Pass written and practical driving test"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Age Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, rent agreement",
                        where_to_obtain="Service provider/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Medical Certificate",
                        document_type="Original",
                        is_mandatory=True,
                        description="Medical fitness certificate from registered doctor",
                        where_to_obtain="Government hospital/Registered medical practitioner"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Apply for Learner's License",
                        description="Submit application for learner's license",
                        estimated_time="1 hour",
                        required_documents=["Age proof, address proof, medical certificate"],
                        online_available=True,
                        offline_location="RTO Office"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Written Test",
                        description="Pass written test on traffic rules",
                        estimated_time="30 minutes",
                        required_documents=["Learner's license application receipt"],
                        online_available=False,
                        offline_location="RTO Office"
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Practical Driving Test",
                        description="Pass practical driving test (after 30 days of learner's license)",
                        estimated_time="30 minutes",
                        required_documents=["Learner's license"],
                        online_available=False,
                        offline_location="RTO Office"
                    )
                ],
                processing_time="30 days after practical test",
                fees="₹200 (learner's), ₹500 (permanent license)",
                online_portal="https://parivahan.gov.in",
                helpline_number="1800-110-321",
                office_locations=[
                    {"type": "RTO Office", "address": "District wise", "timings": "10 AM - 5 PM"}
                ]
            ),
            "pm_kisan": GovernmentServiceInfo(
                service_id="pm_kisan",
                service_name="PM-KISAN Scheme Registration",
                department="Ministry of Agriculture and Farmers Welfare",
                category=ServiceCategory.AGRICULTURE,
                description="Income support scheme for small and marginal farmers",
                eligibility_criteria=[
                    "Small and marginal farmer families",
                    "Landholding up to 2 hectares",
                    "Cultivable land ownership",
                    "Indian citizen"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Aadhaar Card",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card of the farmer",
                        where_to_obtain="UIDAI"
                    ),
                    DocumentRequirement(
                        document_name="Land Records",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Khata/Khatauni, Land ownership documents",
                        where_to_obtain="Village Revenue Office/Tehsil Office"
                    ),
                    DocumentRequirement(
                        document_name="Bank Account Details",
                        document_type="Copy",
                        is_mandatory=True,
                        description="Bank passbook or cancelled cheque",
                        where_to_obtain="Bank"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Registration",
                        description="Register on PM-KISAN portal or visit CSC",
                        estimated_time="30 minutes",
                        required_documents=["Aadhaar, land records, bank details"],
                        online_available=True,
                        offline_location="Common Service Center"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Document Verification",
                        description="Village level verification by revenue officials",
                        estimated_time="7-15 days",
                        required_documents=["All submitted documents"],
                        online_available=False,
                        offline_location="Village Revenue Office"
                    )
                ],
                processing_time="30-45 days",
                fees="Free",
                online_portal="https://pmkisan.gov.in",
                helpline_number="155261",
                office_locations=[
                    {"type": "Common Service Center", "address": "Village level", "timings": "9 AM - 6 PM"},
                    {"type": "Agriculture Office", "address": "Block/District level", "timings": "10 AM - 5 PM"}
                ]
            )
        }
    
    def _initialize_document_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize document templates and formats."""
        return {
            "affidavit": {
                "title": "General Affidavit Format",
                "description": "Standard affidavit format for various government applications",
                "required_elements": [
                    "Deponent's full name and address",
                    "Statement of facts",
                    "Verification clause",
                    "Notary/Magistrate attestation"
                ],
                "stamp_paper_value": "₹10 or ₹20",
                "where_to_get": "Notary office, Court, Sub-registrar office"
            },
            "income_certificate": {
                "title": "Income Certificate Application",
                "description": "Certificate showing annual income of family",
                "required_elements": [
                    "Family income details",
                    "Source of income",
                    "Property details",
                    "Bank statements"
                ],
                "validity": "1 year",
                "where_to_get": "Tehsil office, District Collectorate"
            },
            "domicile_certificate": {
                "title": "Domicile Certificate",
                "description": "Certificate proving residence in a particular state",
                "required_elements": [
                    "Proof of continuous residence",
                    "Birth certificate",
                    "School certificates",
                    "Property documents"
                ],
                "validity": "Permanent",
                "where_to_get": "Tehsil office, District Collectorate"
            }
        }
    
    def _initialize_state_portals(self) -> Dict[str, Dict[str, str]]:
        """Initialize state-specific government portals."""
        return {
            "andhra_pradesh": {
                "name": "AP Land Records",
                "url": "https://webland.ap.gov.in",
                "services": "Land records, Revenue services"
            },
            "telangana": {
                "name": "TS-iPASS",
                "url": "https://ipass.telangana.gov.in",
                "services": "Industrial approvals, Business registration"
            },
            "karnataka": {
                "name": "Sakala Services",
                "url": "https://sakala.karnataka.gov.in",
                "services": "Citizen services, Certificates"
            },
            "tamil_nadu": {
                "name": "TN e-Sevai",
                "url": "https://www.tnesevai.tn.gov.in",
                "services": "Online services, Certificates"
            },
            "maharashtra": {
                "name": "Aaple Sarkar",
                "url": "https://aaplesarkar.mahaonline.gov.in",
                "services": "Citizen services, Online applications"
            },
            "gujarat": {
                "name": "Digital Gujarat",
                "url": "https://digitalgujarat.gov.in",
                "services": "e-Governance services"
            },
            "rajasthan": {
                "name": "e-Mitra",
                "url": "https://emitra.rajasthan.gov.in",
                "services": "Citizen services, Bill payments"
            },
            "uttar_pradesh": {
                "name": "e-Sathi UP",
                "url": "https://esathi.up.gov.in",
                "services": "Income, Caste, Domicile certificates"
            },
            "west_bengal": {
                "name": "e-District",
                "url": "https://edistrict.wb.gov.in",
                "services": "Certificates, Licenses"
            },
            "bihar": {
                "name": "Bihar Online",
                "url": "https://serviceonline.bihar.gov.in",
                "services": "Revenue services, Certificates"
            }
        }
    
    async def get_service_information(
        self,
        service_name: str,
        category: Optional[ServiceCategory] = None
    ) -> ServiceResult:
        """
        Get comprehensive information about a government service.
        
        Args:
            service_name: Name or ID of the service
            category: Service category filter
            
        Returns:
            Service result with detailed service information
        """
        try:
            self.logger.info(f"Getting service information for: {service_name}")
            
            # Search for service by name or ID
            service_info = None
            service_name_lower = service_name.lower().replace(" ", "_")
            
            # Direct lookup
            if service_name_lower in self._government_services:
                service_info = self._government_services[service_name_lower]
            else:
                # Search by partial name match
                for service_id, service in self._government_services.items():
                    if (service_name.lower() in service.service_name.lower() or
                        service_name.lower() in service_id.lower()):
                        service_info = service
                        break
            
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found. Try 'Aadhaar', 'PAN', 'Passport', 'Driving License', or 'PM-KISAN'.",
                    response_time=0.3
                )
            
            # Filter by category if specified
            if category and service_info.category != category:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' is not in category '{category.value}'.",
                    response_time=0.2
                )
            
            service_data = {
                "service_info": {
                    "service_id": service_info.service_id,
                    "service_name": service_info.service_name,
                    "department": service_info.department,
                    "category": service_info.category.value,
                    "description": service_info.description,
                    "eligibility_criteria": service_info.eligibility_criteria,
                    "processing_time": service_info.processing_time,
                    "fees": service_info.fees,
                    "online_portal": service_info.online_portal,
                    "helpline_number": service_info.helpline_number
                },
                "required_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "mandatory": doc.is_mandatory,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain
                    }
                    for doc in service_info.required_documents
                ],
                "application_steps": [
                    {
                        "step": step.step_number,
                        "title": step.title,
                        "description": step.description,
                        "estimated_time": step.estimated_time,
                        "required_documents": step.required_documents,
                        "online_available": step.online_available,
                        "offline_location": step.offline_location
                    }
                    for step in service_info.application_steps
                ],
                "office_locations": service_info.office_locations,
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=service_data,
                error_message=None,
                response_time=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Error getting service information: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get service information: {str(e)}",
                response_time=0.5
            )
    
    async def get_document_requirements(
        self,
        service_name: str,
        user_category: Optional[str] = None
    ) -> ServiceResult:
        """
        Get detailed document requirements for a service.
        
        Args:
            service_name: Name of the government service
            user_category: User category (General, SC/ST, OBC, etc.)
            
        Returns:
            Service result with document requirements
        """
        try:
            self.logger.info(f"Getting document requirements for: {service_name}")
            
            service_info = await self._find_service(service_name)
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found.",
                    response_time=0.3
                )
            
            # Prepare document requirements with additional guidance
            document_data = {
                "service_name": service_info.service_name,
                "total_documents": len(service_info.required_documents),
                "mandatory_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain,
                        "tips": self._get_document_tips(doc.document_name)
                    }
                    for doc in service_info.required_documents if doc.is_mandatory
                ],
                "optional_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain,
                        "tips": self._get_document_tips(doc.document_name)
                    }
                    for doc in service_info.required_documents if not doc.is_mandatory
                ],
                "document_checklist": self._create_document_checklist(service_info.required_documents),
                "common_mistakes": self._get_common_document_mistakes(service_info.service_id),
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=document_data,
                error_message=None,
                response_time=0.6
            )
            
        except Exception as e:
            self.logger.error(f"Error getting document requirements: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get document requirements: {str(e)}",
                response_time=0.5
            )
    
    async def get_application_guidance(
        self,
        service_name: str,
        step_number: Optional[int] = None
    ) -> ServiceResult:
        """
        Get step-by-step application guidance.
        
        Args:
            service_name: Name of the government service
            step_number: Specific step number for detailed guidance
            
        Returns:
            Service result with application guidance
        """
        try:
            self.logger.info(f"Getting application guidance for: {service_name}")
            
            service_info = await self._find_service(service_name)
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found.",
                    response_time=0.3
                )
            
            if step_number:
                # Get specific step guidance
                if step_number > len(service_info.application_steps):
                    return ServiceResult(
                        service_type=ServiceType.GOVERNMENT_SERVICE,
                        success=False,
                        data={},
                        error_message=f"Step {step_number} not found. Service has {len(service_info.application_steps)} steps.",
                        response_time=0.2
                    )
                
                step = service_info.application_steps[step_number - 1]
                guidance_data = {
                    "service_name": service_info.service_name,
                    "current_step": {
                        "step_number": step.step_number,
                        "title": step.title,
                        "description": step.description,
                        "estimated_time": step.estimated_time,
                        "required_documents": step.required_documents,
                        "online_available": step.online_available,
                        "offline_location": step.offline_location,
                        "detailed_instructions": self._get_step_instructions(service_info.service_id, step_number),
                        "common_issues": self._get_step_common_issues(service_info.service_id, step_number)
                    },
                    "next_step": service_info.application_steps[step_number].title if step_number < len(service_info.application_steps) else "Application Complete",
                    "progress": f"{step_number}/{len(service_info.application_steps)}"
                }
            else:
                # Get complete application guidance
                guidance_data = {
                    "service_name": service_info.service_name,
                    "total_steps": len(service_info.application_steps),
                    "estimated_total_time": self._calculate_total_time(service_info.application_steps),
                    "application_steps": [
                        {
                            "step_number": step.step_number,
                            "title": step.title,
                            "description": step.description,
                            "estimated_time": step.estimated_time,
                            "online_available": step.online_available,
                            "key_points": self._get_step_key_points(service_info.service_id, step.step_number)
                        }
                        for step in service_info.application_steps
                    ],
                    "preparation_checklist": self._create_preparation_checklist(service_info),
                    "timeline": self._create_application_timeline(service_info)
                }
            
            guidance_data["last_updated"] = datetime.now().isoformat()
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=guidance_data,
                error_message=None,
                response_time=0.7
            )
            
        except Exception as e:
            self.logger.error(f"Error getting application guidance: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get application guidance: {str(e)}",
                response_time=0.5
            )
    
    async def search_services_by_category(
        self,
        category: ServiceCategory,
        limit: int = 10
    ) -> ServiceResult:
        """
        Search government services by category.
        
        Args:
            category: Service category to search
            limit: Maximum number of results
            
        Returns:
            Service result with matching services
        """
        try:
            self.logger.info(f"Searching services in category: {category.value}")
            
            matching_services = [
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "department": service.department,
                    "description": service.description,
                    "processing_time": service.processing_time,
                    "fees": service.fees,
                    "online_available": any(step.online_available for step in service.application_steps)
                }
                for service in self._government_services.values()
                if service.category == category
            ][:limit]
            
            if not matching_services:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"No services found in category '{category.value}'.",
                    response_time=0.3
                )
            
            search_data = {
                "category": category.value,
                "total_services": len(matching_services),
                "services": matching_services,
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=search_data,
                error_message=None,
                response_time=0.5
            )
            
        except Exception as e:
            self.logger.error(f"Error searching services by category: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to search services: {str(e)}",
                response_time=0.5
            )
    
    async def get_state_portal_info(self, state_name: str) -> ServiceResult:
        """
        Get state-specific government portal information.
        
        Args:
            state_name: Name of the state
            
        Returns:
            Service result with state portal information
        """
        try:
            self.logger.info(f"Getting state portal info for: {state_name}")
            
            state_key = state_name.lower().replace(" ", "_")
            portal_info = self._state_portals.get(state_key)
            
            if not portal_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"State portal information not available for '{state_name}'.",
                    response_time=0.3
                )
            
            portal_data = {
                "state": state_name.title(),
                "portal_name": portal_info["name"],
                "portal_url": portal_info["url"],
                "available_services": portal_info["services"],
                "access_instructions": [
                    "Visit the state portal website",
                    "Register with valid mobile number and email",
                    "Complete profile with required documents",
                    "Browse available services",
                    "Apply online and track application status"
                ],
                "common_services": [
                    "Income Certificate",
                    "Caste Certificate", 
                    "Domicile Certificate",
                    "Birth Certificate",
                    "Death Certificate",
                    "Marriage Certificate"
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=portal_data,
                error_message=None,
                response_time=0.4
            )
            
        except Exception as e:
            self.logger.error(f"Error getting state portal info: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get state portal information: {str(e)}",
                response_time=0.5
            )
    
    # Helper methods
    
    async def _find_service(self, service_name: str) -> Optional[GovernmentServiceInfo]:
        """Find service by name or partial match."""
        service_name_lower = service_name.lower().replace(" ", "_")
        
        # Direct lookup
        if service_name_lower in self._government_services:
            return self._government_services[service_name_lower]
        
        # Partial match
        for service_id, service in self._government_services.items():
            if (service_name.lower() in service.service_name.lower() or
                service_name.lower() in service_id.lower()):
                return service
        
        return None
    
    def _get_document_tips(self, document_name: str) -> List[str]:
        """Get tips for specific document preparation."""
        tips_map = {
            "aadhaar card": [
                "Ensure name matches exactly across all documents",
                "Update address if recently moved",
                "Keep both original and photocopy ready"
            ],
            "proof of identity": [
                "Use government-issued photo ID",
                "Ensure document is not expired",
                "Name should match application form exactly"
            ],
            "proof of address": [
                "Document should be recent (within 3 months)",
                "Address should match current residence",
                "Utility bills, bank statements are commonly accepted"
            ],
            "birth certificate": [
                "Get from municipal corporation where birth was registered",
                "Ensure all details are correct and legible",
                "May need translation if in regional language"
            ]
        }
        
        return tips_map.get(document_name.lower(), ["Ensure document is original and legible", "Keep photocopies ready"])
    
    def _create_document_checklist(self, documents: List[DocumentRequirement]) -> List[Dict[str, Any]]:
        """Create a document preparation checklist."""
        checklist = []
        for doc in documents:
            checklist.append({
                "document": doc.document_name,
                "status": "pending",
                "priority": "high" if doc.is_mandatory else "medium",
                "action": f"Obtain {doc.document_type.lower()} from {doc.where_to_obtain or 'relevant authority'}"
            })
        return checklist
    
    def _get_common_document_mistakes(self, service_id: str) -> List[str]:
        """Get common mistakes for document preparation."""
        common_mistakes = {
            "aadhaar_card": [
                "Name mismatch between documents",
                "Blurred or unclear photocopies",
                "Using expired address proof"
            ],
            "pan_card": [
                "Incorrect date of birth format",
                "Missing signature on application form",
                "Using non-acceptable identity proof"
            ],
            "passport": [
                "Incomplete online application",
                "Missing ECR/ECNR page requirements",
                "Incorrect fee payment"
            ]
        }
        
        return common_mistakes.get(service_id, ["Incomplete documentation", "Unclear photocopies", "Missing signatures"])
    
    def _get_step_instructions(self, service_id: str, step_number: int) -> List[str]:
        """Get detailed instructions for a specific step."""
        # This would contain detailed step-by-step instructions
        # For brevity, returning generic instructions
        return [
            f"Complete step {step_number} as described",
            "Ensure all required documents are ready",
            "Follow the official guidelines carefully",
            "Keep receipt/acknowledgment for future reference"
        ]
    
    def _get_step_common_issues(self, service_id: str, step_number: int) -> List[str]:
        """Get common issues for a specific step."""
        return [
            "Long waiting times during peak hours",
            "Document verification delays",
            "Technical issues with online portals",
            "Incomplete information provided"
        ]
    
    def _get_step_key_points(self, service_id: str, step_number: int) -> List[str]:
        """Get key points for a specific step."""
        return [
            "Carry all original documents",
            "Arrive early to avoid crowds",
            "Keep multiple photocopies",
            "Note down reference numbers"
        ]
    
    def _calculate_total_time(self, steps: List[ServiceStep]) -> str:
        """Calculate total estimated time for all steps."""
        # Simple calculation - in real implementation would parse time strings
        return f"{len(steps) * 2}-{len(steps) * 4} hours"
    
    def _create_preparation_checklist(self, service_info: GovernmentServiceInfo) -> List[str]:
        """Create preparation checklist for the service."""
        return [
            "Gather all required documents",
            "Make photocopies of all documents",
            "Fill application form completely",
            "Arrange for fees payment",
            "Plan visit during working hours",
            "Keep contact numbers handy"
        ]
    
    def _create_application_timeline(self, service_info: GovernmentServiceInfo) -> Dict[str, str]:
        """Create application timeline."""
        return {
            "preparation": "1-2 days",
            "application_submission": "1 day",
            "processing": service_info.processing_time,
            "total_duration": f"Approximately {service_info.processing_time}"
=======
"""
Digital India Platform Integration Service.

This module provides integration with Digital India government services,
including service directories, application assistance, and document requirements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from bharatvoice.core.models import ServiceResult, ServiceType, GovernmentService


class ServiceCategory(str, Enum):
    """Government service categories."""
    IDENTITY_DOCUMENTS = "identity_documents"
    SOCIAL_WELFARE = "social_welfare"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    EMPLOYMENT = "employment"
    AGRICULTURE = "agriculture"
    BUSINESS_REGISTRATION = "business_registration"
    TAX_SERVICES = "tax_services"
    PENSION = "pension"
    CERTIFICATES = "certificates"


class ApplicationStatus(str, Enum):
    """Application status types."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass
class DocumentRequirement:
    """Document requirement information."""
    document_name: str
    document_type: str  # Original, Copy, Self-attested
    is_mandatory: bool
    description: str
    sample_format: Optional[str] = None
    where_to_obtain: Optional[str] = None


@dataclass
class ServiceStep:
    """Service application step."""
    step_number: int
    title: str
    description: str
    estimated_time: str
    required_documents: List[str]
    online_available: bool
    offline_location: Optional[str] = None


@dataclass
class GovernmentServiceInfo:
    """Comprehensive government service information."""
    service_id: str
    service_name: str
    department: str
    category: ServiceCategory
    description: str
    eligibility_criteria: List[str]
    required_documents: List[DocumentRequirement]
    application_steps: List[ServiceStep]
    processing_time: str
    fees: Optional[str]
    online_portal: Optional[str]
    helpline_number: Optional[str]
    office_locations: List[Dict[str, str]]


@dataclass
class ApplicationTracker:
    """Application tracking information."""
    application_id: str
    service_name: str
    applicant_name: str
    submission_date: str
    current_status: ApplicationStatus
    expected_completion: str
    last_updated: str
    next_action_required: Optional[str]
    tracking_url: Optional[str]


class DigitalIndiaService:
    """
    Service for integrating with Digital India government platforms.
    
    Provides access to government service directories, application assistance,
    document requirements, and service tracking capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.base_url = "https://digitalindia.gov.in/api"  # Mock API endpoint
        
        # Initialize service data
        self._government_services = self._initialize_government_services()
        self._document_templates = self._initialize_document_templates()
        self._state_portals = self._initialize_state_portals()
    
    def _initialize_government_services(self) -> Dict[str, GovernmentServiceInfo]:
        """Initialize government service information."""
        return {
            "aadhaar_card": GovernmentServiceInfo(
                service_id="aadhaar_card",
                service_name="Aadhaar Card Application/Update",
                department="Unique Identification Authority of India (UIDAI)",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for new Aadhaar card or update existing information",
                eligibility_criteria=[
                    "Indian resident for 182+ days in 12 months preceding application",
                    "All age groups eligible",
                    "Valid proof of identity and address required"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Proof of Identity",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Valid identity document like passport, driving license, voter ID",
                        where_to_obtain="Issuing authority"
                    ),
                    DocumentRequirement(
                        document_name="Proof of Address",
                        document_type="Original + Copy", 
                        is_mandatory=True,
                        description="Valid address proof like utility bill, bank statement, rent agreement",
                        where_to_obtain="Service provider/Bank/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Date of Birth Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Visit Aadhaar Center",
                        description="Locate nearest Aadhaar enrollment center",
                        estimated_time="30 minutes",
                        required_documents=["All POI, POA, DOB documents"],
                        online_available=False,
                        offline_location="Aadhaar Enrollment Center"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Fill Application Form",
                        description="Complete enrollment form with personal details",
                        estimated_time="15 minutes",
                        required_documents=[],
                        online_available=False
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Biometric Capture",
                        description="Fingerprint and iris scan capture",
                        estimated_time="10 minutes",
                        required_documents=[],
                        online_available=False
                    )
                ],
                processing_time="90 days",
                fees="Free",
                online_portal="https://uidai.gov.in",
                helpline_number="1947",
                office_locations=[
                    {"type": "Enrollment Center", "address": "Various locations", "timings": "9 AM - 5 PM"}
                ]
            ),
            "pan_card": GovernmentServiceInfo(
                service_id="pan_card",
                service_name="PAN Card Application",
                department="Income Tax Department",
                category=ServiceCategory.TAX_SERVICES,
                description="Apply for Permanent Account Number (PAN) card",
                eligibility_criteria=[
                    "Indian citizens and foreign nationals",
                    "Required for income tax purposes",
                    "Mandatory for financial transactions above specified limits"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Identity Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Aadhaar card, passport, driving license, voter ID",
                        where_to_obtain="Respective issuing authority"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, bank statement",
                        where_to_obtain="Service provider/Bank"
                    ),
                    DocumentRequirement(
                        document_name="Date of Birth Proof",
                        document_type="Self-attested copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Application",
                        description="Fill Form 49A online at NSDL/UTIITSL website",
                        estimated_time="20 minutes",
                        required_documents=["Scanned copies of all documents"],
                        online_available=True,
                        offline_location=None
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Document Submission",
                        description="Submit physical documents to PAN center or by post",
                        estimated_time="30 minutes",
                        required_documents=["Original documents + copies"],
                        online_available=False,
                        offline_location="PAN Service Center"
                    )
                ],
                processing_time="15-20 working days",
                fees="₹110 (online), ₹115 (offline)",
                online_portal="https://www.onlineservices.nsdl.com/paam/endUserRegisterContact.html",
                helpline_number="020-27218080",
                office_locations=[
                    {"type": "PAN Service Center", "address": "Various locations", "timings": "9 AM - 6 PM"}
                ]
            ),
            "passport": GovernmentServiceInfo(
                service_id="passport",
                service_name="Passport Application",
                department="Ministry of External Affairs",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for fresh passport or renewal",
                eligibility_criteria=[
                    "Indian citizen",
                    "Valid proof of citizenship",
                    "No criminal background (for certain categories)"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Birth Certificate",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate issued by municipal authority",
                        where_to_obtain="Municipal Corporation/Registrar of Births"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, rent agreement",
                        where_to_obtain="Service provider/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Identity Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, voter ID, driving license",
                        where_to_obtain="Respective issuing authority"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Registration",
                        description="Register on Passport Seva portal and fill application",
                        estimated_time="45 minutes",
                        required_documents=["Digital copies for upload"],
                        online_available=True
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Appointment Booking",
                        description="Book appointment at Passport Seva Kendra",
                        estimated_time="10 minutes",
                        required_documents=[],
                        online_available=True
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Document Verification",
                        description="Visit PSK for document verification and biometrics",
                        estimated_time="2-3 hours",
                        required_documents=["All original documents"],
                        online_available=False,
                        offline_location="Passport Seva Kendra"
                    )
                ],
                processing_time="30-45 days (normal), 7-10 days (tatkal)",
                fees="₹1,500 (36 pages), ₹2,000 (60 pages), ₹3,500 (tatkal)",
                online_portal="https://passportindia.gov.in",
                helpline_number="1800-258-1800",
                office_locations=[
                    {"type": "Passport Seva Kendra", "address": "Various cities", "timings": "9 AM - 5 PM"}
                ]
            ),
            "driving_license": GovernmentServiceInfo(
                service_id="driving_license",
                service_name="Driving License Application",
                department="Ministry of Road Transport and Highways",
                category=ServiceCategory.IDENTITY_DOCUMENTS,
                description="Apply for learner's license or permanent driving license",
                eligibility_criteria=[
                    "Minimum age 18 years (16 for two-wheeler without gear)",
                    "Physically and mentally fit to drive",
                    "Pass written and practical driving test"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Age Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Birth certificate, school certificate, passport",
                        where_to_obtain="Municipal corporation/School/Passport office"
                    ),
                    DocumentRequirement(
                        document_name="Address Proof",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card, utility bill, rent agreement",
                        where_to_obtain="Service provider/Property owner"
                    ),
                    DocumentRequirement(
                        document_name="Medical Certificate",
                        document_type="Original",
                        is_mandatory=True,
                        description="Medical fitness certificate from registered doctor",
                        where_to_obtain="Government hospital/Registered medical practitioner"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Apply for Learner's License",
                        description="Submit application for learner's license",
                        estimated_time="1 hour",
                        required_documents=["Age proof, address proof, medical certificate"],
                        online_available=True,
                        offline_location="RTO Office"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Written Test",
                        description="Pass written test on traffic rules",
                        estimated_time="30 minutes",
                        required_documents=["Learner's license application receipt"],
                        online_available=False,
                        offline_location="RTO Office"
                    ),
                    ServiceStep(
                        step_number=3,
                        title="Practical Driving Test",
                        description="Pass practical driving test (after 30 days of learner's license)",
                        estimated_time="30 minutes",
                        required_documents=["Learner's license"],
                        online_available=False,
                        offline_location="RTO Office"
                    )
                ],
                processing_time="30 days after practical test",
                fees="₹200 (learner's), ₹500 (permanent license)",
                online_portal="https://parivahan.gov.in",
                helpline_number="1800-110-321",
                office_locations=[
                    {"type": "RTO Office", "address": "District wise", "timings": "10 AM - 5 PM"}
                ]
            ),
            "pm_kisan": GovernmentServiceInfo(
                service_id="pm_kisan",
                service_name="PM-KISAN Scheme Registration",
                department="Ministry of Agriculture and Farmers Welfare",
                category=ServiceCategory.AGRICULTURE,
                description="Income support scheme for small and marginal farmers",
                eligibility_criteria=[
                    "Small and marginal farmer families",
                    "Landholding up to 2 hectares",
                    "Cultivable land ownership",
                    "Indian citizen"
                ],
                required_documents=[
                    DocumentRequirement(
                        document_name="Aadhaar Card",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Aadhaar card of the farmer",
                        where_to_obtain="UIDAI"
                    ),
                    DocumentRequirement(
                        document_name="Land Records",
                        document_type="Original + Copy",
                        is_mandatory=True,
                        description="Khata/Khatauni, Land ownership documents",
                        where_to_obtain="Village Revenue Office/Tehsil Office"
                    ),
                    DocumentRequirement(
                        document_name="Bank Account Details",
                        document_type="Copy",
                        is_mandatory=True,
                        description="Bank passbook or cancelled cheque",
                        where_to_obtain="Bank"
                    )
                ],
                application_steps=[
                    ServiceStep(
                        step_number=1,
                        title="Online Registration",
                        description="Register on PM-KISAN portal or visit CSC",
                        estimated_time="30 minutes",
                        required_documents=["Aadhaar, land records, bank details"],
                        online_available=True,
                        offline_location="Common Service Center"
                    ),
                    ServiceStep(
                        step_number=2,
                        title="Document Verification",
                        description="Village level verification by revenue officials",
                        estimated_time="7-15 days",
                        required_documents=["All submitted documents"],
                        online_available=False,
                        offline_location="Village Revenue Office"
                    )
                ],
                processing_time="30-45 days",
                fees="Free",
                online_portal="https://pmkisan.gov.in",
                helpline_number="155261",
                office_locations=[
                    {"type": "Common Service Center", "address": "Village level", "timings": "9 AM - 6 PM"},
                    {"type": "Agriculture Office", "address": "Block/District level", "timings": "10 AM - 5 PM"}
                ]
            )
        }
    
    def _initialize_document_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize document templates and formats."""
        return {
            "affidavit": {
                "title": "General Affidavit Format",
                "description": "Standard affidavit format for various government applications",
                "required_elements": [
                    "Deponent's full name and address",
                    "Statement of facts",
                    "Verification clause",
                    "Notary/Magistrate attestation"
                ],
                "stamp_paper_value": "₹10 or ₹20",
                "where_to_get": "Notary office, Court, Sub-registrar office"
            },
            "income_certificate": {
                "title": "Income Certificate Application",
                "description": "Certificate showing annual income of family",
                "required_elements": [
                    "Family income details",
                    "Source of income",
                    "Property details",
                    "Bank statements"
                ],
                "validity": "1 year",
                "where_to_get": "Tehsil office, District Collectorate"
            },
            "domicile_certificate": {
                "title": "Domicile Certificate",
                "description": "Certificate proving residence in a particular state",
                "required_elements": [
                    "Proof of continuous residence",
                    "Birth certificate",
                    "School certificates",
                    "Property documents"
                ],
                "validity": "Permanent",
                "where_to_get": "Tehsil office, District Collectorate"
            }
        }
    
    def _initialize_state_portals(self) -> Dict[str, Dict[str, str]]:
        """Initialize state-specific government portals."""
        return {
            "andhra_pradesh": {
                "name": "AP Land Records",
                "url": "https://webland.ap.gov.in",
                "services": "Land records, Revenue services"
            },
            "telangana": {
                "name": "TS-iPASS",
                "url": "https://ipass.telangana.gov.in",
                "services": "Industrial approvals, Business registration"
            },
            "karnataka": {
                "name": "Sakala Services",
                "url": "https://sakala.karnataka.gov.in",
                "services": "Citizen services, Certificates"
            },
            "tamil_nadu": {
                "name": "TN e-Sevai",
                "url": "https://www.tnesevai.tn.gov.in",
                "services": "Online services, Certificates"
            },
            "maharashtra": {
                "name": "Aaple Sarkar",
                "url": "https://aaplesarkar.mahaonline.gov.in",
                "services": "Citizen services, Online applications"
            },
            "gujarat": {
                "name": "Digital Gujarat",
                "url": "https://digitalgujarat.gov.in",
                "services": "e-Governance services"
            },
            "rajasthan": {
                "name": "e-Mitra",
                "url": "https://emitra.rajasthan.gov.in",
                "services": "Citizen services, Bill payments"
            },
            "uttar_pradesh": {
                "name": "e-Sathi UP",
                "url": "https://esathi.up.gov.in",
                "services": "Income, Caste, Domicile certificates"
            },
            "west_bengal": {
                "name": "e-District",
                "url": "https://edistrict.wb.gov.in",
                "services": "Certificates, Licenses"
            },
            "bihar": {
                "name": "Bihar Online",
                "url": "https://serviceonline.bihar.gov.in",
                "services": "Revenue services, Certificates"
            }
        }
    
    async def get_service_information(
        self,
        service_name: str,
        category: Optional[ServiceCategory] = None
    ) -> ServiceResult:
        """
        Get comprehensive information about a government service.
        
        Args:
            service_name: Name or ID of the service
            category: Service category filter
            
        Returns:
            Service result with detailed service information
        """
        try:
            self.logger.info(f"Getting service information for: {service_name}")
            
            # Search for service by name or ID
            service_info = None
            service_name_lower = service_name.lower().replace(" ", "_")
            
            # Direct lookup
            if service_name_lower in self._government_services:
                service_info = self._government_services[service_name_lower]
            else:
                # Search by partial name match
                for service_id, service in self._government_services.items():
                    if (service_name.lower() in service.service_name.lower() or
                        service_name.lower() in service_id.lower()):
                        service_info = service
                        break
            
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found. Try 'Aadhaar', 'PAN', 'Passport', 'Driving License', or 'PM-KISAN'.",
                    response_time=0.3
                )
            
            # Filter by category if specified
            if category and service_info.category != category:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' is not in category '{category.value}'.",
                    response_time=0.2
                )
            
            service_data = {
                "service_info": {
                    "service_id": service_info.service_id,
                    "service_name": service_info.service_name,
                    "department": service_info.department,
                    "category": service_info.category.value,
                    "description": service_info.description,
                    "eligibility_criteria": service_info.eligibility_criteria,
                    "processing_time": service_info.processing_time,
                    "fees": service_info.fees,
                    "online_portal": service_info.online_portal,
                    "helpline_number": service_info.helpline_number
                },
                "required_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "mandatory": doc.is_mandatory,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain
                    }
                    for doc in service_info.required_documents
                ],
                "application_steps": [
                    {
                        "step": step.step_number,
                        "title": step.title,
                        "description": step.description,
                        "estimated_time": step.estimated_time,
                        "required_documents": step.required_documents,
                        "online_available": step.online_available,
                        "offline_location": step.offline_location
                    }
                    for step in service_info.application_steps
                ],
                "office_locations": service_info.office_locations,
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=service_data,
                error_message=None,
                response_time=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Error getting service information: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get service information: {str(e)}",
                response_time=0.5
            )
    
    async def get_document_requirements(
        self,
        service_name: str,
        user_category: Optional[str] = None
    ) -> ServiceResult:
        """
        Get detailed document requirements for a service.
        
        Args:
            service_name: Name of the government service
            user_category: User category (General, SC/ST, OBC, etc.)
            
        Returns:
            Service result with document requirements
        """
        try:
            self.logger.info(f"Getting document requirements for: {service_name}")
            
            service_info = await self._find_service(service_name)
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found.",
                    response_time=0.3
                )
            
            # Prepare document requirements with additional guidance
            document_data = {
                "service_name": service_info.service_name,
                "total_documents": len(service_info.required_documents),
                "mandatory_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain,
                        "tips": self._get_document_tips(doc.document_name)
                    }
                    for doc in service_info.required_documents if doc.is_mandatory
                ],
                "optional_documents": [
                    {
                        "name": doc.document_name,
                        "type": doc.document_type,
                        "description": doc.description,
                        "where_to_obtain": doc.where_to_obtain,
                        "tips": self._get_document_tips(doc.document_name)
                    }
                    for doc in service_info.required_documents if not doc.is_mandatory
                ],
                "document_checklist": self._create_document_checklist(service_info.required_documents),
                "common_mistakes": self._get_common_document_mistakes(service_info.service_id),
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=document_data,
                error_message=None,
                response_time=0.6
            )
            
        except Exception as e:
            self.logger.error(f"Error getting document requirements: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get document requirements: {str(e)}",
                response_time=0.5
            )
    
    async def get_application_guidance(
        self,
        service_name: str,
        step_number: Optional[int] = None
    ) -> ServiceResult:
        """
        Get step-by-step application guidance.
        
        Args:
            service_name: Name of the government service
            step_number: Specific step number for detailed guidance
            
        Returns:
            Service result with application guidance
        """
        try:
            self.logger.info(f"Getting application guidance for: {service_name}")
            
            service_info = await self._find_service(service_name)
            if not service_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"Service '{service_name}' not found.",
                    response_time=0.3
                )
            
            if step_number:
                # Get specific step guidance
                if step_number > len(service_info.application_steps):
                    return ServiceResult(
                        service_type=ServiceType.GOVERNMENT_SERVICE,
                        success=False,
                        data={},
                        error_message=f"Step {step_number} not found. Service has {len(service_info.application_steps)} steps.",
                        response_time=0.2
                    )
                
                step = service_info.application_steps[step_number - 1]
                guidance_data = {
                    "service_name": service_info.service_name,
                    "current_step": {
                        "step_number": step.step_number,
                        "title": step.title,
                        "description": step.description,
                        "estimated_time": step.estimated_time,
                        "required_documents": step.required_documents,
                        "online_available": step.online_available,
                        "offline_location": step.offline_location,
                        "detailed_instructions": self._get_step_instructions(service_info.service_id, step_number),
                        "common_issues": self._get_step_common_issues(service_info.service_id, step_number)
                    },
                    "next_step": service_info.application_steps[step_number].title if step_number < len(service_info.application_steps) else "Application Complete",
                    "progress": f"{step_number}/{len(service_info.application_steps)}"
                }
            else:
                # Get complete application guidance
                guidance_data = {
                    "service_name": service_info.service_name,
                    "total_steps": len(service_info.application_steps),
                    "estimated_total_time": self._calculate_total_time(service_info.application_steps),
                    "application_steps": [
                        {
                            "step_number": step.step_number,
                            "title": step.title,
                            "description": step.description,
                            "estimated_time": step.estimated_time,
                            "online_available": step.online_available,
                            "key_points": self._get_step_key_points(service_info.service_id, step.step_number)
                        }
                        for step in service_info.application_steps
                    ],
                    "preparation_checklist": self._create_preparation_checklist(service_info),
                    "timeline": self._create_application_timeline(service_info)
                }
            
            guidance_data["last_updated"] = datetime.now().isoformat()
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=guidance_data,
                error_message=None,
                response_time=0.7
            )
            
        except Exception as e:
            self.logger.error(f"Error getting application guidance: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get application guidance: {str(e)}",
                response_time=0.5
            )
    
    async def search_services_by_category(
        self,
        category: ServiceCategory,
        limit: int = 10
    ) -> ServiceResult:
        """
        Search government services by category.
        
        Args:
            category: Service category to search
            limit: Maximum number of results
            
        Returns:
            Service result with matching services
        """
        try:
            self.logger.info(f"Searching services in category: {category.value}")
            
            matching_services = [
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "department": service.department,
                    "description": service.description,
                    "processing_time": service.processing_time,
                    "fees": service.fees,
                    "online_available": any(step.online_available for step in service.application_steps)
                }
                for service in self._government_services.values()
                if service.category == category
            ][:limit]
            
            if not matching_services:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"No services found in category '{category.value}'.",
                    response_time=0.3
                )
            
            search_data = {
                "category": category.value,
                "total_services": len(matching_services),
                "services": matching_services,
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=search_data,
                error_message=None,
                response_time=0.5
            )
            
        except Exception as e:
            self.logger.error(f"Error searching services by category: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to search services: {str(e)}",
                response_time=0.5
            )
    
    async def get_state_portal_info(self, state_name: str) -> ServiceResult:
        """
        Get state-specific government portal information.
        
        Args:
            state_name: Name of the state
            
        Returns:
            Service result with state portal information
        """
        try:
            self.logger.info(f"Getting state portal info for: {state_name}")
            
            state_key = state_name.lower().replace(" ", "_")
            portal_info = self._state_portals.get(state_key)
            
            if not portal_info:
                return ServiceResult(
                    service_type=ServiceType.GOVERNMENT_SERVICE,
                    success=False,
                    data={},
                    error_message=f"State portal information not available for '{state_name}'.",
                    response_time=0.3
                )
            
            portal_data = {
                "state": state_name.title(),
                "portal_name": portal_info["name"],
                "portal_url": portal_info["url"],
                "available_services": portal_info["services"],
                "access_instructions": [
                    "Visit the state portal website",
                    "Register with valid mobile number and email",
                    "Complete profile with required documents",
                    "Browse available services",
                    "Apply online and track application status"
                ],
                "common_services": [
                    "Income Certificate",
                    "Caste Certificate", 
                    "Domicile Certificate",
                    "Birth Certificate",
                    "Death Certificate",
                    "Marriage Certificate"
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=True,
                data=portal_data,
                error_message=None,
                response_time=0.4
            )
            
        except Exception as e:
            self.logger.error(f"Error getting state portal info: {e}")
            return ServiceResult(
                service_type=ServiceType.GOVERNMENT_SERVICE,
                success=False,
                data={},
                error_message=f"Failed to get state portal information: {str(e)}",
                response_time=0.5
            )
    
    # Helper methods
    
    async def _find_service(self, service_name: str) -> Optional[GovernmentServiceInfo]:
        """Find service by name or partial match."""
        service_name_lower = service_name.lower().replace(" ", "_")
        
        # Direct lookup
        if service_name_lower in self._government_services:
            return self._government_services[service_name_lower]
        
        # Partial match
        for service_id, service in self._government_services.items():
            if (service_name.lower() in service.service_name.lower() or
                service_name.lower() in service_id.lower()):
                return service
        
        return None
    
    def _get_document_tips(self, document_name: str) -> List[str]:
        """Get tips for specific document preparation."""
        tips_map = {
            "aadhaar card": [
                "Ensure name matches exactly across all documents",
                "Update address if recently moved",
                "Keep both original and photocopy ready"
            ],
            "proof of identity": [
                "Use government-issued photo ID",
                "Ensure document is not expired",
                "Name should match application form exactly"
            ],
            "proof of address": [
                "Document should be recent (within 3 months)",
                "Address should match current residence",
                "Utility bills, bank statements are commonly accepted"
            ],
            "birth certificate": [
                "Get from municipal corporation where birth was registered",
                "Ensure all details are correct and legible",
                "May need translation if in regional language"
            ]
        }
        
        return tips_map.get(document_name.lower(), ["Ensure document is original and legible", "Keep photocopies ready"])
    
    def _create_document_checklist(self, documents: List[DocumentRequirement]) -> List[Dict[str, Any]]:
        """Create a document preparation checklist."""
        checklist = []
        for doc in documents:
            checklist.append({
                "document": doc.document_name,
                "status": "pending",
                "priority": "high" if doc.is_mandatory else "medium",
                "action": f"Obtain {doc.document_type.lower()} from {doc.where_to_obtain or 'relevant authority'}"
            })
        return checklist
    
    def _get_common_document_mistakes(self, service_id: str) -> List[str]:
        """Get common mistakes for document preparation."""
        common_mistakes = {
            "aadhaar_card": [
                "Name mismatch between documents",
                "Blurred or unclear photocopies",
                "Using expired address proof"
            ],
            "pan_card": [
                "Incorrect date of birth format",
                "Missing signature on application form",
                "Using non-acceptable identity proof"
            ],
            "passport": [
                "Incomplete online application",
                "Missing ECR/ECNR page requirements",
                "Incorrect fee payment"
            ]
        }
        
        return common_mistakes.get(service_id, ["Incomplete documentation", "Unclear photocopies", "Missing signatures"])
    
    def _get_step_instructions(self, service_id: str, step_number: int) -> List[str]:
        """Get detailed instructions for a specific step."""
        # This would contain detailed step-by-step instructions
        # For brevity, returning generic instructions
        return [
            f"Complete step {step_number} as described",
            "Ensure all required documents are ready",
            "Follow the official guidelines carefully",
            "Keep receipt/acknowledgment for future reference"
        ]
    
    def _get_step_common_issues(self, service_id: str, step_number: int) -> List[str]:
        """Get common issues for a specific step."""
        return [
            "Long waiting times during peak hours",
            "Document verification delays",
            "Technical issues with online portals",
            "Incomplete information provided"
        ]
    
    def _get_step_key_points(self, service_id: str, step_number: int) -> List[str]:
        """Get key points for a specific step."""
        return [
            "Carry all original documents",
            "Arrive early to avoid crowds",
            "Keep multiple photocopies",
            "Note down reference numbers"
        ]
    
    def _calculate_total_time(self, steps: List[ServiceStep]) -> str:
        """Calculate total estimated time for all steps."""
        # Simple calculation - in real implementation would parse time strings
        return f"{len(steps) * 2}-{len(steps) * 4} hours"
    
    def _create_preparation_checklist(self, service_info: GovernmentServiceInfo) -> List[str]:
        """Create preparation checklist for the service."""
        return [
            "Gather all required documents",
            "Make photocopies of all documents",
            "Fill application form completely",
            "Arrange for fees payment",
            "Plan visit during working hours",
            "Keep contact numbers handy"
        ]
    
    def _create_application_timeline(self, service_info: GovernmentServiceInfo) -> Dict[str, str]:
        """Create application timeline."""
        return {
            "preparation": "1-2 days",
            "application_submission": "1 day",
            "processing": service_info.processing_time,
            "total_duration": f"Approximately {service_info.processing_time}"
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
        }