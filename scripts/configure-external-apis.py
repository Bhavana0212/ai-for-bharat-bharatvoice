<<<<<<< HEAD
#!/usr/bin/env python3
"""
External API Configuration Script for BharatVoice Assistant

This script helps configure and validate external API integrations for production deployment.
It provides interactive setup, validation, and testing of API connections.
"""

import asyncio
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bharatvoice.config.settings import get_settings
from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
from bharatvoice.services.external_integrations.weather_service import WeatherService
from bharatvoice.services.external_integrations.digital_india_service import DigitalIndiaService
from bharatvoice.services.payment.upi_service import UPIService
from bharatvoice.services.platform_integrations.platform_manager import PlatformManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalAPIConfigurator:
    """
    Configuration manager for external API integrations.
    """
    
    def __init__(self):
        self.config_dir = Path(__file__).parent.parent / "config" / "production"
        self.env_file = self.config_dir / ".env.production"
        self.api_config_file = self.config_dir / "external_apis.yaml"
        self.validation_results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file."""
        try:
            with open(self.api_config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.api_config_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            return {}
    
    def load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env.production file."""
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    
    def interactive_setup(self):
        """Interactive setup for API configurations."""
        print("🚀 BharatVoice Assistant - External API Configuration Setup")
        print("=" * 60)
        
        config = self.load_config()
        env_vars = self.load_env_vars()
        
        # Setup each service category
        self._setup_indian_railways(config, env_vars)
        self._setup_weather_services(config, env_vars)
        self._setup_digital_india(config, env_vars)
        self._setup_upi_payment(config, env_vars)
        self._setup_platform_integrations(config, env_vars)
        self._setup_entertainment_apis(config, env_vars)
        
        # Save updated environment variables
        self._save_env_vars(env_vars)
        
        print("\n✅ Configuration setup completed!")
        print(f"Environment variables saved to: {self.env_file}")
        
    def _setup_indian_railways(self, config: Dict, env_vars: Dict):
        """Setup Indian Railways API configuration."""
        print("\n🚂 Indian Railways API Configuration")
        print("-" * 40)
        
        # Primary API key
        current_key = env_vars.get('INDIAN_RAILWAYS_API_KEY', '')
        if not current_key:
            print("Indian Railways API is required for train schedules and booking information.")
            print("Get your API key from: https://railwayapi.com/")
            
            api_key = input("Enter your Indian Railways API key: ").strip()
            if api_key:
                env_vars['INDIAN_RAILWAYS_API_KEY'] = api_key
                print("✅ Indian Railways API key configured")
            else:
                print("⚠️  Skipping Indian Railways API configuration")
        else:
            print(f"✅ Indian Railways API key already configured: {current_key[:8]}...")
        
        # Fallback providers
        fallback_keys = [
            ('IRCTC_CONNECT_API_KEY', 'IRCTC Connect API'),
            ('CONFIRMTKT_API_KEY', 'ConfirmTkt API')
        ]
        
        for env_key, service_name in fallback_keys:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name} as fallback? (y/n): ").lower() == 'y'
                if setup:
                    api_key = input(f"Enter {service_name} key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_weather_services(self, config: Dict, env_vars: Dict):
        """Setup weather services configuration."""
        print("\n🌤️  Weather Services Configuration")
        print("-" * 40)
        
        # OpenWeatherMap (Primary)
        if not env_vars.get('OPENWEATHERMAP_API_KEY'):
            print("Weather service is required for local weather and monsoon information.")
            print("Get your free API key from: https://openweathermap.org/api")
            
            api_key = input("Enter your OpenWeatherMap API key: ").strip()
            if api_key:
                env_vars['OPENWEATHERMAP_API_KEY'] = api_key
                print("✅ OpenWeatherMap API configured")
        else:
            print("✅ OpenWeatherMap API already configured")
        
        # Additional weather services
        additional_services = [
            ('IMD_API_KEY', 'Indian Meteorological Department', 'https://mausam.imd.gov.in/'),
            ('ACCUWEATHER_API_KEY', 'AccuWeather', 'https://developer.accuweather.com/'),
            ('WEATHER_UNDERGROUND_API_KEY', 'Weather Underground', 'https://www.wunderground.com/weather/api/')
        ]
        
        for env_key, service_name, url in additional_services:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
                if setup:
                    print(f"Get API key from: {url}")
                    api_key = input(f"Enter {service_name} API key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_digital_india(self, config: Dict, env_vars: Dict):
        """Setup Digital India platform configuration."""
        print("\n🏛️  Digital India Platform Configuration")
        print("-" * 40)
        
        print("Digital India integration requires government API access.")
        print("Contact Digital India team for API credentials.")
        
        digital_india_keys = [
            ('DIGITAL_INDIA_API_KEY', 'Digital India API Key'),
            ('DIGITAL_INDIA_CLIENT_ID', 'Digital India Client ID'),
            ('DIGITAL_INDIA_CLIENT_SECRET', 'Digital India Client Secret')
        ]
        
        for env_key, description in digital_india_keys:
            if not env_vars.get(env_key):
                setup = input(f"Setup {description}? (y/n): ").lower() == 'y'
                if setup:
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
        
        # Service-specific APIs
        service_apis = [
            ('AADHAAR_VERIFICATION_API_KEY', 'Aadhaar Verification'),
            ('PAN_VERIFICATION_API_KEY', 'PAN Verification'),
            ('PASSPORT_API_KEY', 'Passport Services'),
            ('DRIVING_LICENSE_API_KEY', 'Driving License Verification')
        ]
        
        for env_key, service_name in service_apis:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name} API? (y/n): ").lower() == 'y'
                if setup:
                    api_key = input(f"Enter {service_name} API key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_upi_payment(self, config: Dict, env_vars: Dict):
        """Setup UPI payment gateway configuration."""
        print("\n💳 UPI Payment Gateway Configuration")
        print("-" * 40)
        
        print("UPI payment integration requires partnership with payment gateways.")
        
        # Razorpay (Primary)
        razorpay_keys = [
            ('RAZORPAY_KEY_ID', 'Razorpay Key ID'),
            ('RAZORPAY_KEY_SECRET', 'Razorpay Key Secret'),
            ('RAZORPAY_WEBHOOK_SECRET', 'Razorpay Webhook Secret')
        ]
        
        setup_razorpay = input("Setup Razorpay payment gateway? (y/n): ").lower() == 'y'
        if setup_razorpay:
            print("Get credentials from: https://dashboard.razorpay.com/")
            for env_key, description in razorpay_keys:
                if not env_vars.get(env_key):
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
        
        # PayU (Fallback)
        setup_payu = input("Setup PayU as fallback gateway? (y/n): ").lower() == 'y'
        if setup_payu:
            payu_keys = [
                ('PAYU_MERCHANT_KEY', 'PayU Merchant Key'),
                ('PAYU_MERCHANT_SALT', 'PayU Merchant Salt')
            ]
            
            for env_key, description in payu_keys:
                if not env_vars.get(env_key):
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
    
    def _setup_platform_integrations(self, config: Dict, env_vars: Dict):
        """Setup platform integration configuration."""
        print("\n🛵 Platform Integration Configuration")
        print("-" * 40)
        
        # Food delivery platforms
        food_platforms = [
            ('SWIGGY_PARTNER_ID', 'SWIGGY_API_KEY', 'Swiggy', 'https://partner.swiggy.com/'),
            ('ZOMATO_PARTNER_ID', 'ZOMATO_API_KEY', 'Zomato', 'https://developers.zomato.com/'),
            ('UBER_EATS_CLIENT_ID', 'UBER_EATS_CLIENT_SECRET', 'Uber Eats', 'https://developer.uber.com/')
        ]
        
        print("Food Delivery Platform Integration:")
        for id_key, secret_key, platform_name, url in food_platforms:
            setup = input(f"Setup {platform_name} integration? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get credentials from: {url}")
                if not env_vars.get(id_key):
                    partner_id = input(f"Enter {platform_name} Partner/Client ID: ").strip()
                    if partner_id:
                        env_vars[id_key] = partner_id
                
                if not env_vars.get(secret_key):
                    api_key = input(f"Enter {platform_name} API Key/Secret: ").strip()
                    if api_key:
                        env_vars[secret_key] = api_key
                        print(f"✅ {platform_name} configured")
        
        # Ride sharing platforms
        ride_platforms = [
            ('OLA_CLIENT_ID', 'OLA_CLIENT_SECRET', 'Ola', 'https://developers.olacabs.com/'),
            ('UBER_CLIENT_ID', 'UBER_CLIENT_SECRET', 'Uber', 'https://developer.uber.com/'),
            ('RAPIDO_API_KEY', None, 'Rapido', 'https://rapido.bike/partner/')
        ]
        
        print("\nRide Sharing Platform Integration:")
        for id_key, secret_key, platform_name, url in ride_platforms:
            setup = input(f"Setup {platform_name} integration? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get credentials from: {url}")
                if not env_vars.get(id_key):
                    client_id = input(f"Enter {platform_name} Client ID/API Key: ").strip()
                    if client_id:
                        env_vars[id_key] = client_id
                
                if secret_key and not env_vars.get(secret_key):
                    client_secret = input(f"Enter {platform_name} Client Secret: ").strip()
                    if client_secret:
                        env_vars[secret_key] = client_secret
                        print(f"✅ {platform_name} configured")
    
    def _setup_entertainment_apis(self, config: Dict, env_vars: Dict):
        """Setup entertainment and news API configuration."""
        print("\n🏏 Entertainment & News API Configuration")
        print("-" * 40)
        
        # Cricket APIs
        cricket_apis = [
            ('CRICAPI_API_KEY', 'CricAPI', 'https://www.cricapi.com/'),
            ('ESPN_CRICINFO_API_KEY', 'ESPN Cricinfo', 'https://developer.espn.com/')
        ]
        
        for env_key, service_name, url in cricket_apis:
            setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get API key from: {url}")
                api_key = input(f"Enter {service_name} API key: ").strip()
                if api_key:
                    env_vars[env_key] = api_key
                    print(f"✅ {service_name} configured")
        
        # News APIs
        news_apis = [
            ('NEWS_API_KEY', 'NewsAPI', 'https://newsapi.org/'),
            ('TMDB_API_KEY', 'The Movie Database', 'https://www.themoviedb.org/settings/api')
        ]
        
        for env_key, service_name, url in news_apis:
            setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get API key from: {url}")
                api_key = input(f"Enter {service_name} API key: ").strip()
                if api_key:
                    env_vars[env_key] = api_key
                    print(f"✅ {service_name} configured")
    
    def _save_env_vars(self, env_vars: Dict[str, str]):
        """Save environment variables to .env.production file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.env_file, 'w') as f:
            f.write(f"# BharatVoice Assistant Production Environment Variables\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            
            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")
    
    async def validate_api_connections(self):
        """Validate all configured API connections."""
        print("\n🔍 Validating API Connections")
        print("-" * 40)
        
        env_vars = self.load_env_vars()
        
        # Set environment variables for testing
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Test each service
        await self._test_indian_railways_api()
        await self._test_weather_api()
        await self._test_upi_gateway()
        await self._test_platform_apis()
        
        # Print validation summary
        self._print_validation_summary()
    
    async def _test_indian_railways_api(self):
        """Test Indian Railways API connection."""
        try:
            api_key = os.getenv('INDIAN_RAILWAYS_API_KEY')
            if not api_key:
                self.validation_results['indian_railways'] = {
                    'status': 'skipped',
                    'message': 'API key not configured'
                }
                return
            
            service = IndianRailwaysService(api_key=api_key)
            async with service:
                # Test with a known train number
                result = await service.get_train_schedule("12002")  # Shatabdi Express
                
                if result.success:
                    self.validation_results['indian_railways'] = {
                        'status': 'success',
                        'message': 'API connection successful',
                        'response_time': result.response_time
                    }
                else:
                    self.validation_results['indian_railways'] = {
                        'status': 'error',
                        'message': result.error_message or 'API test failed'
                    }
                    
        except Exception as e:
            self.validation_results['indian_railways'] = {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    async def _test_weather_api(self):
        """Test weather API connection."""
        try:
            api_key = os.getenv('OPENWEATHERMAP_API_KEY')
            if not api_key:
                self.validation_results['weather'] = {
                    'status': 'skipped',
                    'message': 'API key not configured'
                }
                return
            
            service = WeatherService(weather_api_key=api_key)
            async with service:
                result = await service.get_weather_info("Delhi")
                
                if result.success:
                    self.validation_results['weather'] = {
                        'status': 'success',
                        'message': 'API connection successful',
                        'response_time': result.response_time
                    }
                else:
                    self.validation_results['weather'] = {
                        'status': 'error',
                        'message': result.error_message or 'API test failed'
                    }
                    
        except Exception as e:
            self.validation_results['weather'] = {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    async def _test_upi_gateway(self):
        """Test UPI payment gateway connection."""
        try:
            razorpay_key = os.getenv('RAZORPAY_KEY_ID')
            razorpay_secret = os.getenv('RAZORPAY_KEY_SECRET')
            
            if not razorpay_key or not razorpay_secret:
                self.validation_results['upi_payment'] = {
                    'status': 'skipped',
                    'message': 'Payment gateway credentials not configured'
                }
                return
            
            # Test basic connectivity (mock for now)
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(razorpay_key, razorpay_secret)
                async with session.get(
                    'https://api.razorpay.com/v1/payments',
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.validation_results['upi_payment'] = {
                            'status': 'success',
                            'message': 'Payment gateway connection successful'
                        }
                    else:
                        self.validation_results['upi_payment'] = {
                            'status': 'error',
                            'message': f'Payment gateway returned status {response.status}'
                        }
                        
        except Exception as e:
            self.validation_results['upi_payment'] = {
                'status': 'error',
                'message': f'Payment gateway connection failed: {str(e)}'
            }
    
    async def _test_platform_apis(self):
        """Test platform integration APIs."""
        # Test Swiggy API
        swiggy_key = os.getenv('SWIGGY_API_KEY')
        if swiggy_key:
            try:
                # Mock test for Swiggy API
                self.validation_results['swiggy'] = {
                    'status': 'success',
                    'message': 'Swiggy API key configured (test connection requires partner approval)'
                }
            except Exception as e:
                self.validation_results['swiggy'] = {
                    'status': 'error',
                    'message': f'Swiggy API test failed: {str(e)}'
                }
        else:
            self.validation_results['swiggy'] = {
                'status': 'skipped',
                'message': 'Swiggy API key not configured'
            }
        
        # Test Ola API
        ola_client_id = os.getenv('OLA_CLIENT_ID')
        if ola_client_id:
            self.validation_results['ola'] = {
                'status': 'success',
                'message': 'Ola API credentials configured (test connection requires partner approval)'
            }
        else:
            self.validation_results['ola'] = {
                'status': 'skipped',
                'message': 'Ola API credentials not configured'
            }
    
    def _print_validation_summary(self):
        """Print validation results summary."""
        print("\n📊 API Validation Summary")
        print("=" * 50)
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for service, result in self.validation_results.items():
            status = result['status']
            message = result['message']
            
            if status == 'success':
                print(f"✅ {service.replace('_', ' ').title()}: {message}")
                success_count += 1
            elif status == 'error':
                print(f"❌ {service.replace('_', ' ').title()}: {message}")
                error_count += 1
            else:
                print(f"⏭️  {service.replace('_', ' ').title()}: {message}")
                skipped_count += 1
        
        print(f"\nSummary: {success_count} successful, {error_count} errors, {skipped_count} skipped")
        
        if error_count > 0:
            print("\n⚠️  Some API connections failed. Please check your credentials and try again.")
        elif success_count > 0:
            print("\n🎉 API validation completed successfully!")
    
    def generate_deployment_checklist(self):
        """Generate deployment checklist for external APIs."""
        checklist_file = self.config_dir / "deployment_checklist.md"
        
        checklist_content = """# External API Deployment Checklist

## Pre-deployment Requirements

### 1. Indian Railways Integration
- [ ] Obtain API key from RailwayAPI.com
- [ ] Test train schedule queries
- [ ] Configure fallback providers (optional)
- [ ] Set up rate limiting and caching

### 2. Weather Services
- [ ] Get OpenWeatherMap API key (free tier available)
- [ ] Configure IMD API access (if available)
- [ ] Test weather data retrieval for major Indian cities
- [ ] Set up monsoon-specific data sources

### 3. Digital India Platform
- [ ] Apply for Digital India API access
- [ ] Obtain government service API credentials
- [ ] Ensure compliance with data localization requirements
- [ ] Set up audit logging for government service access

### 4. UPI Payment Gateway
- [ ] Partner with Razorpay/PayU for UPI integration
- [ ] Complete KYC and compliance requirements
- [ ] Set up webhook endpoints for payment notifications
- [ ] Configure payment limits and security measures
- [ ] Test payment flows in sandbox environment

### 5. Platform Integrations
- [ ] Apply for Swiggy partner program
- [ ] Get Zomato API access for restaurant data
- [ ] Partner with Ola/Uber for ride booking
- [ ] Set up Urban Company service booking
- [ ] Configure webhook endpoints for booking updates

### 6. Entertainment APIs
- [ ] Get CricAPI key for cricket scores
- [ ] Configure NewsAPI for Bollywood news
- [ ] Set up TMDB for movie information

## Security and Compliance

### Data Protection
- [ ] Implement API key encryption
- [ ] Set up secure key management (HashiCorp Vault)
- [ ] Configure audit logging for all API calls
- [ ] Ensure GDPR and Indian data protection compliance

### Rate Limiting and Monitoring
- [ ] Configure rate limiting for all external APIs
- [ ] Set up monitoring and alerting for API failures
- [ ] Implement circuit breakers for resilience
- [ ] Configure retry policies with exponential backoff

### SSL/TLS Configuration
- [ ] Ensure all API calls use HTTPS
- [ ] Validate SSL certificates
- [ ] Configure certificate pinning for critical APIs

## Production Deployment

### Infrastructure
- [ ] Deploy to production Kubernetes cluster
- [ ] Configure horizontal pod autoscaling
- [ ] Set up load balancing for API services
- [ ] Configure persistent storage for caching

### Monitoring and Alerting
- [ ] Set up Prometheus metrics collection
- [ ] Configure Grafana dashboards for API monitoring
- [ ] Set up alerts for API failures and rate limit breaches
- [ ] Configure log aggregation with ELK stack

### Backup and Disaster Recovery
- [ ] Set up automated database backups
- [ ] Configure API key backup and rotation
- [ ] Test disaster recovery procedures
- [ ] Document rollback procedures

## Testing and Validation

### API Testing
- [ ] Run comprehensive API validation tests
- [ ] Test error handling and fallback mechanisms
- [ ] Validate rate limiting and timeout configurations
- [ ] Test webhook endpoints and callbacks

### Load Testing
- [ ] Conduct load testing with realistic traffic patterns
- [ ] Test API performance under high load
- [ ] Validate auto-scaling behavior
- [ ] Test circuit breaker functionality

### Security Testing
- [ ] Conduct security audit of API integrations
- [ ] Test API key security and rotation
- [ ] Validate input sanitization and validation
- [ ] Test for common security vulnerabilities

## Go-Live Checklist

### Final Preparations
- [ ] All API keys configured and validated
- [ ] Monitoring and alerting systems active
- [ ] Backup and disaster recovery tested
- [ ] Security audit completed
- [ ] Load testing passed
- [ ] Documentation updated

### Launch
- [ ] Deploy to production environment
- [ ] Monitor system health and API performance
- [ ] Validate all integrations working correctly
- [ ] Monitor error rates and response times
- [ ] Be prepared for immediate rollback if needed

## Post-Launch

### Ongoing Maintenance
- [ ] Regular API key rotation
- [ ] Monitor API usage and costs
- [ ] Update API integrations as needed
- [ ] Regular security audits
- [ ] Performance optimization based on usage patterns

---

**Note**: This checklist should be customized based on your specific deployment requirements and available API partnerships.
"""
        
        with open(checklist_file, 'w') as f:
            f.write(checklist_content)
        
        print(f"\n📋 Deployment checklist generated: {checklist_file}")


async def main():
    """Main function for API configuration script."""
    configurator = ExternalAPIConfigurator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            configurator.interactive_setup()
        elif command == "validate":
            await configurator.validate_api_connections()
        elif command == "checklist":
            configurator.generate_deployment_checklist()
        else:
            print("Usage: python configure-external-apis.py [setup|validate|checklist]")
    else:
        # Interactive mode
        print("BharatVoice Assistant - External API Configuration")
        print("1. Interactive Setup")
        print("2. Validate API Connections")
        print("3. Generate Deployment Checklist")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            configurator.interactive_setup()
        elif choice == "2":
            await configurator.validate_api_connections()
        elif choice == "3":
            configurator.generate_deployment_checklist()
        else:
            print("Invalid choice")


if __name__ == "__main__":
=======
#!/usr/bin/env python3
"""
External API Configuration Script for BharatVoice Assistant

This script helps configure and validate external API integrations for production deployment.
It provides interactive setup, validation, and testing of API connections.
"""

import asyncio
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bharatvoice.config.settings import get_settings
from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
from bharatvoice.services.external_integrations.weather_service import WeatherService
from bharatvoice.services.external_integrations.digital_india_service import DigitalIndiaService
from bharatvoice.services.payment.upi_service import UPIService
from bharatvoice.services.platform_integrations.platform_manager import PlatformManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExternalAPIConfigurator:
    """
    Configuration manager for external API integrations.
    """
    
    def __init__(self):
        self.config_dir = Path(__file__).parent.parent / "config" / "production"
        self.env_file = self.config_dir / ".env.production"
        self.api_config_file = self.config_dir / "external_apis.yaml"
        self.validation_results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load API configuration from YAML file."""
        try:
            with open(self.api_config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.api_config_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            return {}
    
    def load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env.production file."""
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    
    def interactive_setup(self):
        """Interactive setup for API configurations."""
        print("🚀 BharatVoice Assistant - External API Configuration Setup")
        print("=" * 60)
        
        config = self.load_config()
        env_vars = self.load_env_vars()
        
        # Setup each service category
        self._setup_indian_railways(config, env_vars)
        self._setup_weather_services(config, env_vars)
        self._setup_digital_india(config, env_vars)
        self._setup_upi_payment(config, env_vars)
        self._setup_platform_integrations(config, env_vars)
        self._setup_entertainment_apis(config, env_vars)
        
        # Save updated environment variables
        self._save_env_vars(env_vars)
        
        print("\n✅ Configuration setup completed!")
        print(f"Environment variables saved to: {self.env_file}")
        
    def _setup_indian_railways(self, config: Dict, env_vars: Dict):
        """Setup Indian Railways API configuration."""
        print("\n🚂 Indian Railways API Configuration")
        print("-" * 40)
        
        # Primary API key
        current_key = env_vars.get('INDIAN_RAILWAYS_API_KEY', '')
        if not current_key:
            print("Indian Railways API is required for train schedules and booking information.")
            print("Get your API key from: https://railwayapi.com/")
            
            api_key = input("Enter your Indian Railways API key: ").strip()
            if api_key:
                env_vars['INDIAN_RAILWAYS_API_KEY'] = api_key
                print("✅ Indian Railways API key configured")
            else:
                print("⚠️  Skipping Indian Railways API configuration")
        else:
            print(f"✅ Indian Railways API key already configured: {current_key[:8]}...")
        
        # Fallback providers
        fallback_keys = [
            ('IRCTC_CONNECT_API_KEY', 'IRCTC Connect API'),
            ('CONFIRMTKT_API_KEY', 'ConfirmTkt API')
        ]
        
        for env_key, service_name in fallback_keys:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name} as fallback? (y/n): ").lower() == 'y'
                if setup:
                    api_key = input(f"Enter {service_name} key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_weather_services(self, config: Dict, env_vars: Dict):
        """Setup weather services configuration."""
        print("\n🌤️  Weather Services Configuration")
        print("-" * 40)
        
        # OpenWeatherMap (Primary)
        if not env_vars.get('OPENWEATHERMAP_API_KEY'):
            print("Weather service is required for local weather and monsoon information.")
            print("Get your free API key from: https://openweathermap.org/api")
            
            api_key = input("Enter your OpenWeatherMap API key: ").strip()
            if api_key:
                env_vars['OPENWEATHERMAP_API_KEY'] = api_key
                print("✅ OpenWeatherMap API configured")
        else:
            print("✅ OpenWeatherMap API already configured")
        
        # Additional weather services
        additional_services = [
            ('IMD_API_KEY', 'Indian Meteorological Department', 'https://mausam.imd.gov.in/'),
            ('ACCUWEATHER_API_KEY', 'AccuWeather', 'https://developer.accuweather.com/'),
            ('WEATHER_UNDERGROUND_API_KEY', 'Weather Underground', 'https://www.wunderground.com/weather/api/')
        ]
        
        for env_key, service_name, url in additional_services:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
                if setup:
                    print(f"Get API key from: {url}")
                    api_key = input(f"Enter {service_name} API key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_digital_india(self, config: Dict, env_vars: Dict):
        """Setup Digital India platform configuration."""
        print("\n🏛️  Digital India Platform Configuration")
        print("-" * 40)
        
        print("Digital India integration requires government API access.")
        print("Contact Digital India team for API credentials.")
        
        digital_india_keys = [
            ('DIGITAL_INDIA_API_KEY', 'Digital India API Key'),
            ('DIGITAL_INDIA_CLIENT_ID', 'Digital India Client ID'),
            ('DIGITAL_INDIA_CLIENT_SECRET', 'Digital India Client Secret')
        ]
        
        for env_key, description in digital_india_keys:
            if not env_vars.get(env_key):
                setup = input(f"Setup {description}? (y/n): ").lower() == 'y'
                if setup:
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
        
        # Service-specific APIs
        service_apis = [
            ('AADHAAR_VERIFICATION_API_KEY', 'Aadhaar Verification'),
            ('PAN_VERIFICATION_API_KEY', 'PAN Verification'),
            ('PASSPORT_API_KEY', 'Passport Services'),
            ('DRIVING_LICENSE_API_KEY', 'Driving License Verification')
        ]
        
        for env_key, service_name in service_apis:
            if not env_vars.get(env_key):
                setup = input(f"Setup {service_name} API? (y/n): ").lower() == 'y'
                if setup:
                    api_key = input(f"Enter {service_name} API key: ").strip()
                    if api_key:
                        env_vars[env_key] = api_key
                        print(f"✅ {service_name} configured")
    
    def _setup_upi_payment(self, config: Dict, env_vars: Dict):
        """Setup UPI payment gateway configuration."""
        print("\n💳 UPI Payment Gateway Configuration")
        print("-" * 40)
        
        print("UPI payment integration requires partnership with payment gateways.")
        
        # Razorpay (Primary)
        razorpay_keys = [
            ('RAZORPAY_KEY_ID', 'Razorpay Key ID'),
            ('RAZORPAY_KEY_SECRET', 'Razorpay Key Secret'),
            ('RAZORPAY_WEBHOOK_SECRET', 'Razorpay Webhook Secret')
        ]
        
        setup_razorpay = input("Setup Razorpay payment gateway? (y/n): ").lower() == 'y'
        if setup_razorpay:
            print("Get credentials from: https://dashboard.razorpay.com/")
            for env_key, description in razorpay_keys:
                if not env_vars.get(env_key):
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
        
        # PayU (Fallback)
        setup_payu = input("Setup PayU as fallback gateway? (y/n): ").lower() == 'y'
        if setup_payu:
            payu_keys = [
                ('PAYU_MERCHANT_KEY', 'PayU Merchant Key'),
                ('PAYU_MERCHANT_SALT', 'PayU Merchant Salt')
            ]
            
            for env_key, description in payu_keys:
                if not env_vars.get(env_key):
                    value = input(f"Enter {description}: ").strip()
                    if value:
                        env_vars[env_key] = value
                        print(f"✅ {description} configured")
    
    def _setup_platform_integrations(self, config: Dict, env_vars: Dict):
        """Setup platform integration configuration."""
        print("\n🛵 Platform Integration Configuration")
        print("-" * 40)
        
        # Food delivery platforms
        food_platforms = [
            ('SWIGGY_PARTNER_ID', 'SWIGGY_API_KEY', 'Swiggy', 'https://partner.swiggy.com/'),
            ('ZOMATO_PARTNER_ID', 'ZOMATO_API_KEY', 'Zomato', 'https://developers.zomato.com/'),
            ('UBER_EATS_CLIENT_ID', 'UBER_EATS_CLIENT_SECRET', 'Uber Eats', 'https://developer.uber.com/')
        ]
        
        print("Food Delivery Platform Integration:")
        for id_key, secret_key, platform_name, url in food_platforms:
            setup = input(f"Setup {platform_name} integration? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get credentials from: {url}")
                if not env_vars.get(id_key):
                    partner_id = input(f"Enter {platform_name} Partner/Client ID: ").strip()
                    if partner_id:
                        env_vars[id_key] = partner_id
                
                if not env_vars.get(secret_key):
                    api_key = input(f"Enter {platform_name} API Key/Secret: ").strip()
                    if api_key:
                        env_vars[secret_key] = api_key
                        print(f"✅ {platform_name} configured")
        
        # Ride sharing platforms
        ride_platforms = [
            ('OLA_CLIENT_ID', 'OLA_CLIENT_SECRET', 'Ola', 'https://developers.olacabs.com/'),
            ('UBER_CLIENT_ID', 'UBER_CLIENT_SECRET', 'Uber', 'https://developer.uber.com/'),
            ('RAPIDO_API_KEY', None, 'Rapido', 'https://rapido.bike/partner/')
        ]
        
        print("\nRide Sharing Platform Integration:")
        for id_key, secret_key, platform_name, url in ride_platforms:
            setup = input(f"Setup {platform_name} integration? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get credentials from: {url}")
                if not env_vars.get(id_key):
                    client_id = input(f"Enter {platform_name} Client ID/API Key: ").strip()
                    if client_id:
                        env_vars[id_key] = client_id
                
                if secret_key and not env_vars.get(secret_key):
                    client_secret = input(f"Enter {platform_name} Client Secret: ").strip()
                    if client_secret:
                        env_vars[secret_key] = client_secret
                        print(f"✅ {platform_name} configured")
    
    def _setup_entertainment_apis(self, config: Dict, env_vars: Dict):
        """Setup entertainment and news API configuration."""
        print("\n🏏 Entertainment & News API Configuration")
        print("-" * 40)
        
        # Cricket APIs
        cricket_apis = [
            ('CRICAPI_API_KEY', 'CricAPI', 'https://www.cricapi.com/'),
            ('ESPN_CRICINFO_API_KEY', 'ESPN Cricinfo', 'https://developer.espn.com/')
        ]
        
        for env_key, service_name, url in cricket_apis:
            setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get API key from: {url}")
                api_key = input(f"Enter {service_name} API key: ").strip()
                if api_key:
                    env_vars[env_key] = api_key
                    print(f"✅ {service_name} configured")
        
        # News APIs
        news_apis = [
            ('NEWS_API_KEY', 'NewsAPI', 'https://newsapi.org/'),
            ('TMDB_API_KEY', 'The Movie Database', 'https://www.themoviedb.org/settings/api')
        ]
        
        for env_key, service_name, url in news_apis:
            setup = input(f"Setup {service_name}? (y/n): ").lower() == 'y'
            if setup:
                print(f"Get API key from: {url}")
                api_key = input(f"Enter {service_name} API key: ").strip()
                if api_key:
                    env_vars[env_key] = api_key
                    print(f"✅ {service_name} configured")
    
    def _save_env_vars(self, env_vars: Dict[str, str]):
        """Save environment variables to .env.production file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.env_file, 'w') as f:
            f.write(f"# BharatVoice Assistant Production Environment Variables\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            
            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")
    
    async def validate_api_connections(self):
        """Validate all configured API connections."""
        print("\n🔍 Validating API Connections")
        print("-" * 40)
        
        env_vars = self.load_env_vars()
        
        # Set environment variables for testing
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Test each service
        await self._test_indian_railways_api()
        await self._test_weather_api()
        await self._test_upi_gateway()
        await self._test_platform_apis()
        
        # Print validation summary
        self._print_validation_summary()
    
    async def _test_indian_railways_api(self):
        """Test Indian Railways API connection."""
        try:
            api_key = os.getenv('INDIAN_RAILWAYS_API_KEY')
            if not api_key:
                self.validation_results['indian_railways'] = {
                    'status': 'skipped',
                    'message': 'API key not configured'
                }
                return
            
            service = IndianRailwaysService(api_key=api_key)
            async with service:
                # Test with a known train number
                result = await service.get_train_schedule("12002")  # Shatabdi Express
                
                if result.success:
                    self.validation_results['indian_railways'] = {
                        'status': 'success',
                        'message': 'API connection successful',
                        'response_time': result.response_time
                    }
                else:
                    self.validation_results['indian_railways'] = {
                        'status': 'error',
                        'message': result.error_message or 'API test failed'
                    }
                    
        except Exception as e:
            self.validation_results['indian_railways'] = {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    async def _test_weather_api(self):
        """Test weather API connection."""
        try:
            api_key = os.getenv('OPENWEATHERMAP_API_KEY')
            if not api_key:
                self.validation_results['weather'] = {
                    'status': 'skipped',
                    'message': 'API key not configured'
                }
                return
            
            service = WeatherService(weather_api_key=api_key)
            async with service:
                result = await service.get_weather_info("Delhi")
                
                if result.success:
                    self.validation_results['weather'] = {
                        'status': 'success',
                        'message': 'API connection successful',
                        'response_time': result.response_time
                    }
                else:
                    self.validation_results['weather'] = {
                        'status': 'error',
                        'message': result.error_message or 'API test failed'
                    }
                    
        except Exception as e:
            self.validation_results['weather'] = {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    async def _test_upi_gateway(self):
        """Test UPI payment gateway connection."""
        try:
            razorpay_key = os.getenv('RAZORPAY_KEY_ID')
            razorpay_secret = os.getenv('RAZORPAY_KEY_SECRET')
            
            if not razorpay_key or not razorpay_secret:
                self.validation_results['upi_payment'] = {
                    'status': 'skipped',
                    'message': 'Payment gateway credentials not configured'
                }
                return
            
            # Test basic connectivity (mock for now)
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(razorpay_key, razorpay_secret)
                async with session.get(
                    'https://api.razorpay.com/v1/payments',
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.validation_results['upi_payment'] = {
                            'status': 'success',
                            'message': 'Payment gateway connection successful'
                        }
                    else:
                        self.validation_results['upi_payment'] = {
                            'status': 'error',
                            'message': f'Payment gateway returned status {response.status}'
                        }
                        
        except Exception as e:
            self.validation_results['upi_payment'] = {
                'status': 'error',
                'message': f'Payment gateway connection failed: {str(e)}'
            }
    
    async def _test_platform_apis(self):
        """Test platform integration APIs."""
        # Test Swiggy API
        swiggy_key = os.getenv('SWIGGY_API_KEY')
        if swiggy_key:
            try:
                # Mock test for Swiggy API
                self.validation_results['swiggy'] = {
                    'status': 'success',
                    'message': 'Swiggy API key configured (test connection requires partner approval)'
                }
            except Exception as e:
                self.validation_results['swiggy'] = {
                    'status': 'error',
                    'message': f'Swiggy API test failed: {str(e)}'
                }
        else:
            self.validation_results['swiggy'] = {
                'status': 'skipped',
                'message': 'Swiggy API key not configured'
            }
        
        # Test Ola API
        ola_client_id = os.getenv('OLA_CLIENT_ID')
        if ola_client_id:
            self.validation_results['ola'] = {
                'status': 'success',
                'message': 'Ola API credentials configured (test connection requires partner approval)'
            }
        else:
            self.validation_results['ola'] = {
                'status': 'skipped',
                'message': 'Ola API credentials not configured'
            }
    
    def _print_validation_summary(self):
        """Print validation results summary."""
        print("\n📊 API Validation Summary")
        print("=" * 50)
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for service, result in self.validation_results.items():
            status = result['status']
            message = result['message']
            
            if status == 'success':
                print(f"✅ {service.replace('_', ' ').title()}: {message}")
                success_count += 1
            elif status == 'error':
                print(f"❌ {service.replace('_', ' ').title()}: {message}")
                error_count += 1
            else:
                print(f"⏭️  {service.replace('_', ' ').title()}: {message}")
                skipped_count += 1
        
        print(f"\nSummary: {success_count} successful, {error_count} errors, {skipped_count} skipped")
        
        if error_count > 0:
            print("\n⚠️  Some API connections failed. Please check your credentials and try again.")
        elif success_count > 0:
            print("\n🎉 API validation completed successfully!")
    
    def generate_deployment_checklist(self):
        """Generate deployment checklist for external APIs."""
        checklist_file = self.config_dir / "deployment_checklist.md"
        
        checklist_content = """# External API Deployment Checklist

## Pre-deployment Requirements

### 1. Indian Railways Integration
- [ ] Obtain API key from RailwayAPI.com
- [ ] Test train schedule queries
- [ ] Configure fallback providers (optional)
- [ ] Set up rate limiting and caching

### 2. Weather Services
- [ ] Get OpenWeatherMap API key (free tier available)
- [ ] Configure IMD API access (if available)
- [ ] Test weather data retrieval for major Indian cities
- [ ] Set up monsoon-specific data sources

### 3. Digital India Platform
- [ ] Apply for Digital India API access
- [ ] Obtain government service API credentials
- [ ] Ensure compliance with data localization requirements
- [ ] Set up audit logging for government service access

### 4. UPI Payment Gateway
- [ ] Partner with Razorpay/PayU for UPI integration
- [ ] Complete KYC and compliance requirements
- [ ] Set up webhook endpoints for payment notifications
- [ ] Configure payment limits and security measures
- [ ] Test payment flows in sandbox environment

### 5. Platform Integrations
- [ ] Apply for Swiggy partner program
- [ ] Get Zomato API access for restaurant data
- [ ] Partner with Ola/Uber for ride booking
- [ ] Set up Urban Company service booking
- [ ] Configure webhook endpoints for booking updates

### 6. Entertainment APIs
- [ ] Get CricAPI key for cricket scores
- [ ] Configure NewsAPI for Bollywood news
- [ ] Set up TMDB for movie information

## Security and Compliance

### Data Protection
- [ ] Implement API key encryption
- [ ] Set up secure key management (HashiCorp Vault)
- [ ] Configure audit logging for all API calls
- [ ] Ensure GDPR and Indian data protection compliance

### Rate Limiting and Monitoring
- [ ] Configure rate limiting for all external APIs
- [ ] Set up monitoring and alerting for API failures
- [ ] Implement circuit breakers for resilience
- [ ] Configure retry policies with exponential backoff

### SSL/TLS Configuration
- [ ] Ensure all API calls use HTTPS
- [ ] Validate SSL certificates
- [ ] Configure certificate pinning for critical APIs

## Production Deployment

### Infrastructure
- [ ] Deploy to production Kubernetes cluster
- [ ] Configure horizontal pod autoscaling
- [ ] Set up load balancing for API services
- [ ] Configure persistent storage for caching

### Monitoring and Alerting
- [ ] Set up Prometheus metrics collection
- [ ] Configure Grafana dashboards for API monitoring
- [ ] Set up alerts for API failures and rate limit breaches
- [ ] Configure log aggregation with ELK stack

### Backup and Disaster Recovery
- [ ] Set up automated database backups
- [ ] Configure API key backup and rotation
- [ ] Test disaster recovery procedures
- [ ] Document rollback procedures

## Testing and Validation

### API Testing
- [ ] Run comprehensive API validation tests
- [ ] Test error handling and fallback mechanisms
- [ ] Validate rate limiting and timeout configurations
- [ ] Test webhook endpoints and callbacks

### Load Testing
- [ ] Conduct load testing with realistic traffic patterns
- [ ] Test API performance under high load
- [ ] Validate auto-scaling behavior
- [ ] Test circuit breaker functionality

### Security Testing
- [ ] Conduct security audit of API integrations
- [ ] Test API key security and rotation
- [ ] Validate input sanitization and validation
- [ ] Test for common security vulnerabilities

## Go-Live Checklist

### Final Preparations
- [ ] All API keys configured and validated
- [ ] Monitoring and alerting systems active
- [ ] Backup and disaster recovery tested
- [ ] Security audit completed
- [ ] Load testing passed
- [ ] Documentation updated

### Launch
- [ ] Deploy to production environment
- [ ] Monitor system health and API performance
- [ ] Validate all integrations working correctly
- [ ] Monitor error rates and response times
- [ ] Be prepared for immediate rollback if needed

## Post-Launch

### Ongoing Maintenance
- [ ] Regular API key rotation
- [ ] Monitor API usage and costs
- [ ] Update API integrations as needed
- [ ] Regular security audits
- [ ] Performance optimization based on usage patterns

---

**Note**: This checklist should be customized based on your specific deployment requirements and available API partnerships.
"""
        
        with open(checklist_file, 'w') as f:
            f.write(checklist_content)
        
        print(f"\n📋 Deployment checklist generated: {checklist_file}")


async def main():
    """Main function for API configuration script."""
    configurator = ExternalAPIConfigurator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            configurator.interactive_setup()
        elif command == "validate":
            await configurator.validate_api_connections()
        elif command == "checklist":
            configurator.generate_deployment_checklist()
        else:
            print("Usage: python configure-external-apis.py [setup|validate|checklist]")
    else:
        # Interactive mode
        print("BharatVoice Assistant - External API Configuration")
        print("1. Interactive Setup")
        print("2. Validate API Connections")
        print("3. Generate Deployment Checklist")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            configurator.interactive_setup()
        elif choice == "2":
            await configurator.validate_api_connections()
        elif choice == "3":
            configurator.generate_deployment_checklist()
        else:
            print("Invalid choice")


if __name__ == "__main__":
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
    asyncio.run(main())