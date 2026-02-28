<<<<<<< HEAD
# Regional Context Management System

## Overview

The Regional Context Management System is a comprehensive service designed specifically for the Indian market, providing rich contextual information based on user location. This system integrates multiple data sources to deliver culturally relevant, location-specific information including festivals, weather, local services, and transportation options.

## Features

### 🎭 Indian Festival Calendar
- **National Festivals**: Diwali, Holi, Eid, Dussehra, Ganesh Chaturthi, and more
- **Regional Festivals**: State-specific celebrations like Pongal (Tamil Nadu), Durga Puja (West Bengal), Onam (Kerala)
- **Cultural Significance**: Detailed descriptions and cultural context for each festival
- **Regional Variations**: Recognition of local names and variations of festivals

### 🌤️ Weather Services
- **Climate-Aware**: Understands Indian climate patterns and seasonal variations
- **Monsoon Detection**: Automatic detection of monsoon season (June-September)
- **Regional Adaptation**: Different weather patterns for desert, coastal, and hill regions
- **Air Quality**: Includes air quality index information for Indian cities

### 🏪 Local Service Directory
- **Comprehensive Categories**: Healthcare, education, banking, transport, food, shopping, government, religious
- **Location-Based**: Services within specified radius of user location
- **Indian Context**: Includes traditional Indian businesses like dhabas, sweet shops, cooperative banks
- **Search Functionality**: Query-based service discovery

### 🚌 Transportation Services
- **Multi-Modal**: Railway, bus, auto-rickshaw, taxi, and metro services
- **Indian Railways Integration**: Ready for IRCTC and railway schedule integration
- **Metro Support**: Available for major cities (Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad)
- **Local Transport**: Auto-rickshaws and local bus services

### 🏛️ Government Services
- **Essential Documents**: Aadhaar, PAN, Passport, Driving License, Voter ID
- **Digital India Ready**: Prepared for integration with government portals
- **Document Requirements**: Clear listing of required documents for each service
- **Processing Times**: Expected processing times for government services

### 🗣️ Language and Dialect Support
- **State-Language Mapping**: Automatic detection of local language based on state
- **Dialect Information**: City-specific dialect and accent information
- **Cultural Context**: Understanding of local linguistic preferences

## Architecture

### Core Components

1. **RegionalContextManager**: Main orchestrator class
2. **IndianFestivalCalendar**: Manages cultural events and festivals
3. **WeatherService**: Provides weather and climate information
4. **LocalServiceDirectory**: Manages local business and service information
5. **TransportationService**: Handles transportation options
6. **GovernmentServiceDirectory**: Manages government service information

### Data Models

The system uses comprehensive Pydantic models defined in `bharatvoice.core.models`:

- `RegionalContextData`: Main container for all regional information
- `LocationData`: Geographic location with Indian-specific fields
- `CulturalEvent`: Festival and cultural event information
- `WeatherData`: Weather information with monsoon awareness
- `LocalService`: Local business and service information
- `TransportService`: Transportation service details
- `GovernmentService`: Government service information

## Usage Examples

### Basic Regional Context

```python
from bharatvoice.services.context_management.regional_context_manager import RegionalContextManager
from bharatvoice.core.models import LocationData

# Create location
mumbai = LocationData(
    latitude=19.0760,
    longitude=72.8777,
    city="Mumbai",
    state="Maharashtra",
    country="India"
)

# Get regional context
context_manager = RegionalContextManager()
context = await context_manager.get_regional_context(mumbai)

print(f"Local Language: {context.local_language}")
print(f"Upcoming Festivals: {len(context.cultural_events)}")
print(f"Local Services: {len(context.local_services)}")
print(f"Transport Options: {len(context.transport_options)}")
```

### Festival Information

```python
# Get Hindu festivals specifically
hindu_festivals = await context_manager.get_cultural_events_by_type(
    mumbai, "Hindu Festival"
)

for festival in hindu_festivals:
    print(f"{festival.name}: {festival.significance}")
```

### Service Search

```python
# Find hospitals near location
hospitals = await context_manager.search_local_services(
    mumbai, "hospital"
)

for hospital in hospitals:
    print(f"{hospital.name} - {hospital.distance_km}km away")
```

### Weather Information

```python
# Get weather forecast
forecast = await context_manager.get_weather_forecast(mumbai, days=7)

for day, weather in enumerate(forecast):
    print(f"Day {day+1}: {weather.description}, {weather.temperature_celsius}°C")
```

## State-Language Mapping

The system automatically maps Indian states to their primary languages:

| State | Primary Language |
|-------|------------------|
| Maharashtra | Marathi |
| Tamil Nadu | Tamil |
| West Bengal | Bengali |
| Gujarat | Gujarati |
| Karnataka | Kannada |
| Kerala | Malayalam |
| Punjab | Punjabi |
| Odisha | Odia |
| Telangana/Andhra Pradesh | Telugu |
| Others | Hindi (default) |

## Regional Festivals by State

### Maharashtra
- Gudi Padwa (Marathi New Year)
- Ganpati Festival (Lord Ganesha celebration)

### Tamil Nadu
- Pongal (Harvest festival)
- Tamil New Year (Chithirai celebration)

### West Bengal
- Durga Puja (Goddess Durga worship)
- Poila Boishakh (Bengali New Year)

### Punjab
- Baisakhi (Harvest festival and Sikh New Year)
- Lohri (Winter solstice celebration)

### Kerala
- Onam (Harvest festival and King Mahabali's return)
- Vishu (Malayalam New Year)

### Gujarat
- Navratri (Nine nights of dance and devotion)
- Uttarayan (Kite flying festival)

## Integration Points

### External APIs (Ready for Integration)
- **Indian Railways API**: Train schedules and booking
- **Weather APIs**: Real-time weather data
- **Google Places API**: Local business information
- **Government APIs**: Digital India services
- **Transport APIs**: Real-time transport information

### Caching Strategy
- Festival data: Cached annually with updates
- Weather data: Cached for 1 hour
- Local services: Cached for 24 hours
- Transport schedules: Cached for 6 hours
- Government services: Cached for 7 days

## Error Handling

The system implements robust error handling:

1. **Graceful Degradation**: Returns minimal context on service failures
2. **Timeout Management**: Prevents hanging on slow external services
3. **Retry Logic**: Automatic retry for transient failures
4. **Logging**: Comprehensive logging for debugging and monitoring

## Performance Considerations

- **Concurrent Processing**: All services called concurrently using asyncio
- **Caching**: Intelligent caching to reduce external API calls
- **Lazy Loading**: Services loaded only when needed
- **Connection Pooling**: Efficient HTTP connection management

## Testing

The system includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Property-Based Tests**: Using Hypothesis for edge case testing
- **Mock Testing**: External service simulation

## Future Enhancements

1. **Real-time Integration**: Live APIs for weather, transport, and services
2. **Machine Learning**: Personalized recommendations based on usage patterns
3. **Offline Support**: Cached data for offline functionality
4. **Voice Integration**: Voice-based queries for regional information
5. **Multi-language Support**: Responses in local languages

## Requirements Fulfilled

This implementation addresses the following requirements:

- **2.1**: Indian cultural context understanding
- **2.4**: Regional context and localization
- **4.2**: Weather and local service integration
- **4.5**: Transportation service integration

## Configuration

The system can be configured through environment variables:

```bash
# Weather service configuration
WEATHER_API_KEY=your_weather_api_key
WEATHER_CACHE_TTL=3600

# Local services configuration
PLACES_API_KEY=your_places_api_key
SERVICES_CACHE_TTL=86400

# Transport configuration
TRANSPORT_API_KEY=your_transport_api_key
TRANSPORT_CACHE_TTL=21600
```

## Monitoring and Metrics

The system provides metrics for:

- Response times for each service
- Cache hit/miss ratios
- Error rates by service
- User location distribution
- Popular service categories

## Security Considerations

- **Data Privacy**: No personal data stored without consent
- **API Security**: Secure handling of external API keys
- **Input Validation**: Comprehensive validation of location data
- **Rate Limiting**: Protection against API abuse

=======
# Regional Context Management System

## Overview

The Regional Context Management System is a comprehensive service designed specifically for the Indian market, providing rich contextual information based on user location. This system integrates multiple data sources to deliver culturally relevant, location-specific information including festivals, weather, local services, and transportation options.

## Features

### 🎭 Indian Festival Calendar
- **National Festivals**: Diwali, Holi, Eid, Dussehra, Ganesh Chaturthi, and more
- **Regional Festivals**: State-specific celebrations like Pongal (Tamil Nadu), Durga Puja (West Bengal), Onam (Kerala)
- **Cultural Significance**: Detailed descriptions and cultural context for each festival
- **Regional Variations**: Recognition of local names and variations of festivals

### 🌤️ Weather Services
- **Climate-Aware**: Understands Indian climate patterns and seasonal variations
- **Monsoon Detection**: Automatic detection of monsoon season (June-September)
- **Regional Adaptation**: Different weather patterns for desert, coastal, and hill regions
- **Air Quality**: Includes air quality index information for Indian cities

### 🏪 Local Service Directory
- **Comprehensive Categories**: Healthcare, education, banking, transport, food, shopping, government, religious
- **Location-Based**: Services within specified radius of user location
- **Indian Context**: Includes traditional Indian businesses like dhabas, sweet shops, cooperative banks
- **Search Functionality**: Query-based service discovery

### 🚌 Transportation Services
- **Multi-Modal**: Railway, bus, auto-rickshaw, taxi, and metro services
- **Indian Railways Integration**: Ready for IRCTC and railway schedule integration
- **Metro Support**: Available for major cities (Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad)
- **Local Transport**: Auto-rickshaws and local bus services

### 🏛️ Government Services
- **Essential Documents**: Aadhaar, PAN, Passport, Driving License, Voter ID
- **Digital India Ready**: Prepared for integration with government portals
- **Document Requirements**: Clear listing of required documents for each service
- **Processing Times**: Expected processing times for government services

### 🗣️ Language and Dialect Support
- **State-Language Mapping**: Automatic detection of local language based on state
- **Dialect Information**: City-specific dialect and accent information
- **Cultural Context**: Understanding of local linguistic preferences

## Architecture

### Core Components

1. **RegionalContextManager**: Main orchestrator class
2. **IndianFestivalCalendar**: Manages cultural events and festivals
3. **WeatherService**: Provides weather and climate information
4. **LocalServiceDirectory**: Manages local business and service information
5. **TransportationService**: Handles transportation options
6. **GovernmentServiceDirectory**: Manages government service information

### Data Models

The system uses comprehensive Pydantic models defined in `bharatvoice.core.models`:

- `RegionalContextData`: Main container for all regional information
- `LocationData`: Geographic location with Indian-specific fields
- `CulturalEvent`: Festival and cultural event information
- `WeatherData`: Weather information with monsoon awareness
- `LocalService`: Local business and service information
- `TransportService`: Transportation service details
- `GovernmentService`: Government service information

## Usage Examples

### Basic Regional Context

```python
from bharatvoice.services.context_management.regional_context_manager import RegionalContextManager
from bharatvoice.core.models import LocationData

# Create location
mumbai = LocationData(
    latitude=19.0760,
    longitude=72.8777,
    city="Mumbai",
    state="Maharashtra",
    country="India"
)

# Get regional context
context_manager = RegionalContextManager()
context = await context_manager.get_regional_context(mumbai)

print(f"Local Language: {context.local_language}")
print(f"Upcoming Festivals: {len(context.cultural_events)}")
print(f"Local Services: {len(context.local_services)}")
print(f"Transport Options: {len(context.transport_options)}")
```

### Festival Information

```python
# Get Hindu festivals specifically
hindu_festivals = await context_manager.get_cultural_events_by_type(
    mumbai, "Hindu Festival"
)

for festival in hindu_festivals:
    print(f"{festival.name}: {festival.significance}")
```

### Service Search

```python
# Find hospitals near location
hospitals = await context_manager.search_local_services(
    mumbai, "hospital"
)

for hospital in hospitals:
    print(f"{hospital.name} - {hospital.distance_km}km away")
```

### Weather Information

```python
# Get weather forecast
forecast = await context_manager.get_weather_forecast(mumbai, days=7)

for day, weather in enumerate(forecast):
    print(f"Day {day+1}: {weather.description}, {weather.temperature_celsius}°C")
```

## State-Language Mapping

The system automatically maps Indian states to their primary languages:

| State | Primary Language |
|-------|------------------|
| Maharashtra | Marathi |
| Tamil Nadu | Tamil |
| West Bengal | Bengali |
| Gujarat | Gujarati |
| Karnataka | Kannada |
| Kerala | Malayalam |
| Punjab | Punjabi |
| Odisha | Odia |
| Telangana/Andhra Pradesh | Telugu |
| Others | Hindi (default) |

## Regional Festivals by State

### Maharashtra
- Gudi Padwa (Marathi New Year)
- Ganpati Festival (Lord Ganesha celebration)

### Tamil Nadu
- Pongal (Harvest festival)
- Tamil New Year (Chithirai celebration)

### West Bengal
- Durga Puja (Goddess Durga worship)
- Poila Boishakh (Bengali New Year)

### Punjab
- Baisakhi (Harvest festival and Sikh New Year)
- Lohri (Winter solstice celebration)

### Kerala
- Onam (Harvest festival and King Mahabali's return)
- Vishu (Malayalam New Year)

### Gujarat
- Navratri (Nine nights of dance and devotion)
- Uttarayan (Kite flying festival)

## Integration Points

### External APIs (Ready for Integration)
- **Indian Railways API**: Train schedules and booking
- **Weather APIs**: Real-time weather data
- **Google Places API**: Local business information
- **Government APIs**: Digital India services
- **Transport APIs**: Real-time transport information

### Caching Strategy
- Festival data: Cached annually with updates
- Weather data: Cached for 1 hour
- Local services: Cached for 24 hours
- Transport schedules: Cached for 6 hours
- Government services: Cached for 7 days

## Error Handling

The system implements robust error handling:

1. **Graceful Degradation**: Returns minimal context on service failures
2. **Timeout Management**: Prevents hanging on slow external services
3. **Retry Logic**: Automatic retry for transient failures
4. **Logging**: Comprehensive logging for debugging and monitoring

## Performance Considerations

- **Concurrent Processing**: All services called concurrently using asyncio
- **Caching**: Intelligent caching to reduce external API calls
- **Lazy Loading**: Services loaded only when needed
- **Connection Pooling**: Efficient HTTP connection management

## Testing

The system includes comprehensive tests:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Property-Based Tests**: Using Hypothesis for edge case testing
- **Mock Testing**: External service simulation

## Future Enhancements

1. **Real-time Integration**: Live APIs for weather, transport, and services
2. **Machine Learning**: Personalized recommendations based on usage patterns
3. **Offline Support**: Cached data for offline functionality
4. **Voice Integration**: Voice-based queries for regional information
5. **Multi-language Support**: Responses in local languages

## Requirements Fulfilled

This implementation addresses the following requirements:

- **2.1**: Indian cultural context understanding
- **2.4**: Regional context and localization
- **4.2**: Weather and local service integration
- **4.5**: Transportation service integration

## Configuration

The system can be configured through environment variables:

```bash
# Weather service configuration
WEATHER_API_KEY=your_weather_api_key
WEATHER_CACHE_TTL=3600

# Local services configuration
PLACES_API_KEY=your_places_api_key
SERVICES_CACHE_TTL=86400

# Transport configuration
TRANSPORT_API_KEY=your_transport_api_key
TRANSPORT_CACHE_TTL=21600
```

## Monitoring and Metrics

The system provides metrics for:

- Response times for each service
- Cache hit/miss ratios
- Error rates by service
- User location distribution
- Popular service categories

## Security Considerations

- **Data Privacy**: No personal data stored without consent
- **API Security**: Secure handling of external API keys
- **Input Validation**: Comprehensive validation of location data
- **Rate Limiting**: Protection against API abuse

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
This Regional Context Management System provides a solid foundation for delivering culturally relevant, location-aware services to Indian users, with the flexibility to integrate with real-world APIs and services as needed.