<<<<<<< HEAD
# Natural Language Understanding (NLU) Service

## Overview

The NLU Service is a comprehensive Natural Language Understanding system specifically designed for Indian cultural context. It provides intent recognition, entity extraction, colloquial term understanding, and cultural context interpretation for the BharatVoice Assistant.

## Features

### 1. Intent Recognition for Indian Cultural Context
- **30+ Intent Categories**: Covers greetings, transportation, festivals, food ordering, government services, entertainment, and more
- **Cultural Patterns**: Recognizes Indian communication patterns and cultural nuances
- **Context-Aware Classification**: Uses conversation history to improve intent detection
- **Confidence Scoring**: Provides reliable confidence scores for all classifications

### 2. Entity Extraction for Indian-Specific Terms
- **Location Entities**: Recognizes 100+ Indian cities, states, and landmarks
- **Cultural Entities**: Extracts festivals, deities, cultural terms, and traditions
- **Food Entities**: Identifies 50+ Indian dishes and cuisine types
- **Relationship Entities**: Understands Indian family relationships and social terms
- **Service Entities**: Recognizes government documents, transportation, and services

### 3. Colloquial Term Understanding and Mapping
- **200+ Colloquial Terms**: Maps Hindi, regional, and Indian English colloquialisms
- **Family Relationships**: Understands terms like "mummy", "papa", "bhai", "didi"
- **Cultural Expressions**: Handles greetings like "namaste", "vanakkam", "sat sri akal"
- **Modern Slang**: Processes contemporary terms like "yaar", "bindaas", "jugaad"
- **Regional Variations**: Supports regional variations of common terms

### 4. Cultural Context Interpretation
- **Communication Style Detection**: Identifies formal, casual, family-oriented, or religious contexts
- **Formality Level Analysis**: Determines appropriate response tone based on user language
- **Regional Influence**: Adapts to North, South, East, and West Indian cultural patterns
- **Cultural Sensitivity**: Identifies topics requiring careful cultural handling

## Architecture

### Core Components

#### 1. NLUService
Main orchestrator that coordinates all NLU operations:
```python
nlu_service = NLUService()
result = await nlu_service.process_user_input(
    text="Namaste ji, Mumbai se Delhi ki train ka time kya hai?",
    language=LanguageCode.HINDI
)
```

#### 2. ColloquialTermMapper
Maps colloquial terms to standard meanings:
```python
mapper = ColloquialTermMapper()
mapped_text = await mapper.map_colloquial_terms(
    "Yaar, khana order karna hai", 
    LanguageCode.HINDI
)
# Result: "friend, food order karna hai"
```

#### 3. IndianEntityExtractor
Extracts Indian-specific entities:
```python
extractor = IndianEntityExtractor()
entities = await extractor.extract_entities(
    "Mumbai se Chennai ki train", 
    LanguageCode.HINDI
)
# Extracts: Mumbai (city), Chennai (city), train (transport)
```

#### 4. IndianIntentClassifier
Classifies intents with cultural awareness:
```python
classifier = IndianIntentClassifier()
intent = await classifier.classify_intent("When is Diwali?")
# Result: festival_inquiry with high confidence
```

#### 5. CulturalContextInterpreter
Interprets cultural context and communication style:
```python
interpreter = CulturalContextInterpreter()
context = await interpreter.interpret_cultural_context(
    "Sir, please help me"
)
# Result: formal communication style, high formality level
```

## Intent Categories

### General Intents
- `greeting`: Namaste, hello, how are you
- `farewell`: Goodbye, alvida, see you later
- `help`: Help requests and guidance
- `confirmation`: Yes, haan, theek hai
- `negation`: No, nahi, cancel

### Information Seeking
- `weather_inquiry`: Weather, mausam, rain, temperature
- `time_inquiry`: Current time, what time is it
- `date_inquiry`: Today's date, calendar queries
- `news_inquiry`: News updates and current events

### Transportation
- `train_inquiry`: Railway schedules, IRCTC, train booking
- `bus_inquiry`: Bus schedules and routes
- `flight_inquiry`: Flight information and booking
- `traffic_inquiry`: Traffic conditions and routes

### Cultural and Festivals
- `festival_inquiry`: Diwali, Holi, Eid, festival dates
- `cultural_event`: Cultural celebrations and events
- `religious_inquiry`: Religious information and guidance

### Services
- `food_order`: Food delivery, restaurant orders
- `ride_booking`: Cab, auto, taxi booking
- `payment_upi`: UPI payments, money transfers
- `government_service`: Aadhaar, PAN, passport services

### Entertainment
- `music_request`: Music and song requests
- `cricket_scores`: Cricket matches and scores
- `bollywood_news`: Bollywood updates and news

## Entity Types

### Location Entities
- `city`: Indian cities (Mumbai, Delhi, Bangalore, etc.)
- `state`: Indian states and union territories
- `landmark`: Famous landmarks and places
- `pincode`: Indian postal codes

### Cultural Entities
- `festival`: Indian festivals and celebrations
- `deity`: Hindu, Sikh, and other deities
- `cultural_term`: Cultural concepts and terms

### Food Entities
- `dish`: Indian dishes (biryani, dosa, samosa, etc.)
- `cuisine_type`: Regional cuisines (South Indian, Punjabi, etc.)
- `restaurant`: Restaurant names and types

### Relationship Entities
- `person_name`: Names of people
- `relationship`: Family relationships (mummy, papa, bhai, etc.)

## Usage Examples

### Basic NLU Processing
```python
from bharatvoice.services.response_generation import NLUService
from bharatvoice.core.models import LanguageCode

nlu_service = NLUService()

# Process user input
result = await nlu_service.process_user_input(
    text="Namaste, aaj mausam kaisa hai?",
    language=LanguageCode.HINDI
)

print(f"Intent: {result['intent'].name}")
print(f"Confidence: {result['intent'].confidence}")
print(f"Entities: {len(result['entities'])}")
print(f"Cultural Context: {result['cultural_context']}")
```

### With Conversation Context
```python
from bharatvoice.core.models import ConversationState, UserInteraction

# Create conversation state
conversation_state = ConversationState(
    user_id=user_id,
    current_language=LanguageCode.HINDI,
    conversation_history=[previous_interactions]
)

# Process with context
result = await nlu_service.process_user_input(
    text="What about tomorrow?",
    language=LanguageCode.ENGLISH_IN,
    conversation_state=conversation_state
)
```

### Cultural Appropriateness Validation
```python
# Validate cultural appropriateness
validation = await nlu_service.validate_cultural_appropriateness(
    text="Tell me about religious festivals",
    intent=result['intent'],
    cultural_context=result['cultural_context']
)

if not validation['is_appropriate']:
    print("Cultural concerns:", validation['concerns'])
    print("Suggestions:", validation['suggestions'])
```

## Configuration

### Supported Languages
- Hindi (`hi`)
- Indian English (`en-IN`)
- Tamil (`ta`)
- Telugu (`te`)
- Bengali (`bn`)
- Marathi (`mr`)
- Gujarati (`gu`)
- Kannada (`kn`)
- Malayalam (`ml`)
- Punjabi (`pa`)
- Odia (`or`)

### Regional Contexts
- **North India**: Delhi, Punjab, Haryana, UP, Rajasthan
- **South India**: Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana
- **West India**: Maharashtra, Gujarat, Goa
- **East India**: West Bengal, Odisha, Jharkhand, Bihar

## Performance Characteristics

### Response Times
- Simple intent classification: < 100ms
- Complex entity extraction: < 200ms
- Full NLU processing: < 500ms
- Cultural context analysis: < 50ms

### Accuracy Metrics
- Intent classification accuracy: > 85% for clear patterns
- Entity extraction precision: > 80% for Indian entities
- Cultural context detection: > 90% for clear indicators
- Colloquial term mapping: > 95% for known terms

## Error Handling

The NLU service implements comprehensive error handling:

1. **Graceful Degradation**: Returns unknown intent with low confidence for unclear inputs
2. **Input Validation**: Handles empty, very long, or malformed inputs
3. **Language Fallback**: Falls back to English processing if language-specific processing fails
4. **Cultural Safety**: Provides warnings for culturally sensitive topics

## Testing

### Unit Tests
- Individual component testing
- Edge case handling
- Error condition testing
- Cultural context validation

### Property-Based Tests
- Universal properties across all inputs
- Cultural context recognition consistency
- Confidence score validity
- Intent classification reliability

### Integration Tests
- End-to-end NLU processing
- Conversation context integration
- Regional context integration
- Multi-language processing

## Future Enhancements

1. **Machine Learning Integration**: Add ML models for improved accuracy
2. **Voice Pattern Recognition**: Integrate with speech processing for accent detection
3. **Personalization**: Learn from user interactions for better customization
4. **Extended Regional Support**: Add more regional languages and dialects
5. **Sentiment Analysis**: Add emotional context understanding
6. **Domain-Specific Models**: Specialized models for healthcare, education, etc.

## Dependencies

- `pydantic`: Data validation and settings management
- `asyncio`: Asynchronous processing
- `re`: Regular expression processing
- `datetime`: Time and date handling
- `typing`: Type hints and annotations
- `enum`: Enumeration support
- `uuid`: Unique identifier generation

## API Reference

See the individual module documentation for detailed API reference:
- `nlu_service.py`: Main NLU service implementation
- `nlu_interface.py`: ResponseGenerator interface implementation
- Core models in `bharatvoice.core.models`

## Contributing

When contributing to the NLU service:

1. **Cultural Sensitivity**: Ensure all additions respect Indian cultural diversity
2. **Testing**: Add comprehensive tests for new features
3. **Documentation**: Update documentation for new capabilities
4. **Performance**: Maintain response time requirements
5. **Accuracy**: Validate accuracy improvements with test data

## License

=======
# Natural Language Understanding (NLU) Service

## Overview

The NLU Service is a comprehensive Natural Language Understanding system specifically designed for Indian cultural context. It provides intent recognition, entity extraction, colloquial term understanding, and cultural context interpretation for the BharatVoice Assistant.

## Features

### 1. Intent Recognition for Indian Cultural Context
- **30+ Intent Categories**: Covers greetings, transportation, festivals, food ordering, government services, entertainment, and more
- **Cultural Patterns**: Recognizes Indian communication patterns and cultural nuances
- **Context-Aware Classification**: Uses conversation history to improve intent detection
- **Confidence Scoring**: Provides reliable confidence scores for all classifications

### 2. Entity Extraction for Indian-Specific Terms
- **Location Entities**: Recognizes 100+ Indian cities, states, and landmarks
- **Cultural Entities**: Extracts festivals, deities, cultural terms, and traditions
- **Food Entities**: Identifies 50+ Indian dishes and cuisine types
- **Relationship Entities**: Understands Indian family relationships and social terms
- **Service Entities**: Recognizes government documents, transportation, and services

### 3. Colloquial Term Understanding and Mapping
- **200+ Colloquial Terms**: Maps Hindi, regional, and Indian English colloquialisms
- **Family Relationships**: Understands terms like "mummy", "papa", "bhai", "didi"
- **Cultural Expressions**: Handles greetings like "namaste", "vanakkam", "sat sri akal"
- **Modern Slang**: Processes contemporary terms like "yaar", "bindaas", "jugaad"
- **Regional Variations**: Supports regional variations of common terms

### 4. Cultural Context Interpretation
- **Communication Style Detection**: Identifies formal, casual, family-oriented, or religious contexts
- **Formality Level Analysis**: Determines appropriate response tone based on user language
- **Regional Influence**: Adapts to North, South, East, and West Indian cultural patterns
- **Cultural Sensitivity**: Identifies topics requiring careful cultural handling

## Architecture

### Core Components

#### 1. NLUService
Main orchestrator that coordinates all NLU operations:
```python
nlu_service = NLUService()
result = await nlu_service.process_user_input(
    text="Namaste ji, Mumbai se Delhi ki train ka time kya hai?",
    language=LanguageCode.HINDI
)
```

#### 2. ColloquialTermMapper
Maps colloquial terms to standard meanings:
```python
mapper = ColloquialTermMapper()
mapped_text = await mapper.map_colloquial_terms(
    "Yaar, khana order karna hai", 
    LanguageCode.HINDI
)
# Result: "friend, food order karna hai"
```

#### 3. IndianEntityExtractor
Extracts Indian-specific entities:
```python
extractor = IndianEntityExtractor()
entities = await extractor.extract_entities(
    "Mumbai se Chennai ki train", 
    LanguageCode.HINDI
)
# Extracts: Mumbai (city), Chennai (city), train (transport)
```

#### 4. IndianIntentClassifier
Classifies intents with cultural awareness:
```python
classifier = IndianIntentClassifier()
intent = await classifier.classify_intent("When is Diwali?")
# Result: festival_inquiry with high confidence
```

#### 5. CulturalContextInterpreter
Interprets cultural context and communication style:
```python
interpreter = CulturalContextInterpreter()
context = await interpreter.interpret_cultural_context(
    "Sir, please help me"
)
# Result: formal communication style, high formality level
```

## Intent Categories

### General Intents
- `greeting`: Namaste, hello, how are you
- `farewell`: Goodbye, alvida, see you later
- `help`: Help requests and guidance
- `confirmation`: Yes, haan, theek hai
- `negation`: No, nahi, cancel

### Information Seeking
- `weather_inquiry`: Weather, mausam, rain, temperature
- `time_inquiry`: Current time, what time is it
- `date_inquiry`: Today's date, calendar queries
- `news_inquiry`: News updates and current events

### Transportation
- `train_inquiry`: Railway schedules, IRCTC, train booking
- `bus_inquiry`: Bus schedules and routes
- `flight_inquiry`: Flight information and booking
- `traffic_inquiry`: Traffic conditions and routes

### Cultural and Festivals
- `festival_inquiry`: Diwali, Holi, Eid, festival dates
- `cultural_event`: Cultural celebrations and events
- `religious_inquiry`: Religious information and guidance

### Services
- `food_order`: Food delivery, restaurant orders
- `ride_booking`: Cab, auto, taxi booking
- `payment_upi`: UPI payments, money transfers
- `government_service`: Aadhaar, PAN, passport services

### Entertainment
- `music_request`: Music and song requests
- `cricket_scores`: Cricket matches and scores
- `bollywood_news`: Bollywood updates and news

## Entity Types

### Location Entities
- `city`: Indian cities (Mumbai, Delhi, Bangalore, etc.)
- `state`: Indian states and union territories
- `landmark`: Famous landmarks and places
- `pincode`: Indian postal codes

### Cultural Entities
- `festival`: Indian festivals and celebrations
- `deity`: Hindu, Sikh, and other deities
- `cultural_term`: Cultural concepts and terms

### Food Entities
- `dish`: Indian dishes (biryani, dosa, samosa, etc.)
- `cuisine_type`: Regional cuisines (South Indian, Punjabi, etc.)
- `restaurant`: Restaurant names and types

### Relationship Entities
- `person_name`: Names of people
- `relationship`: Family relationships (mummy, papa, bhai, etc.)

## Usage Examples

### Basic NLU Processing
```python
from bharatvoice.services.response_generation import NLUService
from bharatvoice.core.models import LanguageCode

nlu_service = NLUService()

# Process user input
result = await nlu_service.process_user_input(
    text="Namaste, aaj mausam kaisa hai?",
    language=LanguageCode.HINDI
)

print(f"Intent: {result['intent'].name}")
print(f"Confidence: {result['intent'].confidence}")
print(f"Entities: {len(result['entities'])}")
print(f"Cultural Context: {result['cultural_context']}")
```

### With Conversation Context
```python
from bharatvoice.core.models import ConversationState, UserInteraction

# Create conversation state
conversation_state = ConversationState(
    user_id=user_id,
    current_language=LanguageCode.HINDI,
    conversation_history=[previous_interactions]
)

# Process with context
result = await nlu_service.process_user_input(
    text="What about tomorrow?",
    language=LanguageCode.ENGLISH_IN,
    conversation_state=conversation_state
)
```

### Cultural Appropriateness Validation
```python
# Validate cultural appropriateness
validation = await nlu_service.validate_cultural_appropriateness(
    text="Tell me about religious festivals",
    intent=result['intent'],
    cultural_context=result['cultural_context']
)

if not validation['is_appropriate']:
    print("Cultural concerns:", validation['concerns'])
    print("Suggestions:", validation['suggestions'])
```

## Configuration

### Supported Languages
- Hindi (`hi`)
- Indian English (`en-IN`)
- Tamil (`ta`)
- Telugu (`te`)
- Bengali (`bn`)
- Marathi (`mr`)
- Gujarati (`gu`)
- Kannada (`kn`)
- Malayalam (`ml`)
- Punjabi (`pa`)
- Odia (`or`)

### Regional Contexts
- **North India**: Delhi, Punjab, Haryana, UP, Rajasthan
- **South India**: Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana
- **West India**: Maharashtra, Gujarat, Goa
- **East India**: West Bengal, Odisha, Jharkhand, Bihar

## Performance Characteristics

### Response Times
- Simple intent classification: < 100ms
- Complex entity extraction: < 200ms
- Full NLU processing: < 500ms
- Cultural context analysis: < 50ms

### Accuracy Metrics
- Intent classification accuracy: > 85% for clear patterns
- Entity extraction precision: > 80% for Indian entities
- Cultural context detection: > 90% for clear indicators
- Colloquial term mapping: > 95% for known terms

## Error Handling

The NLU service implements comprehensive error handling:

1. **Graceful Degradation**: Returns unknown intent with low confidence for unclear inputs
2. **Input Validation**: Handles empty, very long, or malformed inputs
3. **Language Fallback**: Falls back to English processing if language-specific processing fails
4. **Cultural Safety**: Provides warnings for culturally sensitive topics

## Testing

### Unit Tests
- Individual component testing
- Edge case handling
- Error condition testing
- Cultural context validation

### Property-Based Tests
- Universal properties across all inputs
- Cultural context recognition consistency
- Confidence score validity
- Intent classification reliability

### Integration Tests
- End-to-end NLU processing
- Conversation context integration
- Regional context integration
- Multi-language processing

## Future Enhancements

1. **Machine Learning Integration**: Add ML models for improved accuracy
2. **Voice Pattern Recognition**: Integrate with speech processing for accent detection
3. **Personalization**: Learn from user interactions for better customization
4. **Extended Regional Support**: Add more regional languages and dialects
5. **Sentiment Analysis**: Add emotional context understanding
6. **Domain-Specific Models**: Specialized models for healthcare, education, etc.

## Dependencies

- `pydantic`: Data validation and settings management
- `asyncio`: Asynchronous processing
- `re`: Regular expression processing
- `datetime`: Time and date handling
- `typing`: Type hints and annotations
- `enum`: Enumeration support
- `uuid`: Unique identifier generation

## API Reference

See the individual module documentation for detailed API reference:
- `nlu_service.py`: Main NLU service implementation
- `nlu_interface.py`: ResponseGenerator interface implementation
- Core models in `bharatvoice.core.models`

## Contributing

When contributing to the NLU service:

1. **Cultural Sensitivity**: Ensure all additions respect Indian cultural diversity
2. **Testing**: Add comprehensive tests for new features
3. **Documentation**: Update documentation for new capabilities
4. **Performance**: Maintain response time requirements
5. **Accuracy**: Validate accuracy improvements with test data

## License

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
This NLU service is part of the BharatVoice Assistant project and follows the project's licensing terms.