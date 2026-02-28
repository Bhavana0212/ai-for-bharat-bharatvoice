# BharatVoice Assistant

An AI-powered multilingual voice assistant specifically designed for the Indian market, supporting 10+ Indian languages with advanced code-switching detection and cultural context understanding.

## Features

- **Multilingual Voice Recognition**: Support for Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, and English (Indian accent)
- **Code-Switching Detection**: Seamless handling of mixed-language conversations
- **Cultural Context Understanding**: Recognition of Indian festivals, local terminology, and cultural references
- **Indian Service Integration**: Integration with Indian Railways, UPI payments, food delivery, and government services
- **Offline Capabilities**: Basic functionality for rural areas with limited connectivity
- **Privacy & Security**: Compliant with Indian data protection laws with local encryption
- **Accessibility Features**: Support for users with visual or hearing impairments

## Architecture

BharatVoice follows a microservices architecture with the following core services:

- **Voice Processing Service**: Audio input/output, noise filtering, voice activity detection
- **Language Engine Service**: Multilingual ASR, code-switching detection, TTS synthesis
- **Context Management Service**: Conversation state, user preferences, regional context
- **Response Generation Service**: Query processing, response generation, external integrations
- **Authentication Service**: User identity and privacy compliance
- **Offline Sync Service**: Data synchronization and offline capability management

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Redis server (for caching)
- SQLite (for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bharatvoice/assistant.git
cd assistant
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run database migrations:
```bash
alembic upgrade head
```

5. Start the development server:
```bash
uvicorn bharatvoice.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run property-based tests
pytest -m property

# Run with coverage
pytest --cov=src --cov-report=html
```

## Development

### Project Structure

```
src/bharatvoice/
├── main.py                 # FastAPI application entry point
├── config/                 # Configuration management
├── core/                   # Core data models and interfaces
├── services/               # Microservices implementation
│   ├── voice_processing/   # Voice Processing Service
│   ├── language_engine/    # Language Engine Service
│   ├── context_management/ # Context Management Service
│   ├── response_generation/# Response Generation Service
│   ├── auth/              # Authentication Service
│   └── offline_sync/      # Offline Sync Service
├── integrations/          # External service integrations
├── utils/                 # Utility functions
└── api/                   # API route definitions
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework
- **hypothesis**: Property-based testing

Run all quality checks:
```bash
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Documentation: https://docs.bharatvoice.ai
- Issues: https://github.com/bharatvoice/assistant/issues
- Email: support@bharatvoice.ai