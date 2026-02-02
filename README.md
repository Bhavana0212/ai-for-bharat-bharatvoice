
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&size=28&pause=1000&color=FF9933&center=true&vCenter=true&width=800&lines=BharatVoice+Assistant;AI+for+Rural+India;Multilingual+Voice+AI+for+Bharat;Offline+%7C+Secure+%7C+Inclusive" />
</p>

<img width="1536" height="1024" alt="ai bharat voice" src="https://github.com/user-attachments/assets/98895c38-62bd-44cc-869d-ee9a5c547158" />


# BharatVoice Assistant

An AI-powered multilingual voice assistant specifically designed for the Indian market, supporting 10+ Indian languages with advanced code-switching detection and cultural context understanding.

## Features

- 🎙️ **Multilingual Voice Recognition**: Support for Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, and English (Indian accent)
- 🔁 **Code-Switching Detection**: Seamless handling of mixed-language conversations
- 🧠 **Cultural Context Understanding**: Recognition of Indian festivals, local terminology, and cultural references
- 🔁 **Indian Service Integration**: Integration with Indian Railways, UPI payments, food delivery, and government services
- 📡 **Offline Capabilities**: Basic functionality for rural areas with limited connectivity
- 🔐 **Privacy & Security**: Compliant with Indian data protection laws with local encryption
- ♿ **Accessibility Features**: Support for users with visual or hearing impairments


## Architecture

BharatVoice follows a microservices architecture with the following core services:

- **Voice Processing Service**: Audio input/output, noise filtering, voice activity detection
- **Language Engine Service**: Multilingual ASR, code-switching detection, TTS synthesis
- **Context Management Service**: Conversation state, user preferences, regional context
- **Response Generation Service**: Query processing, response generation, external integrations
- **Authentication Service**: User identity and privacy compliance
- **Offline Sync Service**: Data synchronization and offline capability management

## 🏗️ System Architecture

```text
User Voice 🎤
     ↓
Voice Processing Service
     ↓
Language Engine (ASR + Code-Switch)
     ↓
Context Management
     ↓
Response Generation
     ↓
TTS 🔊 → User 
```

<h2>🧰 Tech Stack</h2>

<table>
  <tr>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="48"/><br/>
      Python 3.9+
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" width="48"/><br/>
      FastAPI
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" width="48"/><br/>
      Docker
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg" width="48"/><br/>
      Kubernetes
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" width="48"/><br/>
      PostgreSQL
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" width="48"/><br/>
      Redis
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/amazonwebservices/amazonwebservices-original.svg" width="48"/><br/>
      AWS
    </td>
    <td align="center">
      <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="48"/><br/>
      Git
    </td>
  </tr>
</table>



### Quick Start

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

## 🚀 Project Status

🟢 **Production-Ready Architecture**  
🟢 **All Core Services Implemented**  
🟢 **Offline + Multilingual Support**  
🟡 **External APIs: Configurable**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Documentation: https://docs.bharatvoice.ai
- Issues: https://github.com/bharatvoice/assistant/issues

- Email: support@bharatvoice.ai


<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=FF9933&height=5"/>
  <br/>
  🇮🇳 Built with ❤️ for Bharat | AI for Rural Innovation
</p>



