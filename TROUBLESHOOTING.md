<<<<<<< HEAD
# BharatVoice Assistant - Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps resolve common issues with the BharatVoice Assistant system. Issues are organized by category with step-by-step solutions and escalation procedures.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Voice Recognition Issues](#voice-recognition-issues)
3. [Audio Processing Problems](#audio-processing-problems)
4. [Language Detection Issues](#language-detection-issues)
5. [Performance Problems](#performance-problems)
6. [Connectivity Issues](#connectivity-issues)
7. [Authentication Problems](#authentication-problems)
8. [Database Issues](#database-issues)
9. [External Service Integration Issues](#external-service-integration-issues)
10. [Deployment Issues](#deployment-issues)
11. [Monitoring and Logging](#monitoring-and-logging)
12. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostics

### System Health Check

Run the built-in health check to identify issues quickly:

```bash
# Check overall system health
curl -f http://localhost:8000/health

# Detailed health check
curl -f http://localhost:8000/health/detailed

# Check specific components
curl -f http://localhost:8000/health/database
curl -f http://localhost:8000/health/redis
curl -f http://localhost:8000/health/external-services
```

### Log Analysis

Check recent logs for errors:

```bash
# Application logs
tail -f /var/log/bharatvoice/app.log

# System logs
journalctl -u bharatvoice -f

# Error logs only
grep -i error /var/log/bharatvoice/app.log | tail -20
```

### Performance Metrics

Check current performance metrics:

```bash
# Get performance stats
curl -s http://localhost:8000/metrics | grep -E "(response_time|error_rate|active_requests)"

# System resources
htop
df -h
free -h
```

---

## Voice Recognition Issues

### Issue: "Voice recognition not working"

**Symptoms**:
- No transcription returned
- Empty or garbled text output
- High error rates in recognition

**Diagnosis**:
```bash
# Check Whisper model status
python -c "
from bharatvoice.services.language_engine.asr_engine import create_multilingual_asr_engine
engine = create_multilingual_asr_engine()
print('Model loaded successfully')
"

# Test with sample audio
python run_speech_recognition_property_test.py
```

**Solutions**:

1. **Check Audio Input Quality**:
   ```bash
   # Verify audio file format
   file audio_sample.wav
   
   # Check audio properties
   ffprobe -v quiet -show_format -show_streams audio_sample.wav
   ```

2. **Verify Model Dependencies**:
   ```bash
   # Reinstall Whisper
   pip uninstall openai-whisper
   pip install openai-whisper==20231117
   
   # Check CUDA availability (if using GPU)
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Audio Preprocessing Issues**:
   ```python
   # Test audio preprocessing
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor()
   # Test with sample audio
   result = await processor.preprocess_audio(audio_buffer)
   print(f"Preprocessing successful: {len(result.data)} samples")
   ```

4. **Memory Issues**:
   ```bash
   # Check available memory
   free -h
   
   # Reduce model size if needed
   export WHISPER_MODEL_SIZE=base  # instead of large
   ```

### Issue: "Poor recognition accuracy"

**Symptoms**:
- Incorrect transcriptions
- Low confidence scores
- Missing words or phrases

**Solutions**:

1. **Audio Quality Optimization**:
   ```python
   # Enable noise reduction
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor(noise_reduction_factor=0.7)
   filtered_audio = await processor.filter_background_noise(audio_buffer)
   ```

2. **Language Model Tuning**:
   ```python
   # Adjust confidence thresholds
   asr_engine = create_multilingual_asr_engine(
       confidence_threshold=0.6,  # Lower for more permissive recognition
       max_alternatives=5
   )
   ```

3. **Accent Adaptation**:
   ```python
   # Enable accent-specific processing
   result = await asr_engine.recognize_speech(
       audio_buffer,
       language_hint=LanguageCode.HINDI,
       accent_hint=AccentType.NORTH_INDIAN
   )
   ```

### Issue: "Code-switching not detected"

**Symptoms**:
- Mixed language input treated as single language
- Incorrect language detection
- Poor transcription of mixed content

**Solutions**:

1. **Enable Code-Switching Detection**:
   ```python
   # Verify code-switching is enabled
   from bharatvoice.services.language_engine.code_switching_detector import CodeSwitchingDetector
   
   detector = CodeSwitchingDetector()
   switches = await detector.detect_code_switching("Hello नमस्ते how are you")
   print(f"Detected switches: {len(switches)}")
   ```

2. **Language Model Configuration**:
   ```python
   # Configure for mixed language input
   asr_engine = create_multilingual_asr_engine(
       enable_language_detection=True,
       enable_code_switching=True,
       supported_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
   )
   ```

---

## Audio Processing Problems

### Issue: "Audio processing failures"

**Symptoms**:
- Audio files not processed
- Silence detection not working
- Audio format conversion errors

**Diagnosis**:
```bash
# Check audio processing dependencies
python -c "
import librosa
import soundfile
import webrtcvad
print('Audio dependencies loaded successfully')
"

# Test audio processing
python run_audio_property_test.py
```

**Solutions**:

1. **Audio Format Issues**:
   ```python
   # Check supported formats
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor()
   supported_formats = processor.get_supported_formats()
   print(f"Supported formats: {supported_formats}")
   ```

2. **Sample Rate Conversion**:
   ```python
   # Convert sample rate
   import librosa
   
   audio, sr = librosa.load('input.wav', sr=16000)  # Convert to 16kHz
   ```

3. **Voice Activity Detection Issues**:
   ```python
   # Test VAD
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor(vad_aggressiveness=2)
   voice_segments = await processor.detect_voice_activity(audio_buffer)
   print(f"Voice segments detected: {len(voice_segments)}")
   ```

### Issue: "TTS synthesis problems"

**Symptoms**:
- No audio output from TTS
- Poor quality synthesis
- Synthesis timeouts

**Diagnosis**:
```bash
# Test TTS functionality
python run_tts_property_test.py

# Check gTTS connectivity
python -c "
from gtts import gTTS
tts = gTTS('Test', lang='en')
print('gTTS connection successful')
"
```

**Solutions**:

1. **Network Connectivity for gTTS**:
   ```bash
   # Test internet connectivity
   curl -I https://translate.google.com
   
   # Configure proxy if needed
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

2. **TTS Engine Configuration**:
   ```python
   # Configure TTS with fallbacks
   from bharatvoice.services.voice_processing.tts_engine import TTSEngine
   
   tts_engine = TTSEngine(
       sample_rate=22050,
       quality='medium',  # Reduce quality if having issues
       enable_caching=True,
       fallback_enabled=True
   )
   ```

3. **Audio Output Issues**:
   ```python
   # Test audio output
   import pygame
   pygame.mixer.init()
   
   # Verify audio buffer format
   audio_buffer = await tts_engine.synthesize_speech("Test", LanguageCode.ENGLISH_IN)
   print(f"Audio buffer: {len(audio_buffer.data)} samples, {audio_buffer.duration}s")
   ```

---

## Language Detection Issues

### Issue: "Incorrect language detection"

**Symptoms**:
- Wrong language identified
- Low confidence in detection
- Code-switching not recognized

**Solutions**:

1. **Language Detection Model**:
   ```python
   # Test language detection
   from bharatvoice.services.language_engine.service import LanguageEngineService
   
   service = LanguageEngineService()
   detected = await service.detect_language("नमस्ते, how are you?")
   print(f"Detected: {detected}")
   ```

2. **Improve Detection Accuracy**:
   ```python
   # Use longer text samples for better accuracy
   # Minimum 10-15 words recommended
   text = "नमस्ते मित्र, आज का दिन कैसा है? How has your day been so far?"
   detected = await service.detect_language(text)
   ```

3. **Manual Language Hints**:
   ```python
   # Provide language hints when possible
   result = await asr_engine.recognize_speech(
       audio_buffer,
       language_hint=LanguageCode.HINDI
   )
   ```

### Issue: "Translation quality problems"

**Symptoms**:
- Incorrect translations
- Loss of cultural context
- Grammatical errors

**Solutions**:

1. **Translation Engine Configuration**:
   ```python
   # Enable cultural context preservation
   from bharatvoice.services.language_engine.translation_engine import TranslationEngine
   
   translator = TranslationEngine(
       preserve_cultural_context=True,
       enable_colloquial_mapping=True
   )
   ```

2. **Context-Aware Translation**:
   ```python
   # Provide context for better translation
   result = await translator.translate(
       text="I want to order biryani",
       source_lang=LanguageCode.ENGLISH_IN,
       target_lang=LanguageCode.HINDI,
       context={"domain": "food", "region": "north_india"}
   )
   ```

---

## Performance Problems

### Issue: "Slow response times"

**Symptoms**:
- Response times > 5 seconds
- Timeouts on requests
- Poor user experience

**Diagnosis**:
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor system resources
iostat -x 1
vmstat 1
```

**Solutions**:

1. **Database Optimization**:
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Optimize indexes
   ANALYZE;
   REINDEX DATABASE bharatvoice;
   ```

2. **Redis Cache Optimization**:
   ```bash
   # Check Redis performance
   redis-cli info stats
   redis-cli slowlog get 10
   
   # Optimize Redis configuration
   redis-cli config set maxmemory-policy allkeys-lru
   ```

3. **Application Performance**:
   ```python
   # Enable async processing
   import asyncio
   
   # Use connection pooling
   from bharatvoice.database.connection import get_connection_pool
   pool = get_connection_pool(min_size=5, max_size=20)
   ```

4. **Load Balancing**:
   ```bash
   # Check load balancer status
   curl -s http://localhost:8000/metrics | grep active_requests
   
   # Scale workers if needed
   systemctl edit bharatvoice
   # Add: Environment="WORKERS=8"
   ```

### Issue: "High memory usage"

**Symptoms**:
- Memory usage > 80%
- Out of memory errors
- System slowdown

**Solutions**:

1. **Memory Profiling**:
   ```python
   # Profile memory usage
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   memory_info = process.memory_info()
   print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
   ```

2. **Model Memory Optimization**:
   ```python
   # Use smaller models
   asr_engine = create_multilingual_asr_engine(
       model_size="base",  # instead of "large"
       device="cpu"  # if GPU memory is limited
   )
   ```

3. **Cache Management**:
   ```python
   # Configure cache limits
   from bharatvoice.cache.cache_manager import CacheManager
   
   cache_manager = CacheManager(
       max_memory_mb=512,  # Limit cache memory
       ttl_seconds=1800    # Shorter TTL
   )
   ```

---

## Connectivity Issues

### Issue: "Database connection failures"

**Symptoms**:
- Connection refused errors
- Timeout errors
- Pool exhaustion

**Diagnosis**:
```bash
# Test database connectivity
psql -h localhost -U bharatvoice -d bharatvoice -c "SELECT 1;"

# Check connection pool status
python -c "
from bharatvoice.database.connection import get_database_connection
conn = get_database_connection()
print('Database connection successful')
"
```

**Solutions**:

1. **Connection Pool Configuration**:
   ```python
   # Optimize connection pool
   DATABASE_POOL_SIZE = 20
   DATABASE_MAX_OVERFLOW = 30
   DATABASE_POOL_TIMEOUT = 30
   DATABASE_POOL_RECYCLE = 3600
   ```

2. **Database Server Issues**:
   ```bash
   # Check PostgreSQL status
   systemctl status postgresql
   
   # Check PostgreSQL logs
   tail -f /var/log/postgresql/postgresql-*.log
   
   # Restart if needed
   systemctl restart postgresql
   ```

3. **Network Issues**:
   ```bash
   # Test network connectivity
   telnet localhost 5432
   
   # Check firewall rules
   ufw status
   iptables -L
   ```

### Issue: "Redis connection problems"

**Symptoms**:
- Redis connection errors
- Cache misses
- Session data loss

**Solutions**:

1. **Redis Server Status**:
   ```bash
   # Check Redis status
   systemctl status redis-server
   redis-cli ping
   
   # Check Redis logs
   tail -f /var/log/redis/redis-server.log
   ```

2. **Redis Configuration**:
   ```bash
   # Check Redis configuration
   redis-cli config get "*"
   
   # Optimize Redis settings
   redis-cli config set timeout 300
   redis-cli config set tcp-keepalive 60
   ```

3. **Connection Pool Issues**:
   ```python
   # Configure Redis connection pool
   REDIS_MAX_CONNECTIONS = 100
   REDIS_RETRY_ON_TIMEOUT = True
   REDIS_SOCKET_KEEPALIVE = True
   ```

---

## Authentication Problems

### Issue: "JWT token validation failures"

**Symptoms**:
- 401 Unauthorized errors
- Token expired messages
- Invalid signature errors

**Solutions**:

1. **Token Configuration**:
   ```python
   # Check JWT configuration
   from bharatvoice.services.auth.jwt_manager import JWTManager
   
   jwt_manager = JWTManager()
   # Verify secret key is set correctly
   print(f"JWT algorithm: {jwt_manager.algorithm}")
   ```

2. **Token Expiration**:
   ```python
   # Extend token expiration if needed
   JWT_EXPIRATION_HOURS = 24  # Increase from default
   
   # Implement token refresh
   refresh_token = jwt_manager.create_refresh_token(user_id)
   ```

3. **Clock Synchronization**:
   ```bash
   # Ensure system time is synchronized
   timedatectl status
   ntpdate -s time.nist.gov
   ```

### Issue: "Multi-factor authentication problems"

**Symptoms**:
- OTP verification failures
- QR code generation issues
- Backup codes not working

**Solutions**:

1. **OTP Configuration**:
   ```python
   # Test OTP generation
   from bharatvoice.services.auth.mfa_manager import MFAManager
   
   mfa_manager = MFAManager()
   secret = mfa_manager.generate_secret()
   otp = mfa_manager.generate_otp(secret)
   print(f"Generated OTP: {otp}")
   ```

2. **Time Synchronization**:
   ```bash
   # OTP requires accurate time
   chrony sources -v
   systemctl status chrony
   ```

---

## Database Issues

### Issue: "Database migration failures"

**Symptoms**:
- Alembic migration errors
- Schema inconsistencies
- Data corruption

**Solutions**:

1. **Check Migration Status**:
   ```bash
   # Check current migration version
   alembic current
   
   # Check migration history
   alembic history
   
   # Show pending migrations
   alembic show head
   ```

2. **Fix Migration Issues**:
   ```bash
   # Rollback problematic migration
   alembic downgrade -1
   
   # Apply migrations step by step
   alembic upgrade +1
   
   # Force migration (use with caution)
   alembic stamp head
   ```

3. **Database Backup and Recovery**:
   ```bash
   # Create backup before fixing
   pg_dump -h localhost -U bharatvoice bharatvoice > backup.sql
   
   # Restore from backup if needed
   psql -h localhost -U bharatvoice bharatvoice < backup.sql
   ```

### Issue: "Database performance problems"

**Symptoms**:
- Slow query execution
- High CPU usage
- Lock contention

**Solutions**:

1. **Query Optimization**:
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   SELECT pg_reload_conf();
   
   -- Analyze slow queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. **Index Optimization**:
   ```sql
   -- Check missing indexes
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats
   WHERE schemaname = 'public'
   ORDER BY n_distinct DESC;
   
   -- Create indexes for frequently queried columns
   CREATE INDEX CONCURRENTLY idx_user_interactions_user_id 
   ON user_interactions(user_id);
   ```

---

## External Service Integration Issues

### Issue: "Indian Railways API failures"

**Symptoms**:
- Train information not available
- API timeout errors
- Invalid response format

**Solutions**:

1. **API Connectivity**:
   ```bash
   # Test API endpoint
   curl -H "Authorization: Bearer $RAILWAYS_API_KEY" \
        "https://api.railwayapi.com/v2/live/train/12951/date/20240315/"
   ```

2. **Fallback Mechanisms**:
   ```python
   # Implement fallback data sources
   from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
   
   service = IndianRailwaysService(
       enable_fallback=True,
       cache_duration=3600,
       timeout=10
   )
   ```

### Issue: "Weather service problems"

**Symptoms**:
- Weather data not updating
- Location-based queries failing
- API rate limits exceeded

**Solutions**:

1. **API Key Management**:
   ```python
   # Rotate API keys if rate limited
   WEATHER_API_KEYS = [
       "key1", "key2", "key3"
   ]
   
   # Implement key rotation
   service = WeatherService(api_keys=WEATHER_API_KEYS)
   ```

2. **Caching Strategy**:
   ```python
   # Cache weather data to reduce API calls
   weather_data = await service.get_weather(
       city="Delhi",
       cache_duration=1800  # 30 minutes
   )
   ```

---

## Deployment Issues

### Issue: "Docker container startup failures"

**Symptoms**:
- Container exits immediately
- Port binding errors
- Volume mount issues

**Solutions**:

1. **Container Logs**:
   ```bash
   # Check container logs
   docker logs bharatvoice-app
   
   # Follow logs in real-time
   docker logs -f bharatvoice-app
   ```

2. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different port if needed
   docker run -p 8001:8000 bharatvoice-assistant
   ```

3. **Volume Permissions**:
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./logs
   sudo chown -R 1000:1000 ./uploads
   ```

### Issue: "Kubernetes deployment problems"

**Symptoms**:
- Pods not starting
- Service discovery issues
- Resource constraints

**Solutions**:

1. **Pod Diagnostics**:
   ```bash
   # Check pod status
   kubectl get pods -n bharatvoice
   
   # Describe problematic pod
   kubectl describe pod bharatvoice-app-xxx -n bharatvoice
   
   # Check pod logs
   kubectl logs bharatvoice-app-xxx -n bharatvoice
   ```

2. **Resource Issues**:
   ```yaml
   # Adjust resource limits
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "2Gi"
       cpu: "1000m"
   ```

3. **Service Discovery**:
   ```bash
   # Test service connectivity
   kubectl exec -it bharatvoice-app-xxx -n bharatvoice -- curl bharatvoice-service/health
   ```

---

## Monitoring and Logging

### Issue: "Missing or incomplete logs"

**Symptoms**:
- Log files not created
- Missing error information
- Log rotation issues

**Solutions**:

1. **Log Configuration**:
   ```python
   # Configure structured logging
   import structlog
   
   structlog.configure(
       processors=[
           structlog.stdlib.filter_by_level,
           structlog.stdlib.add_logger_name,
           structlog.stdlib.add_log_level,
           structlog.stdlib.PositionalArgumentsFormatter(),
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.processors.StackInfoRenderer(),
           structlog.processors.format_exc_info,
           structlog.processors.JSONRenderer()
       ],
       context_class=dict,
       logger_factory=structlog.stdlib.LoggerFactory(),
       wrapper_class=structlog.stdlib.BoundLogger,
       cache_logger_on_first_use=True,
   )
   ```

2. **Log Rotation**:
   ```bash
   # Configure logrotate
   sudo tee /etc/logrotate.d/bharatvoice << EOF
   /var/log/bharatvoice/*.log {
       daily
       missingok
       rotate 30
       compress
       delaycompress
       notifempty
       create 644 bharatvoice bharatvoice
       postrotate
           systemctl reload bharatvoice
       endscript
   }
   EOF
   ```

### Issue: "Monitoring alerts not working"

**Symptoms**:
- No alerts for system issues
- Prometheus metrics missing
- Grafana dashboards empty

**Solutions**:

1. **Prometheus Configuration**:
   ```yaml
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify metrics endpoint
   curl http://localhost:8000/metrics
   ```

2. **Alert Rules**:
   ```yaml
   # Configure alert rules
   groups:
   - name: bharatvoice
     rules:
     - alert: HighResponseTime
       expr: response_time_seconds > 5
       for: 2m
       labels:
         severity: warning
       annotations:
         summary: "High response time detected"
   ```

---

## Emergency Procedures

### System Recovery

1. **Service Restart**:
   ```bash
   # Restart application service
   systemctl restart bharatvoice
   
   # Restart all related services
   systemctl restart bharatvoice postgresql redis-server nginx
   ```

2. **Database Recovery**:
   ```bash
   # Restore from latest backup
   systemctl stop bharatvoice
   psql -h localhost -U bharatvoice bharatvoice < /opt/bharatvoice/backups/latest.sql
   systemctl start bharatvoice
   ```

3. **Cache Reset**:
   ```bash
   # Clear Redis cache
   redis-cli flushall
   
   # Restart Redis
   systemctl restart redis-server
   ```

### Escalation Procedures

1. **Level 1 - Application Issues**:
   - Check application logs
   - Restart services
   - Verify configuration

2. **Level 2 - System Issues**:
   - Check system resources
   - Analyze performance metrics
   - Contact system administrator

3. **Level 3 - Critical Issues**:
   - Activate incident response team
   - Implement disaster recovery
   - Contact vendor support

### Contact Information

- **Technical Support**: support@bharatvoice.ai
- **Emergency Hotline**: +91-1800-BHARAT-VOICE
- **System Administrator**: sysadmin@bharatvoice.ai
- **On-call Engineer**: oncall@bharatvoice.ai

---

## Preventive Measures

### Regular Maintenance

1. **Daily Tasks**:
   - Check system health
   - Review error logs
   - Monitor performance metrics

2. **Weekly Tasks**:
   - Update system packages
   - Analyze performance trends
   - Review security logs

3. **Monthly Tasks**:
   - Database maintenance
   - Security audits
   - Backup verification

### Monitoring Setup

1. **Health Checks**:
   ```bash
   # Automated health check script
   #!/bin/bash
   curl -f http://localhost:8000/health || echo "Health check failed" | mail -s "BharatVoice Alert" admin@company.com
   ```

2. **Performance Monitoring**:
   ```bash
   # Performance monitoring script
   #!/bin/bash
   RESPONSE_TIME=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8000/health)
   if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
       echo "High response time: $RESPONSE_TIME" | mail -s "Performance Alert" admin@company.com
   fi
   ```

---

**Remember**: When in doubt, check the logs first, then verify system resources, and finally test individual components. Most issues can be resolved by following the systematic approach outlined in this guide.

=======
# BharatVoice Assistant - Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide helps resolve common issues with the BharatVoice Assistant system. Issues are organized by category with step-by-step solutions and escalation procedures.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Voice Recognition Issues](#voice-recognition-issues)
3. [Audio Processing Problems](#audio-processing-problems)
4. [Language Detection Issues](#language-detection-issues)
5. [Performance Problems](#performance-problems)
6. [Connectivity Issues](#connectivity-issues)
7. [Authentication Problems](#authentication-problems)
8. [Database Issues](#database-issues)
9. [External Service Integration Issues](#external-service-integration-issues)
10. [Deployment Issues](#deployment-issues)
11. [Monitoring and Logging](#monitoring-and-logging)
12. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostics

### System Health Check

Run the built-in health check to identify issues quickly:

```bash
# Check overall system health
curl -f http://localhost:8000/health

# Detailed health check
curl -f http://localhost:8000/health/detailed

# Check specific components
curl -f http://localhost:8000/health/database
curl -f http://localhost:8000/health/redis
curl -f http://localhost:8000/health/external-services
```

### Log Analysis

Check recent logs for errors:

```bash
# Application logs
tail -f /var/log/bharatvoice/app.log

# System logs
journalctl -u bharatvoice -f

# Error logs only
grep -i error /var/log/bharatvoice/app.log | tail -20
```

### Performance Metrics

Check current performance metrics:

```bash
# Get performance stats
curl -s http://localhost:8000/metrics | grep -E "(response_time|error_rate|active_requests)"

# System resources
htop
df -h
free -h
```

---

## Voice Recognition Issues

### Issue: "Voice recognition not working"

**Symptoms**:
- No transcription returned
- Empty or garbled text output
- High error rates in recognition

**Diagnosis**:
```bash
# Check Whisper model status
python -c "
from bharatvoice.services.language_engine.asr_engine import create_multilingual_asr_engine
engine = create_multilingual_asr_engine()
print('Model loaded successfully')
"

# Test with sample audio
python run_speech_recognition_property_test.py
```

**Solutions**:

1. **Check Audio Input Quality**:
   ```bash
   # Verify audio file format
   file audio_sample.wav
   
   # Check audio properties
   ffprobe -v quiet -show_format -show_streams audio_sample.wav
   ```

2. **Verify Model Dependencies**:
   ```bash
   # Reinstall Whisper
   pip uninstall openai-whisper
   pip install openai-whisper==20231117
   
   # Check CUDA availability (if using GPU)
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Audio Preprocessing Issues**:
   ```python
   # Test audio preprocessing
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor()
   # Test with sample audio
   result = await processor.preprocess_audio(audio_buffer)
   print(f"Preprocessing successful: {len(result.data)} samples")
   ```

4. **Memory Issues**:
   ```bash
   # Check available memory
   free -h
   
   # Reduce model size if needed
   export WHISPER_MODEL_SIZE=base  # instead of large
   ```

### Issue: "Poor recognition accuracy"

**Symptoms**:
- Incorrect transcriptions
- Low confidence scores
- Missing words or phrases

**Solutions**:

1. **Audio Quality Optimization**:
   ```python
   # Enable noise reduction
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor(noise_reduction_factor=0.7)
   filtered_audio = await processor.filter_background_noise(audio_buffer)
   ```

2. **Language Model Tuning**:
   ```python
   # Adjust confidence thresholds
   asr_engine = create_multilingual_asr_engine(
       confidence_threshold=0.6,  # Lower for more permissive recognition
       max_alternatives=5
   )
   ```

3. **Accent Adaptation**:
   ```python
   # Enable accent-specific processing
   result = await asr_engine.recognize_speech(
       audio_buffer,
       language_hint=LanguageCode.HINDI,
       accent_hint=AccentType.NORTH_INDIAN
   )
   ```

### Issue: "Code-switching not detected"

**Symptoms**:
- Mixed language input treated as single language
- Incorrect language detection
- Poor transcription of mixed content

**Solutions**:

1. **Enable Code-Switching Detection**:
   ```python
   # Verify code-switching is enabled
   from bharatvoice.services.language_engine.code_switching_detector import CodeSwitchingDetector
   
   detector = CodeSwitchingDetector()
   switches = await detector.detect_code_switching("Hello नमस्ते how are you")
   print(f"Detected switches: {len(switches)}")
   ```

2. **Language Model Configuration**:
   ```python
   # Configure for mixed language input
   asr_engine = create_multilingual_asr_engine(
       enable_language_detection=True,
       enable_code_switching=True,
       supported_languages=[LanguageCode.HINDI, LanguageCode.ENGLISH_IN]
   )
   ```

---

## Audio Processing Problems

### Issue: "Audio processing failures"

**Symptoms**:
- Audio files not processed
- Silence detection not working
- Audio format conversion errors

**Diagnosis**:
```bash
# Check audio processing dependencies
python -c "
import librosa
import soundfile
import webrtcvad
print('Audio dependencies loaded successfully')
"

# Test audio processing
python run_audio_property_test.py
```

**Solutions**:

1. **Audio Format Issues**:
   ```python
   # Check supported formats
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor()
   supported_formats = processor.get_supported_formats()
   print(f"Supported formats: {supported_formats}")
   ```

2. **Sample Rate Conversion**:
   ```python
   # Convert sample rate
   import librosa
   
   audio, sr = librosa.load('input.wav', sr=16000)  # Convert to 16kHz
   ```

3. **Voice Activity Detection Issues**:
   ```python
   # Test VAD
   from bharatvoice.services.voice_processing.audio_processor import AudioProcessor
   
   processor = AudioProcessor(vad_aggressiveness=2)
   voice_segments = await processor.detect_voice_activity(audio_buffer)
   print(f"Voice segments detected: {len(voice_segments)}")
   ```

### Issue: "TTS synthesis problems"

**Symptoms**:
- No audio output from TTS
- Poor quality synthesis
- Synthesis timeouts

**Diagnosis**:
```bash
# Test TTS functionality
python run_tts_property_test.py

# Check gTTS connectivity
python -c "
from gtts import gTTS
tts = gTTS('Test', lang='en')
print('gTTS connection successful')
"
```

**Solutions**:

1. **Network Connectivity for gTTS**:
   ```bash
   # Test internet connectivity
   curl -I https://translate.google.com
   
   # Configure proxy if needed
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

2. **TTS Engine Configuration**:
   ```python
   # Configure TTS with fallbacks
   from bharatvoice.services.voice_processing.tts_engine import TTSEngine
   
   tts_engine = TTSEngine(
       sample_rate=22050,
       quality='medium',  # Reduce quality if having issues
       enable_caching=True,
       fallback_enabled=True
   )
   ```

3. **Audio Output Issues**:
   ```python
   # Test audio output
   import pygame
   pygame.mixer.init()
   
   # Verify audio buffer format
   audio_buffer = await tts_engine.synthesize_speech("Test", LanguageCode.ENGLISH_IN)
   print(f"Audio buffer: {len(audio_buffer.data)} samples, {audio_buffer.duration}s")
   ```

---

## Language Detection Issues

### Issue: "Incorrect language detection"

**Symptoms**:
- Wrong language identified
- Low confidence in detection
- Code-switching not recognized

**Solutions**:

1. **Language Detection Model**:
   ```python
   # Test language detection
   from bharatvoice.services.language_engine.service import LanguageEngineService
   
   service = LanguageEngineService()
   detected = await service.detect_language("नमस्ते, how are you?")
   print(f"Detected: {detected}")
   ```

2. **Improve Detection Accuracy**:
   ```python
   # Use longer text samples for better accuracy
   # Minimum 10-15 words recommended
   text = "नमस्ते मित्र, आज का दिन कैसा है? How has your day been so far?"
   detected = await service.detect_language(text)
   ```

3. **Manual Language Hints**:
   ```python
   # Provide language hints when possible
   result = await asr_engine.recognize_speech(
       audio_buffer,
       language_hint=LanguageCode.HINDI
   )
   ```

### Issue: "Translation quality problems"

**Symptoms**:
- Incorrect translations
- Loss of cultural context
- Grammatical errors

**Solutions**:

1. **Translation Engine Configuration**:
   ```python
   # Enable cultural context preservation
   from bharatvoice.services.language_engine.translation_engine import TranslationEngine
   
   translator = TranslationEngine(
       preserve_cultural_context=True,
       enable_colloquial_mapping=True
   )
   ```

2. **Context-Aware Translation**:
   ```python
   # Provide context for better translation
   result = await translator.translate(
       text="I want to order biryani",
       source_lang=LanguageCode.ENGLISH_IN,
       target_lang=LanguageCode.HINDI,
       context={"domain": "food", "region": "north_india"}
   )
   ```

---

## Performance Problems

### Issue: "Slow response times"

**Symptoms**:
- Response times > 5 seconds
- Timeouts on requests
- Poor user experience

**Diagnosis**:
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Monitor system resources
iostat -x 1
vmstat 1
```

**Solutions**:

1. **Database Optimization**:
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Optimize indexes
   ANALYZE;
   REINDEX DATABASE bharatvoice;
   ```

2. **Redis Cache Optimization**:
   ```bash
   # Check Redis performance
   redis-cli info stats
   redis-cli slowlog get 10
   
   # Optimize Redis configuration
   redis-cli config set maxmemory-policy allkeys-lru
   ```

3. **Application Performance**:
   ```python
   # Enable async processing
   import asyncio
   
   # Use connection pooling
   from bharatvoice.database.connection import get_connection_pool
   pool = get_connection_pool(min_size=5, max_size=20)
   ```

4. **Load Balancing**:
   ```bash
   # Check load balancer status
   curl -s http://localhost:8000/metrics | grep active_requests
   
   # Scale workers if needed
   systemctl edit bharatvoice
   # Add: Environment="WORKERS=8"
   ```

### Issue: "High memory usage"

**Symptoms**:
- Memory usage > 80%
- Out of memory errors
- System slowdown

**Solutions**:

1. **Memory Profiling**:
   ```python
   # Profile memory usage
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   memory_info = process.memory_info()
   print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
   ```

2. **Model Memory Optimization**:
   ```python
   # Use smaller models
   asr_engine = create_multilingual_asr_engine(
       model_size="base",  # instead of "large"
       device="cpu"  # if GPU memory is limited
   )
   ```

3. **Cache Management**:
   ```python
   # Configure cache limits
   from bharatvoice.cache.cache_manager import CacheManager
   
   cache_manager = CacheManager(
       max_memory_mb=512,  # Limit cache memory
       ttl_seconds=1800    # Shorter TTL
   )
   ```

---

## Connectivity Issues

### Issue: "Database connection failures"

**Symptoms**:
- Connection refused errors
- Timeout errors
- Pool exhaustion

**Diagnosis**:
```bash
# Test database connectivity
psql -h localhost -U bharatvoice -d bharatvoice -c "SELECT 1;"

# Check connection pool status
python -c "
from bharatvoice.database.connection import get_database_connection
conn = get_database_connection()
print('Database connection successful')
"
```

**Solutions**:

1. **Connection Pool Configuration**:
   ```python
   # Optimize connection pool
   DATABASE_POOL_SIZE = 20
   DATABASE_MAX_OVERFLOW = 30
   DATABASE_POOL_TIMEOUT = 30
   DATABASE_POOL_RECYCLE = 3600
   ```

2. **Database Server Issues**:
   ```bash
   # Check PostgreSQL status
   systemctl status postgresql
   
   # Check PostgreSQL logs
   tail -f /var/log/postgresql/postgresql-*.log
   
   # Restart if needed
   systemctl restart postgresql
   ```

3. **Network Issues**:
   ```bash
   # Test network connectivity
   telnet localhost 5432
   
   # Check firewall rules
   ufw status
   iptables -L
   ```

### Issue: "Redis connection problems"

**Symptoms**:
- Redis connection errors
- Cache misses
- Session data loss

**Solutions**:

1. **Redis Server Status**:
   ```bash
   # Check Redis status
   systemctl status redis-server
   redis-cli ping
   
   # Check Redis logs
   tail -f /var/log/redis/redis-server.log
   ```

2. **Redis Configuration**:
   ```bash
   # Check Redis configuration
   redis-cli config get "*"
   
   # Optimize Redis settings
   redis-cli config set timeout 300
   redis-cli config set tcp-keepalive 60
   ```

3. **Connection Pool Issues**:
   ```python
   # Configure Redis connection pool
   REDIS_MAX_CONNECTIONS = 100
   REDIS_RETRY_ON_TIMEOUT = True
   REDIS_SOCKET_KEEPALIVE = True
   ```

---

## Authentication Problems

### Issue: "JWT token validation failures"

**Symptoms**:
- 401 Unauthorized errors
- Token expired messages
- Invalid signature errors

**Solutions**:

1. **Token Configuration**:
   ```python
   # Check JWT configuration
   from bharatvoice.services.auth.jwt_manager import JWTManager
   
   jwt_manager = JWTManager()
   # Verify secret key is set correctly
   print(f"JWT algorithm: {jwt_manager.algorithm}")
   ```

2. **Token Expiration**:
   ```python
   # Extend token expiration if needed
   JWT_EXPIRATION_HOURS = 24  # Increase from default
   
   # Implement token refresh
   refresh_token = jwt_manager.create_refresh_token(user_id)
   ```

3. **Clock Synchronization**:
   ```bash
   # Ensure system time is synchronized
   timedatectl status
   ntpdate -s time.nist.gov
   ```

### Issue: "Multi-factor authentication problems"

**Symptoms**:
- OTP verification failures
- QR code generation issues
- Backup codes not working

**Solutions**:

1. **OTP Configuration**:
   ```python
   # Test OTP generation
   from bharatvoice.services.auth.mfa_manager import MFAManager
   
   mfa_manager = MFAManager()
   secret = mfa_manager.generate_secret()
   otp = mfa_manager.generate_otp(secret)
   print(f"Generated OTP: {otp}")
   ```

2. **Time Synchronization**:
   ```bash
   # OTP requires accurate time
   chrony sources -v
   systemctl status chrony
   ```

---

## Database Issues

### Issue: "Database migration failures"

**Symptoms**:
- Alembic migration errors
- Schema inconsistencies
- Data corruption

**Solutions**:

1. **Check Migration Status**:
   ```bash
   # Check current migration version
   alembic current
   
   # Check migration history
   alembic history
   
   # Show pending migrations
   alembic show head
   ```

2. **Fix Migration Issues**:
   ```bash
   # Rollback problematic migration
   alembic downgrade -1
   
   # Apply migrations step by step
   alembic upgrade +1
   
   # Force migration (use with caution)
   alembic stamp head
   ```

3. **Database Backup and Recovery**:
   ```bash
   # Create backup before fixing
   pg_dump -h localhost -U bharatvoice bharatvoice > backup.sql
   
   # Restore from backup if needed
   psql -h localhost -U bharatvoice bharatvoice < backup.sql
   ```

### Issue: "Database performance problems"

**Symptoms**:
- Slow query execution
- High CPU usage
- Lock contention

**Solutions**:

1. **Query Optimization**:
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 1000;
   SELECT pg_reload_conf();
   
   -- Analyze slow queries
   SELECT query, mean_time, calls, total_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. **Index Optimization**:
   ```sql
   -- Check missing indexes
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats
   WHERE schemaname = 'public'
   ORDER BY n_distinct DESC;
   
   -- Create indexes for frequently queried columns
   CREATE INDEX CONCURRENTLY idx_user_interactions_user_id 
   ON user_interactions(user_id);
   ```

---

## External Service Integration Issues

### Issue: "Indian Railways API failures"

**Symptoms**:
- Train information not available
- API timeout errors
- Invalid response format

**Solutions**:

1. **API Connectivity**:
   ```bash
   # Test API endpoint
   curl -H "Authorization: Bearer $RAILWAYS_API_KEY" \
        "https://api.railwayapi.com/v2/live/train/12951/date/20240315/"
   ```

2. **Fallback Mechanisms**:
   ```python
   # Implement fallback data sources
   from bharatvoice.services.external_integrations.indian_railways_service import IndianRailwaysService
   
   service = IndianRailwaysService(
       enable_fallback=True,
       cache_duration=3600,
       timeout=10
   )
   ```

### Issue: "Weather service problems"

**Symptoms**:
- Weather data not updating
- Location-based queries failing
- API rate limits exceeded

**Solutions**:

1. **API Key Management**:
   ```python
   # Rotate API keys if rate limited
   WEATHER_API_KEYS = [
       "key1", "key2", "key3"
   ]
   
   # Implement key rotation
   service = WeatherService(api_keys=WEATHER_API_KEYS)
   ```

2. **Caching Strategy**:
   ```python
   # Cache weather data to reduce API calls
   weather_data = await service.get_weather(
       city="Delhi",
       cache_duration=1800  # 30 minutes
   )
   ```

---

## Deployment Issues

### Issue: "Docker container startup failures"

**Symptoms**:
- Container exits immediately
- Port binding errors
- Volume mount issues

**Solutions**:

1. **Container Logs**:
   ```bash
   # Check container logs
   docker logs bharatvoice-app
   
   # Follow logs in real-time
   docker logs -f bharatvoice-app
   ```

2. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different port if needed
   docker run -p 8001:8000 bharatvoice-assistant
   ```

3. **Volume Permissions**:
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./logs
   sudo chown -R 1000:1000 ./uploads
   ```

### Issue: "Kubernetes deployment problems"

**Symptoms**:
- Pods not starting
- Service discovery issues
- Resource constraints

**Solutions**:

1. **Pod Diagnostics**:
   ```bash
   # Check pod status
   kubectl get pods -n bharatvoice
   
   # Describe problematic pod
   kubectl describe pod bharatvoice-app-xxx -n bharatvoice
   
   # Check pod logs
   kubectl logs bharatvoice-app-xxx -n bharatvoice
   ```

2. **Resource Issues**:
   ```yaml
   # Adjust resource limits
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "2Gi"
       cpu: "1000m"
   ```

3. **Service Discovery**:
   ```bash
   # Test service connectivity
   kubectl exec -it bharatvoice-app-xxx -n bharatvoice -- curl bharatvoice-service/health
   ```

---

## Monitoring and Logging

### Issue: "Missing or incomplete logs"

**Symptoms**:
- Log files not created
- Missing error information
- Log rotation issues

**Solutions**:

1. **Log Configuration**:
   ```python
   # Configure structured logging
   import structlog
   
   structlog.configure(
       processors=[
           structlog.stdlib.filter_by_level,
           structlog.stdlib.add_logger_name,
           structlog.stdlib.add_log_level,
           structlog.stdlib.PositionalArgumentsFormatter(),
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.processors.StackInfoRenderer(),
           structlog.processors.format_exc_info,
           structlog.processors.JSONRenderer()
       ],
       context_class=dict,
       logger_factory=structlog.stdlib.LoggerFactory(),
       wrapper_class=structlog.stdlib.BoundLogger,
       cache_logger_on_first_use=True,
   )
   ```

2. **Log Rotation**:
   ```bash
   # Configure logrotate
   sudo tee /etc/logrotate.d/bharatvoice << EOF
   /var/log/bharatvoice/*.log {
       daily
       missingok
       rotate 30
       compress
       delaycompress
       notifempty
       create 644 bharatvoice bharatvoice
       postrotate
           systemctl reload bharatvoice
       endscript
   }
   EOF
   ```

### Issue: "Monitoring alerts not working"

**Symptoms**:
- No alerts for system issues
- Prometheus metrics missing
- Grafana dashboards empty

**Solutions**:

1. **Prometheus Configuration**:
   ```yaml
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify metrics endpoint
   curl http://localhost:8000/metrics
   ```

2. **Alert Rules**:
   ```yaml
   # Configure alert rules
   groups:
   - name: bharatvoice
     rules:
     - alert: HighResponseTime
       expr: response_time_seconds > 5
       for: 2m
       labels:
         severity: warning
       annotations:
         summary: "High response time detected"
   ```

---

## Emergency Procedures

### System Recovery

1. **Service Restart**:
   ```bash
   # Restart application service
   systemctl restart bharatvoice
   
   # Restart all related services
   systemctl restart bharatvoice postgresql redis-server nginx
   ```

2. **Database Recovery**:
   ```bash
   # Restore from latest backup
   systemctl stop bharatvoice
   psql -h localhost -U bharatvoice bharatvoice < /opt/bharatvoice/backups/latest.sql
   systemctl start bharatvoice
   ```

3. **Cache Reset**:
   ```bash
   # Clear Redis cache
   redis-cli flushall
   
   # Restart Redis
   systemctl restart redis-server
   ```

### Escalation Procedures

1. **Level 1 - Application Issues**:
   - Check application logs
   - Restart services
   - Verify configuration

2. **Level 2 - System Issues**:
   - Check system resources
   - Analyze performance metrics
   - Contact system administrator

3. **Level 3 - Critical Issues**:
   - Activate incident response team
   - Implement disaster recovery
   - Contact vendor support

### Contact Information

- **Technical Support**: support@bharatvoice.ai
- **Emergency Hotline**: +91-1800-BHARAT-VOICE
- **System Administrator**: sysadmin@bharatvoice.ai
- **On-call Engineer**: oncall@bharatvoice.ai

---

## Preventive Measures

### Regular Maintenance

1. **Daily Tasks**:
   - Check system health
   - Review error logs
   - Monitor performance metrics

2. **Weekly Tasks**:
   - Update system packages
   - Analyze performance trends
   - Review security logs

3. **Monthly Tasks**:
   - Database maintenance
   - Security audits
   - Backup verification

### Monitoring Setup

1. **Health Checks**:
   ```bash
   # Automated health check script
   #!/bin/bash
   curl -f http://localhost:8000/health || echo "Health check failed" | mail -s "BharatVoice Alert" admin@company.com
   ```

2. **Performance Monitoring**:
   ```bash
   # Performance monitoring script
   #!/bin/bash
   RESPONSE_TIME=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8000/health)
   if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
       echo "High response time: $RESPONSE_TIME" | mail -s "Performance Alert" admin@company.com
   fi
   ```

---

**Remember**: When in doubt, check the logs first, then verify system resources, and finally test individual components. Most issues can be resolved by following the systematic approach outlined in this guide.

>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
For additional support, visit our [community forum](https://community.bharatvoice.ai) or contact our technical support team.