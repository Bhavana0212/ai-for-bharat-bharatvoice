# BharatVoice AI - Streamlit Web Interface Deployment Guide

This guide provides step-by-step instructions for deploying the BharatVoice AI Streamlit web interface in various environments.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [AWS Amplify Deployment](#aws-amplify-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- BharatVoice backend running (see main README.md)

### Installation Steps

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/your-org/bharatvoice-ai.git
   cd bharatvoice-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.streamlit.example .env
   
   # Edit .env and set your configuration
   # For local development, the defaults should work:
   # BACKEND_URL=http://localhost:8000
   # DEBUG=true
   ```

4. **Start the BharatVoice backend** (in a separate terminal):
   ```bash
   cd src/bharatvoice
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

6. **Access the application**:
   - Open your browser and navigate to: `http://localhost:8501`
   - The app should now be running and connected to the backend

### Development Tips

- **Hot Reload**: Streamlit automatically reloads when you save changes to `app.py`
- **Debug Mode**: Set `DEBUG=true` in `.env` to see debug information in the sidebar
- **Clear Cache**: Press `C` in the Streamlit app to clear the cache
- **View Logs**: Check the terminal where you ran `streamlit run app.py` for logs

---

## AWS Amplify Deployment

AWS Amplify provides a simple way to deploy and host the Streamlit web interface with automatic CI/CD.

### Prerequisites

- AWS Account
- Git repository (GitHub, GitLab, or Bitbucket)
- BharatVoice backend deployed and accessible via HTTPS

### Deployment Steps

1. **Prepare your repository**:
   - Ensure `amplify.yml` is in the root directory
   - Ensure `requirements.txt` includes all dependencies
   - Commit and push all changes to your repository

2. **Connect to AWS Amplify**:
   - Log in to the [AWS Amplify Console](https://console.aws.amazon.com/amplify/)
   - Click "New app" → "Host web app"
   - Select your Git provider and authorize access
   - Choose your repository and branch

3. **Configure build settings**:
   - Amplify should automatically detect the `amplify.yml` file
   - Review the build settings and make any necessary adjustments
   - Click "Next"

4. **Set environment variables**:
   - In the "Environment variables" section, add:
     ```
     BACKEND_URL=https://api.bharatvoice.example.com
     DEBUG=false
     CACHE_TTL=3600
     REQUEST_TIMEOUT=30
     ```
   - Replace `https://api.bharatvoice.example.com` with your actual backend URL

5. **Deploy**:
   - Click "Save and deploy"
   - Amplify will build and deploy your application
   - Monitor the build logs for any errors

6. **Configure custom domain** (optional):
   - In the Amplify Console, go to "Domain management"
   - Click "Add domain"
   - Follow the instructions to configure your custom domain

7. **Test the deployment**:
   - Once deployment is complete, click the provided URL
   - Test all features: audio upload, recording, transcription, response generation

### Continuous Deployment

- Amplify automatically rebuilds and redeploys when you push changes to your repository
- You can configure branch-specific deployments for staging and production

---

## Docker Deployment

Docker provides a containerized deployment option that works across different environments.

### Prerequisites

- Docker installed (version 20.10 or higher)
- Docker Compose installed (version 1.29 or higher)
- BharatVoice backend Docker image

### Deployment Steps

1. **Build the Streamlit Docker image**:
   ```bash
   docker build -f Dockerfile.streamlit -t bharatvoice-streamlit:latest .
   ```

2. **Run with Docker Compose**:
   ```bash
   # Start both Streamlit and backend
   docker-compose -f docker-compose.streamlit.yml up -d
   ```

3. **Access the application**:
   - Streamlit UI: `http://localhost:8501`
   - Backend API: `http://localhost:8000`

4. **View logs**:
   ```bash
   # View Streamlit logs
   docker-compose -f docker-compose.streamlit.yml logs -f streamlit
   
   # View backend logs
   docker-compose -f docker-compose.streamlit.yml logs -f backend
   ```

5. **Stop the containers**:
   ```bash
   docker-compose -f docker-compose.streamlit.yml down
   ```

### Production Docker Deployment

For production, you may want to use a reverse proxy (like Nginx) and enable HTTPS:

1. **Create a production docker-compose file** (`docker-compose.prod.yml`):
   ```yaml
   version: '3.8'
   
   services:
     streamlit:
       image: bharatvoice-streamlit:latest
       environment:
         - BACKEND_URL=https://api.bharatvoice.example.com
         - DEBUG=false
         - CACHE_TTL=7200
         - REQUEST_TIMEOUT=60
       restart: always
       networks:
         - bharatvoice-network
     
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf:ro
         - ./ssl:/etc/nginx/ssl:ro
       depends_on:
         - streamlit
       restart: always
       networks:
         - bharatvoice-network
   
   networks:
     bharatvoice-network:
       driver: bridge
   ```

2. **Deploy**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

---

## Environment Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BACKEND_URL` | URL of the BharatVoice backend API | `http://localhost:8000` | Yes |
| `DEBUG` | Enable debug mode (shows debug info in sidebar) | `false` | No |
| `CACHE_TTL` | Cache time-to-live in seconds | `3600` | No |
| `REQUEST_TIMEOUT` | API request timeout in seconds | `30` | No |

### Streamlit Configuration

The `.streamlit/config.toml` file contains Streamlit-specific configuration:

```toml
[server]
port = 8501
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 10  # Maximum file upload size in MB

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Production Configuration Checklist

- [ ] Set `BACKEND_URL` to production backend URL (HTTPS)
- [ ] Set `DEBUG=false`
- [ ] Configure appropriate `CACHE_TTL` for your use case
- [ ] Set `REQUEST_TIMEOUT` based on expected backend response times
- [ ] Enable XSRF protection in `.streamlit/config.toml`
- [ ] Configure CORS settings appropriately
- [ ] Set up SSL/TLS certificates for HTTPS
- [ ] Configure logging and monitoring
- [ ] Set up backup and disaster recovery

---

## Troubleshooting

### Common Issues

#### 1. Backend Connection Failed

**Symptoms**: "Backend unavailable" message, offline mode indicator

**Solutions**:
- Verify `BACKEND_URL` is correct in `.env`
- Check that the backend is running: `curl http://localhost:8000/api/health`
- Check firewall settings
- Verify network connectivity

#### 2. Audio Recording Not Working

**Symptoms**: Recording button doesn't work, no audio captured

**Solutions**:
- Ensure you're using HTTPS (required for microphone access in browsers)
- Check browser permissions for microphone access
- Verify `audio-recorder-streamlit` is installed: `pip show audio-recorder-streamlit`
- Try a different browser (Chrome/Edge recommended)

#### 3. File Upload Fails

**Symptoms**: Error when uploading audio files

**Solutions**:
- Check file size (must be under 10MB)
- Verify file format (WAV, MP3, M4A, OGG only)
- Check `maxUploadSize` in `.streamlit/config.toml`
- Verify backend file size limits

#### 4. Cache Not Working

**Symptoms**: Responses not cached, slow performance

**Solutions**:
- Verify `.streamlit_cache` directory exists and is writable
- Check `CACHE_TTL` configuration
- Clear cache manually: delete `.streamlit_cache` directory
- Check disk space

#### 5. Deployment Fails on AWS Amplify

**Symptoms**: Build fails, deployment errors

**Solutions**:
- Check build logs in Amplify Console
- Verify `amplify.yml` is correct
- Ensure all dependencies are in `requirements.txt`
- Check Python version compatibility (3.9+)
- Verify environment variables are set correctly

### Debug Mode

Enable debug mode to see detailed information:

1. Set `DEBUG=true` in `.env`
2. Restart the Streamlit app
3. Check the sidebar for debug information:
   - Backend URL
   - Cache TTL
   - Request timeout
   - Online status
   - Audio data status
   - Processing status

### Logs

**Local Development**:
- Streamlit logs: Check the terminal where you ran `streamlit run app.py`
- Backend logs: Check the terminal where you ran the backend

**Docker**:
```bash
# Streamlit logs
docker logs bharatvoice-streamlit

# Backend logs
docker logs bharatvoice-backend
```

**AWS Amplify**:
- Build logs: Available in the Amplify Console under "Build history"
- Runtime logs: Available in CloudWatch Logs

### Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md) for general information
2. Review the [API Documentation](API_DOCUMENTATION.md)
3. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
4. Open an issue on GitHub with:
   - Description of the problem
   - Steps to reproduce
   - Error messages and logs
   - Environment details (OS, Python version, etc.)

---

## Performance Optimization

### Tips for Better Performance

1. **Caching**:
   - Increase `CACHE_TTL` for frequently accessed data
   - Use Streamlit's `@st.cache_data` decorator for expensive operations

2. **Network**:
   - Deploy Streamlit and backend in the same region
   - Use a CDN for static assets
   - Enable compression in your reverse proxy

3. **Resource Limits**:
   - Set appropriate `maxUploadSize` in config.toml
   - Configure request timeouts based on expected response times
   - Monitor memory usage and adjust container limits if needed

4. **Monitoring**:
   - Set up application monitoring (e.g., New Relic, Datadog)
   - Monitor API response times
   - Track error rates and user sessions

---

## Security Best Practices

1. **HTTPS**: Always use HTTPS in production
2. **Environment Variables**: Never commit `.env` files to version control
3. **CORS**: Configure CORS appropriately for your domain
4. **XSRF Protection**: Keep `enableXsrfProtection=true` in production
5. **Input Validation**: The app validates file sizes and formats
6. **API Keys**: Store sensitive keys in AWS Secrets Manager or similar
7. **Rate Limiting**: Implement rate limiting on the backend
8. **Regular Updates**: Keep dependencies up to date

---

## Next Steps

After successful deployment:

1. Test all features thoroughly
2. Set up monitoring and alerting
3. Configure backup and disaster recovery
4. Document any custom configuration
5. Train users on how to use the interface
6. Gather feedback and iterate

For more information, see:
- [User Guide](USER_GUIDES.md)
- [API Documentation](API_DOCUMENTATION.md)
- [Developer Documentation](DEVELOPER_DOCUMENTATION.md)
