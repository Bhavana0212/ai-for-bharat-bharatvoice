# BharatVoice AI - Streamlit Web Interface Quick Start Guide

## Get Started in 5 Minutes! 🚀

This guide will help you get the Streamlit web interface up and running quickly.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- BharatVoice backend (see main README.md)

## Step 1: Install Dependencies

```bash
pip install streamlit requests audio-recorder-streamlit python-dotenv
```

## Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.streamlit.example .env

# The default settings should work for local development:
# BACKEND_URL=http://localhost:8000
# DEBUG=true
```

## Step 3: Start the Backend

In a separate terminal window:

```bash
cd src/bharatvoice
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Wait for the message: "Application startup complete."

## Step 4: Run the Streamlit App

```bash
streamlit run app.py
```

You should see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## Step 5: Use the Application

1. **Open your browser** to `http://localhost:8501`

2. **Select a language** from the dropdown (default: Hindi)

3. **Upload or record audio**:
   - Click "Browse files" to upload an audio file (WAV, MP3, M4A, or OGG)
   - OR click the microphone icon to record audio

4. **Process the audio**:
   - Click the "🎯 Process Audio" button
   - Wait for the transcription, response, and TTS audio

5. **View results**:
   - Transcription appears in the blue info box
   - AI response appears in the green success box
   - Audio player appears below for TTS playback
   - Action log shows in the sidebar

## Troubleshooting

### Backend Connection Failed

**Problem**: "Backend unavailable" message appears

**Solution**:
1. Check that the backend is running: `curl http://localhost:8000/api/health`
2. Verify `BACKEND_URL` in `.env` is set to `http://localhost:8000`
3. Check firewall settings

### Audio Recording Not Working

**Problem**: Recording button doesn't work

**Solution**:
1. For local development, use `http://localhost:8501` (not HTTPS)
2. Check browser permissions for microphone access
3. Try Chrome or Edge browser

### File Upload Fails

**Problem**: Error when uploading audio file

**Solution**:
1. Check file size (must be under 10MB)
2. Verify file format (WAV, MP3, M4A, or OGG only)
3. Try a different audio file

## Enable Debug Mode

To see detailed debug information:

1. Edit `.env` and set `DEBUG=true`
2. Restart the Streamlit app
3. Check the sidebar for debug information

## Next Steps

- Read `STREAMLIT_IMPLEMENTATION_SUMMARY.md` for complete feature list
- Read `STREAMLIT_DEPLOYMENT_GUIDE.md` for production deployment
- Run tests: `pytest tests/`
- Explore the code in `app.py`

## Quick Commands Reference

```bash
# Start backend
cd src/bharatvoice && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
streamlit run app.py

# Run tests
pytest tests/ -v

# Check backend health
curl http://localhost:8000/api/health

# View logs
tail -f streamlit_app.log
```

## Support

For more help:
- See `STREAMLIT_DEPLOYMENT_GUIDE.md` for detailed troubleshooting
- Check `TROUBLESHOOTING.md` for common issues
- Review `API_DOCUMENTATION.md` for API details

---

**Enjoy using BharatVoice AI! 🎙️**
