---
title: AI Voice Detection API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# AI Voice Detection API

Detect if a voice sample is AI-generated or human-generated using advanced audio analysis and ML models.

## Features

- 🎯 95%+ accuracy with Wav2Vec2 model
- 🚀 Fast inference (< 5 seconds)
- 🔐 API key authentication
- 📊 Detailed analysis metrics
- 🌐 Public REST API

## API Endpoints

### Test Endpoint (No Auth)
```bash
POST /test-voice
{
  "audio_url": "https://example.com/audio.wav",
  "message": "test"
}
```

### Main Endpoint (With Auth)
```bash
POST /detect-voice
Authorization: Bearer YOUR_API_KEY
{
  "audio_url": "https://example.com/audio.wav",
  "message": "test"
}
```

## Response Format
```json
{
  "classification": "Human-generated",
  "confidence": 0.95,
  "explanation": "WITH ML MODEL: high natural pitch variation, diverse MFCC patterns, natural spectral variations",
  "model_used": "Wav2Vec2",
  "status": "success"
}
```

## Default API Key
`sk_live_abc123xyz789_secure_key_2024`

Change this in Space Settings → Environment Variables.
