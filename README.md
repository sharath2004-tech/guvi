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

Multi-language voice detection system to identify AI-generated vs human-generated speech.

## API Usage

### Endpoint
```
POST /detect-voice
```

### Headers
```
Authorization: Bearer sk_live_abc123xyz789_secure_key_2024
Content-Type: application/json
```

### Request Body
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "message": "Test request"
}
```

### Response
```json
{
  "classification": "AI-generated",
  "confidence": 0.75,
  "explanation": "Analysis based on audio features..."
}
```

## Test the API

Your API will be available at: `https://your-username-ai-voice-detection-api.hf.space`