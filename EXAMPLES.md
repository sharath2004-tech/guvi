# Example API Requests

## Valid Request

```bash
curl -X POST "http://localhost:8000/detect-voice" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "message": "Testing with sample audio"
  }'
```

## Expected Response

```json
{
  "classification": "Human-generated",
  "confidence": 0.6234,
  "explanation": "Analysis based on: natural pitch variation detected, varied MFCC patterns suggesting human speech, natural spectral variations present, natural zero-crossing variations, spectral centroid shows natural distribution. Metrics: pitch_var=1523.45, mfcc_std=18.92, spectral_rolloff_std=678.34"
}
```

## Error Cases

### Missing Authorization Header

```bash
curl -X POST "http://localhost:8000/detect-voice" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.mp3",
    "message": "Test"
  }'
```

Response:
```json
{
  "detail": "Authorization header missing"
}
```

### Invalid API Key

```bash
curl -X POST "http://localhost:8000/detect-voice" \
  -H "Authorization: Bearer wrong-key" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/audio.mp3",
    "message": "Test"
  }'
```

Response:
```json
{
  "detail": "Invalid API key"
}
```

### Invalid Audio URL

```bash
curl -X POST "http://localhost:8000/detect-voice" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/notfound.mp3",
    "message": "Test"
  }'
```

Response:
```json
{
  "detail": "Failed to download audio: 404 Not Found"
}
```

## Python Example

```python
import requests
import json

API_URL = "http://localhost:8000/detect-voice"
API_KEY = "your-secret-api-key-here"

# Test with different audio samples
test_cases = [
    {
        "audio_url": "https://example.com/human-voice.mp3",
        "message": "Testing human voice sample"
    },
    {
        "audio_url": "https://example.com/ai-voice.mp3",
        "message": "Testing AI-generated voice sample"
    }
]

for test in test_cases:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=test
    )
    
    print(f"\nTest: {test['message']}")
    print(f"Status: {response.status_code}")
    print(f"Result: {json.dumps(response.json(), indent=2)}")
```

## JavaScript Example

```javascript
const API_URL = "http://localhost:8000/detect-voice";
const API_KEY = "your-secret-api-key-here";

async function detectVoice(audioUrl, message) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      audio_url: audioUrl,
      message: message
    })
  });
  
  const result = await response.json();
  console.log(result);
  return result;
}

// Usage
detectVoice(
  "https://example.com/sample.mp3",
  "Testing voice detection"
);
```
