# AI-Generated Voice Detection API

Multi-language voice detection system to identify AI-generated vs human-generated speech.

## Features

- **Languages Supported**: Tamil, English, Hindi, Malayalam, Telugu (language-independent detection)
- **Authentication**: Bearer token via Authorization header
- **Audio Processing**: Downloads MP3 from URL, extracts features using wav2vec2 + MFCC
- **Detection Method**: Heuristic-based analysis of pitch variance, spectral characteristics, and MFCC patterns

## Installation

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key in .env file
echo "API_KEY=your-secret-api-key-here" > .env

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage

### Endpoint

```
POST /detect-voice
```

### Headers

```
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

### Request Body

```json
{
  "audio_url": "https://example.com/audio.mp3",
  "message": "Test request for voice detection"
}
```

### Response

```json
{
  "classification": "AI-generated",
  "confidence": 0.75,
  "explanation": "Analysis based on: low pitch variance indicating synthetic monotone, uniform MFCC patterns typical of synthesis. Metrics: pitch_var=850.23, mfcc_std=12.45, spectral_rolloff_std=420.67"
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/detect-voice" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/sample.mp3",
    "message": "Testing endpoint"
  }'
```

### Example Python Request

```python
import requests

response = requests.post(
    "http://localhost:8000/detect-voice",
    headers={"Authorization": "Bearer your-secret-api-key-here"},
    json={
        "audio_url": "https://example.com/sample.mp3",
        "message": "Test request"
    }
)

print(response.json())
```

## Error Handling

### 401 Unauthorized
```json
{"detail": "Authorization header missing"}
```

### 403 Forbidden
```json
{"detail": "Invalid API key"}
```

### 400 Bad Request
```json
{"detail": "URL does not point to an audio file"}
```

## Deployment

### Option 1: Render (Recommended - Free Tier)

1. Create account at [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add `API_KEY=your-secret-key`
5. Deploy

### Option 2: Railway

1. Create account at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select repository
4. Add environment variable: `API_KEY=your-secret-key`
5. Railway auto-detects Python and deploys

### Option 3: Heroku

```bash
# Install Heroku CLI
heroku login
heroku create your-app-name

# Add Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Set API key
heroku config:set API_KEY=your-secret-api-key-here

# Deploy
git push heroku main
```

### Option 4: AWS Lambda + API Gateway

1. Install serverless framework: `npm install -g serverless`
2. Create `serverless.yml`:

```yaml
service: voice-detection-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1

functions:
  api:
    handler: main.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
    environment:
      API_KEY: ${env:API_KEY}
```

3. Add Mangum adapter to requirements.txt: `mangum==0.17.0`
4. Modify main.py to add handler:

```python
from mangum import Mangum
handler = Mangum(app)
```

5. Deploy: `serverless deploy`

## Technical Details

### Audio Feature Extraction

1. **Download**: Fetches MP3 from provided URL
2. **Conversion**: Converts to 16kHz mono waveform using librosa
3. **Feature Extraction**:
   - MFCC (Mel-frequency cepstral coefficients)
   - Spectral centroid & rolloff
   - Zero-crossing rate
   - Pitch variance
   - Wav2Vec2 embeddings

### Detection Logic

AI-generated voices typically exhibit:
- **Lower pitch variance**: More monotone delivery
- **Uniform MFCC patterns**: Less natural variation
- **Smooth spectral characteristics**: Fewer artifacts
- **Consistent zero-crossing rates**: Less natural fluctuation

The system scores these characteristics and classifies based on threshold (>0.5 = AI-generated).

### Model Used

- **facebook/wav2vec2-base**: Pre-trained transformer model for audio feature extraction
- **Open-source**: No paid APIs required
- **Language-independent**: Works across Tamil, English, Hindi, Malayalam, Telugu

## Testing

Run the test script:

```bash
python test_api.py
```

Or use the provided cURL/Python examples above.

## Production Considerations

1. **Model Improvement**: Replace heuristic scoring with trained classifier
2. **Caching**: Add Redis for repeated audio URLs
3. **Rate Limiting**: Implement request throttling
4. **Logging**: Add structured logging for monitoring
5. **Async Processing**: Use background tasks for large files
6. **Model Optimization**: Use quantized models for faster inference

## License

MIT
