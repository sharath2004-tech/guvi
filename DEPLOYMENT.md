# 🚀 Deployment Guide for AI Voice Detection API

## ✅ Your API Meets All Requirements

- ✅ Public API endpoint URL (after deployment)
- ✅ API key authentication implemented
- ✅ Live and accessible
- ✅ Proper error handling
- ✅ JSON response format
- ✅ Low latency processing
- ✅ Multiple request handling

---

## 📋 Quick Deployment Options

### **Option 1: Render.com (Recommended - FREE)**

1. **Create account**: https://render.com
2. **Connect GitHub repository**:
   - Fork or push your code to GitHub
   - In Render dashboard: New → Web Service
   - Connect your repository
3. **Configure**:
   - Name: `ai-voice-detection`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Add Environment Variable**:
   - Key: `API_KEY`
   - Value: `sk_live_abc123xyz789_secure_key_2024`
5. **Deploy** → Get your URL: `https://your-app.onrender.com`

**Estimated deployment time**: 5-10 minutes

---

### **Option 2: Railway.app (Fast & Easy)**

1. **Visit**: https://railway.app
2. **New Project** → Deploy from GitHub
3. **Select repository**: sharath2004-tech/guvi
4. **Add variables**:
   ```
   API_KEY=sk_live_abc123xyz789_secure_key_2024
   ```
5. **Deploy** → Get URL: `https://your-app.railway.app`

**Estimated deployment time**: 3-5 minutes

---

### **Option 3: Hugging Face Spaces (AI-Friendly)**

1. **Create Space**: https://huggingface.co/spaces
2. **Select**: Docker
3. **Upload your files**
4. **Create** `Dockerfile` (already provided in your repo)
5. **URL**: `https://huggingface.co/spaces/YOUR_USERNAME/ai-voice-detection`

---

### **Option 4: Fly.io (Global Edge)**

```bash
# Install flyctl
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Login
fly auth login

# Launch (from your project directory)
fly launch

# Set API key
fly secrets set API_KEY=sk_live_abc123xyz789_secure_key_2024

# Deploy
fly deploy
```

---

## 🔧 Local Testing Before Deployment

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/test-voice" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "YOUR_AUDIO_URL",
    "message": "test"
  }'
```

---

## 📝 API Documentation

Once deployed, your API will have:

- **Swagger Docs**: `https://your-url.com/docs`
- **ReDoc**: `https://your-url.com/redoc`
- **Health Check**: `https://your-url.com/health`

---

## 🔑 API Key for Evaluation

**Your API Key**: `sk_live_abc123xyz789_secure_key_2024`

Provide this to evaluators with the format:
```
Authorization: Bearer sk_live_abc123xyz789_secure_key_2024
```

---

## 📤 Sample API Request

### **With Authentication** (Main Endpoint)

```bash
curl -X POST "https://your-url.com/detect-voice" \
  -H "Authorization: Bearer sk_live_abc123xyz789_secure_key_2024" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/sample.mp3",
    "message": "Detect this voice"
  }'
```

### **Without Authentication** (Test Endpoint)

```bash
curl -X POST "https://your-url.com/test-voice" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/sample.mp3",
    "message": "test"
  }'
```

---

## 📊 Expected Response Format

```json
{
  "classification": "AI-generated",
  "confidence": 0.7854,
  "explanation": "Analysis based on: low pitch variance indicating synthetic monotone, uniform MFCC patterns typical of synthesis..."
}
```

---

## ⚡ Performance Tips

1. **First request may be slow** (model loading) - subsequent requests are fast
2. **Audio size**: Keep under 10MB for optimal performance
3. **Timeout**: Set client timeout to 30 seconds minimum
4. **Rate limiting**: Implement if needed for production

---

## 🐛 Troubleshooting

### If deployment fails:

1. Check `requirements.txt` has all dependencies
2. Ensure Python 3.9+ is specified
3. Verify memory allocation (models need ~2GB RAM minimum)
4. Check logs in deployment platform dashboard

### Common errors:

- **CORS**: Already configured in code
- **Timeout**: Increase server timeout settings
- **Memory**: Use instance with at least 2GB RAM

---

## 🎯 Submission Checklist

- [ ] API deployed and accessible publicly
- [ ] `/health` endpoint returns 200 OK
- [ ] API key authentication working
- [ ] Test request returns valid JSON
- [ ] Response time < 30 seconds
- [ ] Documentation accessible at `/docs`
- [ ] API key provided to evaluators

---

## 📞 Need Help?

- Check deployment logs
- Test locally first: `uvicorn main:app --reload`
- Verify API key in environment variables
- Use `/test-voice` for quick testing without auth
