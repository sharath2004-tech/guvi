# 📋 Evaluation Submission Checklist

## 🎯 SUBMISSION REQUIREMENTS

### ✅ 1. Public API Endpoint URL
- [ ] API deployed to public hosting
- [ ] URL is accessible from anywhere
- [ ] Format: `https://your-app-name.platform.com`

**Your URL (fill after deployment):**
```
https://_____________________________.com
```

---

### ✅ 2. API Endpoints Required

#### Main Detection Endpoint (WITH AUTH):
```
POST https://your-url.com/detect-voice
```

#### Test Endpoint (NO AUTH):
```
POST https://your-url.com/test-voice
```

#### Health Check:
```
GET https://your-url.com/health
```

---

### ✅ 3. API Key Authentication

**API Key to Provide:**
```
sk_live_abc123xyz789_secure_key_2024
```

**Header Format:**
```
Authorization: Bearer sk_live_abc123xyz789_secure_key_2024
```

- [ ] API key is set in environment variables
- [ ] Authentication returns 403 for invalid keys
- [ ] Authentication works with valid key

---

### ✅ 4. Request Format

```json
{
  "audio_url": "https://example.com/audio.mp3",
  "message": "Detect this voice"
}
```

- [ ] Accepts `audio_url` (required)
- [ ] Accepts `message` (required)
- [ ] Validates URL format
- [ ] Handles audio download

---

### ✅ 5. Response Format

```json
{
  "classification": "AI-generated",
  "confidence": 0.7854,
  "explanation": "Analysis based on: low pitch variance..."
}
```

**Required Fields:**
- [ ] `classification` (string: "AI-generated" or "Human-generated")
- [ ] `confidence` (float: 0.0 to 1.0)
- [ ] `explanation` (string: detailed analysis)

---

### ✅ 6. Error Handling

- [ ] Returns proper HTTP status codes
  - 200: Success
  - 400: Bad request
  - 403: Invalid API key
  - 422: Validation error
  - 500: Server error

- [ ] Error responses include details:
```json
{
  "detail": "Error message here"
}
```

---

### ✅ 7. Performance Requirements

- [ ] Response time < 30 seconds (typically 5-15s)
- [ ] Handles multiple concurrent requests
- [ ] Stable under load
- [ ] Low latency for health checks

---

### ✅ 8. Documentation

- [ ] Swagger UI available at `/docs`
- [ ] ReDoc available at `/redoc`
- [ ] Clear API description
- [ ] Request/response examples shown

---

### ✅ 9. Testing Before Submission

Run local tests:
```bash
python test_api.py
```

Manual test with curl:
```bash
curl -X POST "https://your-url.com/detect-voice" \
  -H "Authorization: Bearer sk_live_abc123xyz789_secure_key_2024" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav",
    "message": "test"
  }'
```

- [ ] Health check responds with 200
- [ ] API accepts test request
- [ ] Returns valid JSON
- [ ] All required fields present
- [ ] No errors in logs

---

### ✅ 10. Deployment Checklist

Platform chosen: ________________

- [ ] Code pushed to GitHub
- [ ] Environment variables set
- [ ] Build successful
- [ ] Deployment successful
- [ ] URL is accessible
- [ ] No errors in deployment logs

---

## 📝 SUBMISSION INFORMATION

**To Submit:**

1. **API Endpoint URL:**
   ```
   https://_____________________________.com/detect-voice
   ```

2. **API Key:**
   ```
   sk_live_abc123xyz789_secure_key_2024
   ```

3. **Documentation URL:**
   ```
   https://_____________________________.com/docs
   ```

4. **Sample Request:**
   ```bash
   curl -X POST "https://YOUR-URL.com/detect-voice" \
     -H "Authorization: Bearer sk_live_abc123xyz789_secure_key_2024" \
     -H "Content-Type: application/json" \
     -d '{
       "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav",
       "message": "evaluation test"
     }'
   ```

5. **Expected Response Time:** 5-15 seconds

6. **Supports Multiple Requests:** Yes ✅

---

## 🚨 FINAL PRE-SUBMISSION CHECKS

- [ ] API is **LIVE** and accessible
- [ ] Tested from **different network/device**
- [ ] API key authentication **working**
- [ ] Response format **matches requirements**
- [ ] Error handling **working properly**
- [ ] Performance is **acceptable** (< 30s)
- [ ] No deployment errors
- [ ] Logs show successful requests

---

## 📞 TROUBLESHOOTING

If evaluators report issues:

1. **Check deployment logs** in platform dashboard
2. **Verify environment variables** are set
3. **Test endpoint** yourself from external network
4. **Check rate limits** on hosting platform
5. **Monitor resource usage** (RAM/CPU)

---

## ✅ READY TO SUBMIT?

Once all checkboxes are checked:
1. Fill in your deployed URL above
2. Test one final time
3. Submit your endpoint and API key
4. Monitor deployment during evaluation period

**Good luck! 🎉**
