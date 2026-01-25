import requests
import json

# Configuration
API_URL = "http://localhost:8000/detect-voice"
API_KEY = "your-secret-api-key-here"

# Test request
payload = {
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "message": "Testing voice detection endpoint"
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("Testing API Endpoint...")
print(f"URL: {API_URL}")
print(f"Payload: {json.dumps(payload, indent=2)}\n")

try:
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    
except Exception as e:
    print(f"Error: {str(e)}")
