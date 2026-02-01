"""
Test Script for AI Voice Detection API
Run this to verify your API before deployment
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your deployed URL
API_KEY = "sk_live_abc123xyz789_secure_key_2024"

# Test audio URL (public accessible audio file)
TEST_AUDIO_URL = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav"

def test_health_check():
    """Test health endpoint"""
    print("\n🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print("\n🔍 Testing / endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_without_auth():
    """Test endpoint without authentication"""
    print("\n🔍 Testing /test-voice (no auth)...")
    try:
        payload = {
            "audio_url": TEST_AUDIO_URL,
            "message": "Test without authentication"
        }
        response = requests.post(
            f"{BASE_URL}/test-voice",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_invalid_auth():
    """Test with invalid API key"""
    print("\n🔍 Testing /detect-voice with invalid auth...")
    try:
        payload = {
            "audio_url": TEST_AUDIO_URL,
            "message": "Test with invalid key"
        }
        headers = {
            "Authorization": "Bearer INVALID_KEY",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{BASE_URL}/detect-voice",
            json=payload,
            headers=headers
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 403:
            print("✅ Correctly rejected invalid API key")
            return True
        else:
            print("❌ Should have returned 403")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_with_valid_auth():
    """Test with valid API key"""
    print("\n🔍 Testing /detect-voice with valid auth...")
    try:
        payload = {
            "audio_url": TEST_AUDIO_URL,
            "message": "Test with valid authentication"
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{BASE_URL}/detect-voice",
            json=payload,
            headers=headers,
            timeout=60
        )
        print(f"✅ Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Validate response structure
        required_fields = ["classification", "confidence", "explanation"]
        if all(field in result for field in required_fields):
            print("✅ Response has all required fields")
            return True
        else:
            print("❌ Response missing required fields")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_missing_fields():
    """Test with missing required fields"""
    print("\n🔍 Testing with missing fields...")
    try:
        payload = {
            "message": "Missing audio_url"
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{BASE_URL}/detect-voice",
            json=payload,
            headers=headers
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 422:
            print("✅ Correctly rejected invalid request")
            return True
        else:
            print("❌ Should have returned 422")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🚀 AI Voice Detection API Test Suite")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("No Authentication", test_without_auth),
        ("Invalid API Key", test_with_invalid_auth),
        ("Valid API Key", test_with_valid_auth),
        ("Missing Fields", test_missing_fields),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is ready for deployment!")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before deployment.")

if __name__ == "__main__":
    print("\n⚙️ Make sure your API is running:")
    print("   uvicorn main:app --reload\n")
    
    choice = input("Is your API running? (y/n): ").lower()
    if choice == 'y':
        run_all_tests()
    else:
        print("\n📝 Start your API first with:")
        print("   uvicorn main:app --reload")
