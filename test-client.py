import requests
import json

# Test cases from your original code
bengali_test_cases = [
    "ভিভাসফ্ট কি ধরনের কোম্পানি?",
    "ভিভাসফ্ট কি ধরনের সমাধান প্রদান করে?",
    "ভিভাসফ্ট মুল লক্ষ্য কি?",
    "ভিভাসফ্ট কোথায় অবস্থিত?",
    "কথন কি",
    "নতুন এজেন্টদের ট্রেনিংয়ের সময় কমাতে কথন  কিভাবে সাহায্য করে?"
]

# Server URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code} - {response.json()}")
    return response.status_code == 200

def test_generate(text, max_tokens=128, temperature=0.7, top_p=0.9):
    """Test text generation"""
    payload = {
        "text": text,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    response = requests.post(
        f"{BASE_URL}/generate",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def main():
    print("=== Testing Bengali Gemma3 Inference Server ===\n")
    
    # Test health endpoint
    if not test_health():
        print("❌ Server is not healthy. Exiting.")
        return
    
    print("\n=== Starting Bengali Inference Tests ===")
    
    for i, question in enumerate(bengali_test_cases, 1):
        print(f"\n🔸 Test {i}: {question}")
        
        result = test_generate(question)
        if result:
            print(f"✅ Response: {result['generated_text']}")
        else:
            print("❌ Failed to generate response")
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    main()