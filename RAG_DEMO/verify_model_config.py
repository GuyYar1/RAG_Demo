import requests
import json
import time

def test_model(model_name):
    url = "http://127.0.0.1:5000/chat"
    payload = {
        "message": "What is diabetic retinopathy?",
        "model": model_name
    }
    
    print(f"Requesting with model: '{model_name}'...")
    start = time.time()
    try:
        response = requests.post(url, json=payload)
        duration = time.time() - start
        
        if response.status_code == 200:
            print(f" -> Success! Time: {duration:.2f}s")
            print(f" -> Response length: {len(response.json()['response'])} chars")
        else:
            print(f" -> Error: {response.status_code}")
    except Exception as e:
        print(f" -> Exception: {e}")

if __name__ == "__main__":
    test_model("tinyllama") # Should be fast (~10-30s)
    # test_model("mistral") # Should be slow (~240s) - skipped for speed
