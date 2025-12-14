import json
from app import app
import sys

def test_chat_endpoint():
    print("\n--- Starting E2E Test ---")
    client = app.test_client()
    
    # 1. Test Hello
    resp = client.get('/')
    assert resp.data == b"Hello, World!"
    print("ROOT endpoint: OK")
    
    # 2. Test Chat (Retrieval + Generation)
    print("Testing /chat endpoint (this calls Ollama)...")
    payload = {"message": "What is diabetic retinopathy?"}
    
    try:
        response = client.post('/chat', 
                               data=json.dumps(payload),
                               content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        data = json.loads(response.data)
        
        if response.status_code == 200:
            print("Response received:")
            print(data.get('response', 'No response field'))
            if 'response' in data and len(data['response']) > 10:
                print("CHAT endpoint: OK")
            else:
                print("CHAT endpoint: Response too short or missing")
                sys.exit(1)
        else:
            print(f"CHAT endpoint FAILED: {data}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Test FAILED with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_chat_endpoint()
