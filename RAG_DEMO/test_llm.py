from ollama_service import generate_response
import time

print("Testing Ollama LLM Generation...")
context = "Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina)."
query = "What causes diabetic retinopathy?"

start = time.time()
response = generate_response(context, query)
end = time.time()

print(f"\nResponse ({end-start:.2f}s):")
print("-" * 50)
print(response)
print("-" * 50)
