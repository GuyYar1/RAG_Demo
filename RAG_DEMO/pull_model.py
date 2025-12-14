import ollama
import sys

model_name = "biomistral"
if len(sys.argv) > 1:
    model_name = sys.argv[1]

print(f"Pulling model '{model_name}' via Python API...")
try:
    # stream=True to see progress if possible, but simple pull is fine
    ollama.pull(model_name)
    print(f"Successfully pulled '{model_name}'")
except Exception as e:
    print(f"Failed to pull '{model_name}': {e}")
