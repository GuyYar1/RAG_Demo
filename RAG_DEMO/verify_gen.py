from ollama_service import generate_response
import time

def verify():
    context = "Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina)."
    query = "What is the cause?"
    
    print("Generating response...")
    start = time.time()
    response = generate_response(context, query)
    duration = time.time() - start
    
    with open("gen_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Duration: {duration:.2f}s\n")
        f.write("-" * 20 + "\n")
        f.write(response)
        f.write("\n" + "-" * 20 + "\n")

if __name__ == "__main__":
    verify()
