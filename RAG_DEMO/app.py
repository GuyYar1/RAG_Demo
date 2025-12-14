from flask import Flask, request, jsonify, session
from flask_session import Session
from model_service import get_query_embedding
from text_generator import generate_text_in_chunks, rerank_response, extract_sentences_with_keyword
import logging
import numpy as np
from simple_vector_store import SimpleVectorStore
from data_loader import preprocess_documents

print("Configure logging")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('flask_app.log'), logging.StreamHandler()])

print("Initialize the flask app")
# Initialize the flask app
app = Flask(__name__)
app.secret_key = '@@GYyour_secret_key123%@@'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

print("Load data and models")

# Initialize Vector Store
vector_store = SimpleVectorStore()
if len(vector_store.documents) == 0:
    print("Vector Store empty, running initial data processing...")
    vector_store = preprocess_documents()
else:
    print(f"Vector Store loaded with {len(vector_store.documents)} documents.")

# Define the maximum sequence length (kept for compatibility)
MAX_LENGTH = 512

print("Waiting for request messages .... ")

# Endpoints:
@app.route('/chat', methods=['POST'])
def chat():
    print("got chat a request messages ....")
    user_message = request.json.get('message')
    target_model = request.json.get('model', 'mistral') # Default to mistral if not specified
    logging.info(f"User received message: {user_message} (Model: {target_model})")

    NeedSortRerank = False
    if "@KEY" in user_message:
        user_message = user_message.strip()
        user_message = user_message.replace("@KEY", "")
        NeedSortRerank = True

    if 'conversation' not in session:
        session['conversation'] = []

    # Update conversation history
    session['conversation'].append({'role': 'user', 'content': user_message})
    try:
        # Convert query to embedding (using Ollama via model_service)
        query_embedding = get_query_embedding(user_message)

        # Retrieval from SimpleVectorStore
        # query expects list of arrays
        results = vector_store.query([query_embedding], n_results=5)
        
        # Flatten documents list (results['documents'] is list of list of strings)
        retrieved_docs = results['documents'][0]
        
        # Concatenate retrieved documents
        input_text = " ".join(retrieved_docs)
        
        logging.info(f"Retrieved {len(retrieved_docs)} chunks for context.")

        # Generate text response using Ollama (via text_generator wrapper)
        generated_text = generate_text_in_chunks(input_text, user_message, model=target_model, max_length=MAX_LENGTH)

        # Reranking/Filtering logic (Legacy feature kept)
        if NeedSortRerank:
            print("NeedSortRerank")
            # This function logic might be less relevant with LLM but kept for structure
            # Ideally LLM handles this, but we'll print findings
            results = extract_sentences_with_keyword(retrieved_docs, user_message)
            for result in results:
                doc_index = result['doc_id']
                sentence = result['sentence']
                print(f"Document Index: {doc_index} - Sentence: {sentence}")
        
        logging.info(f"Generated response: {generated_text}")

        # Store bot response in session
        session['conversation'].append({'role': 'system', 'content': generated_text})

        return jsonify({'response': generated_text})

    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return {"error": "Internal Server Error"}, 500

@app.route('/reset', methods=['POST'])
def reset_session():
    print("got reset a request messages ....")
    session.clear()
    return jsonify({'status': 'Session reset.'})


@app.route('/history', methods=['GET'])
def get_history():
    if 'conversation' not in session:
        return jsonify({'history': []})
    return jsonify({'history': session['conversation']})

@app.route('/')
def hello():
    print("App is running")
    return "Hello, World!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
