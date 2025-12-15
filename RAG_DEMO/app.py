import logging
from flask import Flask, request, jsonify, session
from flask_session import Session
from model_service import get_query_embedding
from text_generator import generate_text_in_chunks, rerank_response, extract_sentences_with_keyword
from ollama_service import generate_response_with_validation
from simple_vector_store import SimpleVectorStore
from data_loader import preprocess_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])  # StreamHandler for console logs

# Ensure logs from external modules (like data_loader.py) are captured
logging.getLogger('data_loader').setLevel(logging.INFO)  # Adjust level as needed
logging.getLogger('flask').setLevel(logging.INFO)  # Ensure Flask logs are captured
logging.getLogger('werkzeug').setLevel(logging.INFO)  # Logs from Flask's built-in server

# Create a logger for this app
logger = logging.getLogger(__name__)

# Initialize the flask app
app = Flask(__name__)
app.secret_key = '@@GYyour_secret_key123%@@'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load data and models
vector_store = SimpleVectorStore()

# Check if vector store is empty and preprocess documents if needed
if len(vector_store.documents) == 0:
    logger.info("Vector Store empty, running initial data processing...")
    vector_store = preprocess_documents(vector_store)  # Pass the vector store to preprocess_documents
else:
    logger.info(f"Vector Store loaded with {len(vector_store.documents)} documents.")

# Define the maximum sequence length (kept for compatibility)
MAX_LENGTH = 512

logger.info("Waiting for request messages .... ")

def reorder_context_by_urgency(sorted_docs, query):
    """
    Reorder context chunks to place urgent medical information first.
    This forces the LLM to see critical information early in the context.
    """
    urgent_keywords = [
        'refer', 'referral', 'ophthalmologist', 'specialist',
        'severe', 'proliferative', 'PDR', 'NPDR',
        'PRP', 'panretinal photocoagulation', 'laser',
        'anti-VEGF', 'ranibizumab', 'aflibercept', 'bevacizumab',
        'urgent', 'immediate', 'within', 'weeks'
    ]
    
    query_lower = query.lower()
    
    # Separate docs into urgent and non-urgent
    urgent_docs = []
    routine_docs = []
    
    for doc_tuple in sorted_docs:
        doc_text = doc_tuple[0]["text"].lower()
        
        # Check if doc contains urgent keywords OR matches query terms
        is_urgent = False
        for keyword in urgent_keywords:
            if keyword in doc_text:
                is_urgent = True
                break
        
        # Also check if doc is highly relevant to query
        query_terms = query_lower.split()
        query_match_count = sum(1 for term in query_terms if len(term) > 3 and term in doc_text)
        if query_match_count >= 2:
            is_urgent = True
        
        if is_urgent:
            urgent_docs.append(doc_tuple)
        else:
            routine_docs.append(doc_tuple)
    
    logger.info(f"Reordered context: {len(urgent_docs)} urgent chunks, {len(routine_docs)} routine chunks")
    
    # Return urgent docs first, then routine docs
    return urgent_docs + routine_docs


# Endpoints:
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    target_model = request.json.get('model', 'mistral')  # Default to mistral if not specified
    logger.info(f"User received message: {user_message} (Model: {target_model})")

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
        results = vector_store.query([query_embedding], n_results=20)

        # Assign weights to the retrieved documents
        retrieved_docs = results['documents'][0]
        retrieved_scores = results['scores'][0]
        retrieved_metadatas = results['metadatas'][0]

        # Boost the score for offline documents
        weighted_docs = []
        for i, doc in enumerate(retrieved_docs):
            score = retrieved_scores[i]
            metadata = retrieved_metadatas[i]
            
            # Boost offline documents by 2.5x
            if metadata.get("source") == "offline":
                score *= 2.5
                logger.debug(f"Boosted offline doc {i}: {metadata}")
            
            weighted_docs.append((doc, score, metadata))

        # Sort the documents based on the adjusted score
        sorted_docs = sorted(weighted_docs, key=lambda x: x[1], reverse=True)
        
        # CRITICAL: Reorder context to place urgent information first
        sorted_docs = reorder_context_by_urgency(sorted_docs, user_message)

        # Rebuild the input text for the model with source tracking
        context_parts = []
        for doc, score, metadata in sorted_docs:
            context_parts.append(doc["text"])
        
        input_text = "\n\n".join(context_parts)

        logger.info(f"Retrieved {len(retrieved_docs)} chunks for context.")
        logger.info(f"Offline docs: {sum(1 for _, _, m in sorted_docs if m.get('source') == 'offline')}, Scraped docs: {sum(1 for _, _, m in sorted_docs if m.get('source') == 'scraped')}")
        logger.info(f"Context length: {len(input_text)} characters")
        logger.info(f"First 500 chars of context: {input_text[:500]}")

        # Generate text response using Ollama with validation (via text_generator wrapper)
        # For severe queries, use validated generation to ensure urgent info is included
        severe_terms = ['severe', 'proliferative', 'PDR', 'severe NPDR']
        is_severe_query = any(term.lower() in user_message.lower() for term in severe_terms)
        
        if is_severe_query:
            logger.info("SEVERE QUERY DETECTED - Using validated generation")
            generated_text = generate_response_with_validation(input_text, user_message, model=target_model, max_retries=2)
        else:
            generated_text = generate_text_in_chunks(input_text, user_message, model=target_model, max_length=MAX_LENGTH)
        
        logger.info(f"Generated response (first 300 chars): {generated_text[:300]}")
        logger.info(f"Full response length: {len(generated_text)} characters")
        # Reranking/Filtering logic (Legacy feature kept)
        if NeedSortRerank:
            logger.info("NeedSortRerank")
            results = extract_sentences_with_keyword(retrieved_docs, user_message)
            for result in results:
                doc_index = result['doc_id']
                sentence = result['sentence']
                logger.info(f"Document Index: {doc_index} - Sentence: {sentence}")

        logger.info(f"Generated response: {generated_text}")

        # Store bot response in session
        session['conversation'].append({'role': 'system', 'content': generated_text})

        return jsonify({'response': generated_text})

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {"error": "Internal Server Error"}, 500

@app.route('/reset', methods=['POST'])
def reset_session():
    logger.info("Resetting session.")
    session.clear()
    return jsonify({'status': 'Session reset.'})

@app.route('/history', methods=['GET'])
def get_history():
    if 'conversation' not in session:
        return jsonify({'history': []})
    return jsonify({'history': session['conversation']})

@app.route('/')
def hello():
    logger.info("App is running")
    return "Hello, World!"

# Start the Flask application
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
