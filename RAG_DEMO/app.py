import logging
from flask import Flask, request, jsonify, session
from flask_session import Session
from model_service import get_query_embedding
from simple_vector_store import SimpleVectorStore
from data_loader import preprocess_documents
from profile_extractor import extract_profile_from_query, is_severe_condition
from text_generator import generate_text_in_chunks, rerank_response, extract_sentences_with_keyword
from ollama_service import generate_response_with_validation, generate_response


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
    from data_loader import skip_scraping  # Import the global flag
    vector_store = preprocess_documents(vector_store, skip_scraping=skip_scraping)  # Pass the flag explicitly
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
    
 # Extract user profile from query and merge with existing session profile
    new_profile = extract_profile_from_query(user_message, session['conversation'])
    
    # Merge with existing profile from session (cross-message persistence)
    if 'user_profile' in session:
        existing_profile_dict = session['user_profile']
        # Keep existing data if new extraction is empty
        if not new_profile.age and existing_profile_dict.get('age'):
            new_profile.age = existing_profile_dict['age']
        if not new_profile.gender and existing_profile_dict.get('gender'):
            new_profile.gender = existing_profile_dict['gender']
        if not new_profile.severity_level and existing_profile_dict.get('severity_level'):
            new_profile.severity_level = existing_profile_dict['severity_level']
        # Merge medical history
        for condition in existing_profile_dict.get('medical_history', []):
            if condition not in new_profile.medical_history:
                new_profile.medical_history.append(condition)
        logger.info(f"Merged with existing session profile")
    
    user_profile = new_profile
    logger.info(f"Final user profile: {user_profile}")

    
    try:
        # Convert query to embedding (using Ollama via model_service)
        query_embedding = get_query_embedding(user_message)
        logger.info(f"DEBUG 1: Embedding generated: {len(query_embedding) if hasattr(query_embedding, '__len__') else 'NO_LEN'} dims")

        # =====================================================
        # BACKEND DR RELEVANCE FILTER (FULL DEBUG)
        # =====================================================
        logger.info("DEBUG 2: Starting DR relevance check...")
        test_results = vector_store.query([query_embedding], n_results=3)
        logger.info(f"DEBUG 3: test_results keys: {list(test_results.keys())}")
        logger.info(f"DEBUG 4: test_results['documents'][0] length: {len(test_results.get('documents', [[]])[0])}")
        
        # SAFE score extraction - LOG EVERY STEP
        test_scores_raw = None
        if 'scores' in test_results:
            test_scores_raw = test_results['scores'][0][:3]
            logger.info(f"DEBUG 5A: Found SCORES: {test_scores_raw}")
        elif 'distances' in test_results:
            test_distances = test_results['distances'][0][:3]
            test_scores_raw = [1.0 / (1.0 + float(d)) for d in test_distances]
            logger.info(f"DEBUG 5B: Found DISTANCES: {test_distances} → converted: {test_scores_raw}")
        else:
            test_scores_raw = [0.0, 0.0, 0.0]
            logger.info("DEBUG 5C: NO scores/distances → using [0.0,0.0,0.0]")
        
        # Ensure numeric + pad to 3 - LOG EACH
        test_scores = []
        for i, score in enumerate(test_scores_raw):
            try:
                numeric_score = float(score)
                test_scores.append(numeric_score)
                logger.info(f"DEBUG 6{i}: Raw {score} → float {numeric_score}")
            except Exception as e:
                test_scores.append(0.0)
                logger.info(f"DEBUG 6{i}: FAILED {score} → 0.0 (error: {e})")
        
        while len(test_scores) < 3:
            test_scores.append(0.0)
            logger.info(f"DEBUG 7: Padded to 3 scores: {test_scores}")
        
        logger.info(f"DEBUG 8: FINAL test_scores: {test_scores}")

                # ULTRA STRICT FILTER: ALL top-3 < 0.65 AND avg < 0.60 (blocks random strings)
        all_low = all(s < 0.65 for s in test_scores)
        avg_score = sum(test_scores) / 3
        too_generic = avg_score < 0.60
        
        logger.info(f"DEBUG 9B: all<0.65={all_low}, avg={avg_score:.3f}")


        logger.info(f"DEBUG 9: all<0.55={all_low}, avg={avg_score:.3f} → {'FILTERED' if (all_low and too_generic) else 'PROCEED'}")

        if all_low and too_generic:
            logger.info("🚫 RAG irrelevant: BLOCKED generic query")
            return jsonify({
                'response': '🤖 **Rephrase as diabetic retinopathy question**\n\nExamples:\n• "NPDR level 3 treatment?"\n• "PDR anti-VEGF protocol?"\n• "Severe DR urgent steps?"',
                'profile': getattr(user_profile, 'to_dict', lambda: {} )()
            })
        
        logger.info("✅ RAG relevant: CONTINUING...")


      # Retrieval from SimpleVectorStore - adjust count based on severity
        n_results = 25 if is_severe_condition(user_profile) else 15
        results = vector_store.query([query_embedding], n_results=n_results)


        # Assign weights to the retrieved documents
        retrieved_docs = results['documents'][0]
        
        # Handle both 'scores' and 'distances' keys
        if 'scores' in results:
            retrieved_scores = results['scores'][0]
        elif 'distances' in results:
            retrieved_distances = results['distances'][0]
            retrieved_scores = [1.0 / (1.0 + dist) for dist in retrieved_distances]
        else:
            # Fallback: use uniform scores
            retrieved_scores = [1.0] * len(retrieved_docs)
            logger.warning("No scores or distances found, using uniform scores")
        
        retrieved_metadatas = results['metadatas'][0]
        logger.info(f"DEBUG 10: retrieved_docs length: {len(retrieved_docs)}")
        logger.info(f"DEBUG 11: retrieved_scores length: {len(retrieved_scores)}")
        logger.info(f"DEBUG 12: retrieved_metadatas length: {len(retrieved_metadatas)}")

        
  # Boost the score for offline documents and severity-relevant docs
        weighted_docs = []
        for i, doc in enumerate(retrieved_docs):
            score = retrieved_scores[i]
            metadata = retrieved_metadatas[i]
            
            # Handle doc as either string or dict
            if isinstance(doc, dict):
                doc_text = doc.get("text", "")
            else:
                doc_text = str(doc)  # doc is already a string
            
            doc_text_lower = doc_text.lower()
            
            # Boost offline documents by 3x
            if metadata.get("source") == "offline":
                score *= 3.0
                logger.debug(f"Boosted offline doc {i}: {metadata}")
            
            # Additional boost for severity-relevant content
            if is_severe_condition(user_profile):
                severity_keywords = ['severe', 'proliferative', 'urgent', 'immediate', 'referral', 'ophthalmologist', 'PRP', 'anti-VEGF']
                relevance_boost = sum(1 for kw in severity_keywords if kw.lower() in doc_text_lower)
                if relevance_boost > 0:
                    score *= (1.0 + relevance_boost * 0.2)  # Up to 1.6x boost
                    logger.debug(f"Severity relevance boost for doc {i}: {relevance_boost} keywords")
            
            # Store doc with its text for later use
            doc_dict = {"text": doc_text} if isinstance(doc, str) else doc
            weighted_docs.append((doc_dict, score, metadata))

        # Sort the documents based on the adjusted score
        sorted_docs = sorted(weighted_docs, key=lambda x: x[1], reverse=True)
        
        # CRITICAL: Reorder context to place urgent information first
        sorted_docs = reorder_context_by_urgency(sorted_docs, user_message)

        # Rebuild the input text for the model with source tracking
        context_parts = []
        for doc, score, metadata in sorted_docs:
            # Handle doc as either dict or string
            if isinstance(doc, dict):
                context_parts.append(doc.get("text", ""))
            else:
                context_parts.append(str(doc))
        
        input_text = "\n\n".join(context_parts)

        logger.info(f"Retrieved {n_results} chunks for context (severity-aware).")
        logger.info(f"User profile: {user_profile}")
        logger.info(f"Is severe condition: {is_severe_condition(user_profile)}")
        logger.info(f"Offline docs: {sum(1 for _, _, m in sorted_docs if m.get('source') == 'offline')}, Scraped docs: {sum(1 for _, _, m in sorted_docs if m.get('source') == 'scraped')}")
        logger.info(f"Context length: {len(input_text)} characters")
        logger.info(f"First 500 chars of context: {input_text[:500]}")

       # Generate text response using profile-aware Ollama service
        if is_severe_condition(user_profile):
            logger.info("🔴 SEVERE CONDITION DETECTED - Using validated generation with profile")
            generated_text = generate_response_with_validation(
                input_text, 
                user_message, 
                user_profile,
                model=target_model, 
                max_retries=2
            )
        else:
            logger.info("✓ Standard query - Using profile-aware generation")
            from ollama_service import generate_response
            generated_text = generate_response(
                input_text, 
                user_message, 
                user_profile,
                model=target_model
            )
        
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

        # Store bot response and profile in session
        session['conversation'].append({'role': 'system', 'content': generated_text})
        session['user_profile'] = user_profile.to_dict()  # Persist profile across conversation

        return jsonify({
            'response': generated_text,
            'profile': user_profile.to_dict()
        })

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
def home():
    logger.info("DR Assistance UI loaded")
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>DR Assistance for Doctor (Groq v1.0.0)</title>
    <style>body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
    input, textarea { width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
    button { background: #007bff; color: white; padding: 12px 24px; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; }
    button:hover { background: #0056b3; }
    #response { background: #f8f9fa; padding: 20px; border-left: 4px solid #007bff; margin-top: 20px; min-height: 100px; border-radius: 5px; }
    .char-count { font-size: 12px; color: #666; }</style>
</head>
<body>
    <h1>🚑 DR Assistance for Doctor (Groq v1.0.0)</h1>
    <p><strong>Ask about diabetic retinopathy (max 150 chars):</strong></p>
    <input type="text" id="query" placeholder="e.g. I have severe NPDR level 3/4, what should I do?" maxlength="150">
    <div class="char-count" id="char-count">0/150 chars</div><br>
    <button onclick="askBot()">💬 Ask Doctor Assistant</button>
    <div id="response">Ask a question to get started... (RAG + Groq)</div>

    <script>
        document.getElementById('query').addEventListener('input', function() {
            document.getElementById('char-count').textContent = this.value.length + '/150 chars';
        });
        async function askBot() {
            const query = document.getElementById('query').value.trim();
            if (!query) return alert('Please enter a question');
            document.getElementById('response').innerHTML = '🤔 Thinking...';
            document.getElementById('query').disabled = true;
            try {
                const res = await fetch('/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: query})});
                const data = await res.json();
                document.getElementById('response').innerHTML = `<strong>✅ Answer:</strong><br><pre>${data.response}</pre>`;
            } catch(e) {
                document.getElementById('response').innerHTML = '❌ Error: ' + e.message;
            }
            document.getElementById('query').disabled = false;
            document.getElementById('query').focus();
        }
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askBot();
        });
    </script>
</body>
</html>'''


@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """
    Collect user feedback on responses to improve future answers.
    Expected JSON: {query, response, rating (1-5), comment}
    """
    import json
    import os
    from datetime import datetime
    
    try:
        feedback_data = request.json
        user_query = feedback_data.get('query', '')
        response_text = feedback_data.get('response', '')
        rating = feedback_data.get('rating', 0)
        comment = feedback_data.get('comment', '')
        
        logger.info(f"Feedback received - Rating: {rating}/5, Query: {user_query[:50]}...")
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': user_query,
            'response': response_text,
            'rating': rating,
            'comment': comment,
            'profile': session.get('user_profile', {})
        }
        
        # Append to feedback log file
        feedback_file = 'feedback_log.json'
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_log = json.load(f)
        else:
            feedback_log = []
        
        feedback_log.append(feedback_entry)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_log, f, indent=2)
        
        logger.info(f"Feedback saved to {feedback_file}")
        
        return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
    
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Failed to save feedback'}), 500

# Start the Flask application
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
