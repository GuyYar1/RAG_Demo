from flask import Flask, request, jsonify, session
from flask_session import Session
from model_service import load_w2v_model, get_embedding, load_gpt2_model
from text_generator import generate_text_in_chunks, rerank_response, extract_sentences_with_keyword
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
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

print ("Load data and models")
# Load data and models
documents, document_embeddings, vector_db, w2v_model = preprocess_documents()
# Term: words and raw words
    # Word Types: This refers to the number of unique words or vocabulary items in the corpus.
    # In this case, there are 47,773 distinct words found in the text.
    # Note that this count includes only unique words, not their occurrences.
    # Raw Words: This is the total number of words in the corpus, including repetitions.
    # In this case, there are 1,293,263 total words. This count includes all instances of each word, not just the unique ones

# load w2v_model
#w2v_model = load_w2v_model()
# Access the vocabulary
vocabulary = list(w2v_model.wv.index_to_key)  # list of words in the vocabulary

tokenizer, gpt2_model = load_gpt2_model()

# Define the maximum sequence length
MAX_LENGTH = 1024

print("Waiting for request messages .... ")


# Endpoints:
# /chat: Handles user messages, generates a response using the GPT-2 model, and maintains conversation history in the session.
# /reset: Clears the session, which is useful for resetting the chat context.
@app.route('/chat', methods=['POST'])
def chat():
    print("got chat a request messages ....")
    user_message = request.json.get('message')
    logging.info(f"User received message: {user_message}")

    if 'conversation' not in session:
        session['conversation'] = []

    # Update conversation history
    session['conversation'].append({'role': 'user', 'content': user_message})
    try:
        # Convert query to embedding
        query_embedding = get_embedding(user_message, w2v_model)

        # Compute cosine similarity and retrieve documents
        similarities = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)
        top_k_indices = np.argsort(similarities[0])[-5:][::-1]
        retrieved_docs = [vector_db[idx] for idx in top_k_indices]

        # Concatenate retrieved documents only
        input_text = " ".join(retrieved_docs)

        # Generate text response in chunks
        generated_text = generate_text_in_chunks(input_text, gpt2_model, tokenizer, max_length=MAX_LENGTH)

        # Use the rerank_response function to filter based on 'Crime' genre
        generated_text_rerank = rerank_response(generated_text, user_message)

        # Extract sentences with the keyword
        results = extract_sentences_with_keyword(retrieved_docs, user_message)



        # Print each filtered sentence with its document index
        for result in results:
            doc_index = result['doc_id']
            sentence = result['sentence']
            print(f"Document Index: {doc_index} - Sentence: {sentence}")

        logging.info(f"Generated response: {generated_text_rerank}")

        # Store bot response in session
        session['conversation'].append({'role': 'system', 'content': generated_text_rerank})

        return jsonify({'response': generated_text_rerank})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return {"error": "Internal Server Error"}, 500
@app.route('/reset', methods=['POST'])
def reset_session():
    print("got reset a request messages ....")
    session.clear()
    return jsonify({'status': 'Session reset.'})


@app.route('/history', methods=['GET'])
def get_history():
    if 'conversation' not in session:
        # Return an empty list if there is no conversation history
        return jsonify({'history': []})

    # Return the conversation history
    return jsonify({'history': session['conversation']})


if __name__ == '__main__':
    app.run()
