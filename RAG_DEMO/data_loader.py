import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import re
from collections import OrderedDict
import ollama
import numpy as np
import shutil
import sys
import time
from simple_vector_store import SimpleVectorStore

# Initialize Simple Vector Store
# This creates/loads 'vector_store.pkl' in the current directory
vector_store = SimpleVectorStore()
EMBEDDING_MODEL = "nomic-embed-text"


# Function to scrape and save text from URLs
def scrape_and_save_text(url, output_dir, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(text_content)
        print(f"Saved content from {url} to {filepath}")
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")


# Function to scrape and save corpus
def scrape_and_save_Corpus():
    urls = [
        "https://www.verywellhealth.com/search?q=Diabetic+retinopathy",
        "https://www.verywellhealth.com/search?q=Diabetic+retinopathy",
        "https://app.pulsenotes.com/medicine/diabetes/notes/type-2-diabetes#note-9",
        "https://stanfordhealthcare.org/medical-conditions/eyes-and-vision/diabetic-retinopathy/treatments.html",
        "https://www2.hse.ie/conditions/diabetic-retinopathy/treatment/",
        "https://www.hopkinsmedicine.org/health/conditions-and-diseases/diabetes/diabetic-retinopathy",
        "https://emedicine.medscape.com/article/1225122-treatment#d14",
        "https://www.nhs.uk/conditions/diabetic-retinopathy/",
        "https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy#treatment"
    ]

    output_directory = './scraped_data'
    # Use subset for demo speed, but enough for testing
    for idx, url in enumerate(urls): 
        filename = f"article_{idx}.txt"
        scrape_and_save_text(url, output_directory, filename)


# Function to concatenate corpus into a DataFrame
def concat_corpus(directory):
    concatenated_list = []
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
                concatenated_list.append(file_content)
    return pd.DataFrame(concatenated_list, columns=['Document'])

# Function to create corpus document
def create_corpus_doc():
    directory = './scraped_data'
    documents = concat_corpus(directory)
    print("Corpus created successfully.")
    return documents

def clean_and_deduplicate(text):
    # Remove repeated lines
    lines = text.split('\n')
    seen = OrderedDict.fromkeys(lines)
    cleaned_text = '\n'.join(seen.keys())

    # Additional cleaning if necessary
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    return cleaned_text

def generate_embeddings_ollama(texts):
    """
    Generate embeddings for a list of texts using Ollama.
    """
    embeddings = []
    print(f"Generating embeddings for {len(texts)} chunks...", flush=True)
    for i, text in enumerate(texts):
        if not text or not text.strip():
            print(f"Skipping empty chunk {i}")
            embeddings.append([0.0]*768)
            continue
            
        try:
            # Verbose log
            print(f"Processing chunk {i+1}/{len(texts)}...", end="\r", flush=True)
            
            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            embeddings.append(resp["embedding"])
            # Small delay to be polite to local API
            time.sleep(0.01) 
            
        except Exception as e:
            print(f"\nExample text (first 50 chars): {text[:50]}...")
            print(f"Error embedding chunk {i} via Ollama: {e}")
            # Fallback zero vector
            embeddings.append([0.0]*768) 
    print("\nEmbedding generation complete.", flush=True)
    return embeddings

# Function to chunk text
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    text_len = len(text)
    
    # Handle short text
    if text_len <= chunk_size:
        return [text]
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

# Preprocess documents to generate embeddings and store in SimpleVectorStore
def preprocess_documents():
    print("preprocess_documents: starting...", flush=True)
    
    # 0. Cleanup old data
    output_directory = './scraped_data'
    if os.path.exists(output_directory):
        print("Cleaning up old scraped data...", flush=True)
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. Scrape data
    scrape_and_save_Corpus()
    
    # 2. Load and clean data
    documents_df = create_corpus_doc()
    documents_df['Cleaned_Document'] = documents_df['Document'].apply(clean_and_deduplicate)
    raw_docs = documents_df['Cleaned_Document'].tolist()
    
    print(f"Loaded {len(raw_docs)} raw documents.", flush=True)
    
    # 3. Chunk Documents
    print("Chunking documents...", flush=True)
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for doc_idx, text in enumerate(raw_docs):
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": "web_scrape", 
                "doc_id": str(doc_idx), 
                "chunk_id": str(chunk_idx)
            })
            all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
            
    print(f"Created {len(all_chunks)} chunks from {len(raw_docs)} documents.", flush=True)

    # 4. Generate Embeddings via Ollama
    print(f"Generating embeddings for {len(all_chunks)} chunks using {EMBEDDING_MODEL} (via Ollama)...", flush=True)
    embeddings = generate_embeddings_ollama(all_chunks)
    
    # 5. Store in SimpleVectorStore
    print("Resetting vector store...", flush=True)
    vector_store.reset()
    
    print(f"Adding {len(all_chunks)} items to SimpleVectorStore...", flush=True)
    vector_store.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )
    
    print(f"Stored {len(all_chunks)} chunks in SimpleVectorStore (backed by Pickle).", flush=True)
    return vector_store

    print(f"Stored {len(all_chunks)} chunks in ChromaDB collection 'medical_docs'.", flush=True)
    return collection
