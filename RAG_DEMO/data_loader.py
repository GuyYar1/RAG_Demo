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
import logging
from simple_vector_store import SimpleVectorStore

# Initialize logger for data_loader module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more verbose logs
ch = logging.StreamHandler()  # Logs to the console
ch.setLevel(logging.DEBUG)  # Set the logging level for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')  # Log format
ch.setFormatter(formatter)
logger.addHandler(ch) 

# Optional: Also log to a file (uncomment if needed)
# fh = logging.FileHandler('data_loader.log')
# fh.setLevel(logging.DEBUG)  # Set to DEBUG for file logging
# fh.setFormatter(formatter)
# logger.addHandler(fh)
#

# Initialize Simple Vector Store, already  initit in the app..py
# vector_store = SimpleVectorStore()
EMBEDDING_MODEL = "nomic-embed-text"

# Skip scraping by default (offline files are always safe regardless of this setting)
skip_scraping = True  # False = download new scraped data | True = use existing files only

# Function to scrape and save text from URLs with a distinct prefix
def scrape_and_save_text(url, output_dir, filename, prefix='scraped_'):
    try:
        logger.debug(f"Scraping URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a distinct prefix for new files
        new_filename = f"{prefix}{filename}"
        filepath = os.path.join(output_dir, new_filename)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(text_content)
        
        logger.info(f"Saved content from {url} to {filepath}")
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")

# Function to scrape and save corpus (scraped URLs only)
def scrape_and_save_Corpus():
    urls = [
        "https://www.verywellhealth.com/search?q=Diabetic+retinopathy",
        "https://stanfordhealthcare.org/medical-conditions/eyes-and-vision/diabetic-retinopathy/treatments.html",
        "https://www2.hse.ie/conditions/diabetic-retinopathy/treatment/",
        "https://www.hopkinsmedicine.org/health/conditions-and-diseases/diabetes/diabetic-retinopathy",
        "https://emedicine.medscape.com/article/1225122-treatment#d14",
        "https://www.nhs.uk/conditions/diabetic-retinopathy/",
        "https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy#treatment",
        "https://www.aao.org/preferred-practice-pattern/diabetic-retinopathy-ppp",
        "https://diabetesjournals.org/care/article/48/1/1/312512/2025-Standards-of-Care-in-Diabetes",
        "https://www.nice.org.uk/guidance/ng18",
        "https://www.rcophth.ac.uk/professionals/clinical-guidelines/",
        "https://pubmed.ncbi.nlm.nih.gov/31653148/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7434204/",
        "https://www.nice.org.uk/guidance/ta274",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7100639/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7077224/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7899300/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6805177/",
        "https://www.reviewofoptometry.com/article/evolving-therapies-for-diabetic-macular-edema-options-abound",
        "https://pubmed.ncbi.nlm.nih.gov/31439098/"
    ]
    
    output_directory = './scraped_data'
    if not skip_scraping:
        for idx, url in enumerate(urls): 
            filename = f"article_{idx}.txt"
            scrape_and_save_text(url, output_directory, filename, prefix='scraped_')
    else:
        print("Skipping scraping, using existing files")
        
    output_directory = './scraped_data'
    if not skip_scraping:  # Only scrape if not skipping
        logger.info("Starting to scrape new data...")
        for idx, url in enumerate(urls): 
            filename = f"article_{idx}.txt"
            scrape_and_save_text(url, output_directory, filename, prefix='scraped_')
    else:
        logger.info("Skipping scraping, using existing files.")

# Function to concatenate both offline and scraped corpus into a DataFrame
def concat_corpus(directory):
    concatenated_list = []
    source_list = []
    offline_count = 0
    scraped_count = 0

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
# Concatenate all offline files first (no prefix)
    logger.info("Loading offline files...")
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and not filename.startswith('scraped_'):  # Skip files with 'scraped_' prefix
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
                concatenated_list.append(file_content)
                source_list.append('offline')  # Mark as offline
                offline_count += 1
    
    # Concatenate scraped files (those starting with 'scraped_')
    logger.info("Loading scraped files...")
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('scraped_'):  # Only read scraped files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()
                concatenated_list.append(file_content)
                source_list.append('scraped')  # Mark as scraped
                scraped_count += 1
    
    logger.info(f"Loaded {offline_count} offline files and {scraped_count} scraped files.")
    return pd.DataFrame({'Document': concatenated_list, 'Source': source_list})


# Function to create corpus document from both offline and scraped files
def create_corpus_doc():
    directory = './scraped_data'
    documents_df = concat_corpus(directory)  # This will load both offline and scraped files with source labels
    logger.info("Corpus created successfully.")
    return documents_df

# Function to clean and deduplicate the text
def clean_and_deduplicate(text):
    # Remove repeated lines
    lines = text.split('\n')
    seen = OrderedDict.fromkeys(lines)
    cleaned_text = '\n'.join(seen.keys())

    # Additional cleaning if necessary
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    return cleaned_text

# Function to chunk text
def chunk_text(text, chunk_size=1800, overlap=200):
    chunks = []
    start = 0
    text_len = len(text)
    
    # Handle short text
    if text_len <= chunk_size:
        return [text]
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # If it's the last chunk and smaller than chunk_size, still include it
        if end > text_len:
            chunk = text[start:text_len]
        
        chunks.append(chunk)
        start += (chunk_size - overlap)  # Advance start by chunk_size minus overlap
        
    return chunks


# Function to generate embeddings using Ollama
def generate_embeddings_ollama(texts):
    embeddings = []
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    for i, text in enumerate(texts):
        if not text or not text.strip():
            logger.warning(f"Skipping empty chunk {i}")
            embeddings.append([0.0]*768)
            continue
            
        try:
            logger.debug(f"Processing chunk {i+1}/{len(texts)}...", end="\r", flush=True)
            
            resp = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            embeddings.append(resp["embedding"])
            time.sleep(0.01)  # Small delay to be polite to local API
            
        except Exception as e:
            logger.error(f"Error embedding chunk {i}: {e}")
            embeddings.append([0.0]*768)  # Fallback zero vector
    
    logger.info("Embedding generation complete.")
    return embeddings

def preprocess_documents(vector_store, skip_scraping=False):
    logger.info("Preprocessing documents...")

    # 0. Ensure directory exists
    output_directory = './scraped_data'
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. Clean up ONLY old scraped files before re-scraping (offline files are always preserved)
    if not skip_scraping and os.path.exists(output_directory):
        logger.info("Cleaning up old scraped files only (preserving offline files)...")
        deleted_count = 0
        for filename in os.listdir(output_directory):
            # Only delete files that start with 'scraped_'
            if filename.startswith('scraped_') and filename.endswith('.txt'):
                filepath = os.path.join(output_directory, filename)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    logger.debug(f"Deleted: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")
        logger.info(f"Deleted {deleted_count} old scraped files")

    # 2. Scrape data if not skipping
    if not skip_scraping:
        logger.info("Scraping new data...")
        scrape_and_save_Corpus()
    else:
        logger.info("Skipping scraping and using existing data...")
    
   # 2. Load and clean data
    logger.info("Loading and cleaning data...")
    documents_df = create_corpus_doc()
    documents_df['Cleaned_Document'] = documents_df['Document'].apply(clean_and_deduplicate)
    raw_docs = documents_df['Cleaned_Document'].tolist()
    doc_sources = documents_df['Source'].tolist()  # Extract source labels
    
    logger.info(f"Loaded {len(raw_docs)} raw documents.")

    # 3. Chunk Documents
    logger.info("Chunking documents...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for doc_idx, text in enumerate(raw_docs):
        chunks = chunk_text(text)
        source_label = doc_sources[doc_idx]  # Get the correct source label
        for chunk_idx, chunk in enumerate(chunks):
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": source_label,  # Use actual source: 'offline' or 'scraped'
                "doc_id": str(doc_idx), 
                "chunk_id": str(chunk_idx)
            })
            all_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")
            
    logger.info(f"Created {len(all_chunks)} chunks from {len(raw_docs)} documents.")

    # 4. Generate Embeddings via Ollama
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks using {EMBEDDING_MODEL}...")
    embeddings = generate_embeddings_ollama(all_chunks)

    # 5. Store in SimpleVectorStore (using the existing vector_store)
    logger.info("Resetting vector store...")
    vector_store.reset()

    logger.info(f"Adding {len(all_chunks)} items to SimpleVectorStore...")
    vector_store.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metadatas
    )

    logger.info(f"Stored {len(all_chunks)} chunks in SimpleVectorStore.")
    return vector_store
