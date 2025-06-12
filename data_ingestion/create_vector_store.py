import base64
import os
import time
from pymongo import MongoClient
import numpy as np
import faiss
import json
from tqdm import tqdm
import shutil
from config import MONGO_URI, DB_NAME, ARTICLES_COLLECTION, IMAGES_COLLECTION, VECTOR_STORE_DIR
from embedder import Embedder
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

def connect_to_mongodb():
    """Connect to MongoDB and return database object."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def clear_vector_store():
    """Clear the vector store directory if it exists."""
    if os.path.exists(VECTOR_STORE_DIR):
        print(f"Clearing existing vector store at {VECTOR_STORE_DIR}...")
        shutil.rmtree(VECTOR_STORE_DIR)
        print("Vector store cleared successfully!")

def download_and_process_image(image_url, article_url):
    """Download and process an image from URL."""
    try:
        full_url = urljoin(article_url, image_url)
        response = requests.get(full_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading image {image_url}: {str(e)}")
        return None

def create_vector_store():
    """Create a unified vector store for both text and image embeddings."""
    # Clear existing vector store
    clear_vector_store()
    
    # Initialize
    db = connect_to_mongodb()
    embedder = Embedder()
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # Initialize FAISS index
    embedding_dim = 512
    index = faiss.IndexFlatL2(embedding_dim)
    
    embeddings = []
    metadata_list = []
    
    articles = list(db[ARTICLES_COLLECTION].find())
    print(f"Processing {len(articles)} articles...")
    
    for article in tqdm(articles):
        article_id = str(article['_id'])
        article_url = article.get('url', '')
        
        # Process text chunks
        chunks = embedder.split_text_into_chunks(article['content'])
        chunk_embeddings = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id, embedding, metadata = embedder.process_text_chunk(
                chunk, article_id, chunk_idx, article_url
            )
            chunk_embeddings.append((chunk_id, embedding, metadata))
        
        # Extract all images from HTML content
        soup = BeautifulSoup(article.get('content', ''), 'html.parser')
        img_elements = soup.find_all('img')
        
        image_embeddings = []
        alt_text_embeddings = []
        description_embeddings = []
                
        for img_idx, img in enumerate(img_elements):
            try:
                img_url = img.get('src')
                if not img_url:
                    continue
                    
                # Download and process image
                image_data = download_and_process_image(img_url, article_url)
                if not image_data:
                    continue
                
                # Process image
                image_id, embedding, metadata = embedder.process_image(
                    image_data,
                    article_id,
                    img_idx,
                    img.get('alt', ''),
                    img_url
                )
                image_embeddings.append((image_id, embedding, metadata))
                
                # Process alt text if exists
                if img.get('alt'):
                    alt_text_id, embedding, metadata = embedder.process_alt_text(
                        img.get('alt'),
                        img_url
                    )
                    alt_text_embeddings.append((alt_text_id, embedding, metadata))
                
                # Generate and process image description
                result = embedder.process_image_description(
                    image_data,
                    article_id,
                    img_idx,
                    img_url
                )
                if result:
                    print(f"Generated description: {result[2]['description'][:100]}")
                    desc_id, embedding, metadata = result
                    description_embeddings.append((desc_id, embedding, metadata))
                
            except Exception as e:
                print(f"Error processing image {img_url}: {str(e)}")
            time.sleep(0.5)
        
        # Link chunks and images
        embedder.link_embeddings(chunk_embeddings, image_embeddings)
        
        # Add all embeddings and metadata
        all_embeddings = chunk_embeddings + image_embeddings + alt_text_embeddings + description_embeddings
        embeddings.extend([emb for _, emb, _ in all_embeddings])
        metadata_list.extend([meta for _, _, meta in all_embeddings])
    
    embeddings_array = np.array(embeddings)
    index.add(embeddings_array)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "rag_index.faiss"))
    
    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata_list, f)
    
    print(f"Vector store created successfully!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Text chunks: {sum(1 for m in metadata_list if m['type'] == 'text')}")
    print(f"Images: {sum(1 for m in metadata_list if m['type'] == 'image')}")
    print(f"Alt text: {sum(1 for m in metadata_list if m['type'] == 'alt_text')}")
    print(f"Image descriptions: {sum(1 for m in metadata_list if m['type'] == 'image_description')}")

if __name__ == "__main__":
    create_vector_store() 