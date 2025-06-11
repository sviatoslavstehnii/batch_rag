import os
from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import base64
import faiss
import json
from tqdm import tqdm
import torch
import shutil
from typing import List, Dict, Any, Tuple
from config import IMAGE_EMBEDDING_MODEL, MONGO_URI, DB_NAME, ARTICLES_COLLECTION, IMAGES_COLLECTION, VECTOR_STORE_DIR, TEXT_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def connect_to_mongodb():
    """Connect to MongoDB and return database object."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def initialize_models():
    """Initialize text and image embedding models."""
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
    return text_model, clip_model, clip_processor

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = ' '.join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        
    return chunks

def process_text_chunk(text_model, chunk: str, article_id: str, chunk_idx: int) -> Tuple[str, np.ndarray, Dict]:
    """Process a text chunk and return its ID, embedding, and metadata."""
    chunk_id = f"article_{article_id}_chunk_{chunk_idx}"
    embedding = text_model.encode(chunk)
    metadata = {
        "type": "text",
        "text": chunk,
        "article_id": article_id,
        "linked_images": []
    }
    return chunk_id, embedding, metadata

def process_image(clip_model, clip_processor, image_data: bytes, article_id: str, image_idx: int, alt_text: str) -> Tuple[str, np.ndarray, Dict]:
    """Process an image and return its ID, embedding, and metadata."""
    image_id = f"article_{article_id}_img_{image_idx}"
    
    # Decode and process image
    image = Image.open(io.BytesIO(image_data))
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs).numpy()[0]
    
    metadata = {
        "type": "image",
        "article_id": article_id,
        "linked_chunks": [],
        "alt_text": alt_text
    }
    return image_id, embedding, metadata

def process_alt_text(text_model, alt_text: str) -> Tuple[str, np.ndarray, Dict]:
    """Process an alt text and return its ID, embedding, and metadata."""
    alt_text_id = f"alt_text_{alt_text}"
    embedding = text_model.encode(alt_text)
    metadata = {
        "type": "alt_text",
        "alt_text": alt_text
    }
    return alt_text_id, embedding, metadata
    
def clear_vector_store():
    """Clear the vector store directory if it exists."""
    if os.path.exists(VECTOR_STORE_DIR):
        print(f"Clearing existing vector store at {VECTOR_STORE_DIR}...")
        shutil.rmtree(VECTOR_STORE_DIR)
        print("Vector store cleared successfully!")

def create_vector_store():
    """Create a unified vector store for both text and image embeddings."""
    # Clear existing vector store
    clear_vector_store()
    
    # Initialize
    db = connect_to_mongodb()
    text_model, clip_model, clip_processor = initialize_models()
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
        
        # Process text chunks
        chunks = split_text_into_chunks(article['content'])
        chunk_embeddings = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id, embedding, metadata = process_text_chunk(
                text_model, chunk, article_id, chunk_idx
            )
            chunk_embeddings.append((chunk_id, embedding, metadata))
        
        # Process images
        images = list(db[IMAGES_COLLECTION].find({'article_id': article['_id']}))
        image_embeddings = []
        alt_text_embeddings = []
        
        for img_idx, img in enumerate(images):
            try:
                image_id, embedding, metadata = process_image(
                    clip_model, clip_processor, 
                    base64.b64decode(img['data']), 
                    article_id, img_idx, 
                    img['alt_text']
                )
                image_embeddings.append((image_id, embedding, metadata))
                if img['alt_text']:
                    alt_text_id, embedding, metadata = process_alt_text(
                        text_model, 
                        img['alt_text']
                    )
                    print(f"Alt text: {img['alt_text']}")
                    alt_text_embeddings.append((alt_text_id, embedding, metadata))
            except Exception as e:
                print(f"Error processing image {img['_id']}: {str(e)}")
        

        
        # Link chunks and images
        for chunk_id, _, chunk_metadata in chunk_embeddings:
            chunk_metadata["linked_images"] = [img_id for img_id, _, _ in image_embeddings]
        
        for img_id, _, img_metadata in image_embeddings:
            img_metadata["linked_chunks"] = [chunk_id for chunk_id, _, _ in chunk_embeddings]
        
        # Add all embeddings and metadata
        embeddings.extend([emb for _, emb, _ in chunk_embeddings + image_embeddings + alt_text_embeddings])
        metadata_list.extend([meta for _, _, meta in chunk_embeddings + image_embeddings + alt_text_embeddings])
    
    embeddings_array = np.array(embeddings)
    print(embeddings_array.shape)
    index.add(embeddings_array)
    faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "rag_index.faiss"))
    
    with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata_list, f)
    
    print(f"Vector store created successfully!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Text chunks: {sum(1 for m in metadata_list if m['type'] == 'text')}")
    print(f"Images: {sum(1 for m in metadata_list if m['type'] == 'image')}")
    print(f"Alt text: {sum(1 for m in metadata_list if m['type'] == 'alt_text')}")
if __name__ == "__main__":
    create_vector_store() 