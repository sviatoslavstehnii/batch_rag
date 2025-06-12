import os
import sys
import numpy as np
import faiss
import json
import torch
from typing import List, Dict, Any, Union, Tuple
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import build_prompt, get_image_description
from data_ingestion.config import (
    VECTOR_STORE_DIR,
    TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    MONGO_URI,
    DB_NAME,
    ARTICLES_COLLECTION,
    IMAGES_COLLECTION
)
from pymongo import MongoClient
from google import genai

class MultimodalRAG:
    def __init__(self, top_k: int = 5):
        """Initialize the RAG system.

        """
        self.top_k = top_k
        self.embedding_dim = 512
        
        # Load models
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
        
        # Load vector store
        self.index = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "rag_index.faiss"))
        with open(os.path.join(VECTOR_STORE_DIR, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)

        # Load Gemini model
        self.gemini_model = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    
    
    def generate_answer(self, query: str, image: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Generate an answer using Gemini based on retrieved content."""
        text_emb = None
        image_emb = None
        query_emb = None
        
        image_description = None
        
        
        if query is not None:
            text_emb = self.embed_text(query)
        if image is not None:
            image_emb = self.embed_image(image)
            image_description = get_image_description(image_url=None, image_data=image)

        if text_emb is not None and image_emb is not None:
            query_emb = self.embed_multimodal(query, image)
        elif text_emb is not None:
            query_emb = text_emb
        elif image_emb is not None:
            query_emb = image_emb
            
        # retrieve results
        context = self.search(query_emb)
        
        
        with open("context.json", "w") as f:
            json.dump(context, f)
        
        # Create prompt with both text and image context
        prompt = build_prompt(context, query, image_description)
        contents = [prompt]
        
        try:
            # Generate response
            response = self.gemini_model.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents
            )
            
            # Extract URLs from context
            text_urls = []
            image_urls = []
            for item in context:
                metadata = item['metadata']
                url = metadata.get('url', '')
                if metadata['type'] == 'text' and url:
                    text_urls.append(url)
                elif metadata['type'] in ['image', 'image_description'] and url:
                    image_urls.append(url)
            
            return {
                'answer': response.text,
                'text_sources': list(set(text_urls)),
                'image_sources': list(set(image_urls))
            }
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'text_sources': [],
                'image_sources': []
            }

    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text query using the text model."""
        return self.text_model.encode(text)
    
    def embed_image(self, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Embed image query using CLIP."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs).numpy()[0]
        return embedding
    
    def embed_multimodal(self, text: str, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Combine text and image embeddings."""
        text_emb = self.embed_text(text)
        image_emb = self.embed_image(image)
        # Normalize and average the embeddings
        text_emb = text_emb / np.linalg.norm(text_emb)
        image_emb = image_emb / np.linalg.norm(image_emb)
        return (text_emb + image_emb) / 2
    
    def search(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """Search the vector store for similar items."""
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search for more items than needed to ensure we get enough of each type
        search_k = self.top_k * 2
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Separate results by type
        text_results = []
        image_results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            metadata = self.metadata[idx]
            result = {
                'metadata': metadata,
                'similarity_score': float(1 / (1 + dist)),
            }
            
            if metadata['type'] == 'text':
                text_results.append(result)
            else:
                image_results.append(result)
        
        # return top k results +  top_k/2 images
        return text_results[:self.top_k] + image_results[:self.top_k//2]

    
    def query_text(self, text: str) -> List[Dict[str, Any]]:
        """Query using text only."""
        query_embedding = self.embed_text(text)
        return self.search(query_embedding)
    
    def query_image(self, image: Union[str, bytes, Image.Image]) -> List[Dict[str, Any]]:
        """Query using image only."""
        query_embedding = self.embed_image(image)
        return self.search(query_embedding)
    
    def query_multimodal(self, text: str, image: Union[str, bytes, Image.Image]) -> List[Dict[str, Any]]:
        """Query using both text and image."""
        query_embedding = self.embed_multimodal(text, image)
        return self.search(query_embedding)
    