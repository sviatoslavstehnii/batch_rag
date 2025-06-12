import os
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import base64
import torch
from google import genai
from config import TEXT_EMBEDDING_MODEL, IMAGE_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

class Embedder:
    def __init__(self):
        """Initialize the Embedder with all necessary models."""
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        self.clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)
        self.gemini_model = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = ' '.join(words[i:i + CHUNK_SIZE])
            chunks.append(chunk)
            
        return chunks

    def process_text_chunk(self, chunk: str, article_id: str, chunk_idx: int, url: str) -> Tuple[str, np.ndarray, Dict]:
        """Process a text chunk and return its ID, embedding, and metadata."""
        chunk_id = f"article_{article_id}_chunk_{chunk_idx}"
        embedding = self.text_model.encode(chunk)
        metadata = {
            "type": "text",
            "text": chunk,
            "article_id": article_id,
            "url": url,
            "linked_images": []
        }
        return chunk_id, embedding, metadata

    def process_image(self, image_data: bytes, article_id: str, image_idx: int, alt_text: str, url: str) -> Tuple[str, np.ndarray, Dict]:
        """Process an image and return its ID, embedding, and metadata."""
        image_id = f"article_{article_id}_img_{image_idx}"
        
        # Decode and process image
        image = Image.open(io.BytesIO(image_data))
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs).numpy()[0]
        
        metadata = {
            "type": "image",
            "article_id": article_id,
            "url": url,
            "linked_chunks": [],
            "alt_text": alt_text
        }
        return image_id, embedding, metadata

    def process_alt_text(self, alt_text: str, url: str) -> Tuple[str, np.ndarray, Dict]:
        """Process an alt text and return its ID, embedding, and metadata."""
        alt_text_id = f"alt_text_{alt_text}"
        embedding = self.text_model.encode(alt_text)
        metadata = {
            "type": "alt_text",
            "alt_text": alt_text,
            "url": url
        }
        return alt_text_id, embedding, metadata

    def process_image_description(self, image_data: bytes, article_id: str, image_idx: int, url: str) -> Optional[Tuple[str, np.ndarray, Dict]]:
        """Generate and process image description using Gemini."""
        try:
            # Open and convert image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate description using Gemini
            prompt = "Please provide a description of this image. Focus on the main elements, actions, and context."
            response = self.gemini_model.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            description = response.text

            # Create embedding for the description
            desc_id = f"article_{article_id}_img_{image_idx}_desc"
            embedding = self.text_model.encode(description)
            metadata = {
                "type": "image_description",
                "description": description,
                "article_id": article_id,
                "url": url,
                "linked_images": [f"article_{article_id}_img_{image_idx}"]
            }
            return desc_id, embedding, metadata

        except Exception as e:
            print(f"Error generating image description: {str(e)}")
            time.sleep(10)
            return self.process_image_description(image_data, article_id, image_idx, url)


    def link_embeddings(self, chunk_embeddings: List[Tuple], image_embeddings: List[Tuple]):
        """Link chunks and images by updating their metadata."""
        for chunk_id, _, chunk_metadata in chunk_embeddings:
            chunk_metadata["linked_images"] = [img_id for img_id, _, _ in image_embeddings]
        
        for img_id, _, img_metadata in image_embeddings:
            img_metadata["linked_chunks"] = [chunk_id for chunk_id, _, _ in chunk_embeddings] 