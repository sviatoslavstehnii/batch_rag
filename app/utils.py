import os
from typing import List, Dict, Any
from google import genai
import requests

def build_prompt(context: List[Dict[str, Any]], query: str, image_description: str) -> str:

    
    text_context = ""
    image_context = ""
    for item in context:
        if item['metadata']['type'] == 'text':
            text_context += f"{item['metadata']['text']}\n\n"
        elif item['metadata']['type'] == 'image' or item['metadata']['type'] == 'alt_text':
            image_context += f"Image Description: {get_image_description(item['metadata']['url'])}\n\n"
        elif item['metadata']['type'] == 'image_description':
            image_context += f"Image Description: {item['metadata']['description']}\n\n"
    

    query = f"{query}"
    if image_description is not None:
        query += f" Query Image Description: {image_description}"
    
    return f"""
    Provide a detailed answer to provided query, you might need to use context below to answer the query:
    
    Query: 
    ```
        {query}
    ```
    

    Text Context: 
    ```
        {text_context}
    ```
    
    Images Context: 
    ```
        {image_context}
    ```
    
    Answer: 
    """


def get_image_description(image_url: str, image_data: bytes = None) -> str:
    """Get the description of an image using Gemini."""
    gemini_model = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    if image_data is None:
        response = requests.get(image_url)
        image_data = response.content
    
    prompt = "Please provide a description of this image. Focus on the main elements, actions, and context."
    response = gemini_model.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_data]
    )
    return response.text