from typing import List, Dict, Any

def build_prompt(context: List[Dict[str, Any]], query: str) -> str:
    # context_str = "\n".join([f"Text: {item['metadata']['text']}" for item in context])
    # image_metadata_str = "\n".join([f"Image: {item['metadata']['alt_text']}" for item in context])
    
    text_context = ""
    image_context = ""
    for item in context:
        if item['metadata']['type'] == 'text':
            text_context += f"Text: {item['metadata']['text']}\n"
        elif item['metadata']['type'] == 'image' or item['metadata']['type'] == 'alt_text':
            image_context += f"Image: {item['metadata']['alt_text']}\n"
            
    
    return f"""
    You are a helpful assistant that can answer questions using the following context:
    Text: {text_context}
    Image: {image_context}
    Question: {query}
    Answer:
    """