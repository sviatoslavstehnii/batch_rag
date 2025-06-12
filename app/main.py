import streamlit as st
import sys
import os
from PIL import Image
import io
import base64
import json

# Add the parent directory to the path to import the RAG module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import MultimodalRAG

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = MultimodalRAG(top_k=5)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def display_chat_history():
    """Display the chat history in a nice format."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image" in message:
                st.image(message["image"], caption="Uploaded Image")
            if "text_sources" in message and message["text_sources"]:
                st.markdown("**Text Sources:**")
                for url in message["text_sources"]:
                    st.markdown(f"- [{url}]({url})")
            if "image_sources" in message and message["image_sources"]:
                st.markdown("**Image Sources:**")
                for url in message["image_sources"]:
                    st.markdown(f"- [{url}]({url})")

def main():
    st.title("Multimodal RAG Chat")
    st.write("Ask questions about articles and images using text and/or images!")

    # Display chat history
    display_chat_history()

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg'])
    
    # Text input for query
    query = st.chat_input("Ask a question...")

    if query:
        # Process the query
        if uploaded_file:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)
            
            # Add user message with image to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query if query else "Analyze this image",
                "image": image
            })
            
            # Generate response using multimodal RAG
            with st.spinner("Generating response..."):
                response = st.session_state.rag.generate_answer(query if query else "", image)
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })
            
            # Generate response using text-only RAG
            with st.spinner("Generating response..."):
                response = st.session_state.rag.generate_answer(query, None)

        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response['answer'],
            "text_sources": response['text_sources'],
            "image_sources": response['image_sources']
        })

        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main() 