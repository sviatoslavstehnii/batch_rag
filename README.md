# Multimodal RAG Application

A powerful multimodal Retrieval-Augmented Generation (RAG) application that combines text and image processing capabilities. This application allows users to interact with a knowledge base through a modern Streamlit interface, leveraging advanced AI models for both text and image understanding.

## Project Overview

This application implements a multimodal RAG system that:
- Processes and stores both text and image data
- Uses vector embeddings for efficient similarity search
- Provides an intuitive web interface for querying the knowledge base
- Supports multimodal queries (text and images)
- Integrates with MongoDB for persistent storage
- Utilizes FAISS for efficient vector similarity search

## Tech Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **Streamlit**: Web interface framework
- **MongoDB**: Document database for storing articles and metadata
- **FAISS**: Vector similarity search library
- **Docker**: Containerization for easy deployment

### Key Libraries
- **sentence-transformers**: For text embeddings
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library
- **pymongo**: MongoDB Python driver
- **streamlit**: Web application framework
- **pillow**: Image processing
- **numpy**: Numerical computations
- **pandas**: Data manipulation

## Installation

### Prerequisites
- Docker and Docker Compose
- Git

### Setup Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multimodal-rag
   ```

2. Create a `.env` file as shown in `.env.sample`

3. Start with:
```streamlit run app/main.py```

4. OR Build and start the containers:
   ```bash
   docker-compose up --build
   ```

## Usage

1. Access the application:
   - Open your browser and navigate to `http://localhost:8501`

2. Using the Application:
   - The main interface provides a search bar for text queries
   - Upload images for visual search capabilities
   - View search results with both text and image content
   - Filter and sort results as needed

## Development

### Project Structure
```
multimodal-rag/
├── app/                    # Main application code
│   ├── main.py            # Streamlit application entry point
│   ├── rag.py             # RAG implementation
│   └── utils.py           # Utility functions
├── data_ingestion/        # Data processing scripts
├── vector_store/          # Vector storage implementation
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Service orchestration
└── requirements.txt       # Python dependencies
```
