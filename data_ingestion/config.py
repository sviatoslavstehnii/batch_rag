import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://admin:password123@localhost:27017/')
DB_NAME = os.getenv('MONGO_DB_NAME', 'batch_articles')
ARTICLES_COLLECTION = os.getenv('MONGO_COLLECTION_NAME', 'articles')
IMAGES_COLLECTION = os.getenv('MONGO_IMAGES_COLLECTION_NAME', 'images')

# Vector store settings
VECTOR_STORE_DIR = os.getenv('VECTOR_STORE_DIR', 'vector_store')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'distiluse-base-multilingual-cased')
IMAGE_EMBEDDING_MODEL = os.getenv('IMAGE_EMBEDDING_MODEL', 'openai/clip-vit-base-patch32')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50)) 