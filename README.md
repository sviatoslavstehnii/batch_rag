# The Batch Article Scraper

This project scrapes articles from The Batch newsletter (https://www.deeplearning.ai/the-batch/) and stores them in MongoDB, with articles and images stored in separate collections.

## Prerequisites

- Docker and Docker Compose
- Python 3.8 or higher
- pip (Python package manager)

## Setup

1. Clone this repository
2. Create a `.env` file with the following content:
   ```
   MONGO_URI=mongodb://admin:password123@localhost:27017/
   MONGO_DB_NAME=batch_articles
   MONGO_COLLECTION_NAME=articles
   MONGO_IMAGES_COLLECTION_NAME=images
   MONGO_USERNAME=admin
   MONGO_PASSWORD=password123
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start MongoDB using Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Run the scraper:
   ```bash
   # Normal scraping
   python scrape_batch_articles.py
   
   # Clear database before scraping
   python scrape_batch_articles.py --flush
   ```

## Command Line Options

- `--flush`: Clear all collections before scraping new data

## MongoDB Connection Details

- Host: localhost
- Port: 27017
- Username: admin
- Password: password123
- Database: batch_articles
- Collections: 
  - articles
  - images

## Data Structure

### Articles Collection
Each article document has the following structure:
```json
{
    "title": "Article Title",
    "date": "Publication Date",
    "content": "Article Content",
    "image_count": 2,
    "scraped_at": "Timestamp of when the article was scraped"
}
```

### Images Collection
Each image document has the following structure:
```json
{
    "url": "Original image URL",
    "data": "Base64 encoded image data",
    "alt_text": "Image alt text",
    "article_id": "Reference to the parent article's _id",
    "scraped_at": "Timestamp of when the image was scraped"
}
```

## Notes

- The scraper includes error handling for failed requests and article processing
- Articles and images are stored in separate collections for better performance and management
- Images are linked to their parent articles using the `article_id` field
- The MongoDB container uses a named volume for data persistence
- Environment variables can be customized in the `.env` file
- Use the `--flush` option to clear existing data before scraping 