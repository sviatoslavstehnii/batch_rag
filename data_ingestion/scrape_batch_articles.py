import sys
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime
import os
from bson import BSON
import base64
from urllib.parse import urljoin
import argparse
from config import MONGO_URI, DB_NAME, ARTICLES_COLLECTION, IMAGES_COLLECTION
from PIL import Image
import io
from tqdm import tqdm
import time

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def flush_database(db):
    """Clear all collections in the database."""
    try:
        # Drop all collections
        db[ARTICLES_COLLECTION].drop()
        db[IMAGES_COLLECTION].drop()
        print("Successfully cleared all collections")
    except Exception as e:
        print(f"Error clearing database: {str(e)}")

def process_image(image_data: bytes, max_size: tuple = (800, 800), quality: int = 75) -> bytes:
    """Process image: resize and compress."""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Resize image if it's larger than max_size
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save to bytes with compression
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def download_image(image_url, base_url):
    """Download image and return it as base64 string."""
    try:
        # Handle relative URLs
        full_url = urljoin(base_url, image_url)
        response = requests.get(full_url)
        response.raise_for_status()
        
        # Process image (resize and compress)
        processed_image = process_image(response.content)
        if processed_image is None:
            return None
        
        # Convert to base64
        image_data = base64.b64encode(processed_image).decode('utf-8')
        return image_data
    except Exception as e:
        print(f"Error downloading image {image_url}: {str(e)}")
        return None

def store_image(db, image_data, article_id):
    """Store image in the images collection and return its ID."""
    image_doc = {
        'url': image_data['url'],
        'data': image_data['data'],
        'alt_text': image_data['alt_text'],
        'article_id': article_id,
        'scraped_at': datetime.utcnow()
    }
    result = db[IMAGES_COLLECTION].insert_one(image_doc)
    return result.inserted_id

def get_article_links(page_url, headers):
    """Get all article links from a page."""
    try:
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        article_links = []
        article_elements = soup.find_all('article')
        
        for article in article_elements:
            a_tags = article.find_all('a')
            if len(a_tags) > 1:
                link = urljoin(page_url, a_tags[1]['href'])
                article_links.append(link)
        
        return article_links
    except Exception as e:
        print(f"Error getting article links from {page_url}: {str(e)}")
        return []

def get_next_page_url(base_url, page_num):
    """Get the URL of the next page if it exists."""
    return f"{base_url}/page/{page_num + 1}"

def scrape_article(article_url, headers):
    """Scrape a single article and its images."""
    try:
        response = requests.get(article_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        title = soup.find('h1').text.strip()
        
        content_container = soup.find('div', class_='post_postContent__wGZtc')
        if not content_container:
            print(f"Could not find content container for article: {article_url}")
            return None, []
            
        content = content_container.find_all('p')
        content = [p.text.strip() for p in content]
        content = '\n'.join(content)
        
        images = []
        img_elements = content_container.find_all('img')
        for img in img_elements:
            img_url = img.get('src')
            if img_url:
                image_data = download_image(img_url, article_url)
                if image_data:
                    images.append({
                        'url': img_url,
                        'data': image_data,
                        'alt_text': img.get('alt', '')
                    })
        
        article_data = {
            "title": title,
            "content": content,
            "image_count": len(images),
            "url": article_url,
            "scraped_at": datetime.utcnow()
        }
        
        return article_data, images
    except Exception as e:
        print(f"Error scraping article {article_url}: {str(e)}")
        return None, []

def scrape_articles():
    """Scrape all articles from The Batch website."""
    base_url = "https://www.deeplearning.ai/the-batch/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    all_articles = []
    page_num = 1
    num_pages = 2
    current_page_url = get_next_page_url(base_url, page_num)
    
    print("Starting to scrape all pages...")
    while page_num < num_pages:
        print(f"\nScraping page {page_num}...")
        
        # Get article links from current page
        article_links = get_article_links(current_page_url, headers)
        if not article_links:
            print(f"No articles found on page {page_num}")
            break
            
        print(f"Found {len(article_links)} articles on page {page_num}")
        
        # Scrape each article
        for article_url in tqdm(article_links, desc=f"Processing articles on page {page_num}"):
            article_data, images = scrape_article(article_url, headers)
            if article_data:
                all_articles.append((article_data, images))
            
            time.sleep(0.1)
        
        # Get next page URL
        page_num += 1
        current_page_url = get_next_page_url(base_url, page_num)
            
    print(f"\nFinished scraping {page_num-1} pages")
    return all_articles

def main():
    """Main function to scrape articles and store them in MongoDB."""
    parser = argparse.ArgumentParser(description='Scrape articles from The Batch and store in MongoDB')
    parser.add_argument('--flush', action='store_true', help='Clear all collections before scraping')
    args = parser.parse_args()

    db = connect_to_mongodb()
    
    if args.flush:
        flush_database(db)
    
    articles_with_images = scrape_articles()
    
    if articles_with_images:
        total_images = 0
        print("\nStoring articles and images in MongoDB...")
        for article_data, images in tqdm(articles_with_images, desc="Storing articles"):
            article_result = db[ARTICLES_COLLECTION].insert_one(article_data)
            article_id = article_result.inserted_id
            
            for image in tqdm(images, desc="Storing images", leave=False):
                store_image(db, image, article_id)
                total_images += 1
        
        print(f"\nSuccessfully inserted {len(articles_with_images)} articles and {total_images} images into MongoDB")
    else:
        print("No articles were scraped")

if __name__ == "__main__":
    main() 