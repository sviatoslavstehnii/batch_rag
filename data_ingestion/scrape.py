import requests
import json
import time
from pymongo import MongoClient
from datetime import datetime
import base64
from urllib.parse import urljoin
from PIL import Image
import io
from tqdm import tqdm
from bs4 import BeautifulSoup
from config import (
    MONGO_URI, DB_NAME, ARTICLES_COLLECTION, IMAGES_COLLECTION,
    GHOST_API_BASE, GHOST_API_KEY, DEEPLEARNING_AI_BASE, COOKIE,
    POSTS_PER_PAGE, INITIAL_TAGS
)

class GhostScraper:
    def __init__(self, flush_db=False):
        self.db = self._connect_to_mongodb()
        if flush_db:
            self._flush_database()

    def _connect_to_mongodb(self):
        """Connect to MongoDB and return the database object."""
        client = MongoClient(MONGO_URI)
        return client[DB_NAME]

    def _flush_database(self):
        """Clear all collections in the database."""
        try:
            self.db[ARTICLES_COLLECTION].drop()
            self.db[IMAGES_COLLECTION].drop()
            print("Successfully cleared all collections")
        except Exception as e:
            print(f"Error clearing database: {str(e)}")

    def _process_image(self, image_data: bytes, max_size: tuple = (800, 800), quality: int = 75) -> bytes:
        """Process image: resize and compress."""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            return output.getvalue()
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def _download_image(self, image_url):
        """Download image and return it as base64 string."""
        try:
            full_url = urljoin(DEEPLEARNING_AI_BASE, image_url)
            response = requests.get(full_url)
            response.raise_for_status()
            
            processed_image = self._process_image(response.content)
            if processed_image is None:
                return None
            
            image_data = base64.b64encode(processed_image).decode('utf-8')
            return image_data
        except Exception as e:
            print(f"Error downloading image {image_url}: {str(e)}")
            return None

    def _store_image(self, image_data, article_id):
        """Store image in the images collection and return its ID."""
        image_doc = {
            'url': image_data['url'],
            'data': image_data['data'],
            'alt_text': image_data['alt_text'],
            'article_id': article_id,
            'scraped_at': datetime.utcnow()
        }
        result = self.db[IMAGES_COLLECTION].insert_one(image_doc)
        return result.inserted_id

    def _get_num_articles(self, tag_slug):
        """Given a tag slug, return the number of articles under that tag."""
        resp = requests.get(f'https://www.deeplearning.ai/_next/data/{COOKIE}/the-batch/tag/{tag_slug}.json?slug={tag_slug}')
        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError:
            print(f"  Failed to decode JSON for tag '{tag_slug}': {resp.text}")
            return 0
        try:
            num_posts = data['pageProps']['tag']['count']['posts']
        except KeyError:
            print(f"  Failed to fetch tag '{tag_slug}': {data}")
            return 0
        return num_posts

    def _fetch_posts_for_tag(self, tag_slug, page):
        """Fetches one page of posts for a given tag slug."""
        url = (
            f"{GHOST_API_BASE}/posts/"
            f"?key={GHOST_API_KEY}"
            f"&include=tags%2Cauthors"
            f"&filter=tag:{tag_slug}"
            f"&page={page}"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get("posts", [])

    def _extract_images_from_html(self, html_content):
        """Extract all image URLs and their alt text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        images = []
        
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                images.append({
                    'url': src,
                    'alt_text': img.get('alt', '')
                })
        
        return images

    def _process_post(self, post):
        """Process a single post and store it in MongoDB."""
        # Extract article data
        article_data = {
            "title": post.get("title"),
            "content": post.get("html"),
            "url": urljoin(DEEPLEARNING_AI_BASE, post.get("slug", "")),
            "scraped_at": datetime.utcnow()
        }
        
        # Store article
        article_result = self.db[ARTICLES_COLLECTION].insert_one(article_data)
        article_id = article_result.inserted_id
        
        # Process and store all images from the article content
        images = self._extract_images_from_html(post.get("html", ""))
        
        # Also add feature image if it exists
        if post.get("feature_image"):
            images.append({
                'url': post.get("feature_image"),
                'alt_text': post.get("feature_image_alt", "")
            })
        
        # Download and store all images
        for image_info in images:
            image_data = self._download_image(image_info['url'])
            if image_data:
                self._store_image({
                    'url': image_info['url'],
                    'data': image_data,
                    'alt_text': image_info['alt_text']
                }, article_id)
        
        return article_id

    def crawl_all_tags(self):
        """Crawl Ghost posts by tag and store in MongoDB."""
        tags_to_process = list(INITIAL_TAGS)
        seen_tags = set()
        
        while tags_to_process:
            current_tag = tags_to_process.pop(0)
            if current_tag in seen_tags:
                continue
                
            print(f"\n=== Crawling tag: '{current_tag}' ===")
            
            total_articles = self._get_num_articles(current_tag)
            if total_articles == 0:
                print(f"  No articles found for tag '{current_tag}'.")
                seen_tags.add(current_tag)
                continue
                
            total_pages = (total_articles // POSTS_PER_PAGE) + (1 if total_articles % POSTS_PER_PAGE > 0 else 0)
            print(f"  Found {total_articles} posts under tag '{current_tag}', across {total_pages} pages.")
            
            for page in tqdm(range(1, total_pages + 1), desc=f"Processing pages for tag '{current_tag}'"):
                try:
                    posts = self._fetch_posts_for_tag(current_tag, page)
                except requests.HTTPError as e:
                    print(f"  Failed to fetch tag '{current_tag}' (page {page}): {e}")
                    break
                    
                print(f"  Processing page {page}/{total_pages}, {len(posts)} posts.")
                
                for post in tqdm(posts, desc="Processing posts", leave=False):
                    self._process_post(post)
                
                time.sleep(0.3)  # Avoid hitting API too hard
                
            seen_tags.add(current_tag)
            print(f"=== Finished tag: '{current_tag}' ===")
        
        print("\nAll tags processed. Crawl complete.")

def main():
    scraper = GhostScraper(flush_db=True)
    scraper.crawl_all_tags()

if __name__ == "__main__":
    main() 