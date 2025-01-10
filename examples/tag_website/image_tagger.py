from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
import math
from easyllm_kit.utils import read_json, save_json
import time

app = Flask(__name__)

# Configuration
IMAGES_CSV = "outfit_images.csv"
TAGS_FILE = "image_tags.json"
IMAGES_PER_PAGE = 20

# Add a class to handle tag operations efficiently
class TagManager:
    def __init__(self, tags_file):
        self.tags_file = tags_file
        self.tags = read_json(tags_file) if os.path.exists(tags_file) else {}
        self.last_save = time.time()
        self.is_dirty = False
        self.save_interval = 2  # Save every 2 seconds if there are changes

    def get_tags(self):
        return self.tags

    def add_tag(self, image_url, is_good):
        self.tags[image_url] = is_good
        self.is_dirty = True
        
        # Only save if enough time has passed since last save
        current_time = time.time()
        if current_time - self.last_save >= self.save_interval:
            self.save_tags()

    def save_tags(self):
        if self.is_dirty:
            save_json(self.tags, self.tags_file)
            self.last_save = time.time()
            self.is_dirty = False

# Initialize the TagManager
tag_manager = TagManager(TAGS_FILE)

def extract_url(x):
    """Safely extract URL from f_images field"""
    try:
        images = eval(x)
        if images and len(images) > 0 and isinstance(images[0], dict) and 'url' in images[0]:
            return images[0]['url']
    except:
        pass
    return None

@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    
    # Read the CSV file containing image URLs
    df = pd.read_csv(IMAGES_CSV)

    filtered_df = df[df['labels'].apply(lambda x: False if pd.isna(x) else 'collage' in json.loads(x))]
    
    # Extract image URLs from the f_images column with error handling
    filtered_df['urls'] = filtered_df['f_images'].apply(extract_url)
    
    # Filter out None values and get the list of valid URLs
    all_image_urls = [url for url in filtered_df['urls'].tolist() if url is not None]
    
    # Load existing tags
    tags = tag_manager.get_tags()
    
    # Calculate total pages
    total_images = len(all_image_urls)
    total_pages = math.ceil(total_images / IMAGES_PER_PAGE)
    
    # Get current page's images
    start_idx = (page - 1) * IMAGES_PER_PAGE
    end_idx = start_idx + IMAGES_PER_PAGE
    current_page_urls = all_image_urls[start_idx:end_idx]
    
    # Calculate progress
    tagged_images = len(tags)
    
    return render_template('index.html',
                         image_urls=current_page_urls,
                         tags=tags,
                         total_images=total_images,
                         tagged_images=tagged_images,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/tag', methods=['POST'])
def tag_image():
    data = request.json
    image_url = data['image_url']
    is_good = data['is_good']
    
    # Add tag without immediate save
    tag_manager.add_tag(image_url, is_good)
    
    return jsonify({'status': 'success'})

# Ensure tags are saved when the application exits
import atexit
atexit.register(lambda: tag_manager.save_tags())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)