from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
import math
from easyllm_kit.utils import read_json, save_json

app = Flask(__name__)

# Configuration
IMAGES_CSV = "outfit_images.csv"
TAGS_FILE = "image_tags.json"
IMAGES_PER_PAGE = 40

class TagManager:
    def __init__(self, tags_file):
        self.tags_file = tags_file
        # Load saved tags from file
        self.saved_tags = read_json(tags_file) if os.path.exists(tags_file) else {}
        # Initialize memory tags with saved tags
        self.memory_tags = self.saved_tags.copy()

    def get_tags(self):
        return self.memory_tags

    def add_tag(self, image_url, is_good):
        self.memory_tags[image_url] = is_good

    def save_tags(self):
        if self.memory_tags != self.saved_tags:
            save_json(self.memory_tags, self.tags_file)
            self.saved_tags = self.memory_tags.copy()
            return True
        return False

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
    
    # Extract image URLs from the f_images column with error handling
    df['urls'] = df['f_images'].apply(extract_url)
    
    filtered_df = df[df['labels'].apply(lambda x: False if pd.isna(x) else 'collage' in json.loads(x))]
    # Filter out None values and get the list of valid URLs
    all_image_urls = [url for url in filtered_df['urls'].tolist() if url is not None]
    
    # Calculate total pages
    total_images = len(all_image_urls)
    total_pages = math.ceil(total_images / IMAGES_PER_PAGE)
    
    # Get current page's images
    start_idx = (page - 1) * IMAGES_PER_PAGE
    end_idx = start_idx + IMAGES_PER_PAGE
    current_page_urls = all_image_urls[start_idx:end_idx]
    
    # Calculate progress
    tagged_images = len(tag_manager.get_tags())
    
    return render_template('index.html',
                         image_urls=current_page_urls,
                         tags=tag_manager.get_tags(),
                         total_images=total_images,
                         tagged_images=tagged_images,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/tag', methods=['POST'])
def tag_image():
    data = request.json
    image_url = data['image_url']
    is_good = data['is_good']
    
    # Add tag to memory only
    tag_manager.add_tag(image_url, is_good)
    
    return jsonify({'status': 'success'})

@app.route('/save', methods=['POST'])
def save_tags():
    """Endpoint to save tags to file"""
    success = tag_manager.save_tags()
    return jsonify({'status': 'success' if success else 'no changes'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)