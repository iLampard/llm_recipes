from flask import Flask, render_template, request, jsonify
import os
import json
from pathlib import Path
from easyllm_kit.utils import save_json, read_json

app = Flask(__name__)

# Configuration
TRAIN_DIR = "/data0/shoppal/zhoufan/complete-the-look-dataset/datasets/raw_train"
TEST_DIR = "/data0/shoppal/zhoufan/complete-the-look-dataset/datasets/raw_test"
TAGS_FILE = "image_tags.json"

def load_tags():
    """Load existing tags from JSON file"""
    if os.path.exists(TAGS_FILE):
        return read_json(TAGS_FILE)
    return {}

def save_tags(tags):
    """Save tags to JSON file"""
    save_json(TAGS_FILE, tags)

@app.route('/')
def index():
    # Get all image files from both directories
    train_images = [str(p) for p in Path(TRAIN_DIR).glob('*.jpg')]
    test_images = [str(p) for p in Path(TEST_DIR).glob('*.jpg')]
    
    # Load existing tags
    tags = load_tags()
    
    # Count tagged and untagged images
    total_images = len(train_images) + len(test_images)
    tagged_images = len(tags)
    
    return render_template('index.html',
                         train_images=train_images,
                         test_images=test_images,
                         tags=tags,
                         total_images=total_images,
                         tagged_images=tagged_images)

@app.route('/tag', methods=['POST'])
def tag_image():
    data = request.json
    image_path = data['image_path']
    is_good = data['is_good']
    
    # Load existing tags
    tags = load_tags()
    
    # Update tags
    tags[image_path] = is_good
    
    # Save tags
    save_tags(tags)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)