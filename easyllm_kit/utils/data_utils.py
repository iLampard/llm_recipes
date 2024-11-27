import base64
import json
import os
import random
import re
from pathlib import Path

from PIL import Image
from io import BytesIO
from typing import Union, List, Dict, Any, Optional

from datasets import load_dataset


def process_base64_image(base64_string, output_path, save_format='PNG'):
    """
    Process a base64 encoded image string and save it as PNG.

    Args:
        base64_string (str): The base64 encoded image string
        output_path (str): Path where to save the PNG file
        save_format (str): Format to save the PNG file

    Returns:
        str: Path to the saved image if successful, None if failed
    """
    try:
        # Decode base64 string to bytes
        image_data = base64.b64decode(base64_string)

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_data))

        # Save image as PNG
        image.save(output_path, save_format)

        return output_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def read_image_as_bytes(image_path):
    """Read an image file and return its bytes."""
    with open(image_path, "rb") as image_file:
        return image_file.read()  # Read the image as bytes


def save_json(data: Union[List, dict], filename: str) -> None:
    """
    Save data as JSON to the specified path
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(filename: str) -> Union[List, dict]:
    """
    Read JSON data from the specified file
    """
    with open(filename, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def sample_json_records(
        data: Union[Dict[str, Any], str, Path],
        n_samples: int,
        seed: int = None,
        preserve_keys: bool = True,
        output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sample a fixed number of records from a JSON structure where records are numbered keys.

    Args:
        data: Input JSON data as either a dictionary, string (file path), or Path object
        n_samples: Number of records to sample
        seed: Random seed for reproducibility
        preserve_keys: If True, keeps original keys; if False, renumbers from 0

    Returns:
        Dictionary containing the sampled records
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Load JSON if string or Path is provided
    if isinstance(data, (str, Path)):
        data = read_json(data)

    # Get all keys (excluding any metadata keys that might start with "_")
    keys = [k for k in data.keys() if not k.startswith('_')]

    # Validate sample size
    max_samples = len(keys)
    if n_samples > max_samples:
        raise ValueError(f"Requested {n_samples} samples but only {max_samples} records available")

    # Sample keys
    sampled_keys = random.sample(keys, n_samples)

    # Create new dictionary with sampled records
    if preserve_keys:
        # Keep original keys
        sampled_data = {k: data[k] for k in sampled_keys}
    else:
        # Renumber from 0
        sampled_data = {
            str(i): data[k]
            for i, k in enumerate(sampled_keys)
        }

    if output_file:
        save_json(sampled_data, output_file)

    return sampled_data


def extract_json_from_text(text: str):
    """
    Extract and format the JSON object from a given text.

    Args:
        text (str): The input text containing a JSON object.

    Returns:
        dict: The extracted JSON object as a dictionary, or {'intention': 'error parsing'} if an error occurs.
    """
    try:
        # Use regex to find the JSON part in the text
        json_match = re.search(r'```json\s*\{.*?\}\s*```', text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in the provided text.")

        # Extract the JSON string and clean it
        json_str = json_match.group()
        json_str = json_str.replace('```json', '').replace('```', '').strip()

        # Parse the JSON string to a dictionary
        json_dict = json.loads(json_str)
        return json_dict
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print(text)
        return {'intention': 'error parsing'}


def convert_to_json_list(dataset):
    """
    Convert data in Dataset format to list format
    """
    json_list = []
    for sample in dataset:
        sample_dict = dict(sample)
        for key, value in sample_dict.items():
            if isinstance(value, Image.Image):
                sample_dict[key] = image_to_base64(value)
        json_list.append(sample_dict)
    return json_list


def download_data_from_hf(
        hf_dir: str,
        subset_name: Union[str, List[str], None] = None,
        split: Union[str, List[str], None] = None,
        save_dir: str = "./data"
) -> None:
    """
    Download from huggingface repo and convert all data files into json files
    """
    if subset_name is None:
        subsets = [None]
    elif isinstance(subset_name, str):
        subsets = [subset_name]
    else:
        subsets = subset_name

    if split is None:
        splits = [None]
    elif isinstance(split, str):
        splits = [split]
    else:
        splits = split

    for subset in subsets:
        # Load the dataset
        if subset is None:
            dataset = load_dataset(hf_dir, split=split)
            subset = "main"  # Use "main" as the folder name when there's no subset
        else:
            dataset = load_dataset(hf_dir, subset, split=split)

        for split_name in splits:
            if split is None:
                split_data = dataset[split_name]
            else:
                split_data = dataset

            json_list = convert_to_json_list(split_data)

            split_path = os.path.join(save_dir, subset,
                                      f"{subset}_{split_name}.json" if subset else f"{split_name}.json")
            os.makedirs(os.path.dirname(split_path), exist_ok=True)

            save_json(json_list, split_path)
            print(f"Saved {split_name} split of {subset} subset to {split_path}")
