import base64
import json
import os
from PIL import Image
from io import BytesIO
from typing import Union, List

from datasets import load_dataset


def image_to_base64(image: Image.Image) -> str:
    """
    Convert image to base64 format
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def save_json(filename: str, data: Union[List, dict]) -> None:
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

            split_path = os.path.join(save_dir, subset, f"{subset}_{split_name}.json" if subset else f"{split_name}.json")
            os.makedirs(os.path.dirname(split_path), exist_ok=True)

            save_json(split_path, json_list)
            print(f"Saved {split_name} split of {subset} subset to {split_path}")

