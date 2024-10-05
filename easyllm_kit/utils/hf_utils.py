from typing import Optional, Union, List
import os
from huggingface_hub import HfApi, login
from easyllm_kit.configs.base import Config
from easyllm_kit.utils.data_utils import convert_to_json_list, save_json
from datasets import load_dataset


# Debugging: Print the evaluation metrics after training
def print_evaluation_metrics(trainer):
    eval_result = trainer.evaluate()
    message = f"Evaluation Metrics: {eval_result}"
    return message


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    message = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    return message


def print_trainable_layers(model):
    # print trainable parameters for inspection
    message = "Trainable layers:\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            message += f"\t{name}\n"
    return message.strip()  # Remove trailing newline


class HFHelper:
    def __init__(self, config_dir: Optional[str] = None):
        self.api = HfApi()
        if config_dir is not None:
            self.config = Config.build_from_yaml_file(config_dir)
            self.token = self.config.get("hf_token")

        if self.token:
            login(token=self.token)

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

                save_json(split_path, json_list)
                print(f"Saved {split_name} split of {subset} subset to {split_path}")
