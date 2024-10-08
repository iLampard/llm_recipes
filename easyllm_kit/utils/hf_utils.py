from typing import Union, List
from huggingface_hub import login
from easyllm_kit.configs.base import Config
from easyllm_kit.utils.data_utils import download_data_from_hf


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
    @staticmethod
    def login_from_config(self, config_path: str):
        """
        Login to Hugging Face using a token from a YAML config file.

        Args:
            config_path (str): Path to the YAML config file.
        """

        hf_config = Config.build_from_yaml_file(config_path)

        if not hf_config.hf_token:
            raise Warning("No 'hf_token' found in the config file.")

        login(token=hf_config.hf_token)

    @staticmethod
    def download_data_from_hf(
            hf_dir: str,
            subset_name: Union[str, List[str], None] = None,
            split: Union[str, List[str], None] = None,
            save_dir: str = "./data"
    ) -> None:
        """
        Download from huggingface repo and convert all data files into json files
        """
        download_data_from_hf(hf_dir, subset_name, split, save_dir)
