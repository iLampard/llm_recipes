from typing import Union, List
from huggingface_hub import login
from easyllm_kit.configs.base import Config
from easyllm_kit.utils.data_utils import download_data_from_hf
from easyllm_kit.utils.log_utils import get_logger

logger = get_logger('easyllm_kit')


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
    def login_from_config(config_path: str):
        """
        Login to Hugging Face using a token from a YAML config file.

        Args:
            config_path (str): Path to the YAML config file.
        """

        hf_config = Config.build_from_yaml_file(config_path)

        if not hf_config.hf_token:
            logger.warning("No 'hf_token' found in the config file.")
            return

        try:
            login(token=hf_config.hf_token)
            logger.info("Successfully logged in to Hugging Face.")
        except Exception as e:
            logger.error(f"Failed to log in: {e}")
            raise

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
        try:
            download_data_from_hf(hf_dir, subset_name, split, save_dir)
            logger.info(f"Data downloaded successfully from {hf_dir} to {save_dir}.")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise

    @staticmethod
    def download_model_from_hf(
            model_repo: str,
            save_dir: str = "./models"
    ) -> None:
        """
        Download a model from Hugging Face Hub.

        Args:
            model_repo (str): The model repository ID on Hugging Face Hub.
            save_dir (str): Directory to save the downloaded model.
        """
        from huggingface_hub import snapshot_download
        # Download the entire model, if some files are missing they will be downloaded automatically
        try:
            snapshot_download(repo_id=model_repo, local_dir=save_dir, resume_download=True)
            logger.info(f"Model {model_repo} downloaded successfully to {save_dir}.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
