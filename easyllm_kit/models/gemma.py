from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
import io
import base64
from easyllm_kit.models.base import LLM
from easyllm_kit.utils import get_logger
import torch

logger = get_logger('easyllm_kit')


@LLM.register('gemma2')
class Gemma(LLM):
    model_name = 'gemma2'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_config.model_dir, trust_remote_code=True).to(self.model_config.device)

        # Set model to evaluation mode
        torch.set_grad_enabled(False)
        self.model.eval()

        logger.info(f"Successfully loaded Gemma model")

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> str:
        """
        Generate text based on the input prompt.

        Args:
            prompts (Union[str, List[str]]): The input prompt(s) for text generation.
            **kwargs: Additional arguments for generation.

        Returns:
            str: The generated text.
        """

        if self.model is None:
            raise RuntimeError("Model has not been loaded. Please check the initialization.")

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert single string to list

        # Decode base64 images to PIL Image format
        image_format = kwargs.get('image_format', 'base64')
        image_dir = kwargs.get('image_dir', None)
        images = None

        # Ensure image_dir is a list
        if image_dir is not None:
            if isinstance(image_dir, str):
                image_dir = [image_dir]  # Convert single string to list
            elif not isinstance(image_dir, list):
                raise ValueError("image_dir must be a string or a list of strings.")
            try:
                if image_format == 'base64':
                    images = [Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('RGB') for b64_str in image_dir]
                else:
                    images = [Image.open(image).convert('RGB') for image in image_dir]
            except Exception as e:
                logger.error(f"Error loading images: {e}")
                images = None

        msgs_batch, stop_token_ids = self._prepare_input(prompts, images)

        results = self._generate(msgs_batch, images, stop_token_ids)

        return results

    def _prepare_input(self, prompts, images):
        # Prepare batch messages
        if images:
            # Case with images
            msgs_batch = [
                [{'role': 'user', 'content': [image, question]}]
                for image, question in zip(images, prompts)
            ]
        else:
            # Case without images (text-only)
            msgs_batch = [
                [{'role': 'user', 'content': question}]
                for question in prompts
            ]
        return msgs_batch, None

    def _generate(self, msgs_batch, images, stop_token_ids):
        # Process each input in the batch
        results = []
        for msgs in msgs_batch:
            res = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer
            )
            results.append(res)
        return results if len(results) > 1 else results[0]
