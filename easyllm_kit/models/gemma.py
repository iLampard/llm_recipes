from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List
import io
import base64
from easyllm_kit.models.base import LLM
from easyllm_kit.utils import get_logger
import torch

logger = get_logger('easyllm_kit')


@LLM.register('gemma2')
class Gemma2(LLM):
    model_name = 'gemma2'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_dir, trust_remote_code=True).to(
            self.model_config.device)

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

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Move inputs to the specified device
        inputs = {key: value.to(self.model_config.device) for key, value in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            do_sample=self.generation_config.do_sample,
            max_new_tokens=self.generation_config.max_new_tokens,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            num_beams=self.generation_config.num_beams,
            repetition_penalty=self.generation_config.repetition_penalty,
            length_penalty=self.generation_config.length_penalty
        )

        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return generated_text if len(generated_text) > 1 else generated_text[0]
