from PIL import Image
from transformers import AutoModel, AutoTokenizer
from typing import Union, List
import io
import base64
from easyllm_kit.models.base import LLM


@LLM.register('minicpm')
class MiniCPM(LLM):
    model_name = 'minicpm'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_config.model_dir,
                                               torch_dtype=self.model_config.infer_dtype,
                                               device_map=self.model_config.device_map)
        self.processor = AutoTokenizer.from_pretrained(self.model_config.model_dir)
        return

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> str:
        """
        Generate text based on the input prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_length (int): The maximum length of the generated text.
            temperature (float): Sampling temperature for generation.
            top_p (float): Nucleus sampling parameter.
            num_return_sequences (int): Number of sequences to generate.

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

            if image_format == 'base64':
                images = [Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('RGB') for b64_str in image_dir]
            else:
                images = [Image.open(image).convert('RGB') for image in image_dir]

        # Prepare batch messages
        msgs_batch = [
            [{'role': 'user', 'content': [image, question]}]
            for image, question in zip(images, prompts)
        ]

        # Process each input in the batch
        results = []
        for msgs in msgs_batch:
            res = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.processor
            )
            results.append(res)
        return results if len(results) > 1 else results[0]
