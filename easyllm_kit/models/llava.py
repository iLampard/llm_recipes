
from PIL import Image
from typing import Union, List
from transformers import (
  LlavaForConditionalGeneration,
  AutoProcessor
)
import io
import base64

from easyllm_kit.models.base import LLM


@LLM.register('llava')
class Llava(LLM):
    model_name = 'llava'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_config['model_name'])
        self.processor = AutoProcessor.from_pretrained(self.model_config['model_name'])
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
        image_format = self.kwargs.get('image_format', 'base64')
        image_paths = self.kwargs.get('image_paths', None)
        if image_format == 'base64':
            images = [Image.open(io.BytesIO(base64.b64decode(b64_str))) for b64_str in base64_images]
        else:
            images = [Image.open(image_path) for image_path in image_paths]


        # Prepare prompts using the template
        input_prompts = [
            self.processor.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': text}],
                tokenize=False,
                add_generation_prompt=True
            ) for text in prompts
        ]

        inputs = self.processor(text=input_prompts, images=images, return_tensors="pt")

        # Generate
        generate_ids = self.model.generate(**inputs, max_length=30)
        output = self.processor.batch_decode(generate_ids, 
                                             skip_special_tokens=True, 
                                             clean_up_tokenization_spaces=False)[0]
        return output

