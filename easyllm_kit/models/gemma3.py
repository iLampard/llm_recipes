from transformers import BitsAndBytesConfig, AutoProcessor, Gemma3ForConditionalGeneration, Gemma3ForCausalLM, \
    AutoTokenizer
from typing import Union, List
from easyllm_kit.models.base import LLM
from easyllm_kit.utils import get_logger, format_prompt_with_image
import torch

logger = get_logger('easyllm_kit')


@LLM.register('gemma3')
class Gemma3(LLM):
    model_name = 'gemma3'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the Gemma 3 model and processor."""
        try:
            model_id = self.model_config.model_dir

            if self.model_config.model_full_name == 'gemma3-1b':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = Gemma3ForCausalLM.from_pretrained(
                    model_id, quantization_config=quantization_config,
                    device_map=self.model_config.device_map,
                    trust_remote_code=self.model_config.trust_remote_code
                ).eval()
                # Load processor for handling inputs
                self.processor = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=self.model_config.trust_remote_code
                )


            else:
                # Load model with appropriate settings
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map=self.model_config.device_map,
                    trust_remote_code=self.model_config.trust_remote_code
                ).eval()

                # Load processor for handling inputs
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=self.model_config.trust_remote_code
                )

            logger.info(f"Successfully loaded Gemma 3 model from {model_id}")

        except Exception as e:
            logger.error(f"Error loading Gemma 3 model: {str(e)}")
            raise

    def generate(self, prompt: str, **kwargs) -> Union[str, List[str]]:
        """
        Generate text based on the input prompt.

        Args:
            prompt (str): The input prompt for text generation.
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Please check the initialization.")

        # Handle single image path or multiple image paths
        image_dir = kwargs.get('image_dir', None)
        if image_dir and not isinstance(image_dir, list):
            image_dir = [image_dir]  # Convert single path to list

        generated_texts = []

        try:
            # Prepare messages
            messages = [{'role': 'user', 'content': format_prompt_with_image(prompt, image_dir)}]

            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            # Generate
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k
                )
                generation = generation[0][input_len:]

            # Decode
            generated_text = self.processor.decode(generation, skip_special_tokens=True)
            generated_texts.append(generated_text)

        except Exception as e:
            logger.error(f"Error generating text for prompt '{prompt}': {str(e)}")
            generated_texts.append(f"Error: {str(e)}")

        return generated_texts[0] if len(generated_texts) == 1 else generated_texts
