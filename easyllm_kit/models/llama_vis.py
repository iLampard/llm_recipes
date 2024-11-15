from typing import Union, List, Dict
from easyllm_kit.models.base import LLM
from transformers import MllamaForConditionalGeneration, AutoProcessor
from easyllm_kit.utils import get_logger
import base64
from PIL import Image
import io

logger = get_logger('easyllm_kit')


# ref: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

@LLM.register('llama_vis')
class LlamaVis(LLM):
    model_name = 'llama_vis'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the model and processor."""
        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_config.model_dir,
                torch_dtype=self.model_config.infer_dtype,
                device_map=self.model_config.device_map
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_config.model_dir
            )
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def prepare_inputs(self, prompts: List[str], images: List[Image.Image] = None) -> List[Dict]:
        """Prepare inputs for the model."""
        batch_messages = []

        for idx, text in enumerate(prompts):
            message = {
                "role": "user",
                "content": []
            }

            # Add image if available
            if images and idx < len(images):
                message["content"].append({
                    "type": "image",
                })

            # Add text
            message["content"].append({
                "type": "text",
                "text": text
            })

            batch_messages.append([message])

        return batch_messages

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Generate text based on the input prompts and images.

        Args:
            prompts (Union[str, List[str]]): Text prompts for generation
            **kwargs:
                image_dir (Union[str, List[str], None]): Path(s) to image(s) or base64 strings
                image_format (str): Format of the images ('base64' or 'path')

        Returns:
            Union[str, List[str]]: Generated text(s)
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Please check the initialization.")

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process images if provided
        images = None
        image_dir = kwargs.get('image_dir')
        if image_dir is not None:
            if isinstance(image_dir, str):
                image_dir = [image_dir]

            image_format = kwargs.get('image_format', 'base64')
            try:
                if image_format == 'base64':
                    images = [Image.open(io.BytesIO(base64.b64decode(b64_str))) for b64_str in image_dir]
                else:
                    images = [Image.open(image_path) for image_path in image_dir]
            except Exception as e:
                logger.error(f"Failed to process images: {e}")
                raise

        # Prepare batch inputs
        batch_messages = self.prepare_inputs(prompts, images)

        # Process each batch
        generated_texts = []
        for idx, messages in enumerate(batch_messages):
            try:
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                # Prepare inputs using the processor
                if images is not None:
                    inputs = self.processor(
                        images[idx],
                        input_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to(self.model.device)
                else:
                    inputs = self.processor(
                        None,
                        input_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to(self.model.device)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens
                )

                # Decode outputs
                generated_text = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                )
                generated_texts.append(generated_text)

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise

        # Return single string if input was single string
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts
