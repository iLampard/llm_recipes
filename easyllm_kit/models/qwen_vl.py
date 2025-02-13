from typing import Union, List, Dict
from easyllm_kit.models.base import LLM
from easyllm_kit.utils import get_logger
import base64
from PIL import Image
import io

logger = get_logger('easyllm_kit')


# ref: https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
# It's highly recommanded to use `[decord]` feature for faster video loading.
# pip install qwen-vl-utils[decord]==0.0.8


@LLM.register('qwen_vl')
class QwenVL(LLM):
    model_name = 'qwen_vl'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None
        self.processor = None

        if self.model_config.use_api:
            # ref: https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.10c55a02btl2GS#/model-market/detail/qwen2.5-vl-7b-instruct?tabKey=sdk
            import openai
            self.client = openai.OpenAI(
                api_key=self.model_config.api_key,
                base_url=self.model_config.api_url  # "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self.load_model()

    def load_model(self):
        """Load the model and processor."""
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

    def prepare_inputs(self, prompts: List[str], images: List[Union[str, Image.Image]] = None) -> List[Dict]:
        """Prepare inputs for the model."""
        batch_messages = []

        for idx, text in enumerate(prompts):
            message = {
                "role": "user",
                "content": []
            }

            # Add image if available
            if images and idx < len(images):
                image = images[idx]
                if isinstance(image, Image.Image):
                    # Convert PIL Image to base64 string if necessary
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image = f"data:image;base64,{image_base64}"
                else:
                    image = f"data:image;base64,{image}"
                message["content"].append({
                    "type": "image_url",
                    "image_url": image  # Use the image URL or base64 string
                })

            # Add text
            message["content"].append({
                "type": "text",
                "text": text
            })

            batch_messages.append(message)

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
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process images if provided
        images = self._process_images(kwargs.get('image_dir'), kwargs.get('image_format', 'base64'))

        # Prepare batch inputs
        batch_messages = self.prepare_inputs(prompts, images)

        if self.model_config.use_api:
            completion = self.client.chat.completions.create(
                model=self.model_config.model_full_name,
                max_tokens=self.generation_config.max_length,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                messages=batch_messages
            )
            return completion.model_dump_json()
        else:
            if self.model is None:
                raise RuntimeError("Model has not been loaded. Please check the initialization.")


            # Prepare texts for batch inference
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages
            ]

            # Process vision information
            image_inputs, video_inputs = process_vision_info(batch_messages)

            # Prepare inputs using the processor
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Batch Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.generation_config.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode outputs
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Return single string if input was single string
            return output_texts[0] if len(output_texts) == 1 else output_texts

    def _process_images(self, image_dir, image_format):
        """Helper function to process images."""
        images = None
        if image_dir is not None:
            if isinstance(image_dir, str):
                image_dir = [image_dir]

            try:
                if image_format == 'base64':
                    # Directly use the base64 strings
                    images = image_dir
                else:
                    images = [Image.open(image_path) for image_path in image_dir]
            except Exception as e:
                logger.error(f"Failed to process images: {e}")
                raise
        return images
