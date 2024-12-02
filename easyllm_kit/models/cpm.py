from PIL import Image
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Union, List
import io
import base64
from easyllm_kit.models.base import LLM
from easyllm_kit.utils import get_logger
import torch

logger = get_logger('easyllm_kit')


@LLM.register('minicpm')
class MiniCPM(LLM):
    model_name = 'minicpm'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import LLM as vLLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_dir,
                                              trust_remote_code=True)
            self.model = vLLM(
                model=self.model_config.model_dir,
                trust_remote_code=self.model_config.trust_remote_code
    )
        else:
            from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

            # Get available GPU devices
            if torch.cuda.is_available():
                gpu_device_ids = list(range(torch.cuda.device_count()))
                logger.info(f"Found {len(gpu_device_ids)} GPU devices: {gpu_device_ids}")
            else:
                logger.warning("No GPU devices found, using CPU")
                gpu_device_ids = []

            # Configure memory for each GPU
            max_memory = None
            if self.model_config.max_memory:
                max_memory = {
                    device_id: self.model_config.max_memory.get(str(device_id), "24GiB")
                    for device_id in gpu_device_ids
                }
                if 'cpu' in self.model_config.max_memory:
                    max_memory['cpu'] = self.model_config.max_memory['cpu']

            # Load model configuration
            config = AutoConfig.from_pretrained(
                self.model_config.model_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )

            # Initialize empty model
            with init_empty_weights():
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=self.model_config.trust_remote_code
                )

            # Define modules that should not be split
            no_split_module_classes = ["LlamaDecoderLayer"]

            # Infer device map
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes
            )

            logger.info(f"Determined device map: {device_map}")

            # Ensure critical layers are on the first GPU
            device_map.update({
                "llm.model.embed_tokens": 0,
                "llm.model.layers.0": 0,
                "llm.lm_head": 0,
                "vpm": 0,
                "resampler": 0
            })

            # Load checkpoint with device map
            load_checkpoint_in_model(
                model,
                self.model_config.model_dir,
                device_map=device_map
            )

            # Dispatch model across devices
            self.model = dispatch_model(
                model,
                device_map=device_map
            )

            # Load tokenizer
            self.processor = AutoTokenizer.from_pretrained(
                self.model_config.model_dir,
                trust_remote_code=self.model_config.trust_remote_code
            )

            # Set model to evaluation mode
            torch.set_grad_enabled(False)
            self.model.eval()

            logger.info(f"Successfully loaded MiniCPM model across {len(gpu_device_ids)} GPUs")

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
            try:
                if image_format == 'base64':
                    images = [Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('RGB') for b64_str in image_dir]
                else:
                    images = [Image.open(image).convert('RGB') for image in image_dir]
            except Exception as e:
                logger.error(f"Error loading images: {e}")
                images = None


        msgs_batch, stop_token_ids = self._prepare_input(prompts, 
                                                         images, 
                                                         self.model_config.use_vllm)
        
        results = self._generate(msgs_batch, images, stop_token_ids, self.model_config.use_vllm)

        return results


    def _prepare_input(self, prompts, images, use_vllm):
        if use_vllm:    
            stop_tokens = ['<|im_end|>', '<|endoftext|>']
            stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
            if images:
                msgs_batch = [
                    [{'role': 'user', 'content': f'(<image>./</image>)\n{question}'}]
                    for question in prompts
                ]
            else:
                msgs_batch = [
                    [{'role': 'user', 'content': question}]
                    for question in prompts
                ]
            return msgs_batch, stop_token_ids
        else:
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

    def _generate(self, msgs_batch, images, stop_token_ids, use_vllm):
        if use_vllm:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                stop_token_ids=stop_token_ids
            )

            text_inputs = self.tokenizer.apply_chat_template(msgs_batch,
                                           tokenize=False,
                                           add_generation_prompt=True)
            inputs = {
                    "prompt": text_inputs,
                    "multi_modal_data": {
                        'image': images
                    },
            }
            results = self.model.generate(inputs, 
                                          sampling_params=sampling_params)
        else:
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