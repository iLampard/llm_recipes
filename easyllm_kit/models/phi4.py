from typing import Union, List
from easyllm_kit.models.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyllm_kit.utils import get_logger
from easyllm_kit.utils import print_trainable_parameters

logger = get_logger('easyllm_kit')


@LLM.register('phi4')
@LLM.register('phi-4')
class Phi4(LLM):
    model_name = 'phi4'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        
        # Determine model type based on model_full_name or model_dir
        self.model_type = self._determine_model_type()
        logger.info(f"Initializing Phi4 model: {self.model_type}")
        
        self.load_model()
    
    def _determine_model_type(self):
        """Determine the specific Phi-4 model type"""
        model_full_name = getattr(self.model_config, 'model_full_name', None)
        model_dir = getattr(self.model_config, 'model_dir', '')
        
        if model_full_name:
            if 'reasoning' in model_full_name.lower():
                return 'phi-4-reasoning'
            else:
                return 'phi-4'
        elif 'reasoning' in model_dir.lower():
            return 'phi-4-reasoning'
        else:
            return 'phi-4'

    def load_model(self):
        """
        Load the Phi-4 model and tokenizer. This can be called if you want to reload the model.
        """
        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import LLM as vLLM
            self.model = vLLM(
                model=self.model_config.model_dir,
                tensor_parallel_size=self.model_config.tensor_parallel_size,
                dtype=self.model_config.torch_dtype,
                trust_remote_code=True
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_dir,
                use_fast=self.model_config.use_fast_tokenizer,
                split_special_tokens=self.model_config.split_special_tokens,
                torch_dtype=self.model_config.infer_dtype,
                device_map=self.model_config.device_map,
                trust_remote_code=True
            )

            if self.model_config.new_special_tokens is not None:
                num_added_tokens = self.tokenizer.add_special_tokens(
                    dict(additional_special_tokens=self.model_config.new_special_tokens),
                    replace_additional_special_tokens=False,
                )

                if num_added_tokens > 0 and not self.model_config.resize_vocab:
                    self.model_config.resize_vocab = True
                    logger.warning(
                        'New tokens have been added, changed `resize_vocab` to True.')

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_dir,
                torch_dtype=self.model_config.infer_dtype,
                device_map=self.model_config.device_map,
                trust_remote_code=True
            )
            
            # Only move to device if device_map is not 'auto'
            if self.model_config.device_map != 'auto':
                self.model = self.model.to(self.model_config.device)

            param_stats = print_trainable_parameters(self.model)
            logger.info(param_stats)

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> str:
        """
        Generate text based on the input prompt using Phi-4 model.

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

        # Format messages based on model type
        messages = self._format_messages_for_model(prompts)

        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import SamplingParams

            # Adjust sampling parameters based on model type
            sampling_params = self._get_sampling_params()

            # Apply chat template for Phi-4
            conversations = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            generated_text = self.model.generate([conversations], sampling_params)
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

            # Move inputs to the specified device
            inputs = {key: value.to(self.model_config.device) for key, value in inputs.items()}

            # Adjust generation parameters based on model type
            generation_kwargs = self._get_generation_kwargs()
            
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        generated_text = self.parse_outputs(generated_text, self.model_config.use_vllm)
        return generated_text if len(generated_text) > 1 else generated_text[0]
    
    def _format_messages_for_model(self, prompts: List[str]) -> List[dict]:
        """Format messages based on the specific Phi-4 model type"""
        messages = []
        
        for prompt in prompts:
            if self.model_type == 'phi-4-reasoning':
                # For reasoning model, we might want to add reasoning-specific system prompts
                messages.append({'role': 'user', 'content': prompt})
            else:
                # For base phi-4 model, use standard format
                messages.append({'role': 'user', 'content': prompt})
        
        return messages
    
    def _get_sampling_params(self):
        """Get sampling parameters optimized for the specific model type"""
        from vllm import SamplingParams
        
        base_params = {
            'temperature': self.generation_config.temperature,
            'top_p': self.generation_config.top_p,
            'max_tokens': self.generation_config.max_length
        }
        
        if self.model_type == 'phi-4-reasoning':
            # Reasoning model might benefit from slightly different parameters
            base_params.update({
                'temperature': min(self.generation_config.temperature, 0.3),  # Lower temp for reasoning
                'top_p': max(self.generation_config.top_p, 0.8),  # Higher top_p for reasoning
            })
        
        return SamplingParams(**base_params)
    
    def _get_generation_kwargs(self):
        """Get generation parameters optimized for the specific model type"""
        base_kwargs = {
            'do_sample': self.generation_config.do_sample,
            'max_new_tokens': self.generation_config.max_new_tokens,
            'temperature': self.generation_config.temperature,
            'top_p': self.generation_config.top_p,
            'top_k': self.generation_config.top_k,
            'num_beams': self.generation_config.num_beams,
            'repetition_penalty': self.generation_config.repetition_penalty,
            'length_penalty': self.generation_config.length_penalty
        }
        
        if self.model_type == 'phi-4-reasoning':
            # Reasoning model optimizations
            base_kwargs.update({
                'temperature': min(self.generation_config.temperature, 0.3),
                'top_p': max(self.generation_config.top_p, 0.8),
                'repetition_penalty': max(self.generation_config.repetition_penalty, 1.05),
            })
        
        return base_kwargs

    @staticmethod
    def parse_outputs(outputs, use_vllm):
        """Parse the generated outputs into a structured format."""
        
        parsed_outputs = []
        if use_vllm:
            for request_output in outputs:
                for completion in request_output.outputs:
                    # Clean up the output for Phi-4
                    cleaned_output = completion.text.strip()
                    # Remove any remaining template artifacts
                    if '<|assistant|>' in cleaned_output:
                        cleaned_output = cleaned_output.split('<|assistant|>')[-1]
                    if '<|end|>' in cleaned_output:
                        cleaned_output = cleaned_output.split('<|end|>')[0]
                    parsed_outputs.append(cleaned_output)
        else:
            for output in outputs:
                parsed_output = output.strip()
                # Handle different output formats
                if isinstance(parsed_output, str):
                    # Remove any template artifacts
                    if '<|assistant|>' in parsed_output:
                        parsed_output = parsed_output.split('<|assistant|>')[-1]
                    if '<|end|>' in parsed_output:
                        parsed_output = parsed_output.split('<|end|>')[0]
                    parsed_outputs.append(parsed_output)
                else:
                    parsed_outputs.append(str(parsed_output))
        return parsed_outputs
