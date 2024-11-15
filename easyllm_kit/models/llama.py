from typing import Union, List
from easyllm_kit.models.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyllm_kit.utils import get_logger
from easyllm_kit.utils import print_trainable_parameters

logger = get_logger('easyllm_kit')


# ref: https://github.com/meta-llama/llama3/blob/main/llama/generation.py

@LLM.register('llama3')
class Llama3(LLM):
    model_name = 'llama'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        """
        Load the model and tokenizer. This can be called if you want to reload the model.
        """
        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import LLM as vLLM
            self.model = vLLM(model=self.model_config.model_dir,
                              tensor_parallel_size=self.model_config.tensor_parallel_size)
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_dir,
                use_fast=self.model_config.use_fast_tokenizer,
                split_special_tokens=self.model_config.split_special_tokens,
                torch_dtype=self.model_config.infer_dtype,
                device_map=self.model_config.device_map
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
                trust_remote_code=self.model_config.trust_remote_code
            ).to(self.model_config.device)

            param_stats = print_trainable_parameters(self.model)
            logger.info(param_stats)

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

        messages = [{'role': 'user', 'content': prompt} for prompt in prompts]

        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p
            )

            # Perform inference
            conversations = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
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

        generated_text = self.parse_outputs(generated_text, self.model_config.use_vllm)
        return generated_text if len(generated_text) > 1 else generated_text[0]

    @staticmethod
    def parse_outputs(outputs, use_vllm):
        """Parse the generated outputs into a structured format."""
        
        parsed_outputs = []
        if use_vllm:
            for request_output in outputs:
                for completion in request_output.outputs:
                    cleaned_output = completion.text.split('<|end_header_id|>')[-1]
                    cleaned_output = cleaned_output.strip()
                    if cleaned_output.startswith('"'):
                        cleaned_output = cleaned_output[1:]
                    if cleaned_output.endswith("'"):
                        cleaned_output = cleaned_output[:-1]
                    cleaned_output = f'"{cleaned_output}"'
                    parsed_outputs.append(cleaned_output)
        else:
            for output in outputs:
                parsed_output = output.strip("[]'")
                parsed_output = parsed_output.replace("\\n", "\n")
                lines = parsed_output.split("\n")
                assistant_response = lines[-1].strip()
                parsed_outputs.append({'assistant': assistant_response})
        return parsed_outputs
