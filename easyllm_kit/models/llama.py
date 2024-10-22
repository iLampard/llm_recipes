from typing import Union, List

from sympy.physics.units import temperature
from torch.cuda import max_memory_allocated

from easyllm_kit.models.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyllm_kit.utils import get_logger
from easyllm_kit.utils import print_trainable_parameters

logger = get_logger('easyllm')


# ref: https://github.com/meta-llama/llama3/blob/main/llama/generation.py

@LLM.register('llama3')
class Llama3(LLM):
    model_name = 'llama'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.load_model()

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

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert single string to list

        messages = [{'role': 'user', 'content': prompt} for prompt in prompts]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model_config.device)

        outputs = self.model.generate(**inputs,
                                      do_sample=self.generation_config.do_sample,
                                      max_new_tokens=self.generation_config.max_new_tokens,
                                      temperature=self.generation_config.temperature)

        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_text = self.parse_outputs(generated_text)
        return generated_text if len(generated_text) > 1 else generated_text[0]

    def load_model(self):
        """
        Load the model and tokenizer. This can be called if you want to reload the model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_dir,
            use_fast=self.model_config.use_fast_tokenizer,
            split_special_tokens=self.model_config.split_special_tokens,
            torch_dtype=self.model_config.infer_dtype,
            device_map=self.model_config.device_map,
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

        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_dir,
                                                          torch_dtype=self.model_config.infer_dtype,
                                                          device_map=self.model_config.device_map,
                                                          max_memory=self.model_config.max_memory)
        # Initialize DeepSpeed inference
        if self.model_config.use_deepspeed:
            import deepspeed
            self.model = deepspeed.init_inference(self.model,
                                                mp_size=self.model_config.mp_size,
                                                dtype=self.model_config.infer_dtype,
                                                replace_method='auto',
                                                replace_with_kernel_inject=True)

        param_stats = print_trainable_parameters(self.model)

        logger.info(param_stats)

    @staticmethod
    def parse_outputs(outputs):
        # Parse the output
        parsed_outputs = []
        for output in outputs:
            parsed_output = output.strip("[]'")  # Remove the list brackets and quotes
            parsed_output = parsed_output.replace("\\n", "\n")  # Replace escaped newlines with actual newlines
            # Split the output into lines
            lines = parsed_output.split("\n")
            # Extract the relevant part of the output
            # Assuming the last line is the assistant's response
            assistant_response = lines[-1].strip()

            parsed_outputs.append({'assistant': assistant_response})
        return parsed_outputs
