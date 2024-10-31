from typing import Union, List
from easyllm_kit.models.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, AutoProcessor
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

        messages = [{'role': 'user', 'content': prompt} for prompt in prompts]

        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import SamplingParams

            sampling_params = SamplingParams(temperature=self.generation_config.temperature,
                                             top_p=self.generation_config.top_p)

            # Perform inference
            conversations = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            generated_text = self.model.generate([conversations], sampling_params)
            # generated_text = [output.outputs.CompletionOutput.text for output in outputs]
            # return outputs
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

            outputs = self.model.generate(**inputs,
                                          do_sample=self.generation_config.do_sample,
                                          max_new_tokens=self.generation_config.max_new_tokens,
                                          temperature=self.generation_config.temperature)

            generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        generated_text = self.parse_outputs(generated_text, self.model_config.use_vllm)
        return generated_text if len(generated_text) > 1 else generated_text[0]

    def load_model(self):
        """
        Load the model and tokenizer. This can be called if you want to reload the model.
        """
        # Initialize vllm inference
        if self.model_config.use_vllm:
            from vllm import LLM as vLLM
            print(self.model_config.tensor_parallel_size)
            self.model = vLLM(model=self.model_config.model_dir,
                              tensor_parallel_size=self.model_config.tensor_parallel_size)

            self.tokenizer = self.model.get_tokenizer()

        elif self.model_config.is_multimodal:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_config.model_dir,
                torch_dtype=self.model_config.infer_dtype,
                device_map=self.model_config.device_map
            )
            self.tokenizer = AutoProcessor.from_pretrained(self.model_config.model_dir)
            print('load llama-3.2')
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

            self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_dir,
                                                              torch_dtype=self.model_config.infer_dtype,
                                                              device_map=self.model_config.device_map).to(
                self.model_config.device)

            param_stats = print_trainable_parameters(self.model)

            logger.info(param_stats)

    @staticmethod
    def parse_outputs(outputs, use_vllm):
        # Parse the output
        parsed_outputs = []
        if use_vllm:
            # Iterate through each RequestOutput in the batch
            for request_output in outputs:
                # Extract the 'text' field from each CompletionOutput in 'outputs'
                print(request_output)
                for completion in request_output.outputs:
                    cleaned_output = completion.text.split('<|end_header_id|>')[-1]
                    # Strip any leading or trailing whitespace and newline characters
                    cleaned_output = cleaned_output.strip()
                    # Step 3: Remove the leading double quote, if present
                    if cleaned_output.startswith('"'):
                        cleaned_output = cleaned_output[1:]

                    # Step 4: Remove the trailing single quote, if present
                    if cleaned_output.endswith("'"):
                        cleaned_output = cleaned_output[:-1]

                    # Step 5: Wrap the cleaned output in double quotes
                    cleaned_output = f'"{cleaned_output}"'

                    parsed_outputs.append(cleaned_output)

            return parsed_outputs
        else:
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
