from typing import Union, List

from transformers import T5ForConditionalGeneration, AutoTokenizer

from easyllm_kit.models.base import LLM


@LLM.register('flan-t5')
class FlanT5(LLM):
    model_name = 'flan-t5'

    def __init__(self, config):
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.model = None  # Initialize model attribute
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_dir,
            torch_dtype=self.model_config.infer_dtype,
            device_map=self.model_config.device_map
        )
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_config.model_dir).to(self.model_config.device)

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

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        # Generate text
        outputs = self.model.generate(
            **inputs,
            max_length=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            repetition_penalty=self.generation_config.repetition_penalty,
        )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Return single string if input was single string
        return generated_texts[0] if len(prompts) == 1 else generated_texts