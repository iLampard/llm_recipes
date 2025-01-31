from easyllm_kit.models.base import LLM


@LLM.register("gpt4o")
class GPT4o(LLM):
    model_name = "gpt4o"

    def __init__(self, config):
        import openai

        self.model_config = config["model_config"]
        self.generation_config = config["generation_config"]
        if self.model_config.use_litellm_api:
            self.client = openai.OpenAI(
                api_key=self.model_config.api_key, base_url=self.model_config.api_url
            )
        else:
            self.client = openai.OpenAI(api_key=self.model_config.api_key)

    def generate(self, prompt: str, **kwargs):
        use_default_image_template = kwargs.get('use_default_image_template', False)
        if use_default_image_template:
            prompt_ = self.format_prompt_with_image(prompt, kwargs.get("image"))
        else:
            prompt_ = prompt
        completion = self.client.chat.completions.create(
            model=kwargs.get("model_name", "gpt-4o"),
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            messages=[{"role": "user", "content": prompt_}],
        )
        return completion.choices[0].message.content
