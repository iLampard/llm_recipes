from easyllm_kit.models.base import LLM


@LLM.register('claude_35_sonnet')
class Claude35Sonnet(LLM):
    model_name = 'claude_35_sonnet'

    def __init__(self, config):
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        if self.model_config.use_litellm_api:
            self.client = openai.OpenAI(api_key=self.model_config.api_key, base_url=self.model_config.api_url)
            if self.model_config.model_full_name is None:
                self.model_config.model_full_name = 'vertex-sonnet-3.5'
        else:
            # Initialize Anthropic client
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.model_config['api_key'])
            if self.model_config.model_full_name is None:
                self.model_config.model_full_name = 'claude-3-5-sonnet-20240620'

    def generate(self, prompt: str, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model_config.model_full_name,
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            timeout=self.generation_config.timeout,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content
