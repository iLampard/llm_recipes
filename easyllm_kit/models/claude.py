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
        else:
            # Initialize Anthropic client
            import anthropic
            self.client = anthropic.Client(api_key=self.model_config['api_key'])

    def generate(self, prompt: str, **kwargs):
        if self.model_config.use_litellm_api:
            completion = self.client.chat.completions.create(
                model=self.model_config.model_name if kwargs.get('model_name') is None else kwargs.get('model_name'),
                max_tokens=self.generation_config.max_length,
                temperature=self.generation_config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            # Use Anthropic API format
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.generation_config['max_length'],
                temperature=self.generation_config['temperature']
            )

        return completion.choices[0].message.content
