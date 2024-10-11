from easyllm_kit.models.base import LLM


@LLM.register('claude_3_sonnet')
class Claude3Opus(LLM):
    model_name = 'claude_3_sonnet'

    def __init__(self, config):
        import anthropic
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.client = anthropic.Anthropic(api_key=self.model_config.api_key)

    def generate(self, prompt: str, **kwargs):
        response = self.client.messages.create(
            model=self.model_config.model_name,
            max_tokens=self.generation_config.max_tokens,
            temperature=self.generation_config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
