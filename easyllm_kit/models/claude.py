import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from easyllm_kit.models.base import LLM


@LLM.register('claude_3_sonnet')
class Claude3Opus(LLM):
    model_name = 'claude_3_sonnet'
    def __init__(self, config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=self.config.api_key)

    def generate(self, prompt: str, **kwargs):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
