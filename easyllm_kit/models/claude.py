import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from easyllm_kit.models.base import LLM


@LLM.register('claude_3_opus')
class Claude3Opus(LLM):
    client = anthropic.Anthropic(api_key=self.api_key)
    claude_versions = {
        "claude-2": self._generate_claude_2,
        "claude-3-sonnet-20240229": self._generate_claude_3,
        "claude-3-opus-20240229": self._generate_claude_3
    }

    def generate(self, prompt: str, **kwargs):
        generate_func = self.claude_versions.get(self.model_name, self._generate_claude_2)
        return generate_func(prompt, **kwargs)

    def _generate_claude_2(self, prompt: str, **kwargs) -> str:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            max_tokens_to_sample=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            stop_sequences=[HUMAN_PROMPT]
        )
        return response.completion.strip()

    def _generate_claude_3(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
