from omegaconf import OmegaConf
from registrable import Registrable


class LLM(Registrable):
    def __init__(self, config: OmegaConf):
        self.config = config

    @staticmethod
    def build_from_yaml_file(yaml_dir, **kwargs):
        config = OmegaConf.load(yaml_dir)
        assert config.get('llm_cls_name', None) is not None, "config_cls_name is not set"
        config_cls_name = config.get('llm_cls_name')
        config_cls = LLM.by_name(config_cls_name.lower())
        return config_cls(config)


@LLM.register('gpt4o')
class GPT4o(LLM):
    import openai
    def generate(self, prompt: str, **kwargs):
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            **kwargs
        )
        return response.choices[0].text.strip()


@LLM.register('claude_3_opus')
class Claude3Opus(LLM):
    import anthropic
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

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
