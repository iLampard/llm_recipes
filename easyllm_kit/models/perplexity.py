from easyllm_kit.models.base import LLM


@LLM.register('perplexity')
class Perplexity(LLM):
    model_name = 'perplexity'

    def __init__(self, config):
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.client = openai.OpenAI(api_key=self.model_config.api_key,
                                    base_url=self.model_config.api_url)

    def generate(self, prompt: str, **kwargs):
        completion = self.client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "Be precise and concise."
                },
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
