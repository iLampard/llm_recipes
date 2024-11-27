from easyllm_kit.models.base import LLM


@LLM.register('gemini')
class Gemini(LLM):
    model_name = 'gemini-1.5'

    def __init__(self, config):
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        if self.model_config.use_litellm_api:
            self.client = openai.OpenAI(api_key=self.model_config.api_key,
                                        base_url=self.model_config.api_url)
        else:
            raise NotImplementedError

    def generate(self, prompt: str, **kwargs):
        completion = self.client.chat.completions.create(
            model=kwargs.get('model_name', 'gemini-1.5-pro-002'),
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
