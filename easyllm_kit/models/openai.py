from easyllm_kit.models.base import LLM

@LLM.register('gpt4o')
class GPT4o(LLM):
    model_name = 'gpt4o'
    def __init__(self, config):
        import openai
        self.client = openai.OpenAI(api_key=config.api_key)
        self.config = config

    def generate(self, prompt: str, **kwargs):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
