from easyllm_kit.models.base import LLM


@LLM.register('gpt4o')
class GPT4o(LLM):
    model_name = 'gpt4o'

    def __init__(self, config):
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        self.client = openai.OpenAI(api_key=self.model_config.api_key)
        
    def generate(self, prompt: str, **kwargs):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=self.generation_config.max_length,
            temperature=self.generation_config.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
