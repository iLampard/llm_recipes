from easyllm_kit.models.base import LLM


@LLM.register('gemini')
class Gemini(LLM):
    model_name = 'gemini'

    def __init__(self, config):
        import openai
        self.model_config = config['model_config']
        self.generation_config = config['generation_config']
        if self.model_config.use_litellm_api:
            self.client = openai.OpenAI(api_key=self.model_config.api_key,
                                        base_url=self.model_config.api_url)
            if self.model_config.model_full_name is None:
                self.model_config.model_full_name = 'gemini-1.5-pro-002'
        else:
            from google import genai
            self.client = genai.Client(api_key="YOUR_API_KEY")

    def generate(self, prompt: str, **kwargs):
        if self.model_config.use_litellm_api:
            completion = self.client.chat.completions.create(
                model=self.model_config.model_full_name,
                max_tokens=self.generation_config.max_length,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                timeout=self.generation_config.timeout,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        else:
            response = self.client.models.generate_content(
                model=self.model_config.model_full_name,
                contents=prompt,
            )
            return response
