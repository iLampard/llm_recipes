from easyllm_kit.models.base import LLM
import openai


@LLM.register('gpt4o')
class GPT4o(LLM):
    model_name = 'gpt4o'

    def generate(self, prompt: str, **kwargs):
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            **kwargs
        )
        return response.choices[0].text.strip()
