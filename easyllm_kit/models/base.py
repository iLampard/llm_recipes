from abc import abstractmethod
from registrable import Registrable
import base64
from easyllm_kit.utils import read_image_as_bytes


class LLM(Registrable):
    @staticmethod
    def build_from_config(config, **kwargs):
        LLM_cls = LLM.by_name(config["model_config"].model_name.lower())
        return LLM_cls(config)

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    @staticmethod
    def format_prompt_with_image(prompt: str, image=None):
        """Format prompt with optional image for LiteLLM compatible APIs."""

        prompt_ = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        if image:
            image_base64 = base64.b64encode(read_image_as_bytes(image)).decode("utf-8")
            prompt_.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        return prompt_
