from abc import abstractmethod
from registrable import Registrable


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
