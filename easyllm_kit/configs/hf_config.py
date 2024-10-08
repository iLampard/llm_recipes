from dataclasses import asdict, dataclass, field
from typing import Dict, Any

from easyllm_kit.configs.base import Config


@dataclass
class HFBaseArgs:
    hf_token: str = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@Config.register('hf_config')
class HFConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config: dict, **kwargs):
        hf_config = config['base']
        # Create and return the LLMBaseArgs object
        return HFBaseArgs(**hf_config)
