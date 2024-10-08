from dataclasses import asdict, dataclass, field
from typing import Optional, List, Dict, Any
from easyllm_kit.configs.base import Config


@dataclass
class LLMBaseArgs:
    model_name: str
    use_api: bool = False
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    device: str = "cuda"
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@Config.register('llm_config')
class LLMConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config: dict, **kwargs):
        model_config = config['model']

        # Extract known fields from the config
        known_fields = [
            'model_name', 'use_api', 'api_key', 'api_url', 'device',
            'max_tokens', 'temperature', 'top_p', 'repetition_penalty',
            'stop_sequences'
        ]
        base_args = {k: model_config.get(k) for k in known_fields if k in model_config}

        # Any remaining fields go into extra_params
        extra_params = {k: v for k, v in model_config.items() if k not in known_fields}
        if extra_params:
            base_args['extra_params'] = extra_params

        # Create and return the LLMBaseArgs object
        return LLMBaseArgs(**base_args)
