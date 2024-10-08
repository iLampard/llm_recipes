# EasyLLM_Kit

`easyllm_kit` is a utility library designed to simplify interactions with various large language models (LLMs), providing easy-to-use functions for model deployment, configuration, and inference. 

## Features

- Unified interface for multiple LLM providers (OpenAI, Anthropic, Qwen, Mixtral)
- Support for both API-based and locally-loaded models
- Easy configuration management using YAML files
- Flexible and extensible architecture for adding new models
- Utility functions for common LLM tasks

## Installation

Install the package using pip:

```bash
pip install easyllm_kit
```

## Quick Start

### Loading an LLM with a YAML config file

We provide a yaml config file to define the model and its parameters.
```yaml
config_cls_name: llm_config

model:
  model_name: gpt4o
  use_api: true
  api_key: xx
  api_url: https://api.openai.com/v1/chat/completions
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.1
```

Then we can load the model and generate text with it.
```python
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config
# Load configuration from YAML file
model_config = Config.build_from_yaml_file('config.yaml')

# Build the LLM model
model = LLM.build_from_config(model_config)

# Generate text
response = model.generate('hello')
print(response)
```


## Reference

The following repositories are used in `easyllm_kit`, either in close to original form or as an inspiration:

- [Amazon KDD Cup 2024 Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit)
- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)

