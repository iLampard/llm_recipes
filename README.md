

# EasyLLM_Kit

[![PyPI Version](https://img.shields.io/pypi/v/easyllm_kit.svg)](https://pypi.org/project/easyllm_kit/)
[![Python Version](https://img.shields.io/pypi/pyversions/easyllm_kit.svg)](https://pypi.org/project/easyllm_kit/)
[![Downloads](https://pepy.tech/badge/easyllm_kit)](https://pepy.tech/project/easyllm_kit)



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

#### Load gpt4o from OpenAI API

We provide a yaml config file to define the model and its parameters.
```yaml
config_cls_name: llm_config

model:
  model_name: gpt4o
  use_api: true
  api_key: xx
  api_url: https://api.openai.com/v1/chat/completions

generation:
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.1
```

Then we can load the model and generate text with an optional image (cloth.png).
```python
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config
# Load configuration from YAML file
model_config = Config.build_from_yaml_file('config.yaml')

# Build the LLM model
model = LLM.build_from_config(model_config)

# Generate text
response = model.generate('whats the content of the image?', image='cloth.png')
print(response)
# The image shows a person wearing a black skirt. Below the skirt, there are color options displayed, including black, white, light blue, and green.
```

#### Load llama3.1 from local directory

We load Llama-3.1-70B-Instruct from local directory with vLLM for multi-GPU inference.

```yaml
config_cls_name: llm_config
task: llm_gen # currently not used

model:
  model_name: llama3
  use_api: false
  model_dir: /workspaces/data0/models/huggingface/meta-llama/Meta-Llama-3.1-70B-Instruct
  use_vllm: true
  tensor_parallel_size: 4

generation:
  temperature: 0.3
  top_p: 0.9
  repetition_penalty: 1.1
```

Then we can, using the same interface, to load the model and generate text with it.
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

### Download dataset from HuggingFace

We demonstrate how to download datasets from Hugging Face using the `HFHelper` class. The `HFHelper` class provides a convenient method to log in to Hugging Face and download datasets directly to your local machine. 

To use this functionality, you need to have your Hugging Face token stored in a YAML configuration file. The token allows you to access private datasets and repositories.

```yaml
config_cls_name: hf_config

base:
  hf_token: xxx
```

Then we can download the dataset with the following code.

```python
from easyllm_kit.utils import HFHelper, download_data_from_hf

need_login = True
    
if need_login:
    # have to login
    HFHelper.login_from_config('hf_config.yaml')

hf_dataset_dir = 'weaverbirdllm/famma'
subset_name = ''
split = 'validation'
save_dir = './data'
download_data_from_hf(hf_dataset_dir, subset_name, split, save_dir)
```

### Calling metrics 

We provide a set of metrics to evaluate the performance of the LLM. The `Metrics` class provides a unified interface for accessing these evaluation methods.

Here’s an example of how to use the `Metrics` class to get evaluation methods:

```python
from easyllm_kit.metrics import Metrics

def get_evaluation_methods():
    """
    Get evaluation methods including accuracy, sentence transformers, and other metrics.

    Returns:
    - A dictionary mapping metric names to their respective evaluation functions.
    """
    return {
        "accuracy": Metrics.by_name("accuracy").calculate,
        "bool": Metrics.by_name("accuracy").calculate,
        "hit rate@3": Metrics.by_name("hit_ratio").calculate,
        "rougel": Metrics.by_name("rouge_l").calculate,
        "sent-transformer": lambda generated_text, reference_texts: Metrics.by_name("cosine_similarity").calculate(
            generated_text=generated_text,
            reference_texts=reference_texts,
            model_name="all-MiniLM-L6-v2",
        ),
        "multilingual-sent-transformer": lambda generated_text, reference_texts: Metrics.by_name("cosine_similarity").calculate(
            generated_text=generated_text,
            reference_texts=reference_texts,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        ),
        "micro f1": Metrics.by_name("micro_f1").calculate,
        "ndcg": Metrics.by_name("ndcg").calculate,
        "bleu": Metrics.by_name("bleu").calculate,
        "jp-bleu": lambda generated_text, reference_text: Metrics.by_name("bleu").calculate(
            generated_text=generated_text,
            reference_text=reference_text,
            is_japanese=True,
        ),
    }
```


## Reference

The following repositories are used in `easyllm_kit`, either in close to original form or as an inspiration:

- [Amazon KDD Cup 2024 Starter Kit](https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit)
- [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess)
- [LLamaTuner](https://github.com/jianzhnie/LLamaTuner/)

