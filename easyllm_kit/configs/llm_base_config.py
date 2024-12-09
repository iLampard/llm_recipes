from dataclasses import asdict
from easyllm_kit.configs.base import Config
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union

import torch
from typing_extensions import Self


# ref: https://github.com/hiyouga/LLaMA-Factory/blob/451d271718a8026056d0f7d7b8ab333391d24ad4/src/llamafactory/hparams/model_args.py

@dataclass
class QuantizationArguments:
    r"""
    Arguments pertaining to the quantization method.
    """

    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = field(
        default="bitsandbytes",
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )


@dataclass
class ProcessorArguments:
    r"""
    Arguments pertaining to the image processor.
    """

    image_resolution: int = field(
        default=512,
        metadata={"help": "Keeps the height or width of image below this resolution."},
    )
    video_resolution: int = field(
        default=128,
        metadata={"help": "Keeps the height or width of video below this resolution."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=64,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )


@dataclass
class ExportArguments:
    r"""
    Arguments pertaining to the model export.
    """

    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )


@dataclass
class VllmArguments:
    r"""
    Arguments pertaining to the vLLM worker.
    """

    vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )


@dataclass
class ModelArguments(QuantizationArguments, ProcessorArguments, ExportArguments, VllmArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name: str = field(default='llama3')

    # Optional fields (with default values)
    use_api: bool = field(default=False,
                          metadata={'help': 'Flag to indicate if the model should be accessed via an API.'})
    api_key: Optional[str] = field(default=None, metadata={'help': 'API key for accessing the model if using an API.'})
    api_url: Optional[str] = field(default=None, metadata={'help': 'API URL for accessing the model if using an API.'})

    use_litellm_api: bool = field(default=False, metadata={
        'help': 'Flag to indicate if the model should be accessed via litellm API.'})

    model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )

    device: str = field(default='cuda', metadata={'help': 'Device to run the model on (e.g., "cuda" or "cpu").'})
    device_map: Optional[Dict[str, Any]] = field(default='auto',
                                                 metadata={'help': 'Mapping of device placement for model layers.'})

    max_memory: Optional[Dict[str, int]] = field(
        default=None,
        metadata={"help": "Dictionary specifying maximum memory (in GB) for each device."},
    )

    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether or not to use vLLM for inference."},
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of devices to use for tensor parallelism. Default is 1 (no parallelism). Used for inference initialization on vLLM."},
    )

    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )

    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing."},
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading models from Hugging Face."},
    )

    def __post_init__(self):
        if self.model_dir is None and not self.use_api:
            raise ValueError("Please provide `model_dir`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result


@dataclass
class GenerationArguments:
    """Arguments pertaining to specify the model generation parameters."""

    # Generation strategy
    # 是否采样
    do_sample: Optional[bool] = field(
        default=True,
        metadata={
            'help':
                'Whether or not to use sampling, use greedy decoding otherwise.'
        },
    )
    # Hyperparameters for logit manipulation
    # softmax 函数的温度因子，来调节输出token的分布
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            'help': 'The value used to modulate the next token probabilities.'
        },
    )
    # 核采样参数，top_p最高的前n个（n是变化）概率和为p，从这些n个候选token中随机采样
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
                'The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.'
        },
    )
    # top_k随机搜索中的k个最高概率选择
    top_k: Optional[int] = field(
        default=50,
        metadata={
            'help':
                'The number of highest probability vocabulary tokens to keep for top-k filtering.'
        },
    )
    # 集束搜索的数量
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            'help': 'Number of beams for beam search. 1 means no beam search.'
        },
    )
    # 最大的token数量，会被 max_new_tokens 覆盖
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            'help':
                'The maximum length the generated tokens can have. It can be overridden by max_new_tokens.'
        },
    )
    # 最大的新生成的token数量
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={
            'help':
                'Maximum number of new tokens to be generated in evaluation or prediction loops'
                'if predict_with_generate is set.'
        },
    )
    # 重复性惩罚因子
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
                'The parameter for repetition penalty. 1.0 means no penalty.'
        })
    # 长度惩罚因子
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
                'Exponential penalty to the length that is used with beam-based generation.'
        })
    default_system: Optional[str] = field(
        default=None,
        metadata={'help': 'Default system message to use in chat completion.'},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get('max_new_tokens', None):
            args.pop('max_length', None)
        else:
            args.pop('max_new_tokens', None)
        return args


@Config.register('llm_config')
class LLMConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config: dict, **kwargs):
        model_config = ModelArguments(**config.get('model', {}))
        generation_config = GenerationArguments(**config.get('generation', {}))
        return {'model_config': model_config, 'generation_config': generation_config}
