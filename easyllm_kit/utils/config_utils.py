import ast
import os
from datetime import datetime
from typing import Any


def make_json_compatible_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def convert_str_2_list_or_float(value: str) -> Any:
    """
    Parse a string value into its appropriate Python type.
    
    Args:
        value (str): The string value to parse.
    
    Returns:
        The parsed value in its appropriate type.
    """
    # Try to evaluate as a literal first
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    # If it's not a literal, try other conversions
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            # If all else fails, return the original string
            return value


def generate_output_dir(base_dir, exp_name, **kwargs):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_dir = base_dir
    temp_str = exp_name
    for k, v in kwargs.items():
        temp_str += f'{k}-{v}'
    temp_str += timestamp
    return os.path.join(output_dir, temp_str)

# @Config.register('train_experiment_config')
# class TrainConfig(Config):
#     @staticmethod
#     def parse_from_yaml_config(config, **kwargs):
#         experiment_name = kwargs.get('experiment_name')
#         experiment_config = config.get(experiment_name, None)

#         # setup the base config
#         merged_experiment_config = config['defaults'].copy()

#         # overwrite some config values
#         for key, value in experiment_config.items():
#             merged_experiment_config[key] = value

#         output_dir = merged_experiment_config.get('output_dir', '')
#         merged_experiment_config['output_dir'] = generate_output_dir(output_dir,
#                                                                      experiment_name, **experiment_config)

#         return merged_experiment_config

#     @staticmethod
#     def make_args_to_str(merged_experiment_config):
#         params = []
#         for key, value in merged_experiment_config.items():
#             formatted_value = format_value(value)
#             params.append(f"--{key} {formatted_value}")

#         param_str = ' '.join(params)

#         return param_str


# @Config.register('hf_config')
# class HFConfig(Config):
#     @staticmethod
#     def parse_from_yaml_config(config: dict, **kwargs):
#         assert config.get('hf_token', None) is not None, "HF_TOKEN is not set"
#         return config
