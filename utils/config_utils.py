import os
from datetime import datetime

from omegaconf import OmegaConf
from registrable import Registrable

from .log_utils import default_logger as logger


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    else:
        return str(value)


def generate_output_dir(base_dir, exp_name, **kwargs):
    timestamp = datetime.now().strftime("%m%d-%H%M")
    output_dir = base_dir
    temp_str = exp_name
    for k, v in kwargs.items():
        temp_str += f'{k}-{v}'
    temp_str += timestamp
    return os.path.join(output_dir, temp_str)


class Config(Registrable):
    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        raise NotImplementedError

    @staticmethod
    def build_from_yaml_file(yaml_dir, **kwargs):
        """Load yaml config file from disk.

        Args:
            yaml_fn (str): Path of the yaml config file.

        Returns:
            EasyTPP.Config: Config object corresponding to cls.
        """
        config = OmegaConf.load(yaml_dir)
        config_name = config.get('config_name')
        config_cls = Config.by_name(config_name.lower())
        logger.critical(f'Load config class {config_cls.__name__}')
        return config_cls.parse_from_yaml_config(config, **kwargs)


@Config.register('train_experiment_config')
class TrainConfig(Config):
    @staticmethod
    def parse_from_yaml_config(config, **kwargs):
        experiment_name = kwargs.get('experiment_name')
        experiment_config = config.get(experiment_name, None)

        if experiment_config is None:
            logger.warning(f'Fail to find out experiment {experiment_name} in the config.')

        # setup the base config
        merged_experiment_config = config['defaults'].copy()

        # overwrite some config values
        for key, value in experiment_config.items():
            merged_experiment_config[key] = value

        output_dir = merged_experiment_config.get('output_dir', '')
        merged_experiment_config['output_dir'] = generate_output_dir(output_dir,
                                                                     experiment_name,
                                                                     **experiment_config)

        return merged_experiment_config

    @staticmethod
    def make_args_to_str(merged_experiment_config):
        params = []
        for key, value in merged_experiment_config.items():
            formatted_value = format_value(value)
            params.append(f"--{key} {formatted_value}")

        param_str = ' '.join(params)

        return param_str
