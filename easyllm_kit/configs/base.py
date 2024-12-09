from omegaconf import OmegaConf
from registrable import Registrable

class Config(Registrable):
    @staticmethod
    def parse_from_yaml_config(yaml_config, **kwargs):
        raise NotImplementedError

    @staticmethod
    def build_from_yaml_file(yaml_dir, **kwargs):
        """Load yaml config file from disk.

        Args:
            yaml_dir (str): Path of the yaml config file.

        Returns:
            Config: Config object corresponding to cls.
        """
        config = OmegaConf.load(yaml_dir)
        assert config.get('config_cls_name', None) is not None, "config_cls_name is not set"
        config_cls_name = config.get('config_cls_name')
        config_cls = Config.by_name(config_cls_name.lower())
        return config_cls.parse_from_yaml_config(config, **kwargs)



    
