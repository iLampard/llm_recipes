from easyllm_kit.models import LLM
from easyllm_kit.configs import Config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load various LLMs and generate answers")
    parser.add_argument("--config_dir",
                        default='ds_model_gen.yaml',
                        help="Path to configuration file in yaml format")
    parser.add_argument("--output_dir", help="Path to save the output JSON file")
    args = parser.parse_args()

    config = Config.build_from_yaml_file(args.config_dir)

    model = LLM.build_from_config(config)

    print(model.generate('outfit for office under 200$'))
