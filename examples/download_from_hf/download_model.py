import argparse
from easyllm_kit.utils import HFHelper

if __name__ == "__main__":
    """
    Download the dataset from huggingface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--need_login", type=str, default=True,
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--hf_config_dir", type=str, default="hf_config.yaml",
                        help="The dir of config file on huggingface.")

    parser.add_argument("--model_repo", type=str, default="google/flan-t5-large",
                        help="The dir of model repo on huggingface.")

    parser.add_argument("--save_dir", type=str, default="/workspaces/data/cache/huggingface/hub/models--google--flan-t5-large/",
                        help="The dir to save.")

    args = parser.parse_args()

    if args.need_login:
        # have to login
        HFHelper.login_from_config(args.hf_config_dir)

    HFHelper.download_model_from_hf(args.model_repo, 
                                    args.save_dir)
