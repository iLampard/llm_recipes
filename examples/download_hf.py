import argparse
from easyllm_kit.utils import HFHelper, download_data_from_hf

if __name__ == "__main__":
    """
    Download the dataset from huggingface
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--need_login", type=str, default=False,
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--hf_config_dir", type=str, default="hf_config.yaml",
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--hf_dataset_dir", type=str, default="weaverbirdllm/famma",
                        help="The dir of dataset on huggingface.")

    parser.add_argument("--subset_name", type=str, default='',
                        help="If None, download all the subset.")

    parser.add_argument("--split", type=str, default='validation',
                        help="The split to download: 'train' and 'test' available.")

    parser.add_argument("--save_dir", type=str, default="./data",
                        help="The local dir to save the dataset.", )

    args = parser.parse_args()

    if args.need_login:
        # have to login
        HFHelper.login_from_config(args.hf_config_dir)

    download_data_from_hf(args.hf_dataset_dir, args.subset_name, args.split, args.save_dir)
