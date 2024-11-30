import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory (where 'utils' is located)
parent_dir = os.path.abspath(os.path.join(script_dir, '../..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from easyllm_kit.utils.app import run_app, create_app

if __name__ == "__main__":
    app = create_app('cpm_gen.yaml')
    run_app(app, port=12000)
    