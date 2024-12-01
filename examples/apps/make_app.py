from easyllm_kit.utils.app import run_app, create_app

if __name__ == "__main__":
    app = create_app('cpm_gen.yaml')
    run_app(app, port=12000)
    