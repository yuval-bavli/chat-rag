import os

def get_model_dir(model_name: str) -> str:
    split_name = model_name.split("/")
    models_base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    cur_model_dir = models_base_dir
    for part in split_name:
        cur_model_dir = os.path.join(cur_model_dir, part)

    return cur_model_dir
