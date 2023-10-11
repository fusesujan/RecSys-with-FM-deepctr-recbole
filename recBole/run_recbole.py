import recbole
from recbole.quick_start import run_recbole

if __name__ == "__main__":
    config_dict = {
        "model": "GRU4Rec",
        "dataset": "ml-100k",
        "config_files": ["test.yaml"],
    }

    run_recbole(config_dict)

