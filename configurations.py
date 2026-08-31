from pathlib import Path

def project_settings():
    return {
        "batch_size": 8,
        "epochs": 1,
        "learning_rate": 1e-4,

        "sequence_length": 128,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "feed_forward_dim": 2048,
        "dropout": 0.1,

        "dataset_name": "Helsinki-NLP/opus_books",
        "source_language": "en",
        "target_language": "es",

        "checkpoint_dir": "weights",
        "checkpoint_prefix": "transformer_",
        "resume_from": "latest",

        "tokenizer_pattern": "tokenizer_{lang}.json",
        "experiment_dir": "runs/english_spanish"
    }

def checkpoint_path(settings, epoch):
    directory = Path(f"{settings['dataset_name']}_{settings['checkpoint_dir']}")

    filename = (f"{settings['checkpoint_prefix']}{epoch}.pt")

    return directory / filename

def find_latest_checkpoint(settings):
    directory = Path(f"{settings['dataset_name']}_{settings['checkpoint_dir']}")

    pattern = (f"{settings['checkpoint_prefix']}*.pt")

    checkpoints = sorted(directory.glob(pattern))

    if not checkpoints:
        return None

    return checkpoints[-1]

def tokenizer_path(settings, language):
    filename = settings["tokenizer_pattern"].format(lang=language)

    return Path(filename)