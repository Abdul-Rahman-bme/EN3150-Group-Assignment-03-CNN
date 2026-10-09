"""Paths, saved settings and reproducibility helpers."""

import hashlib
import json
import os
from pathlib import Path


def project_root(value=None):
    if value is not None:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "data/splits").is_dir():
            return root
    raise ValueError("Supply --project-root for an installed copy outside this repository.")


def resolve_path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def portable_path(root, path):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("model", "class_names", "normalization_mean", "normalization_std")
    for key in required:
        if key not in config:
            raise ValueError(f"Missing {key} in {path}")
    from .model_specs import MODEL_NAMES, PRETRAINED_MODELS, PRETRAINED_SPECS, IMAGENET_MEAN, IMAGENET_STD
    if config["model"] not in MODEL_NAMES:
        raise ValueError("Unsupported saved model architecture")
    names = config["class_names"]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("Expected ten distinct saved classes")
    import math
    mean, std = config["normalization_mean"], config["normalization_std"]
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(x) for x in mean + std):
        raise ValueError("Expected finite RGB normalization values")
    if any(x <= 0 for x in std) or config.get("input_size", 64) != 64:
        raise ValueError("Expected positive standard deviations and 64 x 64 inputs")
    if config["model"] in PRETRAINED_MODELS:
        spec = PRETRAINED_SPECS[config["model"]]
        if any(config.get(key) != value for key, value in spec.items()):
            raise ValueError("Saved pretrained weight identifier or source differs from the supported candidate")
        if mean != IMAGENET_MEAN or std != IMAGENET_STD:
            raise ValueError("Pretrained candidates require the saved ImageNet normalization")
        if config.get("fine_tune_all_layers") is not True:
            raise ValueError("Pretrained candidates must fine-tune all layers")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_device(name="auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu or auto")
    return torch.device(name)


def seed_everything(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import random
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def capture_random_state(generator):
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "shuffle": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state, generator):
    import random
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["shuffle"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def new_output_directory(path):
    """Reserve a fresh directory so report commands cannot overwrite results."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
