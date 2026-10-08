"""FP32 experiments with sample-weighted metrics and epoch-boundary resume."""

import copy
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from .data import check_splits, load_split, make_loader, split_fingerprints
from .models import build_model
from .transforms import make_transform
from .utils import (atomic_torch_save, capture_random_state, load_config, portable_path,
                    resolve_path, restore_random_state, seed_everything, select_device, write_json)


def run_epoch(model, loader, criterion, device, optimizer=None, *, progress=True, predictions=False):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_images = 0
    predicted, labels, confidences = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    batches = tqdm(loader, desc="Training" if training else "Validation", leave=False, disable=not progress)
    with torch.set_grad_enabled(training):
        for images, targets in batches:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss detected")
            if training:
                loss.backward()
                optimizer.step()
            size = targets.size(0)
            guesses = logits.argmax(dim=1)
            total_loss += loss.item() * size
            total_correct += (guesses == targets).sum().item()
            total_images += size
            if predictions:
                predicted.extend(guesses.cpu().tolist())
                labels.extend(targets.cpu().tolist())
                confidences.extend(logits.softmax(dim=1).amax(dim=1).cpu().tolist())
            batches.set_postfix(loss=f"{total_loss / total_images:.4f}",
                                accuracy=f"{100 * total_correct / total_images:.2f}%", refresh=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not total_images:
        raise ValueError("Empty loader")
    metrics = {"loss": total_loss / total_images, "accuracy": total_correct / total_images,
               "seconds": time.perf_counter() - start, "images": total_images}
    return (metrics, labels, predicted, confidences) if predictions else metrics


def make_optimizer(model, config):
    settings = config["optimizer_settings"]
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=config["weight_decay"])
    if config["optimizer"] not in ("sgd", "sgd_momentum"):
        raise ValueError("Unknown optimizer")
    return torch.optim.SGD(model.parameters(), lr=settings["lr"],
                           momentum=settings.get("momentum", 0.0), weight_decay=config["weight_decay"])


def save_history(history, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    os.replace(temporary, path)


def completed_run(folder):
    """Only skip folders that contain all expected completed-run artifacts."""
    if not all((folder / name).is_file() for name in ("config.json", "history.csv", "best_weights.pt")):
        return False
    config = load_config(folder / "config.json")
    history = pd.read_csv(folder / "history.csv")
    return history["epoch"].tolist() == list(range(1, config["epochs"] + 1))


def train(root, *, model_name="model_a", optimizer_name="adam", lr=None, run_name=None,
          config_path=None, output_root=None, data_root=None, split_dir=None,
          epochs=30, batch_size=64, seed=42, weight_decay=1e-4, num_workers=0,
          device_name="auto", skip_existing=False, resume=None, progress=True):
    device = select_device(device_name)
    if resume is not None:
        folder = resolve_path(root, resume)
        last_path = folder / "last_checkpoint.pt"
        if not last_path.is_file():
            raise ValueError("Exact resume requires last_checkpoint.pt; old best_weights.pt is evaluation-only.")
        if completed_run(folder):
            raise FileExistsError("This run is complete; choose a fresh run name to train again.")
        # Full checkpoints contain Python and NumPy random state. Load only your own trusted files.
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        config = load_config(folder / "config.json")
        if last.get("format_version") != 1 or last["config"] != config:
            raise ValueError("Last checkpoint and saved configuration disagree")
        if last["epoch"] > config["epochs"]:
            raise ValueError("Checkpoint epoch exceeds configured epochs")
        if str(device) != config["device"] or num_workers != config["num_workers"]:
            raise ValueError("Resume with the saved device and num_workers for repeatable continuation")
        environment = {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}
        if environment != config["environment"] or platform.python_version() != config["python_version"]:
            raise ValueError("Resume requires the saved Python and package versions")
        data_root = resolve_path(root, data_root or config["data_root"])
        split_dir = resolve_path(root, split_dir or config["split_dir"])
    else:
        if (epochs <= 0 or batch_size <= 0 or num_workers < 0 or not np.isfinite(weight_decay)
                or weight_decay < 0 or not 0 <= seed < 2**32):
            raise ValueError("Invalid epoch, batch, worker or weight-decay setting")
        lr = lr if lr is not None else (0.001 if optimizer_name == "adam" else 0.01)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive and finite")
        run_name = run_name or f"{model_name}_{optimizer_name}_lr{lr:g}"
        if run_name in ("", ".", "..") or any(c in run_name for c in "/\\:"):
            raise ValueError("Run name must be a single folder name")
        folder = resolve_path(root, output_root or "outputs/custom_cnn") / run_name
        if folder.exists():
            if skip_existing and completed_run(folder):
                print(f"Skipping completed run: {folder}")
                return folder
            raise FileExistsError(f"Run already exists: {folder}. Use a fresh name, --skip-existing for a completed run, or --resume for a new incomplete run.")
        config = copy.deepcopy(load_config(resolve_path(root, config_path or
                                   "outputs/custom_cnn/model_a_adam_lr0.001/config.json")))
        data_root = resolve_path(root, data_root or "data/raw/eurosat/2750")
        split_dir = resolve_path(root, split_dir or "data/splits")
        # A source config supplies only the immutable dataset setup for a new run.
        config = {key: config[key] for key in ("class_names", "normalization_mean", "normalization_std")}
        settings = {"lr": lr}
        if optimizer_name == "sgd_momentum":
            settings["momentum"] = 0.9
        config.update(model=model_name, optimizer=optimizer_name, optimizer_settings=settings,
                      epochs=epochs, batch_size=batch_size, seed=seed, weight_decay=weight_decay,
                      device=str(device), num_workers=num_workers, input_size=64,
                      data_root=portable_path(root, data_root), split_dir=portable_path(root, split_dir),
                      precision="float32", checkpoint_selection="minimum_validation_loss",
                      augmentation=["horizontal_flip_p0.5", "vertical_flip_p0.5", "random_quarter_turn"],
                      format_version=1, python_version=platform.python_version(),
                      environment={name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")})
    check_splits(split_dir, config["class_names"])
    fingerprints = split_fingerprints(split_dir)
    if resume and fingerprints != config["split_sha256"]:
        raise ValueError("Saved splits changed since this run started")
    config["split_sha256"] = fingerprints
    train_frame = load_split(split_dir, "train", config["class_names"])
    val_frame = load_split(split_dir, "validation", config["class_names"])
    for frame in (train_frame, val_frame):
        for path in frame["relative_path"]:
            if not (data_root / path).is_file():
                raise FileNotFoundError(data_root / path)
    seed_everything(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = make_loader(train_frame, data_root, make_transform(config, True),
                               config["batch_size"], device, training=True,
                               generator=generator, num_workers=num_workers)
    val_loader = make_loader(val_frame, data_root, make_transform(config), config["batch_size"],
                             device, num_workers=num_workers)
    model = build_model(config["model"], len(config["class_names"])).to(device)
    optimizer = make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    if resume:
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        history = last["history"]
        best = last["best_result"]
        best_state = last["best_weights"]
        start_epoch = last["epoch"] + 1
        restore_random_state(last["random_state"], generator)
        # Repair artifacts from an interrupted write using the canonical last checkpoint.
        save_history(history, folder / "history.csv")
        atomic_torch_save(best_state, folder / "best_weights.pt")
    else:
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "config.json", config)
        history, best, best_state, start_epoch = [], {"loss": float("inf"), "epoch": None}, None, 1
    print(f"Run: {folder.name} | {device} | starting epoch {start_epoch}")
    for epoch in range(start_epoch, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, progress=progress)
        val_metrics = run_epoch(model, val_loader, criterion, device, progress=progress)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"], "train_seconds": train_metrics["seconds"],
                        "val_seconds": val_metrics["seconds"], "learning_rate": optimizer.param_groups[0]["lr"]})
        improved = val_metrics["loss"] < best["loss"]
        if improved:
            best = {"epoch": epoch, "loss": val_metrics["loss"], "accuracy": val_metrics["accuracy"]}
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        last = {"format_version": 1, "config": config, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "best_result": best, "best_weights": best_state, "history": history,
                "random_state": capture_random_state(generator)}
        atomic_torch_save(last, folder / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(best_state, folder / "best_weights.pt")
        save_history(history, folder / "history.csv")
        print(f"Epoch {epoch:02d}/{config['epochs']} | train loss {train_metrics['loss']:.4f}, "
              f"acc {100 * train_metrics['accuracy']:.2f}% | val loss {val_metrics['loss']:.4f}, "
              f"acc {100 * val_metrics['accuracy']:.2f}% | train {train_metrics['seconds']:.1f}s, "
              f"val {val_metrics['seconds']:.1f}s" + (" | best saved" if improved else ""))
    print(f"Best validation-loss checkpoint: epoch {best['epoch']}")
    return folder
