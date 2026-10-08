"""Explicit validation or test evaluation of a saved best-weights checkpoint."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import load_split, make_loader, split_fingerprints
from .models import build_model, model_cost
from .training import run_epoch
from .transforms import make_transform
from .utils import load_config, new_output_directory, resolve_path, seed_everything, select_device, sha256, write_json


def classification_metrics(labels, predictions, class_names):
    size = len(class_names)
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(size, dtype=float), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(size, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(size, dtype=float), where=precision + recall != 0)
    normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix, dtype=float), where=support[:, None] != 0)
    metrics = {"accuracy": float(true_positive.sum() / matrix.sum()),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean()), "zero_division": 0}
    report = pd.DataFrame({"class_name": class_names, "precision": precision, "recall": recall,
                           "f1": f1, "support": support})
    return metrics, report, matrix, normalized


def evaluate(root, run, *, split="validation", device_name="auto", data_root=None,
             split_dir=None, num_workers=0, output_dir=None, compare_history=False, progress=True):
    if split not in ("validation", "test"):
        raise ValueError("Only validation and explicit test evaluation are supported")
    if compare_history and split != "validation":
        raise ValueError("History comparison only applies to validation")
    folder = resolve_path(root, run)
    config = load_config(folder / "config.json")
    data_root = resolve_path(root, data_root or config.get("data_root", "data/raw/eurosat/2750"))
    split_dir = resolve_path(root, split_dir or config.get("split_dir", "data/splits"))
    if "split_sha256" in config and split_fingerprints(split_dir) != config["split_sha256"]:
        raise ValueError("Split files differ from the run's saved fingerprints")
    frame = load_split(split_dir, split, config["class_names"])
    device = select_device(device_name)
    seed_everything(config["seed"])
    model = build_model(config["model"], len(config["class_names"])).to(device)
    weights_path = folder / "best_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    loader = make_loader(frame, data_root, make_transform(config), config["batch_size"],
                         device, num_workers=num_workers)
    measured, labels, predicted, confidences = run_epoch(
        model, loader, nn.CrossEntropyLoss(), device, progress=progress, predictions=True)
    metrics, report, matrix, normalized = classification_metrics(labels, predicted, config["class_names"])
    result = {"run": folder.name, "split": split, "device": str(device), "class_names": config["class_names"],
              **measured, **metrics, **model_cost(model, weights_path),
              "weights_sha256": sha256(weights_path), "split_sha256": sha256(split_dir / f"{split}.csv"),
              "timing_scope": "Full inference pass including loading, transfers, loss and prediction collection; CUDA synchronized"}
    if compare_history:
        history = pd.read_csv(folder / "history.csv")
        best = history.loc[history["val_loss"].idxmin()]
        loss_match = bool(np.isclose(measured["loss"], best["val_loss"], rtol=1e-5, atol=1e-6))
        accuracy_match = bool(np.isclose(measured["accuracy"], best["val_accuracy"], rtol=0, atol=1e-8))
        result["history_comparison"] = {"selected_epoch": int(best["epoch"]),
                                        "saved_val_loss": float(best["val_loss"]),
                                        "saved_val_accuracy": float(best["val_accuracy"]),
                                        "loss_difference": float(measured["loss"] - best["val_loss"]),
                                        "loss_matches": loss_match, "accuracy_matches": accuracy_match,
                                        "matches": loss_match and accuracy_match,
                                        "loss_rtol": 1e-5, "loss_atol": 1e-6, "accuracy_atol": 1e-8}
    if output_dir is not None:
        output = new_output_directory(resolve_path(root, output_dir))
        write_json(output / "metrics.json", result)
        report.to_csv(output / "per_class_metrics.csv", index=False)
        for name, values in [("confusion_matrix", matrix), ("confusion_matrix_normalized", normalized)]:
            pd.DataFrame(values, index=config["class_names"], columns=config["class_names"]).to_csv(
                output / f"{name}.csv", index_label="true_class")
        prediction_frame = frame[["relative_path", "label", "class_name"]].copy()
        prediction_frame["predicted_label"] = predicted
        prediction_frame["predicted_class"] = [config["class_names"][value] for value in predicted]
        prediction_frame["confidence"] = confidences
        prediction_frame["correct"] = np.array(labels) == np.array(predicted)
        prediction_frame.to_csv(output / "predictions.csv", index=False)
        save_confusion_plots(matrix, normalized, config["class_names"], output)
    return result


def save_confusion_plots(matrix, normalized, class_names, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, values in [("confusion_matrix", matrix), ("confusion_matrix_normalized", normalized)]:
        fig, ax = plt.subplots(figsize=(11, 9))
        view = ax.imshow(values, cmap="Blues")
        fig.colorbar(view, ax=ax)
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names, xlabel="Predicted class", ylabel="True class")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{values[i, j]:.2f}" if "normalized" in name else str(values[i, j]),
                        ha="center", va="center", color="white" if values[i, j] > values.max() / 2 else "black", fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
