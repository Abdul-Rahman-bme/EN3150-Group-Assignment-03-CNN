"""History-only plotting: no image, torchvision, torch or model imports."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_history(run):
    frame = pd.read_csv(Path(run) / "history.csv")
    required = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "train_seconds"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid history: {run}")
    if not frame[list(required)].notna().all().all():
        raise ValueError("Missing history metrics")
    return frame


def history_summary(run, history=None):
    run = Path(run)
    history = load_history(run) if history is None else history
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    best = history.loc[history["val_loss"].idxmin()]
    return {"run_folder": run.name, "model": config["model"], "optimizer": config["optimizer"],
            "learning_rate": config["optimizer_settings"]["lr"], "selected_epoch": int(best["epoch"]),
            "selected_val_loss": float(best["val_loss"]),
            "selected_val_accuracy_percent": float(100 * best["val_accuracy"]),
            "mean_train_seconds": float(history["train_seconds"].mean()),
            "mean_train_seconds_excluding_first": (float(history["train_seconds"].iloc[1:].mean())
                                                    if len(history) > 1 else None)}


def plot_history(run, output):
    history = load_history(run)
    best = history.loc[history["val_loss"].idxmin()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, scale, title in zip(axes, ["loss", "accuracy"], [1, 100], ["Loss", "Accuracy (%)"]):
        for split, label in [("train", "Training"), ("val", "Validation")]:
            ax.plot(history["epoch"], scale * history[f"{split}_{metric}"], label=label)
        ax.axvline(best["epoch"], color="grey", linestyle="--", label=f"Selected epoch: {int(best['epoch'])}")
        ax.set(title=f"{Path(run).name}: {title}", xlabel="Epoch", ylabel=title)
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return history_summary(run, history)


def plot_comparison(runs, output_dir):
    output_dir = Path(output_dir)
    rows = [history_summary(run) for run in runs]
    models = sorted({row["model"] for row in rows})
    if not models:
        raise ValueError("No saved histories found")
    fig, axes = plt.subplots(len(models), 2, squeeze=False, figsize=(13, 4.5 * len(models)))
    for run, row in zip(runs, rows):
        history = load_history(run)
        index = models.index(row["model"])
        label = f"{row['optimizer']} lr={row['learning_rate']:g} ({Path(run).name})"
        axes[index, 0].plot(history["epoch"], history["val_loss"], label=label)
        axes[index, 1].plot(history["epoch"], 100 * history["val_accuracy"], label=label)
    for index, model in enumerate(models):
        for ax, title in zip(axes[index], ["Validation loss", "Validation accuracy (%)"]):
            ax.set(title=f"{model}: {title}", xlabel="Epoch", ylabel=title)
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "optimizer_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "optimizer_comparison.csv", index=False)
    return rows
