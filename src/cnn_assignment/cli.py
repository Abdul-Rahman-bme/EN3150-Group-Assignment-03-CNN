"""Each subcommand loads its own inputs; plotting stays independent of torch."""

import argparse
import json
from datetime import datetime, timezone
from .utils import load_config, new_output_directory, project_root, resolve_path, write_json

DEFAULT_CONFIG = "outputs/custom_cnn/model_a_adam_lr0.001/config.json"


def build_parser():
    parser = argparse.ArgumentParser(description="EN3150 EuroSAT experiment tools")
    parser.add_argument("--project-root", help="Project location; relative input paths resolve here")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Check and reuse saved splits and normalization")
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.add_argument("--data-root", default="data/raw/eurosat/2750")
    prepare.add_argument("--split-dir", default="data/splits")
    prepare.add_argument("--check-images", action="store_true", help="Read and hash every image; no model evaluation")
    prepare.add_argument("--check-archive", action="store_true")
    prepare.add_argument("--archive", default="data/raw/eurosat/EuroSAT.zip")
    prepare.add_argument("--output-dir", help="Optional fresh folder for preparation summary")

    train = commands.add_parser("train", help="Start a fresh run or resume a new incomplete run")
    train.add_argument("--model", choices=["model_a", "model_b", "model_c"], default="model_a")
    train.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"], default="adam")
    train.add_argument("--lr", type=float)
    train.add_argument("--run-name")
    train.add_argument("--config", default=DEFAULT_CONFIG, help="Source of class order and normalization for new runs")
    train.add_argument("--output-root", default="outputs/custom_cnn")
    train.add_argument("--data-root")
    train.add_argument("--split-dir")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    existing = train.add_mutually_exclusive_group()
    existing.add_argument("--skip-existing", action="store_true")
    existing.add_argument("--resume", metavar="RUN_FOLDER", help="Use saved config and full last checkpoint")
    add_runtime_arguments(train)

    plot = commands.add_parser("plot", help="Plot saved histories or compare optimizers")
    plot.add_argument("--run", help="One run; omit to compare all saved runs")
    plot.add_argument("--runs-root", default="outputs/custom_cnn")
    plot.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"], help="Filter comparison runs")
    plot.add_argument("--output-dir", help="Fresh report folder; defaults to a timestamped folder")

    validate = commands.add_parser("validate", help="Evaluate best weights on validation images")
    test = commands.add_parser("evaluate-test", help="Explicit final test evaluation; writes metrics and predictions")
    for command in (validate, test):
        command.add_argument("--run", required=True)
        command.add_argument("--data-root")
        command.add_argument("--split-dir")
        command.add_argument("--output-dir", help="Fresh report folder; defaults to a timestamped folder")
        add_runtime_arguments(command)
    validate.add_argument("--compare-history", action="store_true", help="Check selected epoch loss and accuracy; fail on mismatch")
    return parser


def add_runtime_arguments(parser):
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")


def report_path(root, command, supplied=None):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return resolve_path(root, supplied or f"outputs/reports/{command}_{timestamp}")


def execute(args, parser):
    root = project_root(args.project_root)
    if getattr(args, "num_workers", 0) < 0:
        raise ValueError("num_workers must be nonnegative")
    if args.command == "prepare":
        from .data import check_archive, check_images, check_splits, load_split
        config = load_config(resolve_path(root, args.config))
        split_dir = resolve_path(root, args.split_dir)
        data_root = resolve_path(root, args.data_root)
        result = check_splits(split_dir, config["class_names"])
        # Verify saved paths without opening images or changing any files.
        for name in result["sizes"]:
            frame = load_split(split_dir, name, config["class_names"])
            for relative in frame["relative_path"]:
                if not (data_root / relative).is_file():
                    raise FileNotFoundError(data_root / relative)
        result.update(normalization_mean=config["normalization_mean"], normalization_std=config["normalization_std"],
                      splits_reused=True, normalization_reused=True)
        if args.check_images:
            result["images_checked"] = check_images(data_root, split_dir, config["class_names"])
        if args.check_archive:
            result["archive_md5"] = check_archive(resolve_path(root, args.archive))
        if args.output_dir:
            output = new_output_directory(resolve_path(root, args.output_dir))
            write_json(output / "preparation_summary.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        if args.resume:
            # Avoid accepting overrides that would silently be ignored on resume.
            options = {part.split("=", 1)[0] for part in parser._argv if part.startswith("--")}
            overrides = options & {"--model", "--optimizer", "--lr", "--run-name", "--config", "--output-root",
                                   "--epochs", "--batch-size", "--seed", "--weight-decay"}
            if overrides:
                raise ValueError("Resume uses the saved configuration; remove " + ", ".join(sorted(overrides)))
        from .training import train
        train(root, model_name=args.model, optimizer_name=args.optimizer, lr=args.lr, run_name=args.run_name,
              config_path=args.config, output_root=args.output_root, data_root=args.data_root,
              split_dir=args.split_dir, epochs=args.epochs, batch_size=args.batch_size,
              seed=args.seed, weight_decay=args.weight_decay, num_workers=args.num_workers,
              device_name=args.device, skip_existing=args.skip_existing, resume=args.resume,
              progress=not args.no_progress)
    elif args.command == "plot":
        from .plotting import history_summary, plot_comparison, plot_history
        output_path = report_path(root, "plot", args.output_dir)
        if args.run:
            if args.optimizer:
                raise ValueError("--optimizer filters comparisons; use it without --run")
            run = resolve_path(root, args.run)
            history_summary(run)  # Read inputs before reserving output.
            output = new_output_directory(output_path)
            summary = plot_history(run, output / "learning_curves.png")
            write_json(output / "summary.json", summary)
        else:
            runs_root = resolve_path(root, args.runs_root)
            runs = [path for path in sorted(runs_root.iterdir()) if path.is_dir()
                    and (path / "history.csv").is_file() and (path / "config.json").is_file()]
            if args.optimizer:
                runs = [run for run in runs if history_summary(run)["optimizer"] == args.optimizer]
            if not runs:
                raise ValueError("No matching saved histories")
            for run in runs:
                history_summary(run)
            output = new_output_directory(output_path)
            plot_comparison(runs, output)
        print(f"Plots saved to: {output}")
    else:
        from .evaluation import evaluate
        split = "test" if args.command == "evaluate-test" else "validation"
        output = report_path(root, args.command, args.output_dir)
        if output.exists():
            raise FileExistsError(f"Output folder already exists: {output}")
        result = evaluate(root, args.run, split=split, device_name=args.device,
                          data_root=args.data_root, split_dir=args.split_dir, num_workers=args.num_workers,
                          output_dir=output, compare_history=getattr(args, "compare_history", False),
                          progress=not args.no_progress)
        print(json.dumps(result, indent=2))
        print(f"Evaluation saved to: {output}")
        if args.command == "validate" and args.compare_history and not result["history_comparison"]["matches"]:
            raise ValueError("Validation does not match saved history; see the saved comparison details")


def main(argv=None):
    import sys
    parser = build_parser()
    parser._argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(parser._argv)
    try:
        execute(args, parser)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
