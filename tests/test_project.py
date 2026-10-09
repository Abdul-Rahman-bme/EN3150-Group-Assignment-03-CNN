"""Focused checks against saved artifacts. No real training or test evaluation."""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cnn_assignment.cli import main
from cnn_assignment.data import check_splits, load_split
from cnn_assignment.evaluation import classification_metrics
from cnn_assignment.models import build_model, model_cost
from cnn_assignment.training import run_epoch, train
from cnn_assignment.transforms import make_transform
from cnn_assignment.utils import (capture_random_state, load_config, project_root,
                                  restore_random_state, seed_everything, sha256)

ROOT = project_root()
RUNS = ROOT / "outputs/custom_cnn"
CONFIG = RUNS / "model_a_adam_lr0.001/config.json"


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def test_model_shapes_parameters_macs_and_all_saved_checkpoints(self):
        for name, parameters, macs in [("model_a", 94762, 41288960), ("model_b", 12965, 5141760),
                                       ("model_c", 46373, 18561536)]:
            model = build_model(name)
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(2, 3, 64, 64))
            self.assertEqual(tuple(output.shape), (2, 10))
            self.assertTrue(torch.isfinite(output).all())
            cost = model_cost(model)
            self.assertEqual(cost["trainable_parameters"], parameters)
            self.assertEqual(cost["estimated_fp32_parameter_bytes"], 4 * parameters)
            self.assertEqual(cost["conv_linear_macs_per_image"], macs)
        folders = sorted(RUNS.glob("*/config.json"))
        self.assertGreaterEqual(len(folders), 6)
        for config_path in folders:
            config = load_config(config_path)
            model = build_model(config["model"], len(config["class_names"]))
            state = torch.load(config_path.parent / "best_weights.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)

    def test_saved_split_integrity_and_immutable_artifacts(self):
        config = load_config(CONFIG)
        checked = check_splits(ROOT / "data/splits", config["class_names"])
        self.assertEqual(checked["sizes"], {"train": 18900, "validation": 4050, "test": 4050})
        self.assertEqual(checked["unique_pixel_hashes"], 27000)
        manifest = json.loads((ROOT / "outputs/refactor_checks/source_manifest.json").read_text(encoding="utf-8-sig"))
        for record in manifest:
            if record["path"] != "test.ipynb":
                self.assertEqual(sha256(ROOT / record["path"]), record["sha256"], record["path"])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            split_dir = Path(directory)
            frame = pd.read_csv(ROOT / "data/splits/validation.csv")
            frame.loc[0, "class_name"] = "wrong"
            frame.to_csv(split_dir / "validation.csv", index=False)
            with self.assertRaisesRegex(ValueError, "label mapping"):
                load_split(split_dir, "validation", config["class_names"])

    def test_saved_normalization_and_augmentation_shape(self):
        config = load_config(CONFIG)
        image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        transform = make_transform(config)
        tensor = transform(image)
        expected = (torch.full((3,), 128 / 255) - torch.tensor(config["normalization_mean"])) / torch.tensor(config["normalization_std"])
        torch.testing.assert_close(tensor[:, 0, 0], expected)
        self.assertTrue(torch.equal(tensor, transform(image)))
        self.assertEqual(tuple(make_transform(config, training=True)(image).shape), (3, 64, 64))

    def test_sample_weighted_metrics_include_final_batch(self):
        logits = torch.tensor([[4., 0.], [0., 4.], [0., 4.]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = run_epoch(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"), progress=False)
        self.assertAlmostEqual(result["loss"], nn.CrossEntropyLoss()(logits, labels).item(), places=6)
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["images"], 3)

    def test_metrics_and_zero_division(self):
        result, report, matrix, normalized = classification_metrics([0, 0, 1], [0, 1, 1], ["a", "b", "c"])
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(matrix.tolist(), [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertEqual(report.loc[2, "f1"], 0)
        self.assertAlmostEqual(result["macro_f1"], 4 / 9)
        self.assertEqual(normalized[0].tolist(), [.5, .5, 0])

    def test_random_and_shuffle_restore(self):
        seed_everything(42)
        generator = torch.Generator().manual_seed(42)
        state = capture_random_state(generator)

        def draw():
            return random.random(), np.random.rand(), torch.rand(1), torch.randperm(7, generator=generator)

        expected = draw()
        restore_random_state(state, generator)
        actual = draw()
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_completed_run_protection_skip_and_old_resume_limitation(self):
        with self.assertRaises(FileExistsError):
            train(ROOT, device_name="cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            skipped = train(ROOT, skip_existing=True, device_name="cpu")
        self.assertEqual(skipped, RUNS / "model_a_adam_lr0.001")
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            train(ROOT, resume="outputs/custom_cnn/model_a_adam_lr0.001", device_name="cpu")

    def test_resume_with_simulated_epochs_without_training(self):
        """Exercise the actual save/resume loop with fake epochs; no gradients or optimizer steps."""
        for model_name in ("model_a", "model_c"):
            with self.subTest(model=model_name):
                self.check_simulated_resume(model_name)

    def check_simulated_resume(self, model_name):
        calls = 0

        def fake_epoch(model, loader, criterion, device, optimizer=None, **kwargs):
            nonlocal calls
            calls += 1
            draws = random.random() + np.random.rand() + torch.rand(1).item()
            if optimizer is not None:
                draws += torch.randperm(7, generator=loader.generator)[0].item()
                # Synthetic state checks optimizer and model restoration without a training step.
                parameter = next(model.parameters())
                previous = optimizer.state[parameter].get("counter", torch.tensor(0.)).item()
                optimizer.state[parameter]["counter"] = torch.tensor(previous + draws)
                optimizer.state[parameter].setdefault("step", torch.tensor(0.)).add_(1)
                optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, draws)
                optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, draws * draws)
                with torch.no_grad():
                    model.classifier.bias.add_(draws)
            return {"loss": float(draws + model.classifier.bias[0].item()), "accuracy": .5,
                    "seconds": 1., "images": 7}

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            output_root = Path(directory)
            with patch("cnn_assignment.training.run_epoch", side_effect=fake_epoch), contextlib.redirect_stdout(io.StringIO()):
                whole = train(ROOT, model_name=model_name, run_name="whole", output_root=output_root, epochs=3, device_name="cpu")
            calls = 0

            def interrupted(*args, **kwargs):
                if calls == 2:
                    raise RuntimeError("Simulated interruption at epoch boundary")
                return fake_epoch(*args, **kwargs)

            with patch("cnn_assignment.training.run_epoch", side_effect=interrupted), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    train(ROOT, model_name=model_name, run_name="resumed", output_root=output_root, epochs=3, device_name="cpu")
            checkpoint_path = output_root / "resumed/last_checkpoint.pt"
            snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(snapshot["epoch"], 1)
            self.assertEqual(set(snapshot["random_state"]), {"python", "numpy", "torch", "shuffle", "cuda"})
            self.assertTrue(snapshot["optimizer_state"]["state"])
            with patch("cnn_assignment.training.run_epoch", side_effect=fake_epoch), contextlib.redirect_stdout(io.StringIO()):
                resumed = train(ROOT, resume=output_root / "resumed", device_name="cpu")
            pd.testing.assert_frame_equal(pd.read_csv(whole / "history.csv"), pd.read_csv(resumed / "history.csv"))
            full = torch.load(whole / "last_checkpoint.pt", map_location="cpu", weights_only=False)
            continued = torch.load(resumed / "last_checkpoint.pt", map_location="cpu", weights_only=False)
            source = load_config(CONFIG)
            self.assertEqual(continued["config"]["model"], model_name)
            for key in ("class_names", "normalization_mean", "normalization_std"):
                self.assertEqual(continued["config"][key], source[key])
            self.assertEqual(continued["config"]["split_dir"], "data/splits")
            self.assertEqual(full["best_result"], continued["best_result"])
            for key in full["model_state"]:
                self.assertTrue(torch.equal(full["model_state"][key], continued["model_state"][key]), key)
            for key in full["optimizer_state"]["state"]:
                for field in full["optimizer_state"]["state"][key]:
                    self.assertTrue(torch.equal(full["optimizer_state"]["state"][key][field],
                                                continued["optimizer_state"]["state"][key][field]))
            # A completed new run also rejects resume.
            with self.assertRaises(FileExistsError):
                train(ROOT, resume=resumed, device_name="cpu")
            # Recover a final canonical checkpoint whose history/weights write was interrupted.
            repair = output_root / "repair"
            repair.mkdir()
            for name in ("config.json", "last_checkpoint.pt"):
                shutil.copy2(whole / name, repair / name)
            pd.read_csv(whole / "history.csv").iloc[:1].to_csv(repair / "history.csv", index=False)
            with patch("cnn_assignment.training.run_epoch", side_effect=AssertionError("No new epochs needed")), contextlib.redirect_stdout(io.StringIO()):
                train(ROOT, resume=repair, device_name="cpu")
            pd.testing.assert_frame_equal(pd.read_csv(whole / "history.csv"), pd.read_csv(repair / "history.csv"))
            self.assertTrue((repair / "best_weights.pt").is_file())

    def test_notebook_outputs_preserved_and_sources_compile(self):
        from IPython.core.inputtransformer2 import TransformerManager

        notebook = json.loads((ROOT / "test.ipynb").read_text(encoding="utf-8"))
        # Per-ID hashes freeze the supplied saved outputs independently of cell
        # order. Only the obsolete section 7.20 cells may be absent.
        baseline = json.loads((ROOT / "tests/fixtures/notebook_preservation.json").read_text(encoding="utf-8"))
        import hashlib
        cells_by_id = {cell["id"]: cell for cell in notebook["cells"]}
        self.assertEqual(len(cells_by_id), len(notebook["cells"]), "Duplicate notebook cell IDs")
        historical = baseline["cell_records_sha256_by_id"]
        required = set(historical) - set(baseline["optional_cell_ids"])
        self.assertFalse(required - cells_by_id.keys(),
                         f"Missing required historical cells: {sorted(required - cells_by_id.keys())}")
        for cell_id in historical.keys() & cells_by_id.keys():
            fields = {key: value for key, value in cells_by_id[cell_id].items() if key != "source"}
            actual = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
            self.assertEqual(actual, historical[cell_id], f"Historical cell changed: {cell_id}")
        self.assertEqual(notebook["metadata"], baseline["metadata"])
        transformer = TransformerManager()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell_id=cell["id"]):
                    source = transformer.transform_cell("".join(cell["source"]))
                    compile(source, f"test.ipynb[cell {cell['id']}]", "exec")

    def test_cli_from_another_working_directory_and_plot_without_torch(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            outside = Path(directory)
            help_check = subprocess.run([sys.executable, "-m", "cnn_assignment", "--help"],
                                        cwd=outside, env=environment, capture_output=True, text=True)
            self.assertEqual(help_check.returncode, 0, help_check.stderr)
            self.assertIn("evaluate-test", help_check.stdout)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--project-root", str(ROOT), "prepare"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                main(["train", "--resume", str(RUNS / "model_a_adam_lr0.001"), "--epochs", "31"])
            self.assertEqual(failure.exception.code, 2)
            script = "from cnn_assignment.cli import main; import sys; main(['--project-root', sys.argv[2], 'plot', '--runs-root', sys.argv[3], '--output-dir', sys.argv[1]]); assert 'torch' not in sys.modules; assert 'cnn_assignment.data' not in sys.modules; assert 'cnn_assignment.models' not in sys.modules"
            plotted = subprocess.run([sys.executable, "-c", script, str(outside / "plots"), str(outside), str(RUNS)],
                                     cwd=outside, env=environment, capture_output=True, text=True)
            self.assertEqual(plotted.returncode, 0, plotted.stderr)
            self.assertTrue((outside / "plots/optimizer_comparison.png").is_file())
            summary = pd.read_csv(outside / "plots/optimizer_comparison.csv")
            saved_histories = [path for path in RUNS.glob("*/history.csv")
                               if (path.parent / "config.json").is_file()]
            self.assertEqual(len(summary), len(saved_histories))
            self.assertAlmostEqual(summary.loc[summary["run_folder"] == "model_a_adam_lr0.001", "selected_val_loss"].iloc[0], .1721100697215692)
            # A plot command rejects a report folder it already wrote.
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                main(["plot", "--output-dir", str(outside / "plots")])
            self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    (ROOT / "tmp").mkdir(exist_ok=True)
    torch.set_num_threads(2)
    unittest.main()
