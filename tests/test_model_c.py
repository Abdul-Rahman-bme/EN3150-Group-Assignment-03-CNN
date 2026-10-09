"""Model C integration checks using synthetic inputs and mocked epoch metrics."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn

from cnn_assignment.cli import build_parser, main
from cnn_assignment.evaluation import evaluate
from cnn_assignment.models import build_model, model_cost
from cnn_assignment.plotting import plot_comparison, plot_history
from cnn_assignment.training import make_optimizer
from cnn_assignment.utils import load_config, project_root, write_json

ROOT = project_root()
SOURCE_CONFIG = ROOT / "outputs/custom_cnn/model_a_adam_lr0.001/config.json"


class ModelCChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def test_block_order_widths_and_biases(self):
        model = build_model("model_c")
        for block, inputs, outputs in zip(model.features, (3, 64, 128), (64, 128, 256)):
            depthwise, pointwise, batchnorm, relu, pool = block.block
            self.assertEqual([type(layer) for layer in block.block],
                             [nn.Conv2d, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d])
            self.assertEqual((depthwise.in_channels, depthwise.out_channels, depthwise.groups),
                             (inputs, inputs, inputs))
            self.assertEqual(depthwise.kernel_size, (3, 3))
            self.assertEqual(depthwise.padding, (1, 1))
            self.assertEqual((pointwise.in_channels, pointwise.out_channels), (inputs, outputs))
            self.assertEqual(pointwise.kernel_size, (1, 1))
            self.assertIsNone(depthwise.bias)
            self.assertIsNone(pointwise.bias)
            self.assertEqual(batchnorm.num_features, outputs)
            self.assertTrue(batchnorm.affine)
            self.assertTrue(relu.inplace)
            self.assertEqual(pool.kernel_size, 2)
        self.assertEqual(model.pool.output_size, 1)
        self.assertEqual((model.classifier.in_features, model.classifier.out_features), (256, 10))
        self.assertIsNotNone(model.classifier.bias)

    def test_output_shape_for_single_partial_and_full_batches(self):
        model = build_model("model_c").eval()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        generator = torch.Generator().manual_seed(42)
        with torch.inference_mode():
            for batch_size in (1, 3, 64):
                with self.subTest(batch_size=batch_size):
                    images = torch.randn(batch_size, 3, 64, 64, generator=generator)
                    logits = model(images)
                    self.assertEqual(tuple(logits.shape), (batch_size, 10))
                    self.assertTrue(torch.isfinite(logits).all().item())
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_trainable_parameter_breakdown_and_cost(self):
        model = build_model("model_c")
        blocks = [sum(p.numel() for p in block.parameters() if p.requires_grad)
                  for block in model.features]
        self.assertEqual(blocks, [347, 9024, 34432])
        classifier = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        self.assertEqual(classifier, 2570)
        # Existing bias settings yield 46,373; the requested 47,525 is inconsistent.
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(total, 46373)
        self.assertEqual(total, sum(blocks) + classifier)
        cost = model_cost(model)
        self.assertEqual(cost["trainable_parameters"], total)
        self.assertEqual(cost["estimated_fp32_parameter_bytes"], 185492)
        self.assertEqual(cost["conv_linear_macs_per_image"], 18561536)

    def test_four_cli_run_settings_without_training(self):
        model = build_model("model_c")
        for name, lr, momentum in (("adam", .001, 0), ("sgd", .01, 0),
                                   ("sgd_momentum", .01, .9), ("sgd", .003, 0)):
            with self.subTest(optimizer=name, lr=lr):
                argv = ["train", "--model", "model_c", "--optimizer", name, "--lr", str(lr)]
                args = build_parser().parse_args(argv)
                self.assertEqual((args.epochs, args.batch_size, args.seed, args.weight_decay),
                                 (30, 64, 42, .0001))
                with patch("cnn_assignment.training.train") as mocked_train:
                    main(argv)
                settings = mocked_train.call_args.kwargs
                self.assertEqual(settings["model_name"], "model_c")
                self.assertEqual(settings["optimizer_name"], name)
                self.assertEqual(settings["lr"], lr)
                config = {"optimizer": name, "optimizer_settings": {"lr": lr, "momentum": momentum},
                          "weight_decay": args.weight_decay}
                optimizer = make_optimizer(model, config)
                self.assertIsInstance(optimizer, torch.optim.Adam if name == "adam" else torch.optim.SGD)
                self.assertEqual(optimizer.param_groups[0]["lr"], lr)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], .0001)
                if name != "adam":
                    self.assertEqual(optimizer.param_groups[0]["momentum"], momentum)

    def test_checkpoint_round_trip_restores_weights_buffers_and_outputs(self):
        original = build_model("model_c").eval()
        with torch.no_grad():
            original.features[0].block[2].running_mean.fill_(.5)
            original.features[0].block[2].running_var.fill_(1.5)
            original.classifier.bias.copy_(torch.arange(10, dtype=torch.float32))
        images = torch.randn(3, 3, 64, 64, generator=torch.Generator().manual_seed(42))
        with torch.inference_mode():
            expected = original(images)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            checkpoint = Path(directory) / "best_weights.pt"
            torch.save(original.state_dict(), checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = build_model("model_c").eval()
            result = restored.load_state_dict(state, strict=True)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            for name, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
            with torch.inference_mode():
                actual = restored(images)
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Strict loading must catch damaged checkpoints and the wrong architecture.
            incomplete = dict(state)
            del incomplete["classifier.bias"]
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                build_model("model_c").load_state_dict(incomplete, strict=True)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                build_model("model_b").load_state_dict(state, strict=True)

    def test_validation_and_comparison_with_synthetic_saved_run(self):
        config = load_config(SOURCE_CONFIG)
        config["model"] = "model_c"
        history = pd.DataFrame({"epoch": [1, 2], "train_loss": [1., .6], "val_loss": [.8, .5],
                                "train_accuracy": [.5, .7], "val_accuracy": [.6, .8],
                                "train_seconds": [2., 1.]})
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            output = Path(directory)
            run = output / "model_c_adam_lr0.001"
            run.mkdir()
            write_json(run / "config.json", config)
            self.assertEqual(load_config(run / "config.json"), config)
            history.to_csv(run / "history.csv", index=False)
            model = build_model("model_c")
            torch.save(model.state_dict(), run / "best_weights.pt")
            frame = pd.DataFrame({"relative_path": ["synthetic.png"], "label": [0],
                                  "class_name": [config["class_names"][0]]})
            # Exercise saved-config and strict weight loading; no dataset inference.
            measured = {"loss": .5, "accuracy": .8, "seconds": 1., "images": 1}
            with patch("cnn_assignment.evaluation.load_split", return_value=frame) as split, \
                    patch("cnn_assignment.evaluation.make_loader") as loader, \
                    patch("cnn_assignment.evaluation.run_epoch",
                          return_value=(measured, [0], [0], [.9])) as epoch:
                result = evaluate(ROOT, run, split="validation", device_name="cpu", compare_history=True)
            self.assertEqual(split.call_args.args[1], "validation")
            self.assertEqual(loader.call_args.args[3], 64)
            self.assertEqual(epoch.call_args.args[0].classifier.in_features, 256)
            self.assertEqual(result["trainable_parameters"], 46373)
            self.assertTrue(result["history_comparison"]["matches"])
            summary = plot_history(run, output / "curves.png")
            self.assertEqual(summary["model"], "model_c")
            original_run = ROOT / "outputs/custom_cnn/model_b_adam_lr0.001"
            plot_comparison([original_run, run], output)
            report = pd.read_csv(output / "optimizer_comparison.csv")
            self.assertEqual(set(report["model"]), {"model_b", "model_c"})
            self.assertTrue((output / "optimizer_comparison.png").is_file())
            with contextlib.redirect_stdout(io.StringIO()):
                main(["plot", "--run", str(run), "--output-dir", str(output / "cli_plot")])
            self.assertTrue((output / "cli_plot/learning_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
