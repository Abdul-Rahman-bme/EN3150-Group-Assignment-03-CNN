"""Pretrained workflow checks; only synthetic inputs and simulated epochs."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from cnn_assignment.cli import build_parser, main
from cnn_assignment.evaluation import evaluate
from cnn_assignment.model_specs import PRETRAINED_MODELS, PRETRAINED_SPECS, pretrained_settings
from cnn_assignment.models import PretrainedWeightsError, build_model, load_saved_state, model_cost
from cnn_assignment.plotting import plot_comparison, plot_history
from cnn_assignment.training import train
from cnn_assignment.transforms import make_transform
from cnn_assignment.utils import load_config, project_root, seed_everything, write_json

ROOT = project_root()
SOURCE = ROOT / 'outputs/custom_cnn/model_a_adam_lr0.001/config.json'


def source_file(name):
    filename = PRETRAINED_SPECS[name]['pretrained_source'].rsplit('/', 1)[1]
    candidates = [ROOT / 'outputs/pretrained_weights' / filename,
                  ROOT / 'tmp/pretrained_weights' / filename,
                  ROOT / 'tmp/pretrained_download_cache/hub/checkpoints' / filename,
                  Path(torch.hub.get_dir()) / 'checkpoints' / filename]
    return next((path for path in candidates if path.is_file()), None)


class PretrainedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def config(self, name):
        config = load_config(SOURCE)
        config.update(model=name, input_size=64, **pretrained_settings(name))
        return config

    def test_shapes_finite_gradients_and_backbone_update(self):
        for name in PRETRAINED_MODELS:
            with self.subTest(model=name):
                seed_everything(42)
                model = build_model(name).eval()
                for batch in (1, 20, 64):
                    with torch.inference_mode():
                        logits = model(torch.randn(batch, 3, 64, 64))
                    self.assertEqual(tuple(logits.shape), (batch, 10))
                    self.assertTrue(torch.isfinite(logits).all().item())
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
                parameter = next(model.parameters())
                before = parameter.detach().clone()
                loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                                    for p in model.parameters()))
                optimizer.step()
                self.assertFalse(torch.equal(before, parameter))
                cost = model_cost(model)
                self.assertEqual(cost['total_parameters'], cost['trainable_parameters'])
                expected = {'mobilenet_v2': (2236682, 24461312), 'shufflenet_v2_x0_5': (352042, 3230848)}
                self.assertEqual((cost['total_parameters'], cost['conv_linear_macs_per_image']), expected[name])

    def test_genuine_backbone_matches_every_official_non_classifier_tensor(self):
        for name in PRETRAINED_MODELS:
            path = source_file(name)
            if path is None:
                self.skipTest('Genuine source checkpoint not cached; run the documented verification setup')
            with self.subTest(model=name):
                model = build_model(name, pretrained=True, pretrained_weights_file=path, progress=False)
                official = torch.load(path, map_location='cpu', weights_only=True)
                prefix = 'classifier.' if name == 'mobilenet_v2' else 'fc.'
                for key, tensor in model.state_dict().items():
                    if not key.startswith(prefix):
                        if key in official:
                            self.assertTrue(torch.equal(tensor, official[key]), key)
                        else:
                            # The old official ShuffleNet checkpoint predates BN counters.
                            self.assertTrue(key.endswith('num_batches_tracked'), key)
                            self.assertEqual(tensor.item(), 0)
                head = model.classifier[1] if name == 'mobilenet_v2' else model.fc
                self.assertEqual(head.out_features, 10)
                self.assertFalse(torch.equal(head.weight, official[prefix + ('1.weight' if name == 'mobilenet_v2' else 'weight')][:10]))
                self.assertTrue(model.pretrained_initialization['pretrained_backbone_loaded'])
                self.assertEqual(len(model.pretrained_initialization['pretrained_source_sha256']), 64)

    def test_download_failure_and_wrong_manual_checkpoint_stop(self):
        for name in PRETRAINED_MODELS:
            with patch('cnn_assignment.models._load_pretrained_state', side_effect=OSError('synthetic download failure')):
                with self.assertRaisesRegex(PretrainedWeightsError, 'No random fallback.*official file'):
                    build_model(name, pretrained=True)
            with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
                path = Path(temporary) / 'wrong.pth'
                torch.save(build_model(name).state_dict(), path)
                with self.assertRaisesRegex(PretrainedWeightsError, 'SHA-256 prefix mismatch'):
                    build_model(name, pretrained=True, pretrained_weights_file=path)

    def test_imagenet_transforms_are_64_pixels_and_custom_transforms_unchanged(self):
        image = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
        for name in PRETRAINED_MODELS:
            config = self.config(name)
            transform = make_transform(config)
            actual = transform(image)
            expected = (torch.full((3,), 128 / 255) - torch.tensor(config['normalization_mean'])) / torch.tensor(config['normalization_std'])
            torch.testing.assert_close(actual[:, 0, 0], expected)
            self.assertEqual(tuple(actual.shape), (3, 64, 64))
            self.assertEqual(tuple(make_transform(config, True)(image).shape), (3, 64, 64))
            self.assertNotIn('Resize', str(transform))
            self.assertNotIn('CenterCrop', str(transform))
        original = load_config(SOURCE)
        self.assertNotEqual(original['normalization_mean'], self.config('mobilenet_v2')['normalization_mean'])

    def test_strict_offline_restoration_including_batchnorm_buffers(self):
        for name in PRETRAINED_MODELS:
            seed_everything(42)
            model = build_model(name).train()
            with torch.no_grad():
                model(torch.randn(3, 3, 64, 64))
            model.eval()
            image = torch.randn(1, 3, 64, 64)
            with torch.inference_mode():
                expected = model(image)
            with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
                path = Path(temporary) / 'best_weights.pt'
                torch.save(model.state_dict(), path)
                with patch('cnn_assignment.models._load_pretrained_state', side_effect=AssertionError('Network forbidden')):
                    restored = build_model(name, pretrained=False).eval()
                state = torch.load(path, map_location='cpu', weights_only=True)
                load_saved_state(restored, state)
                for key, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
                self.assertTrue(any(key.endswith('num_batches_tracked') for key in state))
                with torch.inference_mode():
                    torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
                damaged = dict(state)
                del damaged[next(key for key in state if key.endswith('running_mean'))]
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    load_saved_state(restored, damaged)
                damaged = dict(state)
                del damaged[next(key for key in state if key.endswith('num_batches_tracked'))]
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    load_saved_state(restored, damaged)

    def test_cli_initial_commands_and_explicit_test_routing_without_execution(self):
        for name in PRETRAINED_MODELS:
            argv = ['train', '--model', name, '--optimizer', 'adam', '--lr', '0.0001',
                    '--weight-decay', '0.0001', '--epochs', '30', '--batch-size', '64', '--seed', '42']
            with patch('cnn_assignment.training.train') as mocked:
                main(argv)
            kwargs = mocked.call_args.kwargs
            self.assertEqual((kwargs['epochs'], kwargs['batch_size'], kwargs['seed'], kwargs['lr']), (30, 64, 42, .0001))
            self.assertTrue(kwargs['progress'])
            parsed = build_parser().parse_args(['train', '--model', name, '--pretrained-weights-file', 'official.pth'])
            self.assertEqual(parsed.pretrained_weights_file, 'official.pth')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['train', '--resume', 'not-a-run', '--pretrained-weights-file', 'official.pth'])
        with patch('cnn_assignment.evaluation.evaluate', return_value={}) as mocked, contextlib.redirect_stdout(io.StringIO()):
            main(['evaluate-test', '--run', 'outputs/pretrained_cnn/example'])
        self.assertEqual(mocked.call_args.kwargs['split'], 'test')

    def test_validation_and_history_only_plotting_without_download(self):
        for name in PRETRAINED_MODELS:
            with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
                root = Path(temporary)
                run = root / name
                run.mkdir()
                config = self.config(name)
                write_json(run / 'config.json', config)
                self.assertEqual(load_config(run / 'config.json'), config)
                history = pd.DataFrame({'epoch': [1, 2], 'train_loss': [1., .6], 'val_loss': [.8, .5],
                                        'train_accuracy': [.5, .7], 'val_accuracy': [.6, 1.], 'train_seconds': [2., 1.]})
                history.to_csv(run / 'history.csv', index=False)
                torch.save(build_model(name).state_dict(), run / 'best_weights.pt')
                frame = pd.DataFrame({'relative_path': ['synthetic.png'], 'label': [0],
                                      'class_name': [config['class_names'][0]]})
                measured = {'loss': .5, 'accuracy': 1., 'seconds': 1., 'images': 1}
                with patch('cnn_assignment.models._load_pretrained_state', side_effect=AssertionError('Network forbidden')), \
                        patch('cnn_assignment.evaluation.load_split', return_value=frame), \
                        patch('cnn_assignment.evaluation.make_loader'), \
                        patch('cnn_assignment.evaluation.run_epoch', return_value=(measured, [0], [0], [.9])):
                    result = evaluate(ROOT, run, split='validation', device_name='cpu', compare_history=True)
                self.assertTrue(result['history_comparison']['matches'])
                self.assertEqual(result['pretrained_weights'], PRETRAINED_SPECS[name]['pretrained_weights'])
                self.assertEqual(result['actual_weights_file_bytes'], (run / 'best_weights.pt').stat().st_size)
                self.assertEqual(plot_history(run, root / 'curves.png')['model'], name)
                plot_comparison([run, SOURCE.parent], root)
                self.assertTrue((root / 'optimizer_comparison.png').is_file())
                with contextlib.redirect_stdout(io.StringIO()):
                    main(['plot', '--additional-runs-root', str(root), '--output-dir', str(root / 'combined')])
                combined = pd.read_csv(root / 'combined/optimizer_comparison.csv')
                self.assertEqual(len(combined), 13)
                self.assertIn(name, set(combined['model']))

    def test_epoch_boundary_resume_with_genuine_initialization_and_simulated_epochs(self):
        for name in PRETRAINED_MODELS:
            path = source_file(name)
            if path is None:
                self.skipTest('Genuine checkpoint not cached; workflow test never downloads automatically')
            with self.subTest(model=name), tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
                root = Path(temporary)
                Image.new('RGB', (64, 64)).save(root / 'synthetic.png')
                frame = pd.DataFrame({'relative_path': ['synthetic.png'], 'label': [0],
                                      'class_name': [self.config(name)['class_names'][0]]})
                calls = 0

                def fake_epoch(model, loader, criterion, device, optimizer=None, **kwargs):
                    nonlocal calls
                    calls += 1
                    draw = random.random() + np.random.rand() + torch.rand(1).item()
                    if optimizer is not None:
                        draw += torch.randperm(7, generator=loader.generator)[0].item()
                        parameter = next(model.parameters())
                        state = optimizer.state[parameter]
                        state.setdefault('step', torch.tensor(0.)).add_(1)
                        state['exp_avg'] = torch.full_like(parameter, draw)
                        state['exp_avg_sq'] = torch.full_like(parameter, draw * draw)
                        with torch.no_grad():
                            parameter.add_(draw / 100)
                            next(module for module in model.modules() if isinstance(module, nn.BatchNorm2d)).running_mean.add_(draw)
                    return {'loss': float(draw), 'accuracy': .5, 'seconds': 1., 'images': 1}

                with patch('cnn_assignment.training.load_split', return_value=frame), contextlib.redirect_stdout(io.StringIO()):
                    with patch('cnn_assignment.training.run_epoch', side_effect=fake_epoch):
                        whole = train(ROOT, model_name=name, run_name='whole', output_root=root, data_root=root,
                                      epochs=3, device_name='cpu', pretrained_weights_file=path)
                    calls = 0

                    def interrupted(*args, **kwargs):
                        if calls == 2:
                            raise RuntimeError('Simulated epoch-boundary interruption')
                        return fake_epoch(*args, **kwargs)

                    with patch('cnn_assignment.training.run_epoch', side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError, 'Simulated epoch-boundary'):
                            train(ROOT, model_name=name, run_name='resumed', output_root=root, data_root=root,
                                  epochs=3, device_name='cpu', pretrained_weights_file=path)
                    with patch('cnn_assignment.models._load_pretrained_state', side_effect=AssertionError('Network forbidden')), \
                            patch('cnn_assignment.training.run_epoch', side_effect=fake_epoch):
                        resumed = train(ROOT, resume=root / 'resumed', device_name='cpu')
                a = torch.load(whole / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
                b = torch.load(resumed / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
                self.assertEqual(a['history'], b['history'])
                self.assertEqual(a['best_result'], b['best_result'])
                for key in a['model_state']:
                    self.assertTrue(torch.equal(a['model_state'][key], b['model_state'][key]), key)
                for param in a['optimizer_state']['state']:
                    for key in a['optimizer_state']['state'][param]:
                        self.assertTrue(torch.equal(a['optimizer_state']['state'][param][key], b['optimizer_state']['state'][param][key]))
                config = load_config(resumed / 'config.json')
                self.assertEqual(config['optimizer_settings']['lr'], .0001)
                self.assertTrue(config['pretrained_backbone_loaded'])
                summary = json.loads((resumed / 'model_summary.json').read_text())
                self.assertEqual(summary['actual_weights_file_bytes'], (resumed / 'best_weights.pt').stat().st_size)
                self.assertEqual(summary['total_parameters'], summary['trainable_parameters'])


if __name__ == '__main__':
    unittest.main()
