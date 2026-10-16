"""Candidate architecture equivalence and real synthetic optimizer update."""
import ast
import json
from pathlib import Path
import unittest

import torch
from torch import nn
from cnn_assignment.models import build_model, load_saved_state, model_cost
from cnn_assignment.cli import build_parser
from cnn_assignment.training import make_optimizer


class CompactChecks(unittest.TestCase):
    def test_notebook_equivalence_cost_and_update(self):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/03_model_B_compact_cnn.ipynb').read_text())
        nodes = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                nodes += [n for n in tree.body if isinstance(n, ast.ClassDef)
                          and n.name in ('DSBlock', 'CompactEuroSAT')]
        self.assertEqual([n.name for n in nodes], ['DSBlock', 'CompactEuroSAT'])
        namespace = {'torch': torch, 'nn': nn}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<historical classes only>', 'exec'), namespace)
        historical = namespace['CompactEuroSAT']().eval()
        model = build_model('compact_eurosat').eval()
        load_saved_state(model, historical.state_dict())
        for batch in (1, 20, 64):
            x = torch.randn(batch, 3, 64, 64)
            with torch.no_grad():
                torch.testing.assert_close(model(x), historical(x), rtol=0, atol=0)
        cost = model_cost(model)
        self.assertEqual(cost['trainable_parameters'], 11466)
        self.assertEqual(cost['conv_linear_macs_per_image'], 3882784)
        self.assertEqual(cost['estimated_fp32_parameter_bytes'], 45864)
        model.train()
        before = next(model.parameters()).detach().clone()
        optimizer = make_optimizer(model, {'optimizer': 'adam',
            'optimizer_settings': {'lr': .001}, 'weight_decay': .0001})
        loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.parameters())))

    def test_cli_controlled_protocol(self):
        args = build_parser().parse_args(['train', '--model', 'compact_eurosat',
            '--optimizer', 'adam', '--lr', '.001', '--run-name', 'compact_common_seed42_v1'])
        self.assertEqual((args.epochs, args.batch_size, args.seed, args.weight_decay), (30, 64, 42, .0001))


if __name__ == '__main__':
    unittest.main()
