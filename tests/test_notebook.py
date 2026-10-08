"""Fresh-kernel notebook checks; no training or test-set evaluation.

Run with the notebook's environment: python -m unittest discover -s tests -p test_notebook.py -v
Requires nbclient, nbformat and ipykernel (provided by the Jupyter environment).
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
SETUP_CELL_ID = 'a351791e'
AUDIT_CELL_IDS = {'dc6fda09', 'cbee5fe6'}  # Full-dataset quality/hash audits.
ANALYSIS_CELL_IDS = {
    '8b928a05': 'experiment settings',
    '6b7a8944': 'saved experiment reader',
    '8c294ea0': 'Model A Adam history',
    '6f91b7ab': 'Model A Adam curves',
    '358c07f9': 'Model B Adam history',
    'b2c22e01': 'Model B Adam curves',
    '5b81687e': 'Adam comparison',
    '64d202d7': 'Model A SGD history',
    '9a2ce807': 'Model A SGD curves',
    '0dd7d495': 'saved checkpoint validation',
    'ce45d6dc': 'Model A momentum history',
    '9a1c0504': 'Model A momentum curves',
    'e65a94c4': 'Model B SGD history',
    '70cfc86b': 'Model B SGD curves',
    '4251e991': 'Model B momentum history',
    'a7c4e86b': 'Model B momentum curves',
    '7fd68929': 'optimizer comparison',
    '8fbf1bbb': 'saved history/checkpoint reconnect',
}
RERUN_CELL_IDS = ['5b81687e', '7fd68929', '0dd7d495', '8fbf1bbb']
GUARD = '''
import cnn_assignment.training as training_module
import cnn_assignment.evaluation as evaluation_module

def forbid_training(*args, **kwargs):
    raise AssertionError("Notebook verification must not launch training")

training_module.train = forbid_training
original_run_epoch = training_module.run_epoch
def validation_only_epoch(model, loader, criterion, device, optimizer=None, _original=original_run_epoch, **kwargs):
    assert optimizer is None, "Notebook verification must not update weights"
    return _original(model, loader, criterion, device, optimizer=optimizer, **kwargs)
training_module.run_epoch = validation_only_epoch
evaluation_module.run_epoch = validation_only_epoch
original_evaluate = evaluation_module.evaluate
def validation_only_evaluate(*args, _original=original_evaluate, **kwargs):
    assert kwargs.get("split", "validation") == "validation", "Test evaluation is forbidden"
    return _original(*args, **kwargs)
evaluation_module.evaluate = validation_only_evaluate
torch.set_num_threads(2)
'''


def execute(cells, cwd):
    # Execute copies, never replace historical outputs in the source notebook.
    notebook = nbformat.v4.new_notebook(cells=cells)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False)
    def show_progress(cell, cell_index, **kwargs):
        print(f"  Kernel cell {cell_index + 1}/{len(cells)}: {cell.source.splitlines()[0]}", flush=True)
    client.on_cell_start = show_progress
    client.execute(cwd=str(cwd), env=environment, cleanup_kc=True)
    return notebook


class NotebookChecks(unittest.TestCase):
    def required_code_cells(self, source):
        cells = {cell.id: cell for cell in source.cells}
        self.assertEqual(len(cells), len(source.cells), 'Duplicate notebook cell IDs')
        for cell_id in {SETUP_CELL_ID, *ANALYSIS_CELL_IDS, *AUDIT_CELL_IDS}:
            self.assertIn(cell_id, cells, f'Missing required cell: {ANALYSIS_CELL_IDS.get(cell_id, cell_id)}')
            self.assertEqual(cells[cell_id].cell_type, 'code', cell_id)
        return cells

    def test_fresh_kernel_root_notebook_analysis(self):
        source = nbformat.read(ROOT / 'test.ipynb', as_version=4)
        self.required_code_cells(source)
        # Root: sequential setup, previews, architecture checks and analysis.
        # Skip only the original full-dataset image quality/hash audits: those
        # do not exercise imports or saved-model analysis and can take minutes.
        # Guards prohibit actual training and test-set inference throughout.
        sequential = []
        for cell in source.cells:
            if cell.cell_type == 'code' and cell.id not in AUDIT_CELL_IDS:
                sequential.append(nbformat.v4.new_code_cell(cell.source))
                if cell.id == SETUP_CELL_ID:
                    sequential.append(nbformat.v4.new_code_cell(GUARD))
        root_result = execute(sequential, ROOT)
        self.check_result(root_result)
        self.assertEqual(source, nbformat.read(ROOT / 'test.ipynb', as_version=4))

    def test_fresh_kernel_notebooks_independent_analysis(self):
        source = nbformat.read(ROOT / 'test.ipynb', as_version=4)
        cells = self.required_code_cells(source)
        # Subfolder: reset the user namespace before every analysis cell, so
        # earlier analysis, exploration and training cells cannot supply inputs.
        independent = [nbformat.v4.new_code_cell(cells[SETUP_CELL_ID].source),
                       nbformat.v4.new_code_cell(GUARD)]
        for cell_id in ANALYSIS_CELL_IDS:
            independent.extend([
                nbformat.v4.new_code_cell('get_ipython().run_line_magic("reset", "-f")'),
                nbformat.v4.new_code_cell(cells[SETUP_CELL_ID].source),
                nbformat.v4.new_code_cell(cells[cell_id].source),
            ])
        # Exercise rerunning comparisons and validation within the same setup.
        for cell_id in RERUN_CELL_IDS:
            independent.append(nbformat.v4.new_code_cell(cells[cell_id].source))
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as temporary:
            notebooks_dir = Path(temporary) / 'notebooks'
            notebooks_dir.mkdir()
            subfolder_result = execute(independent, notebooks_dir)
        self.check_result(subfolder_result)
        # Saved outputs, metadata, ids and execution counts are unchanged.
        after = nbformat.read(ROOT / 'test.ipynb', as_version=4)
        self.assertEqual(source, after)
        summary = {
            'fresh_kernel_notebooks_independent_analysis': 'passed',
            'analysis_cells_by_id': ANALYSIS_CELL_IDS,
            'comparison_validation_and_reconnect_reruns': 'passed',
            'validation_split': 'validation',
            'checkpoint': 'model_a_sgd_lr0.01',
            'validation_matches_saved_history': True,
            'historical_notebook_preserved': True,
            'training_launched': False,
            'test_set_evaluated': False,
        }
        report = ROOT / 'outputs/reports/notebook_verification.json'
        report.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    def check_result(self, result):
        for cell in result.cells:
            self.assertFalse(any(output.output_type == 'error' for output in cell.outputs))
        text = '\n'.join(output.get('text', '') for cell in result.cells for output in cell.outputs)
        self.assertIn('Validation matches saved history: True', text)


if __name__ == '__main__':
    unittest.main()
