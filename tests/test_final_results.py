"""Audit saved final results and notebook analysis without model execution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

import nbformat
import numpy as np
import pandas as pd

from test_notebook import execute

ROOT = Path(__file__).resolve().parents[1]
AUDIT = json.loads((ROOT / 'docs/final_results_audit.json').read_text(encoding='utf-8'))
ANALYSIS_IDS = ['3901d8bb', 'fdae80a8', '6e992b3c', 'd73db9fc', '4dcd2a0b',
                '5ae337df', 'eaa3ea2e', 'e9c04540', '2fc30bec', 'd317fe2d',
                '45348b2b', 'cbff3f02', '6ecdeb9c', 'e4e5e35d', 'ab27f52e']
GUARD = '''
import cnn_assignment.models as models_module
import cnn_assignment.training as training_module
import cnn_assignment.evaluation as evaluation_module
import cnn_assignment.data as data_module
def forbidden(*args, **kwargs):
    raise AssertionError("Final analysis must read saved files, never execute models")
models_module.build_model = forbidden
training_module.build_model = forbidden
evaluation_module.build_model = forbidden
training_module.train = forbidden
training_module.run_epoch = forbidden
evaluation_module.evaluate = forbidden
evaluation_module.run_epoch = forbidden
data_module.EuroSATSplit = forbidden
torch.load = forbidden
torch.nn.Module.__call__ = forbidden
torch.hub.load_state_dict_from_url = forbidden
'''
ASSERT_ANALYSIS = '''
import json
audit = json.loads((PROJECT_ROOT / "docs/final_results_audit.json").read_text())
expected_custom = {"model_a": "Model A", "model_c": "Model C"}
display_names = ["Standard A", "Final lightweight B", "MobileNetV2", "ShuffleNetV2 ×0.5"]
for record, display_name in zip(audit["records"], display_names):
    metrics = record["metrics"]
    row = final_test_comparison.set_index("model").loc[display_name]
    assert abs(row["test_accuracy_percent"] - 100*metrics["accuracy"]) < 1e-10
    assert abs(row["weights_size_MB"] - metrics["actual_weights_file_bytes"]/1e6) < 1e-10
    assert abs(row["MACs_millions"] - metrics["conv_linear_macs_per_image"]/1e6) < 1e-10
    errors = largest_confusions[largest_confusions["model"] == display_name]
    assert errors["error_count"].tolist() == [e["count"] for e in record["largest_confusions"]]
    assert errors["true_class"].tolist() == [e["true_class"] for e in record["largest_confusions"]]
    assert errors["predicted_class"].tolist() == [e["predicted_class"] for e in record["largest_confusions"]]
    np.testing.assert_allclose(errors["percent_of_true_class"], [e["percent"] for e in record["largest_confusions"]], rtol=0, atol=1e-10)
    expected_f1 = [100*float(p["f1"]) for p in record["per_class"]]
    np.testing.assert_allclose(class_f1_percent[display_name], expected_f1, rtol=0, atol=1e-10)
    if record["model"] in expected_custom:
        name = expected_custom[record["model"]]
        selected = selected_custom_models.set_index("model").loc[name]
        assert selected["selected_epoch"] == record["selected_epoch"]
        cost = efficiency_comparison.set_index("model").loc[name]
        assert cost["trainable_parameters"] == metrics["trainable_parameters"]
        assert cost["conv_linear_MACs"] == metrics["conv_linear_macs_per_image"]
        assert abs(cost["FP32_parameter_storage_KiB"] - metrics["trainable_parameters"]*4/1024) < 1e-10
        observed = training_time_comparison.set_index("model").loc[name]
    else:
        observed = pretrained_training_times.set_index("model").loc[display_name]
        assert observed["selected_epoch"] == record["selected_epoch"]
    for actual, expected in [("mean_train_seconds", "mean_seconds"), ("mean_train_seconds_excluding_first", "mean_excluding_first_seconds"), ("median_train_seconds", "median_seconds"), ("total_train_minutes", "total_minutes")]:
        assert abs(observed[actual] - record["training_time"][expected]) < 1e-8
print("Saved final analysis agrees with verified reports and histories.")
'''


class FinalResultChecks(unittest.TestCase):
    def test_predictions_confusions_and_metrics_agree(self):
        split = pd.read_csv(ROOT / 'data/splits/test.csv')
        for record in AUDIT['records']:
            with self.subTest(model=record['model']):
                folder = ROOT / record['report']
                m = json.loads((folder / 'metrics.json').read_text())
                self.assertEqual(m, record['metrics'])
                predictions = pd.read_csv(folder / 'predictions.csv')
                pd.testing.assert_series_equal(predictions.relative_path, split.relative_path)
                pd.testing.assert_series_equal(predictions.label, split.label)
                classes = m['class_names']
                matrix = pd.crosstab(predictions.label, predictions.predicted_label).reindex(index=range(10), columns=range(10), fill_value=0).to_numpy()
                saved = pd.read_csv(folder / 'confusion_matrix.csv', index_col=0)
                self.assertEqual(saved.index.tolist(), classes)
                self.assertEqual(saved.columns.tolist(), classes)
                np.testing.assert_array_equal(matrix, saved.to_numpy())
                support = matrix.sum(axis=1)
                precision = np.divide(matrix.diagonal(), matrix.sum(axis=0), out=np.zeros(10), where=matrix.sum(axis=0)!=0)
                recall = matrix.diagonal() / support
                f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(10), where=precision+recall!=0)
                per_class = pd.read_csv(folder / 'per_class_metrics.csv')
                self.assertEqual(per_class.class_name.tolist(), classes)
                np.testing.assert_array_equal(per_class.support, support)
                np.testing.assert_allclose(per_class[['precision','recall','f1']], np.stack([precision,recall,f1],axis=1), rtol=0, atol=1e-12)
                normalized = pd.read_csv(folder / 'confusion_matrix_normalized.csv', index_col=0)
                np.testing.assert_allclose(normalized, matrix / support[:,None], rtol=0, atol=1e-12)
                self.assertAlmostEqual(m['accuracy'], matrix.trace()/matrix.sum(), places=12)
                for key, values in [('precision',precision),('recall',recall),('f1',f1)]:
                    self.assertAlmostEqual(m['macro_'+key], values.mean(), places=12)
                self.assertEqual(m['split'], 'test')
                self.assertEqual(m['images'], 4050)
                self.assertEqual(m['mac_input_shape'], [1,3,64,64])
                self.assertEqual(m['split_sha256'], hashlib.sha256((ROOT/'data/splits/test.csv').read_bytes()).hexdigest())

    def test_selected_epochs_timing_and_model_file_sizes(self):
        for record in AUDIT['records']:
            folder = ROOT / record['run_path']
            config = json.loads((folder/'config.json').read_text())
            history = pd.read_csv(folder/'history.csv')
            self.assertEqual(history.epoch.tolist(), list(range(1,31)))
            self.assertEqual(config['epochs'], 30)
            selected = history.loc[history.val_loss.idxmin()]
            self.assertEqual(selected.epoch, record['selected_epoch'])
            self.assertAlmostEqual(selected.val_accuracy, record['selected_val_accuracy'], places=14)
            times = history.train_seconds
            for value,key in [(times.mean(),'mean_seconds'),(times.iloc[1:].mean(),'mean_excluding_first_seconds'),(times.median(),'median_seconds'),(times.sum()/60,'total_minutes')]:
                self.assertAlmostEqual(value,record['training_time'][key],places=8)
            m=record['metrics']
            self.assertEqual((folder/'best_weights.pt').stat().st_size,m['actual_weights_file_bytes'])
            self.assertAlmostEqual(m['actual_weights_file_MB'],m['actual_weights_file_bytes']/1e6)
            self.assertAlmostEqual(m['actual_weights_file_MiB'],m['actual_weights_file_bytes']/1048576)
            if not record['model'].startswith('model_'):
                summary=json.loads((folder/'model_summary.json').read_text())
                for key in ['total_parameters','trainable_parameters','conv_linear_macs_per_image','actual_weights_file_bytes']:
                    self.assertEqual(summary[key],m[key])

    def test_source_hashes_and_all_historical_notebook_outputs(self):
        for path,digest in AUDIT['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest,path)
        notebook=json.loads((ROOT/'test.ipynb').read_text(encoding='utf-8'))
        baseline=json.loads((ROOT/'tests/fixtures/final_notebook_preservation.json').read_text())
        self.assertEqual(notebook['metadata'],baseline['metadata'])
        cells={c['id']:c for c in notebook['cells']}
        for cell_id,digest in baseline['cell_records_sha256_by_id'].items():
            fields={k:v for k,v in cells[cell_id].items() if k!='source'}
            self.assertEqual(hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest(),digest,cell_id)
        self.assertEqual(len(cells),len(notebook['cells']))

    def test_document_values_links_and_sharing_rules(self):
        document=(ROOT/'docs/final_results.md').read_text(encoding='utf-8')
        for r in AUDIT['records']:
            self.assertIn(f"{r['metrics']['accuracy']*100:.6f}",document)
            self.assertIn(f"{r['metrics']['actual_weights_file_bytes']:,}",document)
            self.assertIn(f"{r['training_time']['mean_seconds']:.9f}",document)
        for target in re.findall(r'\]\(([^)]+)\)',document):
            if not target.startswith('https:'):
                self.assertTrue((ROOT/'docs'/target).is_file(),target)
        shared=['docs/final_results.md','docs/final_results_audit.json','outputs/README.md']
        local=['outputs/archive/manifest.json','data/raw/example.jpg','notes.md','outputs/reports/notebook_20261007T025636510579Z/metrics.json']
        for r in AUDIT['records']:
            shared.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/r['report']).iterdir())
            shared += [r['run_path']+'/config.json',r['run_path']+'/history.csv']
            if not r['model'].startswith('model_'):
                shared.append(r['run_path']+'/model_summary.json')
            local += [r['run_path']+'/best_weights.pt',r['run_path']+'/last_checkpoint.pt']
        for paths,expected in [(shared,False),(local,True)]:
            result=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],input=('\0'.join(paths)+'\0').encode(),capture_output=True,cwd=ROOT)
            self.assertIn(result.returncode,[0,1],result.stderr)
            actual=set(result.stdout.decode().rstrip('\0').split('\0')) if result.stdout else set()
            self.assertEqual(actual,set(paths) if expected else set(),result.stderr)

    def test_fresh_kernel_saved_analysis_without_any_model_execution(self):
        source=nbformat.read(ROOT/'test.ipynb',as_version=4)
        cells={c.id:c for c in source.cells}
        selected=[nbformat.v4.new_code_cell(cells['a351791e'].source),nbformat.v4.new_code_cell(GUARD)]
        selected += [nbformat.v4.new_code_cell(cells[cell_id].source) for cell_id in ANALYSIS_IDS]
        selected.append(nbformat.v4.new_code_cell(ASSERT_ANALYSIS))
        result=execute(selected,ROOT)
        text='\n'.join(o.get('text','') for c in result.cells for o in c.outputs)
        self.assertIn('Saved final analysis agrees',text)
        self.assertEqual(source,nbformat.read(ROOT/'test.ipynb',as_version=4))

    def test_pretrained_timing_directly_after_setup(self):
        source=nbformat.read(ROOT/'test.ipynb',as_version=4)
        cells={c.id:c for c in source.cells}
        (ROOT/'tmp').mkdir(exist_ok=True)
        result=execute([nbformat.v4.new_code_cell(cells['a351791e'].source),
                        nbformat.v4.new_code_cell(GUARD),
                        nbformat.v4.new_code_cell(cells['ab27f52e'].source),
                        nbformat.v4.new_code_cell('assert pretrained_training_times.selected_epoch.tolist() == [27, 29]')],ROOT/'tmp')
        self.assertFalse(any(o.output_type=='error' for c in result.cells for o in c.outputs))
        self.assertEqual(source,nbformat.read(ROOT/'test.ipynb',as_version=4))


if __name__=='__main__':
    unittest.main()
