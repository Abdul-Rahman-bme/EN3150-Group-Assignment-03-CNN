"""Rebuild report evidence/figures from source and saved artifacts, without training.

Run from the repository root: python report/generate.py
Assertions deliberately fail if metrics, memberships or architectural counts disagree.
Historical hashes are compared with both literal and CRLF serializations, without
rewriting their files. Checkpoint sizes remain recorded evidence when files are absent.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cnn_assignment.data import check_splits
from cnn_assignment.models import build_model, model_cost

OUT = ROOT / 'report'
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
AUDIT = OUT / 'audit'
for directory in (FIG, TABLE, AUDIT):
    directory.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
BASELINE = json.loads((AUDIT / 'baseline_refs.json').read_text())
plt.rcParams.update({'font.size': 10, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()


def git_bytes(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + '\n')


def rel(path):
    """Return a repository-relative path with stable separators on every OS."""
    return path.relative_to(ROOT).as_posix()


def tex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def tabular(path, headers, rows, widths=None):
    spec = widths or ('l' + 'r' * (len(headers)-1))
    text = '\\begin{tabular}{' + spec + '}\n\\toprule\n'
    text += ' & '.join(tex(x) for x in headers) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(tex(x) for x in row) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    path.write_text(text)


provenance = []


def record(model, field, value, source, evidence='verified from saved artifact'):
    provenance.append({'model': model, 'report_value': field, 'value': value,
                       'source_artifact': source, 'verification': evidence})


classes = load('outputs/custom_cnn/model_a_adam_lr0.001/config.json')['class_names']
split_check = check_splits(ROOT / 'data/splits', classes)
frames = {name: pd.read_csv(ROOT / f'data/splits/{name}.csv') for name in ('train', 'validation', 'test')}
hash_rows = []
for name, frame in frames.items():
    path = ROOT / f'data/splits/{name}.csv'
    data = path.read_bytes()
    crlf = hashlib.sha256(data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    hash_rows.append({'split': name, 'literal_sha256': hashlib.sha256(data).hexdigest(), 'crlf_sha256': crlf})
    record('dataset', name + '_count', len(frame), rel(path))
pd.DataFrame(hash_rows).to_csv(AUDIT / 'split_hashes.csv', index=False)
counts = pd.DataFrame({n: f.groupby('class_name').size().reindex(classes) for n, f in frames.items()})
counts.to_csv(TABLE / 'class_distribution.csv')
tabular(TABLE / 'class_distribution.tex', ['Class', 'Train', 'Validation', 'Test'],
        [[c, *map(int, counts.loc[c])] for c in classes])
ax = counts.plot.bar(figsize=(10, 4), color=['#345995', '#e2a03f', '#58a684'])
ax.set_ylabel('Images'); ax.set_xlabel(''); plt.xticks(rotation=35, ha='right')
plt.tight_layout(); plt.savefig(FIG / 'class_distribution.pdf'); plt.close()

# Reconstruct the historical compact notebook's sorted-folder random split.
all_images = pd.concat(frames.values()).sort_values('relative_path').reset_index(drop=True)
t, remaining = train_test_split(all_images, test_size=.30, stratify=all_images.label, random_state=42)
v, e = train_test_split(remaining, test_size=.50, stratify=remaining.label, random_state=42)
reconstruction = {name: set(f.relative_path) == set(frames[name].relative_path)
                  for name, f in zip(frames, (t, v, e))}
assert all(reconstruction.values())
dump(AUDIT / 'split_audit.json', {**split_check, 'compact_algorithm_reconstructed_membership': reconstruction,
    'pixel_verification': 'Not rerun: raw images absent. Saved hashes are unique; no image-content claim.'})

# Each config is selected by minimum validation loss, never test accuracy.
optimizer_rows = []
for path in sorted((ROOT / 'outputs/custom_cnn').glob('*/config.json')):
    c = json.loads(path.read_text()); h = pd.read_csv(path.parent / 'history.csv')
    if c['model'] == 'model_d':
        continue  # Model D is a verified final run, not part of the 12-run A/B optimizer sweep.
    assert h.epoch.tolist() == list(range(1, 31))
    best = h.loc[h.val_loss.idxmin()]
    row = {'run': path.parent.name, 'model': c['model'], 'optimizer': c['optimizer'],
           'lr': c['optimizer_settings']['lr'], 'momentum': c['optimizer_settings'].get('momentum', 0),
           'selected_epoch': int(best.epoch), 'val_loss': best.val_loss, 'val_accuracy': best.val_accuracy,
           'mean_train_seconds': h.train_seconds.mean(), 'mean_train_seconds_excluding_first': h.train_seconds.iloc[1:].mean(),
           'total_train_minutes': h.train_seconds.sum()/60}
    optimizer_rows.append(row)
    for key, value in row.items():
        record(c['model'], path.parent.name + ':' + key, value,
               rel(path) + '; ' + rel(path.parent/'history.csv'))
opt = pd.DataFrame(optimizer_rows)
opt.to_csv(TABLE / 'optimizer_comparison.csv', index=False)
tabular(TABLE / 'optimizer_comparison.tex', ['Code', 'Optimizer', 'LR', 'Mom.', 'Epoch', 'Val. loss', 'Val. acc. (%)', 'Train s'],
        [[r.model, r.optimizer, f'{r.lr:g}', f'{r.momentum:g}', r.selected_epoch,
          f'{r.val_loss:.4f}', f'{100*r.val_accuracy:.2f}', f'{r.mean_train_seconds:.2f}'] for r in opt.itertuples()])
best_custom = opt.loc[opt.groupby('model').val_loss.idxmin()]
best_custom.to_csv(TABLE / 'validation_selections.csv', index=False)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for ax, name in zip(axes, ['model_a', 'model_b', 'model_c']):
    d = opt[opt.model == name]
    ax.bar(np.arange(len(d)), d.val_accuracy*100, color='#345995')
    ax.set_xticks(np.arange(len(d)), [f'{r.optimizer}\n{r.lr:g}' for r in d.itertuples()], rotation=30, ha='right', fontsize=8)
    ax.set_title(name); ax.set_ylim(70,100); ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Selected-checkpoint validation accuracy (%)')
fig.tight_layout(); fig.savefig(FIG / 'optimizer_comparison.pdf'); plt.close(fig)

names = {'model_a': 'Standard A', 'model_c': 'Selected lightweight B',
         'model_d': 'Model D', 'mobilenet_v2': 'MobileNetV2',
         'shufflenet_v2_x0_5': 'ShuffleNetV2 x0.5'}
master, confusions = [], {}
metric_paths = sorted((ROOT / 'outputs/reports').glob('evaluate-test_*/metrics.json'))
metric_paths.append(ROOT / 'outputs/reports/model_d_numpy126_final_test/metrics.json')
for metric_path in metric_paths:
    m = json.loads(metric_path.read_text())
    run = m['run']
    run_path = next((ROOT / 'outputs' / p / run for p in ['custom_cnn', 'pretrained_cnn']
                     if (ROOT / 'outputs' / p / run).is_dir()), None)
    assert run_path is not None
    c = json.loads((run_path / 'config.json').read_text())
    code = c['model']; assert code in names
    h = pd.read_csv(run_path/'history.csv'); selected = h.loc[h.val_loss.idxmin()]
    assert h.epoch.tolist() == list(range(1,31))
    assert (c['seed'],c['batch_size'],c['weight_decay']) == (42,64,.0001)
    assert c['class_names'] == classes
    assert m['class_names'] == classes and m['split'] == 'test'
    assert m['split_sha256'] == next(x['crlf_sha256'] for x in hash_rows if x['split']=='test')
    if 'split_sha256' in c:
        assert c['split_sha256'] == {x['split']: x['crlf_sha256'] for x in hash_rows}
    p = pd.read_csv(metric_path.parent/'predictions.csv')
    pd.testing.assert_series_equal(p.relative_path, frames['test'].relative_path)
    pd.testing.assert_series_equal(p.label, frames['test'].label)
    assert p.predicted_label.isin(range(10)).all()
    assert p.predicted_class.tolist() == [classes[i] for i in p.predicted_label]
    assert p.correct.tolist() == (p.label == p.predicted_label).tolist()
    cm = np.zeros((10,10),dtype=int); np.add.at(cm,(p.label,p.predicted_label),1)
    saved = pd.read_csv(metric_path.parent/'confusion_matrix.csv',index_col=0)
    assert saved.index.tolist() == saved.columns.tolist() == classes
    np.testing.assert_array_equal(cm,saved)
    precision = np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(10),where=cm.sum(0)!=0)
    recall = cm.diagonal()/cm.sum(1)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(10),where=precision+recall!=0)
    recomputed = dict(accuracy=cm.trace()/cm.sum(),macro_precision=precision.mean(),macro_recall=recall.mean(),macro_f1=f1.mean())
    for key,value in recomputed.items():
        assert abs(value-m[key]) < 1e-12
    per = pd.read_csv(metric_path.parent/'per_class_metrics.csv')
    np.testing.assert_allclose(per[['precision','recall','f1']], np.stack([precision,recall,f1],axis=1),rtol=0,atol=1e-12)
    normalized = cm/cm.sum(1)[:,None]
    np.testing.assert_allclose(normalized,pd.read_csv(metric_path.parent/'confusion_matrix_normalized.csv',index_col=0),rtol=0,atol=1e-12)
    cost = model_cost(build_model(code))
    for key in ['total_parameters','trainable_parameters','conv_linear_macs_per_image']:
        assert cost[key] == m[key]
    architecture = ('standard' if code == 'model_a' else
                    'residual depthwise' if code == 'model_d' else 'depthwise / mobile')
    row = {'code': code, 'model': names[code], 'architecture': architecture,
           'parameters': cost['trainable_parameters'], 'total_parameters': cost['total_parameters'],
           'fp32_parameter_KB': cost['trainable_parameters']*4/1000,
           'recorded_weight_bytes': m['actual_weights_file_bytes'], 'recorded_weight_MB': m['actual_weights_file_bytes']/1e6,
           'recorded_weight_KB': m['actual_weights_file_bytes']/1000,
           'weight_size_status': 'recorded in saved metrics; trained file unavailable for remeasurement',
           'macs': cost['conv_linear_macs_per_image'], 'selected_epoch': int(selected.epoch),
           'val_loss': selected.val_loss, 'val_accuracy': selected.val_accuracy,
           **recomputed, 'mean_train_seconds': h.train_seconds.mean(),
           'mean_train_seconds_excluding_first': h.train_seconds.iloc[1:].mean(),
           'test_pass_seconds': m['seconds'], 'report': rel(metric_path.parent),
           'run_path': rel(run_path)}
    master.append(row); confusions[code] = cm
    for key,value in row.items():
        evidence = 'independently recomputed from predictions' if key in recomputed else 'saved artifact / source calculation'
        if key.startswith('recorded_weight'): evidence='recorded, not remeasured: checkpoint absent'
        record(code,key,value,rel(metric_path)+'; '+rel(run_path)+'/history.csv',evidence)
    fig,axes = plt.subplots(1,2,figsize=(9,3.3))
    for ax, key in zip(axes,['loss','accuracy']):
        ax.plot(h.epoch,h['train_'+key],label='Training'); ax.plot(h.epoch,h['val_'+key],label='Validation')
        ax.axvline(selected.epoch,color='grey',ls=':',label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel=key.title()); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(names[code]); fig.tight_layout(); fig.savefig(FIG / f'{code}_curves.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,6.4)); im=ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(10),classes,rotation=55,ha='right',fontsize=8); ax.set_yticks(range(10),classes,fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xlabel='Predicted class',ylabel='True class',title=names[code]+' — saved test predictions')
    fig.colorbar(im,ax=ax,shrink=.75); fig.tight_layout(); fig.savefig(FIG/f'{code}_confusion.pdf'); plt.close(fig)
    per.to_csv(TABLE/f'{code}_per_class.csv',index=False)
    tabular(TABLE/f'{code}_per_class.tex',['Class','Precision','Recall','F1','Support'],
            [[r.class_name,f'{r.precision:.4f}',f'{r.recall:.4f}',f'{r.f1:.4f}',r.support] for r in per.itertuples()])

master.sort(key=lambda r:list(names).index(r['code']))
df=pd.DataFrame(master); df.to_csv(TABLE/'master_comparison.csv',index=False)
tabular(TABLE/'master_comparison.tex',['Model','Params','Weights MB*','MACs M','Val. %','Test %','Prec. %','Recall %','F1 %','Train s'],
        [[r.model,f'{r.parameters:,}',f'{r.recorded_weight_MB:.6f}',f'{r.macs/1e6:.3f}',
          *[f'{100*x:.2f}' for x in [r.val_accuracy,r.accuracy,r.macro_precision,r.macro_recall,r.macro_f1]],
          f'{r.mean_train_seconds:.2f}'] for r in df.itertuples()])
tabular(TABLE/'custom_comparison.tex',['Model','Params','FP32 KB','Weights KB*','Train s','Test %'],
        [[r.model,f'{r.parameters:,}',f'{r.fp32_parameter_KB:.3f}',f'{r.recorded_weight_KB:.3f}',f'{r.mean_train_seconds:.2f}',f'{100*r.accuracy:.2f}']
         for r in df.iloc[:2].itertuples()])
tabular(TABLE/'training_timing.tex',['Model','Mean s','Mean excl. first s','Test pass s'],
        [[r.model,f'{r.mean_train_seconds:.3f}',f'{r.mean_train_seconds_excluding_first:.3f}',f'{r.test_pass_seconds:.3f}'] for r in df.itertuples()])
plot_labels = ['A', 'Selected B', 'Model D', 'MobileNetV2', 'ShuffleNet']
plot_colors = ['#345995', '#58a684', '#d46b5c', '#e2a03f', '#9564a4']
positions = np.arange(len(df))
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,field,label,scale in zip(axes,['accuracy','recorded_weight_MB','macs'],['Test accuracy (%)','Recorded state dictionary (MB)','Conv/linear MACs (millions)'],[100,1,1e-6]):
    values=df[field]*scale; ax.bar(positions,values,color=plot_colors)
    ax.set_xticks(positions,plot_labels,rotation=25,ha='right'); ax.set_ylabel(label)
    if field=='recorded_weight_MB':ax.set_yscale('log')
    if field=='accuracy':ax.set_ylim(90,100)
    for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom',fontsize=8)
fig.tight_layout();fig.savefig(FIG/'final_comparison.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,3.8))
axes[0].bar(positions,df.parameters,color=plot_colors);axes[0].set_yscale('log')
axes[0].set_ylabel('Trainable parameters (log scale)')
axes[1].bar(positions-.18,df.fp32_parameter_KB/1000,.36,label='FP32 parameters',color='#58a684')
axes[1].bar(positions+.18,df.recorded_weight_MB,.36,label='Recorded state dictionary',color='#e2a03f')
axes[1].set_yscale('log');axes[1].set_ylabel('Storage MB (log scale)');axes[1].legend(fontsize=8)
for ax in axes:
    ax.set_xticks(positions,plot_labels,rotation=25,ha='right')
fig.tight_layout();fig.savefig(FIG/'parameter_storage.pdf');plt.close(fig)

# Leaf-level formulas and dimensions independently checked against tensor counts.
def layer_table(name,model):
    rows=[]; handles=[]
    def hook(key,module,inputs,output):
        p=sum(x.numel() for x in module.parameters(recurse=False)); formula='0'; macs=0
        if isinstance(module,nn.Conv2d):
            k=module.kernel_size; bias=module.out_channels if module.bias is not None else 0
            manual=module.out_channels*(module.in_channels//module.groups)*k[0]*k[1]+bias
            formula=f'{module.out_channels} x {module.in_channels//module.groups} x {k[0]} x {k[1]} + {bias}'
            macs=output.numel()*(module.in_channels//module.groups)*k[0]*k[1]
        elif isinstance(module,nn.BatchNorm2d):manual=2*module.num_features;formula=f'2 x {module.num_features}'
        elif isinstance(module,nn.Linear):
            bias=module.out_features if module.bias is not None else 0
            manual=module.in_features*module.out_features+bias;formula=f'{module.in_features} x {module.out_features} + {bias}'
            macs=module.in_features*module.out_features
        else:manual=0
        assert manual==p
        description=type(module).__name__
        if isinstance(module,nn.Conv2d):description+=f' k{module.kernel_size[0]} s{module.stride[0]} p{module.padding[0]} g{module.groups}'
        rows.append({'layer':key,'operation':description,'output':' x '.join(map(str,output.shape[1:])),
                     'formula':formula,'parameters':p,'macs':macs})
    for key,m in model.named_modules():
        if not list(m.children()):handles.append(m.register_forward_hook(lambda m,i,o,k=key:hook(k,m,i,o)))
    model.eval()
    with torch.no_grad():model(torch.zeros(1,3,64,64))
    for h in handles:h.remove()
    assert sum(r['parameters'] for r in rows)==sum(p.numel() for p in model.parameters())
    assert sum(r['macs'] for r in rows)==model_cost(model)['conv_linear_macs_per_image']
    pd.DataFrame(rows).to_csv(TABLE/f'{name}_layers.csv',index=False)
    tabular(TABLE/f'{name}_layers.tex',['Operation','Output C x H x W','Formula','Params'],
        [[r['operation'],r['output'],r['formula'],f"{r['parameters']:,}"] for r in rows])
    return model_cost(model)

for name in ['model_a','model_c','compact_eurosat','model_d']:layer_table(name,build_model(name))

d_cost=model_cost(build_model('model_d'))
assert d_cost['trainable_parameters']==15442 and d_cost['conv_linear_macs_per_image']==4728768
compact_cost=model_cost(build_model('compact_eurosat'))
dump(AUDIT/'candidate_costs.json',{'compact_eurosat':compact_cost,'model_d':d_cost,'nipun_ref':BASELINE['nipun'],
    'model_d_results_status':'Verified from committed config, history, metrics, predictions, per-class results and confusion matrices; checkpoint file is not committed',
    'compact_results_status':'Historical notebook output; controlled common-protocol run blocked by absent images'})

# Branch audit uses the original remote refs, prior to local synchronization.
branches=[]
for ref in ['main','heshan','integration/heshan-main-20261009','data/eurosat-pipeline','setup/project-structure','model/standard-cnn','simple_model','nipun']:
    full=BASELINE[ref]; comparison=BASELINE['main']+'...'+full
    behind,ahead=map(int,git('rev-list','--left-right','--count',comparison).split())
    unique=git('diff','--name-only',comparison)
    branches.append({'branch':ref,'sha':full,'ahead':ahead,'behind':behind,
        'merged':ahead==0,'unique_files':unique.replace('\n','; '),
        'recommendation': 'preserve; already integrated' if ahead==0 else
            ('synchronized locally; preserve historical notebook; candidate pending controlled run' if ref=='simple_model' else
             'historical branch later integrated; final experiment artifacts are now on main')})
pd.DataFrame(branches).to_csv(AUDIT/'branch_audit.csv',index=False)
pd.DataFrame(provenance).to_csv(AUDIT/'report_data_sources.csv',index=False)
source_paths = set()
for pattern in ['outputs/custom_cnn/*/config.json', 'outputs/custom_cnn/*/history.csv',
                'outputs/pretrained_cnn/*/config.json', 'outputs/pretrained_cnn/*/history.csv',
                'outputs/reports/evaluate-test_*/*.csv', 'outputs/reports/evaluate-test_*/metrics.json',
                'outputs/reports/model_d_numpy126_final_test/*.csv',
                'outputs/reports/model_d_numpy126_final_test/metrics.json',
                'data/splits/*.csv', 'src/cnn_assignment/*.py', 'notebooks/03_model_B_compact_cnn.ipynb']:
    source_paths.update(ROOT.glob(pattern))
dump(AUDIT/'source_file_hashes.json', {rel(p):hashlib.sha256(git_bytes('show', 'HEAD:' + rel(p))).hexdigest()
                                     for p in sorted(source_paths)})
by_code = {row['code']: row for row in master}
dump(AUDIT/'derived_values.json', {
    'selected_B_vs_A':{
        'parameter_reduction_percent':100*(1-master[1]['parameters']/master[0]['parameters']),
        'MAC_reduction_percent':100*(1-master[1]['macs']/master[0]['macs']),
        'accuracy_difference_percentage_points':100*(master[1]['accuracy']-master[0]['accuracy']),
        'additional_correct_test_images':round(4050*(master[1]['accuracy']-master[0]['accuracy']))},
    'MobileNetV2_parameter_ratio_to_B':by_code['mobilenet_v2']['parameters']/by_code['model_c']['parameters'],
    'ShuffleNetV2_parameter_ratio_to_B':by_code['shufflenet_v2_x0_5']['parameters']/by_code['model_c']['parameters'],
    'custom_normalization_source':'outputs/custom_cnn/model_a_adam_lr0.001/config.json',
    'layer_formula_sources':'tables/*_layers.csv',
    'numeric_constant_sources':{'seed_batch_epochs_lr_weight_decay':'saved run configs; optimizer_comparison.csv',
                              'split_proportions_and_limits':'EN3150_Assignment_03_In23.pdf',
                              'group_number':'user supplied 54',
                              'team_names':'README.md and git show c6c64fe:README.md'},
    'historical_compact_notebook':'notebooks/03_model_B_compact_cnn.ipynb cells 3, 14, 16, 24, 26, 28-30',
    'model_D_evidence':'outputs/custom_cnn/model_d_residual_numpy126 plus outputs/reports/model_d_numpy126_final_test (verified and included in master table)'})
dump(AUDIT/'generation.json',{'base_main':git('rev-parse','origin/main'),
    'assignment_pdf':'EN3150_Assignment_03_In23.pdf','assignment_sha256':hashlib.sha256((ROOT/'EN3150_Assignment_03_In23.pdf').read_bytes()).hexdigest(),
    'python':sys.version,'torch':torch.__version__,
    'scope':'Saved-artifact reanalysis and synthetic architecture checks; no dataset training or test inference performed.',
    'required_models':master})
print(df[['model','parameters','macs','val_accuracy','accuracy','macro_f1','mean_train_seconds']].to_string(index=False))
print('Report evidence generated; all consistency assertions passed.')
