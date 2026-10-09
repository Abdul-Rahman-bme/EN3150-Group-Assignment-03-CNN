# EN3150: reproducible EuroSAT CNN experiments

I use this project to prepare saved data, train custom CNNs, plot histories and
evaluate selected checkpoints independently from the terminal. Each command
loads its own files. No notebook cells need to run first.

The original EuroSAT RGB dataset, saved splits and six completed experiments
remain in place. The refactor uses their saved configurations and notebook code
as the reference. New reports go into fresh folders under `outputs/reports/`.

## Windows setup

Use the existing `ml_env_fixed` environment. Install this local project once in
editable mode so new terminals can import it without setting `PYTHONPATH`.
No dependency upgrade or reinstall is needed. In PowerShell:

```powershell
conda activate ml_env_fixed
$Project = 'D:\2_ML PROJECTS\38. CNN_PR\EN3150-Assignment03-CNN'
$PreviousDistutilsSetting = $env:SETUPTOOLS_USE_DISTUTILS
try {
    $env:SETUPTOOLS_USE_DISTUTILS = 'stdlib'
    python -m pip install --no-index --no-deps --no-build-isolation -e "$Project"
} finally {
    $env:SETUPTOOLS_USE_DISTUTILS = $PreviousDistutilsSetting
}
python -m cnn_assignment --help
```

The setup command installs only this project's editable package and entry point.
`--no-index` disables package-index access, `--no-deps` skips dependencies, and
`--no-build-isolation` uses the environment's existing build tools. The temporary
distutils setting works around the inspected Python 3.11 environment's existing
setuptools import assertion and is restored afterward. Source edits are available
immediately; keep the project at this location or reinstall after moving it.

In a fresh PowerShell terminal, activate the environment and run:

```powershell
conda activate ml_env_fixed
python -m cnn_assignment --help
```

The editable installation was verified in `ml_env_fixed` from a fresh shell
outside the project directory with `PYTHONPATH` and `SETUPTOOLS_USE_DISTUTILS`
removed: `python -m cnn_assignment --help` exited successfully, and the package
import resolved to this project's `src/cnn_assignment/`.

Commands work from any current directory. The package finds the project from its
own file location. Relative command paths resolve against the project, including
output paths. To select another project location, put
`--project-root "D:\path\to\project"` **before** the command name.
The `cnn-assignment` entry point is also available in the activated environment.

For `test.ipynb`, select `ml_env_fixed` as the kernel
and run its first code cell. It finds the project from the kernel's working
directory (the project root or `notebooks/`, including deeper subfolders) and
adds `src/` to that kernel's import path. The notebook does not need a terminal
`PYTHONPATH` setting or an editable installation.

After restarting the notebook kernel, run the first code cell, then any saved
history, learning-curve, optimizer-comparison or saved-checkpoint validation
cell in section 7. Each reads its saved inputs directly; data exploration,
loader construction and training setup are not prerequisites.
Run all cells in order for the complete analysis. Notebook execution reuses
the completed runs and never starts training. Every plot or validation invocation
saves to a fresh folder under `outputs/reports/`, including when rerunning a cell.
The notebook's existing outputs remain the historical experiment record.

The inspected environment contains torch 2.5.1, torchvision 0.20.1, NumPy 1.26.4,
pandas 2.3.3, matplotlib 3.10.9, Pillow 12.3.0 and tqdm 4.66.1. `requirements.txt`
is a dependency inventory; use the existing environment. New run configs record
Python and package versions for later reproduction.

## Prepare and check saved data

```powershell
python -m cnn_assignment prepare
python -m cnn_assignment prepare --check-images --check-archive --output-dir outputs/reports/data_audit_v1
```

`prepare` verifies CSV structure, class order, saved pixel hashes, disjointness,
70/15/15 stratification and the presence of every saved image path. It reuses the
saved splits and exact normalization from the selected config. `--check-images`
also opens all images, checks 64 × 64 RGB format and compares their pixel hashes.
This optional dataset audit includes test images but does no model evaluation.
`--check-archive` checks the ZIP against the notebook's original MD5.

Preparation does not download or extract the dataset, regenerate splits or
recompute normalization. Missing files cause an error. Restore the existing
dataset and split files before continuing. Use `--data-root`, `--split-dir`,
`--archive` or `--config` to point to explicitly supplied locations.

The dataset is 27,000 images in ten classes, with 18,900 training, 4,050 validation
and 4,050 test rows. Paths in each CSV are relative to `data/raw/eurosat/2750/`.
The saved label order is AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River and SeaLake.

## Train a fresh experiment

These commands start training when you choose to run them:

```powershell
python -m cnn_assignment train --model model_a --optimizer adam --lr 0.001 --run-name model_a_adam_lr0.001_repeat1
python -m cnn_assignment train --model model_b --optimizer sgd --lr 0.01 --run-name model_b_sgd_lr0.01_repeat1
python -m cnn_assignment train --model model_b --optimizer sgd_momentum --lr 0.01 --run-name model_b_momentum_repeat1
```

Defaults are 30 epochs, batch size 64, seed 42, weight decay 1e-4 and zero workers.
Adam defaults to LR 0.001; SGD and momentum SGD default to LR 0.01. Momentum is
0.9 for `sgd_momentum`. `--epochs`, `--batch-size`, `--seed`, `--weight-decay`,
`--output-root`, `--num-workers` and `--device` are explicit options for new runs.
Device `auto` selects CUDA if available, otherwise CPU. `--device cpu` forces CPU.

Training uses raw logits, CrossEntropyLoss and FP32. It resets initialization,
Python/NumPy/torch random seeds and a separate training shuffle generator before
each new run. Horizontal and vertical flips and random quarter turns are the
only augmentation. Validation has tensor conversion and saved normalization.
Inputs stay 64 × 64; there is no colour jitter or resizing. Epoch metrics weight
each batch by its sample count. Batch progress bars and epoch summaries are
enabled by default; `--no-progress` hides batch bars. CUDA timing is synchronized.
Best weights are selected strictly by lowest validation loss. Training never
constructs a test loader or evaluates test images.

Default run names retain the original convention, for example
`model_a_sgd_momentum_lr0.01`. Any existing run folder is protected. To deliberately
skip a completed run, use:

```powershell
python -m cnn_assignment train --model model_a --optimizer adam --lr 0.001 --skip-existing
```

This skips the completed original Adam run without training or writing to it.
An incomplete existing folder still raises an error. Use a fresh run name for a
repeat. There is no overwrite flag.

## Model C experiments

Model C (`model_c`, `WideLightweightCNN`) uses the same saved splits, class order,
normalization, augmentation, training loop and checkpoint selection as Models A
and B. Its four planned runs are below. These commands are instructions for later
execution; adding the model does not start them or create run artifacts.

```powershell
python -m cnn_assignment train --model model_c --optimizer adam --lr 0.001 --epochs 30 --batch-size 64 --seed 42 --weight-decay 0.0001
python -m cnn_assignment train --model model_c --optimizer sgd --lr 0.01 --epochs 30 --batch-size 64 --seed 42 --weight-decay 0.0001
python -m cnn_assignment train --model model_c --optimizer sgd_momentum --lr 0.01 --epochs 30 --batch-size 64 --seed 42 --weight-decay 0.0001
python -m cnn_assignment train --model model_c --optimizer sgd --lr 0.003 --epochs 30 --batch-size 64 --seed 42 --weight-decay 0.0001
```

Momentum is 0.9 for the third run. Default run folders are
`model_c_adam_lr0.001`, `model_c_sgd_lr0.01`,
`model_c_sgd_momentum_lr0.01` and `model_c_sgd_lr0.003` under
`outputs/custom_cnn/`. Once runs exist, the existing commands support them:

```powershell
python -m cnn_assignment train --resume outputs/custom_cnn/model_c_adam_lr0.001
python -m cnn_assignment validate --run outputs/custom_cnn/model_c_adam_lr0.001 --compare-history
python -m cnn_assignment plot --run outputs/custom_cnn/model_c_adam_lr0.001
python -m cnn_assignment plot
```

The resume command applies to an incomplete run with a full last checkpoint.
Comparison reports automatically include Model C histories alongside existing
models and distinguish the two plain SGD learning rates.

## Resume a new incomplete run

```powershell
python -m cnn_assignment train --resume outputs/custom_cnn/model_a_adam_lr0.001_repeat1
```

New runs save `config.json`, per-epoch `history.csv`, `best_weights.pt` and
`last_checkpoint.pt`. The full last checkpoint contains the current model and
optimizer state, completed epoch, best result and best weights, canonical history,
Python/NumPy/torch/CUDA random states and the training shuffle generator state.
Writes use temporary files and replacement. The last checkpoint is saved first;
resume restores history and best weights from it if an artifact write was
interrupted. Resume continues the original epoch budget and uses the saved
configuration; model/optimizer/LR/epoch overrides are rejected.

Resume requires the same Python/package versions, device type, worker count and
unchanged split fingerprints. If the original run used workers or CPU explicitly,
supply matching options, for example `--num-workers 2 --device cpu`. Dataset and
split directory overrides allow relocation, while split contents must match.
For reproducible continuation, also keep the images and hardware unchanged.
The saved random states support epoch-boundary resume; interrupted work within
an epoch is repeated. cuDNN uses the notebook's deterministic settings, but
different hardware or PyTorch kernels can still change floating-point results.

**The six original `best_weights.pt` files support evaluation only.** They do not
contain optimizer, random, shuffle or last-epoch states, so exact resume from
those files is impossible. Completed runs cannot be resumed to extend training.
Full new checkpoints contain Python/NumPy objects and are loaded as trusted local
artifacts; evaluation loads only tensor weights with `weights_only=True`.

## Plot existing results

```powershell
python -m cnn_assignment plot --run outputs/custom_cnn/model_a_adam_lr0.001
python -m cnn_assignment plot
python -m cnn_assignment plot --optimizer adam --output-dir outputs/reports/adam_comparison_v1
```

A single-run plot shows training and validation loss/accuracy and marks the
minimum-validation-loss epoch. Comparisons group curves by model and optimizer,
and save a CSV with selected validation metrics and mean training times, including
the mean excluding the first epoch. Plotting imports no torch/model/dataset
modules and works with only the saved configs and histories, even without images
or weights. Use `--runs-root` for another experiment collection.

Every CLI plot and evaluation writes to a new timestamped folder by default.
An explicit `--output-dir` must not already exist. Choose another suffix for
subsequent reports; completed experiment plots and results are never overwritten.

## Validate a saved checkpoint

```powershell
python -m cnn_assignment validate --run outputs/custom_cnn/model_a_sgd_lr0.01 --compare-history
python -m cnn_assignment validate --run outputs/custom_cnn/model_b_adam_lr0.001 --device cpu
```

Evaluation loads architecture, class order, normalization and batch size from
that run's `config.json`, then strictly loads its `best_weights.pt`. It does not
update model weights or BatchNorm statistics. `--compare-history` compares loss
and accuracy with the minimum-loss row, reports the difference and exits with an
error on mismatch. Loss tolerance is `rtol=1e-5, atol=1e-6`; accuracy tolerance is
`atol=1e-8`. CPU/CUDA accumulation can differ slightly.

## Explicit final test evaluation

Run this only after selecting a checkpoint using validation results:

```powershell
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_a_adam_lr0.001 --output-dir outputs/reports/final_test_model_a_v1
```

This command alone selects the test split. It saves `metrics.json` with accuracy,
macro precision, macro recall and macro F1; `per_class_metrics.csv` with precision,
recall, F1 and support; raw and row-normalized confusion matrices as CSV and PNG;
and `predictions.csv` with each relative image path, true and predicted labels,
class names, confidence and correctness. Undefined precision/recall/F1 is zero.
Reports include checkpoint/split hashes and measured inference-pass timing.
No test evaluation was performed during the refactor.

## Model cost and timing

| Model | Trainable parameters | Estimated FP32 parameter bytes | Conv/linear MACs per image | Original weights-file bytes |
| --- | ---: | ---: | ---: | ---: |
| StandardCNN (`model_a`) | 94,762 | 379,048 | 41,288,960 | 387,218 |
| LightweightCNN (`model_b`) | 12,965 | 51,860 | 5,141,760 | 61,016 |
| WideLightweightCNN (`model_c`) | 46,373 | 185,492 | 18,561,536 | Not trained |

FP32 parameter storage is four bytes per trainable parameter. Actual weight-file
size includes BatchNorm buffers and serialization overhead. MACs count only
convolution and linear layers for one 64 × 64 image; they exclude pooling,
normalization, activations and data loading. Measured seconds depend on hardware
and include the whole data-loader pass, transfers and loss computation. Evaluation
also includes prediction collection. These quantities are reported separately.

Both models retain the original layer names, channel widths 3 → 32 → 64 → 128,
pooling and Linear(128, 10) classifier. The lightweight blocks have depthwise and
pointwise convolutions followed by one BatchNorm/ReLU/pooling sequence, with no
extra activation or normalization between the convolutions.

Model C reuses those lightweight blocks with widths 3 → 64 → 128 → 256,
adaptive average pooling to 1 × 1, flattening and Linear(256, 10). Both
convolutions in each block keep `bias=False`; BatchNorm keeps its affine
parameters and the classifier keeps its bias. Code checks verify the 46,373
trainable parameters and MAC count for a 64 × 64 input. The blocks contribute
347, 9,024 and 34,432 parameters, and the classifier contributes 2,570.

## Checks and files

```powershell
python -m unittest discover -s "$Project\tests" -v
python -m cnn_assignment validate --run outputs/custom_cnn/model_a_sgd_lr0.01 --compare-history --no-progress
```

The focused tests check model shapes, parameter/MAC counts, strict loading of all
six checkpoints, split integrity, normalization, sample-weighted metrics, metric
formulas, random/shuffle restoration, completed-run protection, simulated resume
and CLI operation from another directory. Resume tests substitute fake epoch
metrics and synthetic states; they perform no gradients or optimizer steps.
The separate validation command checks one real saved model on validation data.

The refactor check run passed all 10 tests. Model A's saved SGD checkpoint matched
epoch 27 exactly on all 4,050 validation images: loss 0.3795960882857994 and
accuracy 0.8720987654320987 (87.2099%). See
`outputs/refactor_checks/validation_model_a_sgd/metrics.json` and
`outputs/refactor_checks/verification_summary.json`. Recreated optimizer and
single-run plots are under `outputs/refactor_checks/optimizer_plots/` and
`outputs/refactor_checks/sgd_plot/`.

`src/cnn_assignment/` contains the CLI, data, transforms, models, training,
evaluation, plotting and utility modules. Dataset classes and transform classes
are importable for Windows worker spawning; zero workers is the default.
`pyproject.toml` defines packaging and the `cnn-assignment` entry point.
`test.ipynb` remains the analysis document, with its 79 cells and historical
outputs preserved. Revised cells import shared code, read completed experiments
and save recreated plots in a fresh report folder.

`outputs/refactor_checks/` holds the original file-hash manifest and verification
reports. The manifest's notebook hash records its pre-refactor version; the
notebook's source was intentionally updated while retaining its outputs. Split
CSVs and all files under `outputs/custom_cnn/` must still match their hashes.
Git ignores raw data, caches, environments and binary weights/generated figures.
Source, the notebook, split CSVs and small config/history/summary files remain
trackable. Ignoring a file does not delete it or untrack one already in Git.
