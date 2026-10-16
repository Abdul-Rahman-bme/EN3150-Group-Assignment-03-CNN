# EN3150 Group Project: Resource-Constrained CNN Classification

This project compares standard and depthwise-separable custom CNNs with
fine-tuned MobileNetV2 and ShuffleNetV2 for EuroSAT land-cover classification.
The comparison covers accuracy, model storage, training time and computation
for 64 x 64 RGB inputs.

## Team members

| Index number | Name |
| --- | --- |
| 230507R | Rahman M.F.A. |
| 230521E | Ranasinghe D.P.H. |
| 230544C | Rathnayake M.A.G.K.N. |
| 230108U | Colombage D.M. |

## Dataset and models

[EuroSAT](https://github.com/phelber/EuroSAT) contains 27,000 labeled Sentinel-2
images across ten land-cover classes. This project uses the RGB version and
saved stratified splits: 18,900 training (70%), 4,050 validation (15%) and
4,050 test images (15%). The saved class order is AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential,
River and SeaLake.

| Report role | Code identifier | Architecture |
| --- | --- | --- |
| Standard Model A | `model_a` | StandardCNN |
| Final lightweight Model B | `model_c` | WideLightweightCNN |
| Initial lightweight baseline | `model_b` | LightweightCNN |

Final Model B was selected using validation results. Its 46,373 trainable
parameters satisfy the 100,000-parameter cap. Existing identifiers, run paths
and historical notebook A/B/C labels remain unchanged.

## Final test results

Values come from the four saved final test reports linked in the
[output index](outputs/README.md#final-test-results-and-selected-pretrained-runs).
Each run used seed 42 and 30 epochs; checkpoints were selected by minimum
validation loss before test evaluation.

| Model | Test accuracy | Macro F1 | Parameters | Saved weights (MB) | MACs/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard Model A (`model_a`) | 93.90% | 93.75% | 94,762 | 0.387218 | 41,288,960 |
| Final lightweight Model B (`model_c`) | 94.37% | 94.18% | 46,373 | 0.196521 | 18,561,536 |
| MobileNetV2 | 98.12% | 98.02% | 2,236,682 | 9.180270 | 24,461,312 |
| ShuffleNetV2 x0.5 | 96.37% | 96.21% | 352,042 | 1.542078 | 3,230,848 |

MB means 1,000,000 bytes. Sizes include saved buffers and serialization overhead.
MACs count convolution/linear operations for one 64 x 64 image; they are not
measured inference speed. These single-seed test results are separate from
validation and published ImageNet results.

## Project structure

```text
data/
  raw/eurosat/             # Archive and images; obtained separately
  splits/                 # Saved train.csv, validation.csv and test.csv
docs/                     # Results, selection notes and troubleshooting
outputs/
  custom_cnn/             # Twelve completed custom experiments
  pretrained_cnn/         # Two completed pretrained experiments
  reports/                # Selected metrics, predictions and plots
  archive/                # Local historical artifacts; ignored
  README.md               # Shared-result index and local-only artifact paths
src/cnn_assignment/       # CLI and shared workflow
notebooks/                # Abdul's original preparation and standard-CNN notebooks
tests/                    # Workflow and saved-result checks
test.ipynb                # Analysis notebook with historical outputs
pyproject.toml
requirements.txt
```

## Setup

Use Python 3.11 and run commands from the repository root. The recorded training
environment used PyTorch 2.5.1 and Torchvision 0.20.1. A clean-machine installation
has **not** been verified; `requirements.txt` pins the tested torch/torchvision pair; the remaining
dependency inventory is not an environment lock.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install:

```sh
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install numpy==1.26.4 pandas matplotlib Pillow tqdm nbclient nbformat ipykernel jupyterlab
python -m pip install --no-deps -e .
python -m cnn_assignment --help
```

Choose the platform-appropriate CPU/CUDA installation command from the
[official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
Exact resume requires the saved Python/package versions, device and worker count;
the setup above is not an exact reconstruction of every recorded dependency.
See [environment-specific troubleshooting](docs/troubleshooting.md) for the
existing Conda environment and Windows certificate issues.

Open `jupyter lab test.ipynb` with the project environment as its kernel.
Run the setup cell first; section 9 reads final reports in order. Earlier
validation cells require images and checkpoints. Historical outputs and
first-person notebook explanations are retained.

## Dataset acquisition

Obtain the **RGB** archive through the
[official EuroSAT repository](https://github.com/phelber/EuroSAT#dataset),
which links the current Zenodo distribution. Extract or arrange the JPEG class
folders into this layout, preserving filenames and pixel content:

```text
data/raw/eurosat/
  EuroSAT.zip             # Optional; needed only for --check-archive
  2750/
    AnnualCrop/AnnualCrop_1.jpg
    Forest/Forest_1.jpg
    ...                   # All ten class directories listed above
data/splits/
  train.csv
  validation.csv
  test.csv
```

Reuse the committed split CSVs and saved normalization. Images must match the
CSV paths and pixel hashes. The optional archive check expects the original
archive checksum; another packaging of the same images can fail that check.

```sh
python -m cnn_assignment prepare
python -m cnn_assignment prepare --check-images
```

`prepare` checks existing images, saved split integrity and configuration. It
does **not** download/extract data, generate missing splits or recalculate
normalization. Missing images or CSVs must be supplied separately.

## Commands

Paths below are relative to the project. Plotting uses saved configs/histories
and works without raw images or checkpoints:

```sh
python -m cnn_assignment plot --run outputs/custom_cnn/model_c_adam_lr0.001
python -m cnn_assignment plot
python -m cnn_assignment plot --runs-root outputs/pretrained_cnn
python -m cnn_assignment plot --additional-runs-root outputs/pretrained_cnn
```

Training is explicit and requires images. Use new run names; existing folders
are protected. These examples start new experiments:

```sh
python -m cnn_assignment train --model model_a --optimizer adam --lr 0.001 --run-name model_a_adam_repeat1
python -m cnn_assignment train --model model_c --optimizer adam --lr 0.001 --run-name model_c_adam_repeat1
python -m cnn_assignment train --model mobilenet_v2 --optimizer adam --lr 0.0001 --run-name mobilenet_v2_adam_repeat1
python -m cnn_assignment train --model shufflenet_v2_x0_5 --optimizer adam --lr 0.0001 --run-name shufflenet_v2_x0_5_adam_repeat1
```

Defaults are 30 epochs, batch size 64, seed 42, weight decay 0.0001 and zero
workers. `--device auto` selects CUDA when available; `--device cpu` forces CPU.
Training retains 64 x 64 inputs and geometric augmentation. Custom models use
saved dataset normalization; pretrained models use ImageNet normalization and
fine-tune all layers. Fresh pretrained runs load genuine ImageNet weights before
creating a ten-class head; loading failures stop the run. Use
`--pretrained-weights-file` for a verified manual source download.

Resume an **incomplete** run from its full epoch-boundary checkpoint:

```sh
python -m cnn_assignment train --resume outputs/custom_cnn/model_c_adam_repeat1
```

Resume uses `last_checkpoint.pt` and the saved configuration. Match its device
and worker count; completed runs cannot be extended. The original six custom
`best_weights.pt` files lack optimizer/RNG/last-epoch state and **cannot provide
exact training resume**. Pretrained resume/evaluation load trained checkpoints
without downloading ImageNet weights again.

Validation/evaluation require both images and the run's trained
`best_weights.pt` checkpoint:

```sh
python -m cnn_assignment validate --run outputs/custom_cnn/model_c_adam_lr0.001 --compare-history
```

Final-test evaluation is separate and explicit, used only after selecting a
configuration using validation. Existing reports can be read without rerunning it:

```sh
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_c_adam_lr0.001
```

Plots/evaluations create fresh timestamped report folders. An explicit
`--output-dir` must not already exist. `--no-progress` disables batch progress
bars for training/validation/evaluation.

Saved-file and guarded notebook-analysis checks:

```sh
python -m unittest discover -s tests -p test_final_results.py -v
```

These checks perform no model evaluation or training; local checkpoint hash/size
assertions require saved weights. The optional broader suite is
`python -m unittest discover -s tests -v`; it also needs dataset/checkpoint
artifacts, includes synthetic updates and real validation checks, and does not
launch dataset training or final-test evaluation.

## Shared files and further reading

Git includes source, tests, the notebook, split CSVs, all 12 custom configs and
histories, selected pretrained configs/histories/summaries, and the reports/plots
allowlisted in [outputs/README.md](outputs/README.md). New reports stay ignored
until deliberately selected.

**Raw images and model weights are not included in Git.** Obtain EuroSAT
separately and request trained checkpoints from the group, preserving their
documented paths. Full resume checkpoints, source ImageNet weights, archives,
temporary reports and personal `notes.md` also remain local-only.

- [Final results and remaining assignment requirements](docs/final_results.md)
- [Custom CNN validation experiments](docs/custom_cnn_results.md)
- [Pretrained model selection and initial settings](docs/pretrained_model_selection.md)
- [Selected output index](outputs/README.md)
- [Environment-specific troubleshooting](docs/troubleshooting.md)

The upstream [data-preparation notebook](notebooks/01_dataset_preparation.ipynb)
and [standard-CNN notebook](notebooks/02_model_A_standard_cnn.ipynb) are retained
unchanged, including their saved outputs. Their standalone workflow contains
dataset-download, split-generation, training and test-evaluation cells; it is
distinct from the saved-file CLI workflow. See [integration decisions](docs/integration.md)
before comparing results or running those notebooks.
