# Integration of the completed workflows with upstream main

This integration starts from `upstream/main` commit `0056ee8` and merges
`heshan` commit `4bdd4d5`. The original branches and local experiments are
left unchanged. No dataset training or test evaluation is required for the merge.

## Contributor implementations

Abdul's `notebooks/01_dataset_preparation.ipynb` and
`notebooks/02_model_A_standard_cnn.ipynb`, their metadata and saved outputs remain
byte-identical to upstream. The upstream data/dataset/experiment documentation
and `src/__init__.py` also remain intact. The README restores the upstream team
names and index numbers. Original upstream ignore rules are retained alongside
the exact shared-result allowlist.

There are two distinct classes named `StandardCNN`, kept in their original
namespaces rather than treated as interchangeable:

| Feature | Abdul's original notebook baseline | Shared CLI `model_a` |
| --- | --- | --- |
| Channels | 32 / 64 / 128 | 32 / 64 / 128 |
| Convolutions | Standard 3 x 3, with biases | Standard 3 x 3, no biases |
| Normalization in network | No BatchNorm | BatchNorm after each convolution |
| Head | Flatten 8,192 -> 256 -> 10, ReLU/dropout 0.5 | Global average pooling, 128 -> 10 |
| Trainable parameters | 2,193,226 | 94,762 |
| Training | Adam 0.001, 20 epochs, no geometric augmentation | Saved 30-epoch optimizer experiments with geometric augmentation |
| Selection | Maximum validation accuracy | Minimum validation loss |
| Trained weights | `models/model_A_best.pth` (local-only) | Existing `outputs/custom_cnn/.../best_weights.pt` (local-only) |

The notebook's saved 92.27% test accuracy is a historical, rounded baseline
result, not a fifth full-precision final report. It is not substituted into
the completed four-model comparison. No notebook output is relabeled and no
checkpoint is loaded into the competing architecture. The final report still
uses Model A = `model_a`, final Model B = `model_c`, and `model_b` as the initial
lightweight baseline. The group should agree on this report convention before
submitting the final PDF.

Abdul's notebooks rebuild split indices with seed-42 stratified scikit-learn
splits and calculate training normalization. The CLI reuses the committed CSVs
and saved normalization. The integration checks compare split membership from
the same sorted EuroSAT filenames without reading images or rebuilding files.
Historical train/validation/test splits, configs, histories and result paths
are not rewritten to standardize the two pipelines.

The membership check passed for train, validation and test. This confirms the
same image memberships, not identical row order or identical augmentation.

## Dependency conflict

The merge conflict was in `requirements.txt`: upstream pinned `torch==2.13.0`
and `torchvision==0.28.0`, while heshan used an inventory and had completed
experiments on torch 2.5.1 / torchvision 0.20.1. The shared file uses the latter
tested pair and includes both contributors' dependencies (`scikit-learn`,
`ipykernel`, Pillow and notebook-check tools included). No notebook code changes
were needed for these APIs. The upstream file is preserved exactly as
`requirements-upstream.txt` for provenance, not as the default installation.
A fresh environment with the union dependencies has not been installed/tested;
this is not a clean-machine reproduction claim.

## Artifacts and submission gaps

All tracked heshan configs, histories, split CSVs, original notebook outputs and
saved final reports retain their paths and bytes. Raw images, trained/source
weights, full resume checkpoints, archives and personal notes stay ignored.
They remain available in the original local workspace; none is pushed as part
of the integration. Missing ignored artifacts in a clean worktree are an
environment limitation, not a model implementation failure.

Six indexed validation reports were never committed because local permissions
prevent reading them. Their index links and allowlist remain preserved; those
six files remain a sharing gap. No permissions or historical folders are changed.
The detailed [final-results audit](final_results.md) lists remaining dataset-sheet
confirmation, final PDF/group information, packaged code and Moodle submission.
Team identifiers are now present in the shared README, but they still need to
be included and confirmed in the final submission PDF.

Checks and integration evidence are recorded in
[integration_checks.json](integration_checks.json). They avoid dataset training
and test-set evaluation. Original notebook training/evaluation cells are checked
as source, never run during integration.

All 30 targeted tests passed: 28 initially and two artifact-dependent tests on
a focused rerun after linking the existing local image directory. The initial
three subtest errors were missing raw-image paths, not implementation failures.
The link and locally copied ignored weights are not committed. The separate
notebook suite's real validation/preview passes were not repeated; static
compilation, output preservation and guarded fresh-kernel saved-analysis checks
passed instead. Abdul's exact class definition also produced finite ten-class
outputs for synthetic batches of 1, 20 and 64 on the tested package pair.
