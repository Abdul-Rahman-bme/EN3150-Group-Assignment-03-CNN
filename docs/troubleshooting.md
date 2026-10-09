# Environment-specific troubleshooting

These notes describe issues observed in the existing Windows/Conda environment.
They are not general setup requirements or evidence of a verified clean-machine
installation. Run commands from the repository root; use
[README.md](../README.md) for standard setup.

## Existing Conda environment and distutils assertion

The local `ml_env_fixed` environment already contains the training dependencies.
An editable install encountered a setuptools/distutils import assertion with
Python 3.11. This PowerShell workaround temporarily selects standard-library
distutils and restores the previous setting:

```powershell
conda activate ml_env_fixed
$previousDistutilsSetting = $env:SETUPTOOLS_USE_DISTUTILS
try {
    $env:SETUPTOOLS_USE_DISTUTILS = 'stdlib'
    python -m pip install --no-index --no-deps --no-build-isolation -e .
} finally {
    $env:SETUPTOOLS_USE_DISTUTILS = $previousDistutilsSetting
}
python -m cnn_assignment --help
```

The offline flags reuse installed dependencies/build tools; this cannot populate
a new environment. The workaround applies to the inspected Python 3.11
environment, which still has standard-library distutils. It should not be
applied blindly to Python versions that removed that module.

## Windows TLS error downloading ImageNet weights

Python in this environment reported `[ASN1: NOT_ENOUGH_DATA]` during certificate
initialization (`pip_system_certs`/truststore) and automatic weight downloads.
Fresh pretrained runs stop on download/loading/hash failure, without switching
to random initialization.

Obtain the exact checkpoints from official PyTorch sources. Windows `curl.exe`
successfully downloaded the official ShuffleNet checkpoint in this environment
without disabling TLS verification. For example:

```powershell
New-Item -ItemType Directory -Force outputs/pretrained_weights
curl.exe --fail --location --output outputs/pretrained_weights/mobilenet_v2-7ebf99e0.pth https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth
curl.exe --fail --location --output outputs/pretrained_weights/shufflenetv2_x0.5-f707e7126e.pth https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
```

Do not replace already verified files unnecessarily. Pass the file to a fresh
run with `--pretrained-weights-file`; the loader verifies the official SHA-256
prefix/state dictionary and records the full source hash:

```sh
python -m cnn_assignment train --model shufflenet_v2_x0_5 --optimizer adam --lr 0.0001 --run-name shufflenet_v2_x0_5_manual_repeat1 --pretrained-weights-file outputs/pretrained_weights/shufflenetv2_x0.5-f707e7126e.pth
```

This command starts training when explicitly run. Source weights stay ignored.
Resume/evaluation restore trained checkpoints without ImageNet downloads. See
[the selection notes](pretrained_model_selection.md) for identifiers and evidence.

## Git report-folder permission warnings

Notebook validation uses fresh `notebook_validation_*` folders. Earlier checks
also generated such folders; current checks redirect disposable reports to
`tmp/`. The `.gitignore` directory rule excludes unselected notebook-validation
folders before exact selected-report exceptions, letting Git prune scratch
folders without reading their contents.

The six currently reported inaccessible folders are:

```text
outputs/reports/notebook_validation_387lz32a/
outputs/reports/notebook_validation_ebqujor6/
outputs/reports/notebook_validation_hj60y4kr/
outputs/reports/notebook_validation_o3ekd1ua/
outputs/reports/notebook_validation_r6hd0lhs/
outputs/reports/notebook_validation_zj2251yj/
```

These are selected in the output index and exact allowlist. The archive manifest
records complete, unique validation content (including timing). Direct reads are
currently denied, so these cannot be reclassified as confirmed disposable
reports. Their exceptions are preserved. Git must still inspect selected
untracked folders, so their warnings remain under current permissions. Ignoring
those directories would hide selected files; a broad rule cannot solve both needs.

Sandbox access can also make other tracked reports appear missing when host Git
can read them. This is separate from the six host permission warnings. No folders
were deleted and no permissions were changed. Check host-context status before
treating warnings as deletions; do not stage apparent sandbox-only deletions.
