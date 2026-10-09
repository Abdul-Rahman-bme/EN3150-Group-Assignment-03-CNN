# My model naming and pretrained candidates

The initial-stage notes below preserve my selection rationale and setup. Both
30-epoch runs and their final test reports have since been completed; see
[my final results](final_results.md) for the saved EuroSAT metrics.

## Names I use in the final report

| Report name | Existing code identifier | Role | Trainable parameters |
| --- | --- | --- | ---: |
| Required standard Model A | `model_a` / StandardCNN | Standard custom CNN | 94,762 |
| Required final lightweight Model B | `model_c` / WideLightweightCNN | Selected final lightweight custom CNN | 46,373 |
| Initial lightweight baseline | `model_b` / LightweightCNN | Earlier lightweight design | 12,965 |

I selected `model_c` using the saved validation results. It improved over the
initial `model_b` in all four matching configurations. With Adam 0.001, it reached
94.00% validation accuracy at its minimum-validation-loss checkpoint, close to
`model_a` at 94.123457%. Its 46,373 parameters satisfy the final lightweight
model's 100,000-parameter cap. The [custom CNN results](custom_cnn_results.md)
retain the historical A/B/C labels used when those experiments were run.

This is a report naming convention. I keep all code identifiers, experiment
folder names and saved artifacts unchanged. The two pretrained candidates are
separate comparisons; I do not apply the custom lightweight model's parameter
cap to them.

## The two pretrained candidates

I chose MobileNetV2 because its inverted residual blocks provide an established
compact pretrained backbone. I chose ShuffleNetV2 x0.5 as a smaller candidate
that uses depthwise convolutions and channel shuffling. I will compare their
EuroSAT validation results after fine-tuning; I do not assume the ImageNet ranking
will carry over to satellite images.

| Candidate / CLI identifier | Explicit weights | Published ImageNet-1K top-1 / top-5 accuracy | Published original 1,000-class parameters |
| --- | --- | --- | ---: |
| MobileNetV2 / `mobilenet_v2` | `MobileNet_V2_Weights.IMAGENET1K_V2` | 72.154% / 90.822% | 3,504,872 |
| ShuffleNetV2 x0.5 / `shufflenet_v2_x0_5` | `ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1` | 60.552% / 81.746% | 1,366,792 |

These are published ImageNet results for the original models with Torchvision's
224-pixel evaluation crops. They are **not my EuroSAT results**, and the original
parameter counts include the 1,000-class classifiers. Sources:
[official Torchvision 0.20 MobileNetV2 documentation](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.mobilenet_v2.html)
and [official Torchvision 0.20 ShuffleNetV2 x0.5 documentation](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.shufflenet_v2_x0_5.html).

## Initial fine-tuning setup

I first load the explicit ImageNet checkpoint, then replace only the final
classifier with a newly initialized ten-class linear layer. I fine-tune all
parameters, including the backbone, and allow BatchNorm statistics to update
while training. Validation and evaluation use evaluation mode.

I keep inputs at 64 x 64 RGB. I use tensor conversion and ImageNet normalization:
mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
I do not call the default weight transforms, resize to 224 pixels, or center-crop.
Training retains the existing random horizontal/vertical flips and quarter turns.
Custom-model normalization remains unchanged.

I reuse the exact saved train/validation/test CSVs and class order: AnnualCrop,
Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River and SeaLake. The splits contain 18,900 / 4,050 / 4,050 images.
Reading split metadata or checking its hashes does not evaluate the test set.

My initial configuration is Adam with learning rate 0.0001 and weight decay
0.0001, 30 epochs, batch size 64 and seed 42. I use FP32, cross-entropy loss and
the minimum-validation-loss checkpoint. This is an initial configuration, not a
proven optimum. Each initial run uses one seed.

## Measured model cost and synthetic verification

I verified the official source hashes and compared every available backbone
tensor with the source checkpoint before any update: 312 tensors for MobileNetV2
and 280 for ShuffleNetV2. The old official ShuffleNet checkpoint omits 56
`num_batches_tracked` counters; Torchvision/PyTorch's legacy loader initializes
these to zero. All backbone parameters and saved running means/variances match
the official file. My own saved checkpoints must include every buffer, including
the counters, and restoration rejects missing or unexpected keys.

The following values were measured after replacing the ImageNet classifiers.
MACs are convolution/linear multiply-accumulate counts obtained with forward
hooks for **one 64 x 64 RGB image**, not the published 224-pixel GFLOPS and not
measured inference speed. Pooling, normalization, activations, channel shuffling
and data loading are excluded.

| Candidate | Total parameters | Trainable parameters | Conv/linear MACs per image | Synthetic saved state bytes | Decimal MB | Binary MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MobileNetV2 | 2,236,682 | 2,236,682 | 24,461,312 | 9,180,270 | 9.180270 | 8.754988 |
| ShuffleNetV2 x0.5 | 352,042 | 352,042 | 3,230,848 | 1,542,078 | 1.542078 | 1.470640 |

These are actual file sizes for disposable synthetic verification state
dictionaries, saved using the same atomic best-weights writer. They are not
EuroSAT-trained checkpoints. Future runs record their own actual best-weights
and full resume-checkpoint sizes in `model_summary.json`; serialization overhead
and BatchNorm buffers are included. Parameter-storage estimates exclude both.
One byte is eight bits; one MB is 1,000,000 bytes; one MiB is 1,048,576 bytes.

I checked evaluation and training outputs for synthetic batches of 1, 20 (the
training split's final partial batch at batch size 64), and 64. Every output had
ten finite logits. Each batch had finite cross-entropy and gradients for all
parameters and completed an Adam update, including a backbone update. Saved
weights restored every parameter and BatchNorm buffer, and the restored outputs
were bitwise equal to the saved model's evaluation outputs.

| Candidate | GPU | Peak allocated memory at batch 64 | Peak reserved memory at batch 64 |
| --- | --- | ---: | ---: |
| MobileNetV2 | GTX 1650 Max-Q, approximately 4 GiB | 462,156,288 bytes | 532,676,608 bytes (508 MiB) |
| ShuffleNetV2 x0.5 | GTX 1650 Max-Q, approximately 4 GiB | 81,952,256 bytes | 106,954,752 bytes (102 MiB) |

The configured all-layer FP32/Adam update fits the available GPU in this
synthetic check. These peaks describe this process's PyTorch allocator, not
all applications' GPU use or an entire 30-epoch EuroSAT run. The complete evidence,
including source SHA-256 hashes and all tested batch sizes, is in
[pretrained_verification.json](pretrained_verification.json). I did not run
dataset training or test-set evaluation.

## Download failures and local source weights

Automatic Python HTTPS loading in this Windows environment failed with
`ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`. The implementation reports the original
error and stops. It never retries with an unverified random backbone and never
disables TLS verification. I obtained ShuffleNet's official checkpoint with
Windows curl and verified the official hash prefix and the full state. MobileNet's
official V2 checkpoint was already cached. Both source files are now available
locally under `outputs/pretrained_weights/`, which remains ignored by Git.

For another machine, I can download the exact files from the official sources:

- [MobileNetV2 V2 checkpoint](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth).
- [ShuffleNetV2 x0.5 V1 checkpoint](https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth).

I save them with the filenames shown below and supply `--pretrained-weights-file`.
The loader checks the official SHA-256 filename prefix (and records the full
SHA-256), restores the original classifier/backbone strictly, then creates the
new ten-class head. A wrong, incomplete or inaccessible file stops the run.
For example, after creating a fresh local source directory:

```powershell
New-Item -ItemType Directory -Force outputs/pretrained_weights
curl.exe --fail --location --output outputs/pretrained_weights/mobilenet_v2-7ebf99e0.pth https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth
curl.exe --fail --location --output outputs/pretrained_weights/shufflenetv2_x0.5-f707e7126e.pth https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth
```

The sources already exist in this workspace, so I do not need to run these
download commands again. Omitting `--pretrained-weights-file` uses the explicit
Torchvision weight enum and cache/download path. This environment's TLS issue
may still prevent uncached automatic downloads.

## Exact commands for my initial runs

I run these explicitly from the project root after activating `ml_env_fixed`.
They use the already verified local official source files. Neither command has
been executed as part of this implementation.

```powershell
conda activate ml_env_fixed
python -m cnn_assignment train --model mobilenet_v2 --optimizer adam --lr 0.0001 --weight-decay 0.0001 --epochs 30 --batch-size 64 --seed 42 --device cuda --num-workers 0 --output-root outputs/pretrained_cnn --run-name mobilenet_v2_adam_lr0.0001_initial1 --pretrained-weights-file outputs/pretrained_weights/mobilenet_v2-7ebf99e0.pth
python -m cnn_assignment train --model shufflenet_v2_x0_5 --optimizer adam --lr 0.0001 --weight-decay 0.0001 --epochs 30 --batch-size 64 --seed 42 --device cuda --num-workers 0 --output-root outputs/pretrained_cnn --run-name shufflenet_v2_x0_5_adam_lr0.0001_initial1 --pretrained-weights-file outputs/pretrained_weights/shufflenetv2_x0.5-f707e7126e.pth
```

Fresh run folders are protected against overwriting. Progress bars show batches
and epoch summaries unless `--no-progress` is supplied. Configs record the explicit
weight enum, official source URL and SHA-256, 64-pixel input, normalization,
augmentation, all-layer fine-tuning, environment, split hashes, training settings
and post-replacement parameter/MAC counts. The existing cross-entropy loop uses
sample-weighted metrics and synchronizes CUDA timing.

## Resume, validation, plots and the explicit final test

An interrupted run resumes from its complete epoch-boundary state, including
optimizer state, BatchNorm buffers, Python/NumPy/torch/CUDA random states and the
training shuffle generator. I use the original device and worker count. I do
not resupply the ImageNet file or override the saved training settings:

```powershell
python -m cnn_assignment train --resume outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1 --device cuda --num-workers 0
python -m cnn_assignment validate --run outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1 --compare-history
python -m cnn_assignment plot --run outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1
python -m cnn_assignment plot --runs-root outputs/pretrained_cnn
python -m cnn_assignment plot --additional-runs-root outputs/pretrained_cnn
```

The same commands accept the ShuffleNet run path. Single-run plots mark the
minimum-loss checkpoint. Comparisons read only saved histories; the final command
includes the existing custom runs and both new models without moving folders.
Every report is written to a fresh folder. Resume and evaluation build the
architecture with no ImageNet loading, then restore the saved state; they work
without source weights or network access. Synthetic tests forbid the source
loader during those operations and compare uninterrupted/resumed states.

After selecting a final checkpoint using validation only, the existing explicit
test command supports either model. This example is an instruction for that later
stage, **not a command I ran**:

```powershell
python -m cnn_assignment evaluate-test --run outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1
```

## Repeating the checks

```powershell
python -m unittest discover -s tests -v
python tests/verify_pretrained_gpu.py --weights-dir outputs/pretrained_weights
```

The optional GPU verification uses synthetic tensors only and writes disposable
states to a fresh `tmp/pretrained_gpu_check_*` folder. The genuine-weight unit
checks use cached/manual official checkpoints; if unavailable on another machine,
those two checks skip rather than downloading during the unit suite. The workflow
tests use synthetic state or simulated epoch metrics and never train the dataset.
Notebook checks execute copies, forbid dataset training and test evaluation,
and direct reports into scratch space without replacing historical notebook
outputs or verification records.
