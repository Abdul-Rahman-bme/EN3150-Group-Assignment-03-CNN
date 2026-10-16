# EN3150 Group Project: Resource-Constrained CNN Classification

This project investigates convolutional neural networks for resource-constrained
image classification using the **EuroSAT RGB land-cover dataset**.

The work includes:

- A standard custom CNN used as the baseline (**Model A**).
- An initial lightweight depthwise-separable CNN (**initial Model B / `model_b`**).
- A wider depthwise-separable CNN selected as the final lightweight assignment
  model (**final Model B / `model_c`**).
- An additional compact depthwise-separable CNN developed by **Mokshan**
  (**`compact_eurosat`**).
- An additional residual depthwise-separable CNN developed by **Nipun**
  (**Model D / `model_d`**).
- Fine-tuned **MobileNetV2**.
- Fine-tuned **ShuffleNetV2 x0.5**.

The experiments compare:

- Classification accuracy
- Precision, recall and F1 score
- Validation performance
- Test performance
- Trainable parameter count
- Model storage
- Convolution/linear MACs
- Training time
- Optimizer behaviour
- Architecture complexity
- Resource/accuracy trade-offs
- Verification and reproducibility evidence

The main custom-model objective is to investigate how much model size and
computation can be reduced while maintaining useful classification accuracy.

All custom models use **64 × 64 RGB inputs** and classify the ten EuroSAT
land-cover classes.

---

## Team Members

| Index number | Name |
|---|---|
| 230507R | Rahman M.F.A. |
| 230521E | Ranasinghe D.P.H. |
| 230544C | Rathnayake M.A.G.K.N. |
| 230108U | Colombage D.M. |

Additional model contributions are explicitly identified in the model-history
sections below, including the CompactEuroSAT experiment developed by
**Mokshan** and the residual Model D experiment developed by **Nipun**.

---

# 1. Dataset

## 1.1 EuroSAT RGB

The project uses the **EuroSAT RGB dataset**, which contains 27,000 labelled
Sentinel-2 satellite images covering ten land-cover classes.

Only the RGB version of the dataset is used.

Each image is processed as:

```text
64 × 64 × 3
```

The ten classes are:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

The project uses the following saved stratified split:

| Split | Number of images | Percentage |
|---|---:|---:|
| Training | 18,900 | 70% |
| Validation | 4,050 | 15% |
| Testing | 4,050 | 15% |
| **Total** | **27,000** | **100%** |

The same saved split is used for the controlled final experiments.

The saved class order is:

```text
AnnualCrop
Forest
HerbaceousVegetation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
```

The project does not regenerate the split during normal reproduction. The
saved CSV files and their fingerprints are retained so that experiments can
use the same data partition.

---

# 2. Project Objective

The main objective is to investigate the trade-off between classification
performance and computational/resource requirements.

A conventional CNN can achieve good accuracy but may require substantially
more parameters and computation.

Depthwise-separable convolution provides a way of reducing the cost of
convolution by separating spatial filtering from channel mixing.

The project therefore investigates several architectures with different
resource requirements.

The main questions investigated are:

1. How well does a standard CNN perform?
2. How much can the model be reduced using depthwise-separable convolution?
3. Does increasing the width of a lightweight model recover accuracy?
4. How does optimizer choice affect the custom architectures?
5. How do the custom models compare with pretrained networks?
6. Can an even smaller architecture maintain useful accuracy?
7. Does adding residual connections provide a useful accuracy/resource
   trade-off?
8. How do parameter count, storage and MACs differ between architectures?
9. What evidence is available to support the reported results?

---

# 3. Model Naming and History

There are several historical model names in the project because the architectures
were developed incrementally.

The source-code identifiers are intentionally preserved so that existing
checkpoints, notebooks and experiment folders remain valid.

The relationship between the historical names and the final assignment naming
is:

| Assignment / historical role | Code identifier | Architecture |
|---|---|---|
| Standard Model A | `model_a` | `StandardCNN` |
| Initial lightweight baseline | `model_b` | `LightweightCNN` |
| Final lightweight Model B | `model_c` | `WideLightweightCNN` |
| Mokshan additional compact model | `compact_eurosat` | `CompactEuroSAT` |
| Nipun additional model | `model_d` | `ResidualDepthwiseCNN` |
| Pretrained reference | `mobilenet_v2` | MobileNetV2 |
| Pretrained reference | `shufflenet_v2_x0_5` | ShuffleNetV2 x0.5 |

## Important naming clarification

The architecture implemented as:

```text
model_c
```

was selected as the final lightweight architecture for the assignment.

Therefore:

```text
model_c = final assignment Model B
```

The earlier:

```text
model_b
```

is retained as the **initial lightweight baseline**.

This naming is preserved deliberately because changing the identifiers would
break compatibility with the existing experiment folders, saved configurations,
histories and checkpoints.

---

# 4. Complete Model Inventory

The repository currently contains five custom architectures and two pretrained
architectures.

| Model | Code identifier | Type | Parameters | MACs/image | Test accuracy |
|---|---|---|---:|---:|---:|
| Standard Model A | `model_a` | Custom standard CNN | 94,762 | 41,288,960 | 93.90% |
| Initial lightweight Model B | `model_b` | Custom depthwise-separable CNN | 12,965 | 5,141,760 | Not part of final test set |
| Final lightweight Model B | `model_c` | Custom wide depthwise-separable CNN | 46,373 | 18,561,536 | 94.37% |
| CompactEuroSAT | `compact_eurosat` | Additional compact CNN | 11,466 | 3,882,784 | 94.72%* |
| Model D | `model_d` | Additional residual depthwise CNN | 15,442 | 4,728,768 | 93.63%* |
| MobileNetV2 | `mobilenet_v2` | Pretrained | 2,236,682 | 24,461,312 | 98.12% |
| ShuffleNetV2 x0.5 | `shufflenet_v2_x0_5` | Pretrained | 352,042 | 3,230,848 | 96.37% |

\* CompactEuroSAT and Model D are additional experiments and are discussed
separately because their experiment/evidence status differs from the four
controlled final test reports.

---

# 5. Model Development History

The project did not begin with the final lightweight architecture.

Several architectures were developed and evaluated to understand the effect
of width, depthwise-separable convolution and residual connections.

The repository preserves these experiments rather than showing only the
final model.

---

# 6. Model A — Standard CNN

## 6.1 Purpose

Model A is the standard-convolution baseline.

It provides a reference point for determining how much resource reduction can
be achieved by replacing standard convolutions with depthwise-separable
convolutions.

## 6.2 Architecture

The architecture is:

```text
Input
64 × 64 × 3

        ↓

3 × 3 Standard Convolution
3 → 32 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

3 × 3 Standard Convolution
32 → 64 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

3 × 3 Standard Convolution
64 → 128 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Global Average Pooling

        ↓

Linear
128 → 10

        ↓

10 class logits
```

The implementation disables convolution biases because BatchNorm follows the
convolution.

## 6.3 Resource cost

| Metric | Model A |
|---|---:|
| Trainable parameters | 94,762 |
| Conv/linear MACs per 64 × 64 image | 41,288,960 |
| Saved weights | 387,218 bytes |
| Saved weights | 0.387218 MB |

The model is below the 100,000-parameter limit, but it is substantially larger
than the lightweight alternatives.

## 6.4 Optimizer experiments

Four configurations were tested:

| Optimizer | Learning rate | Momentum |
|---|---:|---:|
| Adam | 0.001 | N/A |
| SGD | 0.01 | 0 |
| SGD with momentum | 0.01 | 0.9 |
| SGD | 0.003 | 0 |

All runs used:

```text
Seed = 42
Batch size = 64
Weight decay = 0.0001
Epoch budget = 30
```

## 6.5 Validation results

The checkpoint selection rule was minimum validation loss.

| Optimizer | LR | Selected epoch | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| Adam | 0.001 | 30 | 0.172110 | 94.123% |
| SGD | 0.003 | 29 | 0.459452 | 83.630% |
| SGD | 0.01 | 27 | 0.379596 | 87.210% |
| SGD + momentum | 0.01 | 30 | 0.174506 | 94.198% |

Adam was selected because it produced the lowest validation loss:

```text
Adam:
Validation loss = 0.172110

SGD + momentum:
Validation loss = 0.174506
```

Although momentum SGD had slightly higher validation accuracy at its selected
epoch, the project uses validation loss as the selection criterion.

## 6.6 Final test result

After selecting the configuration using the validation data, the selected
checkpoint was evaluated on the untouched test set.

| Metric | Result |
|---|---:|
| Test loss | 0.173574 |
| Test accuracy | 93.901235% |
| Macro precision | 93.900989% |
| Macro recall | 93.702222% |
| Macro F1 | 93.749068% |

## 6.7 Evidence

The following evidence is retained:

- Model implementation
- Model configuration
- Training history
- Validation results
- Test metrics
- Test predictions
- Per-class metrics
- Confusion matrix
- Normalized confusion matrix
- Model parameter count
- MAC calculation
- Saved weights
- SHA-256/file-size information

Relevant locations include:

```text
outputs/custom_cnn/model_a_*
outputs/reports/
docs/custom_cnn_results.md
docs/final_results.md
```

---

# 7. Initial Lightweight Model B

## 7.1 Purpose

The initial lightweight Model B was the first custom architecture designed
using depthwise-separable convolutions.

The purpose was to reduce parameter count and computation compared with the
standard Model A.

## 7.2 Architecture

```text
Input
64 × 64 × 3

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
3 → 32 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
32 → 64 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
64 → 128 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Global Average Pooling

        ↓

Linear
128 → 10
```

## 7.3 Resource cost

| Metric | Initial Model B |
|---|---:|
| Trainable parameters | 12,965 |
| Conv/linear MACs per image | 5,141,760 |

Compared with Model A, this architecture is substantially smaller.

## 7.4 Optimizer experiments

The same four optimizer configurations were tested.

| Optimizer | Learning rate | Selected epoch | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| Adam | 0.001 | 27 | 0.244486 | 91.556% |
| SGD | 0.003 | 28 | 0.651173 | 78.296% |
| SGD | 0.01 | 28 | 0.483142 | 83.531% |
| SGD + momentum | 0.01 | 25 | 0.234703 | 91.827% |

The best configuration according to the validation-loss selection rule was:

```text
SGD
Learning rate = 0.01
Momentum = 0.9
```

with:

```text
Validation loss = 0.234703
Validation accuracy = 91.827%
```

## 7.5 Role in the final project

This architecture remained in the repository as an initial lightweight
baseline.

It was not selected as the final assignment Model B.

The main reason for continuing the architecture search was that the model was
very small, but its validation performance was lower than desired.

The later wider `model_c` architecture provided a better balance between
resource usage and classification accuracy.

---

# 8. Final Lightweight Model B — Wide Lightweight CNN

## 8.1 Purpose

The final lightweight Model B was developed by widening the depthwise-separable
architecture.

The intention was to recover representational capacity while keeping the model
below the 100,000-parameter requirement.

The source identifier remains:

```text
model_c
```

but the final assignment role is:

```text
Model B
```

## 8.2 Architecture

```text
Input
64 × 64 × 3

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
3 → 64 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
64 → 128 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Depthwise 3 × 3
+
Pointwise 1 × 1
128 → 256 channels
BatchNorm
ReLU
2 × 2 MaxPool

        ↓

Global Average Pooling

        ↓

Linear
256 → 10
```

The depthwise convolution performs spatial filtering independently for each
input channel.

The pointwise 1 × 1 convolution then performs channel mixing.

## 8.3 Parameter calculation

For a depthwise-separable block:

```text
Depthwise parameters
= Cin × 3 × 3

Pointwise parameters
= Cin × Cout × 1 × 1
```

Therefore:

```text
Depthwise + pointwise
= Cin × 9 + Cin × Cout
```

BatchNorm contributes two trainable affine parameters per output channel.

The final classifier contributes:

```text
256 × 10 + 10
= 2,570 parameters
```

The total model contains:

```text
46,373 trainable parameters
```

This satisfies the assignment requirement:

```text
46,373 < 100,000
```

## 8.4 Resource cost

| Metric | Final Model B (`model_c`) |
|---|---:|
| Trainable parameters | 46,373 |
| Conv/linear MACs per image | 18,561,536 |
| Saved weights | 196,521 bytes |
| Saved weights | 0.196521 MB |
| Estimated FP32 parameter storage | approximately 185.5 KB |

## 8.5 Optimizer investigation

Four configurations were evaluated.

| Optimizer | Learning rate | Selected epoch | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| Adam | 0.001 | 28 | 0.177609 | 94.000% |
| SGD | 0.003 | 29 | 0.588112 | 79.605% |
| SGD | 0.01 | 29 | 0.416601 | 85.012% |
| SGD + momentum | 0.01 | 29 | 0.218274 | 92.321% |

The selected configuration was:

```text
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.0001
Batch size: 64
Seed: 42
Epoch budget: 30
Selected epoch: 28
```

The selected validation result was:

```text
Validation loss = 0.177608554
Validation accuracy = 94.000000%
```

Adam was selected because it achieved the lowest validation loss among the
tested configurations.

## 8.6 Final test evaluation

The final test evaluation was performed after the model/configuration had
been selected using validation data.

The test set contains:

```text
4,050 images
```

Final result:

| Metric | Final Model B |
|---|---:|
| Test loss | 0.179913 |
| Test accuracy | 94.370370% |
| Macro precision | 94.225456% |
| Macro recall | 94.251111% |
| Macro F1 | 94.182223% |

## 8.7 Comparison with Model A

| Metric | Model A | Final Model B |
|---|---:|---:|
| Test accuracy | 93.901% | **94.370%** |
| Parameters | 94,762 | **46,373** |
| MACs/image | 41,288,960 | **18,561,536** |
| Saved weights | 0.387218 MB | **0.196521 MB** |

Relative to Model A, final Model B uses approximately:

- **51.06% fewer trainable parameters**
- **55.04% fewer conv/linear MACs**
- **49.25% less saved-weight storage**

The recorded test accuracy is approximately:

```text
0.469 percentage points higher
```

than Model A.

This is an observation from a single seed and should not be interpreted as
proof that the architecture is universally more accurate.

## 8.8 Model B verification

The final Model B implementation was checked for:

- Correct depthwise convolution configuration
- Correct pointwise convolution configuration
- Correct channel widths
- Correct ordering of operations
- Correct convolution bias configuration
- Correct BatchNorm configuration
- Correct output shape
- Correct trainable parameter count
- Correct MAC count
- Correct optimizer configuration
- Checkpoint round-trip loading
- Validation workflow
- Comparison workflow

The corresponding architecture/workflow tests passed.

## 8.9 Model B evidence

The repository retains:

```text
outputs/custom_cnn/model_c_*
outputs/reports/
docs/custom_cnn_results.md
docs/final_results.md
report/audit/
```

The saved evidence includes configurations, training histories, validation
results, final test results, predictions, confusion matrices and derived
resource measurements.

---

# 9. Mokshan's CompactEuroSAT CNN

## 9.1 Purpose

A separate compact CNN experiment was developed by **Mokshan Colombage**.

The purpose was to investigate whether a very small depthwise-separable network
could achieve strong EuroSAT performance with substantially fewer parameters
and MACs.

The architecture is registered in the shared repository as:

```text
compact_eurosat
```

with the class:

```text
CompactEuroSAT
```

The original notebook is retained so that the historical experiment is not
lost.

## 9.2 Architecture

```text
Input
64 × 64 × 3

        ↓

Standard 3 × 3 Stem
3 → 16

        ↓

Depthwise-Separable Block
16 → 32
Stride = 2

        ↓

Depthwise-Separable Block
32 → 48
Stride = 2

        ↓

Residual Depthwise-Separable Block
48 → 48

        ↓

Depthwise-Separable Block
48 → 80
Stride = 2

        ↓

Global Average Pooling

        ↓

Dropout
0.10

        ↓

Linear
80 → 10
```

The model therefore combines:

- A small standard convolutional stem
- Depthwise-separable convolution
- A residual block
- Global average pooling
- A small classifier

## 9.3 Resource cost

| Metric | CompactEuroSAT |
|---|---:|
| Trainable parameters | 11,466 |
| Conv/linear MACs | 3,882,784 |
| FP32 parameter storage | 45,864 bytes |
| Historical checkpoint size | approximately 0.06554 MiB |

This is smaller than the final Model B in both parameter count and MACs.

## 9.4 Historical notebook configuration

The historical notebook used:

```text
Dataset: EuroSAT RGB
Images: 27,000
Split: 70/15/15 stratified
Input: 64 × 64 RGB
Seed: 42
Batch size: 128
Optimizer: Adam
Learning rate: 0.001
Epoch budget: 60
Scheduler: Cosine Annealing
Weighted loss: False
Automatic mixed precision: Enabled on CUDA
Device: CUDA / Tesla T4
Precision: FP32 model parameters
```

The notebook recorded:

```text
PyTorch 2.11.0
```

## 9.5 Historical training/validation/test result

The notebook reported:

| Metric | Result |
|---|---:|
| Training accuracy | 95.63% |
| Validation accuracy | 94.64% |
| Test accuracy | **94.72%** |
| Test loss | 0.1583 |
| Balanced accuracy | 94.576% |
| Macro precision | 94.538% |
| Macro recall | 94.576% |
| Macro F1 | 94.542% |
| Weighted F1 | 94.707% |

## 9.6 Historical per-class results

The notebook's final classification report recorded:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| AnnualCrop | 90.71% | 93.33% | 92.00% |
| Forest | 98.46% | 99.78% | 99.12% |
| HerbaceousVegetation | 93.89% | 92.22% | 93.05% |
| Highway | 92.92% | 90.93% | 91.91% |
| Industrial | 96.53% | 96.53% | 96.53% |
| Pasture | 91.75% | 96.33% | 93.98% |
| PermanentCrop | 90.91% | 90.67% | 90.79% |
| Residential | 97.36% | 98.44% | 97.90% |
| River | 93.30% | 89.07% | 91.13% |
| SeaLake | 99.55% | 98.44% | 98.99% |

## 9.7 Historical latency measurement

The notebook also measured model-only batch-one latency on the Tesla T4.

| Metric | Result |
|---|---:|
| Batch size | 1 |
| Median model latency | 0.921 ms |
| P95 model latency | 1.053 ms |
| Device | CUDA / Tesla T4 |
| Precision | FP32 |

These measurements exclude preprocessing and host-to-device transfer.

They therefore should not be interpreted as end-to-end application latency or
as expected latency on an arbitrary edge device.

## 9.8 Important comparability note

The CompactEuroSAT result of **94.72%** should not be directly ranked against
the final controlled Model B result as if both experiments used exactly the
same protocol.

The historical notebook used:

- A 60-epoch budget
- Batch size 128
- Its own experiment configuration
- The historical notebook's training workflow

The final shared experiments use:

- Saved project splits
- 30-epoch common protocol
- Batch size 64
- Shared CLI workflow

Therefore the CompactEuroSAT result is retained as an **additional historical
experiment** rather than being used to replace the controlled final Model B.

This distinction is important because otherwise the comparison could give the
impression that the 94.72% result was obtained under exactly the same
conditions as the final four-model comparison.

## 9.9 CompactEuroSAT evidence

The repository retains:

```text
notebooks/03_model_B_compact_cnn.ipynb
src/cnn_assignment/compact.py
tests/test_compact.py
report/audit/candidate_costs.json
```

The architecture was integrated into the shared codebase and regression tests
were added.

The CompactEuroSAT tests passed:

```text
2 / 2 tests passed
```

---

# 10. Nipun's Model D — Residual Depthwise CNN

## 10.1 Purpose

Model D was introduced as an additional architecture experiment by **Nipun**.

The model investigates the combination of:

- Depthwise-separable convolution
- Residual shortcut connections
- Small channel widths
- Global average pooling
- A compact classifier

The goal was to determine whether a residual lightweight network could provide
a useful accuracy/resource trade-off.

The code identifier is:

```text
model_d
```

and the architecture class is:

```text
ResidualDepthwiseCNN
```

## 10.2 Architecture

```text
Input
64 × 64 × 3

        ↓

3 × 3 Standard Stem
3 → 24
BatchNorm
ReLU

        ↓

Residual Depthwise-Separable Block
24 → 24
Stride = 2

        ↓

Residual Depthwise-Separable Block
24 → 48
Stride = 2

        ↓

Residual Depthwise-Separable Block
48 → 96
Stride = 2

        ↓

Global Average Pooling

        ↓

Linear
96 → 10
```

## 10.3 Residual block

Each residual block contains:

1. 3 × 3 depthwise convolution
2. BatchNorm
3. ReLU
4. 1 × 1 pointwise convolution
5. BatchNorm
6. Shortcut path
7. Final ReLU

The residual operation is:

```text
y = ReLU(F(x) + S(x))
```

where:

```text
F(x) = depthwise-separable feature path
S(x) = size/channel-matched shortcut
```

When the input and output dimensions differ, the shortcut performs the required
projection/downsampling.

## 10.4 Resource cost

| Metric | Model D |
|---|---:|
| Trainable parameters | 15,442 |
| Conv/linear MACs | 4,728,768 |
| Estimated FP32 parameter storage | 61,768 bytes |
| Recorded best-weight file | 82,062 bytes |
| Recorded weight size | 0.082062 MB |

Model D is therefore substantially smaller than the final Model B.

## 10.5 Training configuration

Model D used the project's common experiment configuration:

```text
Optimizer: Adam
Learning rate: 0.001
Epochs: 30
Batch size: 64
Seed: 42
Device: CUDA
Saved project splits
Existing project preprocessing/augmentation
Checkpoint selection: minimum validation loss
```

## 10.6 Training result

At epoch 30, the recorded training/validation results were:

| Metric | Epoch 30 |
|---|---:|
| Training loss | 0.1956 |
| Training accuracy | 93.30% |
| Validation loss | 0.1802 |
| Validation accuracy | 94.17% |

The recorded epoch-30 timing was:

| Measurement | Time |
|---|---:|
| Training | 21.7 s |
| Validation | 3.0 s |

These values describe the last recorded epoch and are not average timing
benchmarks.

## 10.7 Selected checkpoint validation

The selected checkpoint was epoch 30.

The saved checkpoint was reloaded and compared with the saved validation
history.

| Validation metric | Result |
|---|---:|
| Images | 4,050 |
| Loss | 0.180202 |
| Accuracy | 94.17% |
| Macro precision | 94.08% |
| Macro recall | 93.98% |
| Macro F1 | 93.98% |
| Selected epoch | 30 |
| Saved-history consistency | Passed |

The re-evaluated validation result matched the recorded history within the
configured numerical tolerances.

## 10.8 Final test result

The selected checkpoint was evaluated on the separate 4,050-image test set.

| Test metric | Model D |
|---|---:|
| Test accuracy | **93.63%** |
| Test loss | 0.175938 |
| Macro precision | 93.43% |
| Macro recall | 93.42% |
| Macro F1 | 93.40% |
| Complete test-pass time | 3.79 s |

The 3.79-second value includes:

- Image loading
- Device transfers
- Loss calculation
- Prediction collection
- CUDA synchronization

It is therefore **not** a pure model inference latency measurement.

## 10.9 Comparison with final Model B

| Metric | Final Model B | Model D |
|---|---:|---:|
| Test accuracy | **94.37%** | 93.63% |
| Trainable parameters | 46,373 | **15,442** |
| Saved weights | 0.196521 MB | **0.082062 MB** |
| Conv/linear MACs | 18,561,536 | **4,728,768** |

Relative to final Model B, Model D provides approximately:

- **66.7% fewer trainable parameters**
- **74.5% fewer conv/linear MACs**
- **58.2% smaller recorded weight file**
- **0.74 percentage-point lower test accuracy**

This makes Model D an additional useful point on the
accuracy/resource trade-off curve.

## 10.10 What Model D does and does not prove

The experiment demonstrates that a residual depthwise-separable architecture
can achieve useful EuroSAT accuracy with a very small model footprint.

However, the experiment does **not** prove that residual connections themselves
caused an improvement.

A proper residual ablation would require an otherwise-identical architecture
with the shortcut removed.

Therefore the result should be interpreted as:

> Model D is a compact residual depthwise CNN that achieved 93.63% test
> accuracy with 15,442 parameters and 4.73 M conv/linear MACs.

It should not be interpreted as proof that:

> residual connections are responsible for a specific accuracy improvement.

## 10.11 Model D evidence and reproducibility status

The repository contains:

```text
src/cnn_assignment/models.py
src/cnn_assignment/model_specs.py
docs/model_d_results.md
tests/test_model_d.py
report/audit/candidate_costs.json
```

The Model D architecture was independently checked for:

- Forward-pass shape
- Parameter count
- Model registration
- Resource calculation

The dedicated Model D architecture tests passed:

```text
Forward-pass shape test
Parameter-count test
```

The recorded best-weight SHA-256 is:

```text
8e5f2b0da974d9b719039e44527a3e6eb09e1b46cc9b20ded25a95d177e2e470
```

The recorded validation split SHA-256 is:

```text
a88a3e0c7d45eb44079a74245d9503457713aa8af4657ffec2f8fd9cfc32bf41
```

The recorded test split SHA-256 is:

```text
6902f9b300a12d30bd6aeffd2432f9dee36efb7083f7237c78db35121a460928
```

The original Model D checkpoint is not committed to Git because model weights
are intentionally excluded from the repository.

Therefore:

- The architecture is reproducible from the repository.
- The parameter/MAC measurements are reproducible.
- The documented experiment results are retained as evidence.
- Reproducing the exact recorded training/test result requires obtaining the
  original checkpoint and dataset artifacts.

---

# 11. Pretrained Models

Two lightweight pretrained networks were also evaluated.

These models were included to provide a reference against which the custom
networks could be compared.

The pretrained models were initialized from ImageNet weights and then
fine-tuned for the ten EuroSAT classes.

The final classifier was replaced with a ten-class classifier.

All layers were fine-tuned.

---

# 12. MobileNetV2

MobileNetV2 was selected as a pretrained reference model.

The recorded pretrained configuration used:

```text
ImageNet pretrained weights
Adam
Learning rate = 0.0001
Batch size = 64
Weight decay = 0.0001
Seed = 42
30 epoch budget
Fine-tuning of all layers
ImageNet normalization
```

The selected checkpoint was:

```text
Epoch = 27
Validation loss = 0.057098039
Validation accuracy = 98.024691%
```

Final test result:

| Metric | MobileNetV2 |
|---|---:|
| Test loss | 0.055913 |
| Test accuracy | **98.123457%** |
| Macro precision | 98.112113% |
| Macro recall | 97.948889% |
| Macro F1 | 98.024607% |
| Parameters | 2,236,682 |
| Saved weights | 9.180270 MB |
| MACs/image | 24,461,312 |

MobileNetV2 provides the highest accuracy among the controlled final
experiments.

However, it is substantially larger than the custom lightweight models.

---

# 13. ShuffleNetV2 x0.5

ShuffleNetV2 x0.5 was selected as another pretrained lightweight reference.

The recorded configuration used:

```text
ImageNet pretrained weights
Adam
Learning rate = 0.0001
Batch size = 64
Weight decay = 0.0001
Seed = 42
30 epoch budget
Fine-tuning of all layers
ImageNet normalization
```

The selected checkpoint was:

```text
Epoch = 29
Validation loss = 0.112947836
Validation accuracy = 96.197531%
```

Final test result:

| Metric | ShuffleNetV2 x0.5 |
|---|---:|
| Test loss | 0.100200 |
| Test accuracy | **96.370370%** |
| Macro precision | 96.211199% |
| Macro recall | 96.231111% |
| Macro F1 | 96.213641% |
| Parameters | 352,042 |
| Saved weights | 1.542078 MB |
| MACs/image | 3,230,848 |

ShuffleNetV2 has substantially fewer MACs than the other pretrained model, but
its parameter count and storage still exceed the custom lightweight
Model B requirement.

---

# 14. Final Controlled Test Results

The four main final test reports use:

- The same EuroSAT test split
- 4,050 test images
- 64 × 64 RGB inputs
- Seed 42
- The documented final training/evaluation workflow
- Validation-based checkpoint selection

The final controlled comparison is:

| Model | Test loss | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Standard Model A | 0.173574 | 93.901% | 93.901% | 93.702% | 93.749% |
| Final lightweight Model B | 0.179913 | **94.370%** | 94.225% | 94.251% | 94.182% |
| MobileNetV2 | 0.055913 | **98.123%** | 98.112% | 97.949% | 98.025% |
| ShuffleNetV2 x0.5 | 0.100200 | **96.370%** | 96.211% | 96.231% | 96.214% |

---

# 15. Final Resource Comparison

| Model | Parameters | Saved weights | MACs/image |
|---|---:|---:|---:|
| Standard Model A | 94,762 | 0.387218 MB | 41,288,960 |
| Final Model B | 46,373 | 0.196521 MB | 18,561,536 |
| MobileNetV2 | 2,236,682 | 9.180270 MB | 24,461,312 |
| ShuffleNetV2 x0.5 | 352,042 | 1.542078 MB | 3,230,848 |

The MAC count includes convolution and linear multiply-accumulate operations
for one 64 × 64 input.

It does not include:

- BatchNorm
- ReLU
- Pooling
- Channel shuffling
- Data loading
- Preprocessing
- Device transfer

Therefore MAC count is a computational estimate and is not equivalent to
measured inference speed.

---

# 16. Additional Model Comparison

The additional experiments are shown separately because their protocols and
evidence availability differ.

| Model | Parameters | MACs/image | Test accuracy | Status |
|---|---:|---:|---:|---|
| Final Model B (`model_c`) | 46,373 | 18,561,536 | 94.37% | Final controlled model |
| CompactEuroSAT | 11,466 | 3,882,784 | 94.72%* | Historical additional experiment |
| Model D | 15,442 | 4,728,768 | 93.63%* | Additional experiment |

\* These results should not be treated as identical-protocol replacements for
the four-model final comparison.

---

# 17. Full Custom Optimizer Experiment

A total of **12 custom CNN experiments** were completed.

There were four configurations for each of:

- Model A / `model_a`
- Initial lightweight Model B / `model_b`
- Wide lightweight Model C / `model_c`

All runs used:

```text
Seed = 42
Batch size = 64
Weight decay = 0.0001
Epoch budget = 30
```

## 17.1 Complete results

| Model | Optimizer | LR | Selected epoch | Validation loss | Validation accuracy |
|---|---|---:|---:|---:|---:|
| A | Adam | 0.001 | 30 | 0.172110 | 94.123% |
| A | SGD | 0.003 | 29 | 0.459452 | 83.630% |
| A | SGD | 0.01 | 27 | 0.379596 | 87.210% |
| A | SGD + momentum 0.9 | 0.01 | 30 | 0.174506 | 94.198% |
| Initial B | Adam | 0.001 | 27 | 0.244486 | 91.556% |
| Initial B | SGD | 0.003 | 28 | 0.651173 | 78.296% |
| Initial B | SGD | 0.01 | 28 | 0.483142 | 83.531% |
| Initial B | SGD + momentum 0.9 | 0.01 | 25 | 0.234703 | 91.827% |
| Final B / C | Adam | 0.001 | 28 | 0.177609 | 94.000% |
| Final B / C | SGD | 0.003 | 29 | 0.588112 | 79.605% |
| Final B / C | SGD | 0.01 | 29 | 0.416601 | 85.012% |
| Final B / C | SGD + momentum 0.9 | 0.01 | 29 | 0.218274 | 92.321% |

## 17.2 Interpretation

The experiments show a clear difference between Adam, plain SGD and
momentum SGD under the tested 30-epoch budget.

For the final lightweight architecture:

```text
Adam 0.001
Validation loss = 0.177609
Validation accuracy = 94.000%

SGD + momentum 0.01
Validation loss = 0.218274
Validation accuracy = 92.321%

SGD 0.01
Validation loss = 0.416601
Validation accuracy = 85.012%

SGD 0.003
Validation loss = 0.588112
Validation accuracy = 79.605%
```

Adam therefore provided the best validation-loss result for the final
lightweight architecture.

However, these experiments use one seed per configuration. They do not
establish that Adam is universally superior for the dataset or architecture.

---

# 18. Selected Model Configurations

The final selected configurations were:

| Model | Optimizer | Learning rate | Selected epoch | Validation loss | Validation accuracy |
|---|---|---:|---:|---:|---:|
| Model A | Adam | 0.001 | 30 | 0.172110 | 94.123% |
| Final Model B | Adam | 0.001 | 28 | 0.177609 | 94.000% |
| MobileNetV2 | Adam | 0.0001 | 27 | 0.057098 | 98.025% |
| ShuffleNetV2 x0.5 | Adam | 0.0001 | 29 | 0.112948 | 96.198% |

The final test set was not used for selecting these configurations.

---

# 19. Training Time

The saved training histories contain training timing information.

For the four controlled final experiments:

| Model | Selected epoch | Mean training time/epoch | Median time/epoch | Total training time |
|---|---:|---:|---:|---:|
| Model A | 30 | 26.7768 s | 25.2465 s | 13.388 min |
| Final Model B | 28 | 30.5324 s | 23.7360 s | 15.266 min |
| MobileNetV2 | 27 | 42.1579 s | 32.8674 s | 21.079 min |
| ShuffleNetV2 x0.5 | 29 | 37.2639 s | 25.8794 s | 18.632 min |

The timing values come from the saved training histories.

They include training data loading and processing and exclude validation time.

These timings were collected on the available hardware at different times and
are therefore not treated as a controlled hardware benchmark.

---

# 20. Per-Class Test Results

The following results are from the four controlled final test reports.

The support values are the same across all models.

| Class | Support | Model A P/R/F1 | Final Model B P/R/F1 | MobileNetV2 P/R/F1 | ShuffleNetV2 P/R/F1 |
|---|---:|---|---|---|---|
| AnnualCrop | 450 | 95.91 / 88.67 / 92.15 | 92.46 / 92.67 / 92.56 | 96.92 / 98.00 / 97.46 | 95.10 / 94.89 / 94.99 |
| Forest | 450 | 97.18 / 99.56 / 98.35 | 98.23 / 98.89 / 98.56 | 98.68 / 99.78 / 99.23 | 97.81 / 99.33 / 98.57 |
| HerbaceousVegetation | 450 | 88.84 / 95.56 / 92.08 | 90.87 / 92.89 / 91.87 | 97.58 / 98.67 / 98.12 | 96.00 / 96.00 / 96.00 |
| Highway | 375 | 95.42 / 88.80 / 91.99 | 95.56 / 86.13 / 90.60 | 98.12 / 97.33 / 97.72 | 92.88 / 93.87 / 93.37 |
| Industrial | 375 | 97.50 / 93.60 / 95.51 | 96.34 / 98.13 / 97.23 | 98.92 / 97.33 / 98.12 | 98.13 / 98.13 / 98.13 |
| Pasture | 300 | 94.39 / 95.33 / 94.86 | 91.28 / 97.67 / 94.36 | 97.27 / 95.00 / 96.12 | 93.85 / 96.67 / 95.24 |
| PermanentCrop | 375 | 85.64 / 89.07 / 87.32 | 92.00 / 85.87 / 88.83 | 97.81 / 95.20 / 96.49 | 93.22 / 91.73 / 92.47 |
| Residential | 450 | 94.68 / 98.89 / 96.74 | 98.66 / 98.00 / 98.33 | 97.82 / 99.78 / 98.79 | 99.11 / 99.33 / 99.22 |
| River | 375 | 90.79 / 89.33 / 90.05 | 88.19 / 93.60 / 90.82 | 98.66 / 98.40 / 98.53 | 96.67 / 92.80 / 94.69 |
| SeaLake | 450 | 98.66 / 98.22 / 98.44 | 98.67 / 98.67 / 98.67 | 99.34 / 100.00 / 99.67 | 99.33 / 99.56 / 99.45 |

Values are:

```text
Precision / Recall / F1
```

and are shown as percentages.

---

# 21. Confusion Analysis

The largest directional errors in the controlled final test results were:

| Model | True class → Predicted class | Count | Error relative to true class |
|---|---|---:|---:|
| Model A | AnnualCrop → PermanentCrop | 30 | 6.67% |
| Model A | PermanentCrop → HerbaceousVegetation | 28 | 7.47% |
| Model A | Industrial → Residential | 16 | 4.27% |
| Model A | Highway → PermanentCrop | 12 | 3.20% |
| Model A | Highway → River | 10 | 2.67% |
| Final Model B | PermanentCrop → HerbaceousVegetation | 25 | 6.67% |
| Final Model B | Highway → River | 20 | 5.33% |
| Final Model B | PermanentCrop → AnnualCrop | 12 | 3.20% |
| Final Model B | AnnualCrop → PermanentCrop | 10 | 2.22% |
| Final Model B | PermanentCrop → River | 9 | 2.40% |
| MobileNetV2 | Industrial → Residential | 7 | 1.87% |
| MobileNetV2 | Pasture → AnnualCrop | 7 | 2.33% |
| MobileNetV2 | PermanentCrop → AnnualCrop | 6 | 1.60% |
| MobileNetV2 | PermanentCrop → HerbaceousVegetation | 5 | 1.33% |
| MobileNetV2 | AnnualCrop → Pasture | 4 | 0.89% |
| ShuffleNetV2 | River → Highway | 15 | 4.00% |
| ShuffleNetV2 | PermanentCrop → HerbaceousVegetation | 13 | 3.47% |
| ShuffleNetV2 | AnnualCrop → PermanentCrop | 12 | 2.67% |
| ShuffleNetV2 | Highway → PermanentCrop | 9 | 2.40% |
| ShuffleNetV2 | AnnualCrop → Pasture | 8 | 1.78% |

PermanentCrop appears frequently in the largest error directions.

For Model A, the largest error direction is:

```text
AnnualCrop → PermanentCrop
```

with 30 errors.

For final Model B, the largest error direction is:

```text
PermanentCrop → HerbaceousVegetation
```

with 25 errors.

For ShuffleNetV2, the largest directional error is:

```text
River → Highway
```

with 15 errors.

These observations describe the saved predictions. They do not establish the
visual cause of each error without inspecting the corresponding images.

---

# 22. Accuracy, Memory and Computation Discussion

## 22.1 Final Model B versus Model A

Final Model B achieved:

```text
94.37% test accuracy
```

compared with:

```text
93.90% for Model A
```

while using:

```text
46,373 parameters
```

instead of:

```text
94,762 parameters
```

and:

```text
18.56 M MACs/image
```

instead of:

```text
41.29 M MACs/image
```

This demonstrates a useful accuracy/resource trade-off in the recorded
experiment.

However, the accuracy difference is only approximately 0.47 percentage points
and both models were evaluated using a single seed.

Therefore it should not be described as statistically significant or guaranteed
to repeat across other seeds.

---

# 23. Pretrained versus Custom Models

MobileNetV2 achieved the highest accuracy:

```text
98.12%
```

but uses:

```text
2,236,682 parameters
9.18 MB saved weights
```

The final custom Model B achieved:

```text
94.37%
```

with:

```text
46,373 parameters
0.1965 MB saved weights
```

Therefore the pretrained model provides substantially higher accuracy, but at
a much larger model footprint.

ShuffleNetV2 provides another interesting trade-off:

```text
96.37% accuracy
352,042 parameters
1.54 MB weights
3.23 M MACs
```

Its MAC count is substantially lower than final Model B, but its parameter
count and storage are significantly larger.

This demonstrates why a single metric such as parameter count is not enough
to describe model efficiency.

---

# 24. Important Resource-Metric Distinctions

The project intentionally reports several different resource measures.

## Parameters

Trainable parameter count measures the number of learned scalar values.

It does not directly measure:

- Runtime
- Energy
- Activation memory
- Hardware efficiency

## Saved weights

Saved weight size measures the actual serialized model file.

The file includes buffers and serialization overhead, so:

```text
file size ≠ parameter count × 4 bytes
```

exactly.

## MACs

MACs represent convolution/linear multiply-accumulate operations.

The project counts:

```text
Convolution MACs
+
Linear MACs
```

for one:

```text
1 × 3 × 64 × 64
```

input.

MACs do not directly establish real-world inference speed.

A model with fewer MACs can still be slower on a particular device because of
memory access, operator support, parallelism and hardware implementation.

## Measured latency

Latency measurements are reported separately when available.

For example, CompactEuroSAT's historical Tesla T4 measurement is model-only
batch-one latency.

It should not be compared directly with the full test-pass timing of Model D,
because the measurement scopes are different.

---

# 25. Hardware-Aware Activation Function

The custom models use ReLU:

```text
ReLU(x) = max(0, x)
```

ReLU was selected because it is computationally simple.

It primarily requires:

- A comparison with zero
- A clamp of negative values to zero

It does not require:

- Exponentials
- Division
- Complex nonlinear functions

This makes ReLU suitable for lightweight neural-network implementations.

The implementation also uses in-place ReLU where appropriate, which can reduce
additional activation storage.

However, in-place ReLU does not eliminate the need to store the feature maps
themselves.

The pretrained architectures retain their original activation design.

This is a hardware-aware architectural justification rather than a claim that
ReLU has been experimentally proven to provide lower power consumption on a
specific edge device.

---

# 26. Evidence and Verification

The project maintains evidence at several levels.

## 26.1 Dataset evidence

The repository retains:

- Dataset preparation code
- Saved train/validation/test CSV files
- Class ordering
- Split information
- Dataset fingerprints
- Image/path checks
- Saved normalization information

## 26.2 Model implementation evidence

The repository contains source implementations for:

```text
StandardCNN
LightweightCNN
WideLightweightCNN
CompactEuroSAT
ResidualDepthwiseCNN
```

as well as the pretrained model integration.

## 26.3 Training evidence

For the controlled custom experiments, the repository retains:

- Model configuration
- Optimizer
- Learning rate
- Seed
- Batch size
- Weight decay
- Epoch history
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Training timing

The complete 12-run custom experiment record is documented in:

```text
docs/custom_cnn_results.md
```

## 26.4 Final test evidence

For the four controlled final models, the repository retains or indexes:

- Test metrics
- Per-class metrics
- Confusion matrices
- Normalized confusion matrices
- Test predictions
- Saved configurations
- Training histories
- Selected checkpoints where available
- Resource measurements

## 26.5 CompactEuroSAT evidence

The repository retains:

```text
notebooks/03_model_B_compact_cnn.ipynb
src/cnn_assignment/compact.py
tests/test_compact.py
report/audit/candidate_costs.json
```

The historical notebook contains the reported training, validation and test
results.

The integrated architecture also has regression tests.

## 26.6 Model D evidence

The repository retains:

```text
src/cnn_assignment/models.py
src/cnn_assignment/model_specs.py
docs/model_d_results.md
tests/test_model_d.py
report/audit/candidate_costs.json
```

The Model D documentation records:

- Architecture
- Training configuration
- Validation results
- Test results
- Parameter count
- MAC count
- Checkpoint hash
- Dataset hashes
- Timing information

The architecture tests passed.

---

# 27. Repository-Level Verification

The project includes automated tests and analysis checks.

The verification work covers:

- Model architecture construction
- Forward-pass shapes
- Parameter counts
- MAC calculations
- Checkpoint loading
- Saved result consistency
- Final result calculations
- Notebook analysis
- Source preservation
- Merge-conflict checks
- Formatting checks

The CompactEuroSAT regression tests passed:

```text
2 / 2
```

The Model D architecture checks passed:

```text
Forward-pass shape test
Parameter-count test
```

The final-result audit also checks saved reports and histories without
performing new model training or test inference.

---

# 28. Result Recalculation and Audit

The saved test predictions were checked against the saved test CSV ordering.

The project audit reconstructed:

- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion counts
- Normalized confusion matrices

The saved histories were also used to determine selected epochs.

The selection rule is:

```text
Minimum validation loss
```

If two epochs have identical loss, the first matching epoch is retained.

The test set was not used to select the checkpoint.

This separation is important because repeatedly selecting architectures or
hyperparameters using the test set would make the final test accuracy an
optimistically biased estimate.

---

# 29. Single-Seed Limitation

The controlled experiments use:

```text
Seed = 42
```

for the recorded runs.

Therefore the results demonstrate the behaviour of the tested configurations
on the saved split under this seed.

They do not establish:

- Multi-seed mean accuracy
- Standard deviation
- Statistical significance
- General robustness across random initializations

Repeated experiments with multiple seeds would strengthen the conclusions.

---

# 30. Model D Limitation

The Model D experiment does not include a matched ablation without residual
connections.

Therefore the experiment cannot isolate the effect of the residual shortcut.

A stronger future experiment would compare:

```text
ResidualDepthwiseCNN
vs.
Same architecture without residual shortcuts
```

using exactly the same:

- Dataset
- Split
- Seed
- Optimizer
- Learning rate
- Epoch budget
- Batch size

That would allow the residual connection itself to be evaluated more directly.

---

# 31. Edge-Device Limitations

The project reports parameter count, model storage and MACs.

It does not yet establish actual performance on a specific edge device.

Important future measurements would include:

- End-to-end inference latency
- Peak RAM usage
- Activation memory
- Energy per inference
- Power consumption
- Quantized model size
- INT8 inference speed
- Target-device throughput

The current MAC measurements should therefore be interpreted as architectural
computational estimates rather than direct hardware benchmarks.

---

# 32. Reproducibility

## 32.1 Python environment

The main recorded shared workflow uses:

```text
Python 3.11
PyTorch 2.5.1
Torchvision 0.20.1
```

The complete dependency list is provided in:

```text
requirements.txt
```

The exact historical CompactEuroSAT notebook used a different recorded
environment and is therefore documented separately.

## 32.2 Dataset

The RGB EuroSAT dataset must be obtained separately.

Expected structure:

```text
data/raw/eurosat/
    EuroSAT.zip
    2750/
        AnnualCrop/
        Forest/
        HerbaceousVegetation/
        Highway/
        Industrial/
        Pasture/
        PermanentCrop/
        Residential/
        River/
        SeaLake/

data/splits/
    train.csv
    validation.csv
    test.csv
```

The committed split CSVs should be reused.

## 32.3 Model weights

Raw model weights are not committed to Git.

This is intentional.

The repository therefore contains:

- Source code
- Configurations
- Histories
- Reports
- Analysis
- Tests

but not every trained checkpoint.

Exact checkpoint reproduction requires obtaining the corresponding local
checkpoint artifacts.

---

# 33. Setup

Create a Python environment:

```powershell
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the recorded PyTorch versions:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Install the remaining dependencies:

```powershell
python -m pip install numpy==1.26.4 pandas matplotlib Pillow tqdm nbclient nbformat ipykernel jupyterlab
```

Install the project:

```powershell
python -m pip install --no-deps -e .
```

Check the command-line interface:

```powershell
python -m cnn_assignment --help
```

For a platform-specific CUDA installation, use the appropriate PyTorch
installation instructions for the available hardware.

---

# 34. Dataset Preparation

The project does not automatically download the complete dataset.

After obtaining the RGB EuroSAT dataset and placing it in the expected
directory, run:

```powershell
python -m cnn_assignment prepare
```

To check image integrity:

```powershell
python -m cnn_assignment prepare --check-images
```

The preparation command checks:

- Existing images
- Saved split integrity
- Configuration
- Dataset paths

It does not automatically generate new splits or recalculate the saved
normalization.

---

# 35. Training Commands

## Model A

```powershell
python -m cnn_assignment train --model model_a --optimizer adam --lr 0.001 --run-name model_a_adam_repeat1
```

## Initial lightweight Model B

```powershell
python -m cnn_assignment train --model model_b --optimizer adam --lr 0.001 --run-name model_b_adam_repeat1
```

## Final lightweight Model B

Remember that the final assignment Model B uses the source identifier
`model_c`.

```powershell
python -m cnn_assignment train --model model_c --optimizer adam --lr 0.001 --run-name model_c_adam_repeat1
```

## CompactEuroSAT

```powershell
python -m cnn_assignment train --model compact_eurosat --optimizer adam --lr 0.001 --run-name compact_eurosat_repeat1
```

## Model D

```powershell
python -m cnn_assignment train --model model_d --optimizer adam --lr 0.001 --epochs 30 --device auto --run-name model_d_residual_adam_v1
```

## MobileNetV2

```powershell
python -m cnn_assignment train --model mobilenet_v2 --optimizer adam --lr 0.0001 --run-name mobilenet_v2_adam_repeat1
```

## ShuffleNetV2 x0.5

```powershell
python -m cnn_assignment train --model shufflenet_v2_x0_5 --optimizer adam --lr 0.0001 --run-name shufflenet_v2_x0_5_adam_repeat1
```

---

# 36. Validation

Validation can be performed using:

```powershell
python -m cnn_assignment validate --run outputs/custom_cnn/model_c_adam_lr0.001 --compare-history
```

For Model D:

```powershell
python -m cnn_assignment validate --run outputs/custom_cnn/model_d_residual_adam_v1 --compare-history
```

Validation uses the validation split and can compare the evaluated result with
the saved training history.

---

# 37. Final Test Evaluation

Final test evaluation is intentionally separate from validation.

For final Model B:

```powershell
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_c_adam_lr0.001
```

For Model A:

```powershell
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_a_adam_lr0.001
```

For Model D:

```powershell
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_d_residual_adam_v1
```

The final test set should only be used after the model/configuration has been
selected using validation data.

---

# 38. Plotting

Plot a specific run:

```powershell
python -m cnn_assignment plot --run outputs/custom_cnn/model_c_adam_lr0.001
```

Plot the default saved results:

```powershell
python -m cnn_assignment plot
```

Plot pretrained runs:

```powershell
python -m cnn_assignment plot --runs-root outputs/pretrained_cnn
```

Plot additional runs:

```powershell
python -m cnn_assignment plot --additional-runs-root outputs/pretrained_cnn
```

---

# 39. Resume Training

Only incomplete runs can be resumed.

Example:

```powershell
python -m cnn_assignment train --resume outputs/custom_cnn/model_c_adam_repeat1
```

Resume requires the full epoch-boundary checkpoint.

Older saved `best_weights.pt` files do not contain the full optimizer/RNG/epoch
state required for exact training resume.

---

# 40. Automated Tests

The final-result checks can be run using:

```powershell
python -m unittest discover -s tests -p test_final_results.py -v
```

The broader test suite is:

```powershell
python -m unittest discover -s tests -v
```

The tests are designed to verify saved artifacts and model behaviour without
unnecessarily launching full training runs.

---

# 41. Project Structure

```text
EN3150-Group-Assignment-03-CNN/
│
├── data/
│   ├── raw/
│   │   └── eurosat/
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── docs/
│   ├── custom_cnn_results.md
│   ├── final_results.md
│   ├── model_d_results.md
│   ├── pretrained_model_selection.md
│   └── troubleshooting.md
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_A_standard_cnn.ipynb
│   └── 03_model_B_compact_cnn.ipynb
│
├── outputs/
│   ├── custom_cnn/
│   ├── pretrained_cnn/
│   ├── reports/
│   └── README.md
│
├── report/
│   ├── main.tex
│   ├── figures/
│   ├── tables/
│   └── audit/
│
├── src/
│   └── cnn_assignment/
│       ├── compact.py
│       ├── models.py
│       ├── model_specs.py
│       ├── training.py
│       ├── evaluation.py
│       ├── data.py
│       └── cli.py
│
├── tests/
│   ├── test_compact.py
│   ├── test_model_d.py
│   ├── test_final_results.py
│   ├── test_pretrained.py
│   └── other project tests
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── test.ipynb
```

---

# 42. Evidence File Index

## Main result documentation

```text
docs/custom_cnn_results.md
```

Contains the complete 12-run custom CNN validation experiment.

```text
docs/final_results.md
```

Contains the consolidated final test results, per-class metrics, confusion
analysis, resource comparisons and assignment audit.

```text
docs/model_d_results.md
```

Contains the detailed Model D experiment and evidence record.

```text
docs/pretrained_model_selection.md
```

Contains pretrained-model selection and initialization information.

```text
outputs/README.md
```

Contains the saved-result index and artifact paths.

---

# 43. Important Artifact Policy

Raw images and trained model weights are not committed to Git.

The repository therefore intentionally separates:

### Version-controlled evidence

- Source code
- Tests
- Configuration files
- Training histories
- Reports
- Analysis
- Saved split CSVs
- Documentation
- Notebook source/output where appropriate

### Local-only artifacts

- Raw dataset images
- Large model weights
- Full resume checkpoints
- Temporary reports
- Source ImageNet checkpoint files
- Local archives
- Personal notes

This keeps the Git repository manageable while retaining the numerical and
experimental evidence required to understand the project.

---

# 44. Assignment Requirement Audit

The following table records the current project status.

| Requirement | Status | Evidence |
|---|---|---|
| Dataset other than CIFAR-10 | Complete | EuroSAT RGB |
| Dataset ≤64 × 64 | Complete | 64 × 64 RGB |
| 70/15/15 split | Complete | 18,900 / 4,050 / 4,050 |
| Standard custom CNN | Complete | Model A |
| Depthwise-separable custom CNN | Complete | Initial B and final B |
| Lightweight model ≤100k parameters | Complete | Final B = 46,373 |
| Activation-function justification | Complete | ReLU hardware-aware discussion |
| Optimizer investigation | Complete | 12 custom experiments |
| SGD comparison | Complete | Saved SGD runs |
| Momentum comparison | Complete | Saved momentum runs |
| ≥20 epochs custom training | Complete | 30-epoch recorded runs |
| Training/validation histories | Complete | Saved `history.csv` files |
| Final test accuracy | Complete | Four final test reports |
| Confusion matrices | Complete | Saved final reports |
| Precision/recall/F1 | Complete | Saved final reports |
| Custom-model size comparison | Complete | Parameter/MAC/storage tables |
| Two pretrained networks | Complete | MobileNetV2 + ShuffleNetV2 |
| Fine-tuning pretrained networks | Complete | All-layer fine-tuning |
| Pretrained test evaluation | Complete | Final test reports |
| Final custom/pretrained comparison | Complete | Final results section |
| Additional lightweight exploration | Complete | CompactEuroSAT + Model D |
| Model D integration | Complete | `model_d` implementation |
| CompactEuroSAT integration | Complete | `compact_eurosat` implementation |
| Model architecture tests | Complete | Compact + Model D checks |
| Final PDF report | To be confirmed | Must be finalized before submission |
| Dataset information entered in assignment sheet | To be confirmed | Confirmation/receipt required |
| GitHub/profile evidence | To be confirmed | Include required profile/repository information |
| Final Moodle submission | To be confirmed | Must be completed separately |
| Clean-machine full reproduction | Not yet established | Environment is documented, but exact clean-machine reproduction has not been certified |

---

# 45. Interpretation of the Main Results

The main controlled experiment shows the following:

### Model A

```text
93.90% accuracy
94,762 parameters
41.29 M MACs
```

This provides the standard custom CNN reference.

### Final Model B

```text
94.37% accuracy
46,373 parameters
18.56 M MACs
```

It provides a better recorded accuracy/resource balance than Model A while
remaining well below the 100,000-parameter requirement.

### MobileNetV2

```text
98.12% accuracy
2.24 M parameters
24.46 M MACs
9.18 MB weights
```

It achieves the highest accuracy but is much larger in parameter count and
storage.

### ShuffleNetV2 x0.5

```text
96.37% accuracy
352,042 parameters
3.23 M MACs
1.54 MB weights
```

It achieves a strong accuracy/computation trade-off but does not satisfy the
same strict lightweight parameter budget as the custom final Model B.

### CompactEuroSAT

```text
94.72% historical test accuracy
11,466 parameters
3.88 M MACs
```

It demonstrates that an even smaller architecture can achieve strong accuracy,
but its historical experiment protocol differs from the final controlled
pipeline.

### Model D

```text
93.63% test accuracy
15,442 parameters
4.73 M MACs
```

It provides another compact point in the accuracy/resource trade-off and
demonstrates the feasibility of combining residual shortcuts with
depthwise-separable convolution.

---

# 46. Overall Conclusion

The experiments show that there is no single model that dominates every
resource and accuracy metric.

The pretrained MobileNetV2 achieves the highest recorded accuracy, but it is
substantially larger than the custom lightweight networks.

The final custom Model B provides a strong compromise:

```text
94.37% test accuracy
46,373 parameters
18.56 M MACs
0.1965 MB saved weights
```

It improves slightly over the standard Model A while using substantially fewer
parameters, MACs and saved-weight storage.

The additional CompactEuroSAT experiment demonstrates another possible
direction toward extremely small models:

```text
11,466 parameters
3.88 M MACs
94.72% historical test accuracy
```

However, this result was produced using a different historical experiment
protocol and therefore is not treated as a direct replacement for the final
controlled Model B.

Model D provides another compact architecture:

```text
15,442 parameters
4.73 M MACs
93.63% test accuracy
```

Its residual design gives another architecture point for analysing the
accuracy/resource trade-off, but a matched non-residual ablation would be
required to isolate the contribution of the residual connections.

Overall, the project demonstrates that depthwise-separable and compact CNN
architectures can substantially reduce model size and estimated convolutional
computation while retaining useful EuroSAT classification performance.

The results also demonstrate why model selection for resource-constrained
deployment should consider several factors simultaneously:

```text
Accuracy
+
Parameter count
+
Storage
+
Computation
+
Memory
+
Latency
+
Hardware characteristics
```

A lower parameter count or MAC count alone does not guarantee faster or more
energy-efficient inference on a particular edge device.

---

# 47. Final Evidence Summary

The final project contains evidence covering:

```text
Dataset
    ↓
Saved splits
    ↓
Model architectures
    ↓
Parameter calculations
    ↓
MAC calculations
    ↓
Optimizer experiments
    ↓
Training histories
    ↓
Validation selection
    ↓
Checkpoint verification
    ↓
Final test evaluation
    ↓
Per-class metrics
    ↓
Confusion matrices
    ↓
Resource comparison
    ↓
Additional model experiments
    ↓
Automated architecture tests
    ↓
Result/audit documentation
```

The repository therefore retains not only the final accuracy numbers but also
the experimental path used to reach the final architecture and the evidence
supporting the reported results.

---

# 48. Further Documentation

For more detailed information, see:

```text
docs/custom_cnn_results.md
docs/final_results.md
docs/model_d_results.md
docs/pretrained_model_selection.md
outputs/README.md
docs/troubleshooting.md
```

The notebooks are also retained for the original dataset/model workflows and
historical experiment records.

```text
notebooks/01_dataset_preparation.ipynb
notebooks/02_model_A_standard_cnn.ipynb
notebooks/03_model_B_compact_cnn.ipynb
```

---

# 49. Final Note on Reproducibility and Claims

The numerical results in this README are recorded experimental results.

Where results were obtained under different protocols, they are explicitly
identified rather than being presented as directly comparable.

Where a result depends on a local checkpoint that is not committed to Git, the
README identifies that limitation.

Where a conclusion has not been experimentally isolated, such as the specific
effect of residual connections, the README does not claim causality.

This distinction is maintained deliberately so that the project documents both
the successful results and the limitations of the evidence.