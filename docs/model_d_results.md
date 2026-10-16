# Model D — Residual Depthwise CNN: Experiment Results

## 1. Purpose

Model D is an additional experimental CNN for **EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification**. It investigates whether depthwise-separable convolution combined with residual shortcuts can reduce model storage and convolution/linear computation while retaining useful accuracy on **EuroSAT RGB** land-cover classification.

**Status:** Architecture implemented; 30-epoch training completed; best checkpoint validated; final test evaluation completed.

This is an **additional experiment**, not a replacement for the assignment's selected Model B (`model_c` in the source code).

## 2. Dataset and evaluation protocol

- **Dataset:** EuroSAT RGB, 27,000 images, 10 land-cover classes, each 64 × 64 × 3.
- **Saved stratified splits:** 18,900 training images (70%), 4,050 validation images (15%), and 4,050 test images (15%).
- **Class order:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.
- **Inputs and preprocessing:** The project's existing training augmentations and saved RGB mean/standard deviation were reused. The test set was evaluated without training augmentation.
- **Checkpoint selection:** Minimum validation loss; selected epoch **30**.

The project-provided split CSVs were reused rather than regenerated. Final test evaluation was performed on the selected checkpoint; test accuracy was **not** used to select that checkpoint.

## 3. Architecture

The new `ResidualDepthwiseCNN` is registered as **`model_d`** in `src/cnn_assignment/models.py` and `src/cnn_assignment/model_specs.py`.

| Stage | Operation | Output channels |
| --- | --- | ---: |
| Input | 64 × 64 RGB image | 3 |
| Stem | 3 × 3 standard convolution, BatchNorm, ReLU | 24 |
| Block 1 | Residual depthwise-separable block, stride 2 | 24 |
| Block 2 | Residual depthwise-separable block, stride 2 | 48 |
| Block 3 | Residual depthwise-separable block, stride 2 | 96 |
| Pooling | Global average pooling | 96 features |
| Classifier | Fully connected layer | 10 logits |

Each residual depthwise-separable block uses a 3 × 3 **depthwise convolution**, a 1 × 1 **pointwise convolution**, normalization, and a shortcut path, followed by ReLU. A channel-matching/downsampling shortcut is used where needed.

The shortcut adds the transformed path to a compatible version of the input:

\[
 y = \operatorname{ReLU}(F(x) + S(x)).
\]

Depthwise-separable convolutions reduce computation relative to full convolutions, while shortcut connections provide an additional path for feature and gradient flow. Whether they improve accuracy must be established experimentally, rather than assumed.

### Model cost (measured by the project's tooling)

| Metric | Model D |
| --- | ---: |
| Total/trainable parameters | **15,442** |
| Estimated FP32 parameter bytes | **61,768 bytes** |
| Actual saved `best_weights.pt` file | **82,062 bytes (0.082062 MB)** |
| Convolution/linear MACs per 64 × 64 image | **4,728,768 (~4.73 M)** |

**Measurement scope:** The project's MAC counter includes convolution and linear-layer multiply-accumulates; it omits other operations and does not measure device inference latency. The saved checkpoint is larger than the raw FP32 parameters because it includes buffers and serialization overhead.

## 4. Training configuration and outcome

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 30 |
| Batch size | 64 |
| Random seed | 42 (project CLI default) |
| Device | CUDA |
| Run name | `model_d_residual_adam_v1` |
| Checkpoint rule | Lowest validation loss |

The final training epoch reported:

| Epoch 30 metric | Value |
| --- | ---: |
| Training loss | 0.1956 |
| Training accuracy | 93.30% |
| Validation loss | 0.1802 |
| Validation accuracy | 94.17% |
| Training time, epoch 30 only | 21.7 s |
| Validation time, epoch 30 only | 3.0 s |

These timing values describe the **last epoch**, not the average time over 30 epochs.

## 5. Best-checkpoint validation

The saved checkpoint was reloaded and checked against its recorded validation history:

| Validation metric | Result |
| --- | ---: |
| Images | 4,050 |
| Loss | **0.180202** |
| Accuracy | **94.17%** |
| Macro precision | **94.08%** |
| Macro recall | **93.98%** |
| Macro F1 | **93.98%** |
| Selected epoch | **30** |
| Saved-history consistency | **Passed (`matches: true`)** |

The re-evaluated validation loss and accuracy matched the saved epoch-30 history within the configured numerical tolerances.

## 6. Final test results

The selected best-validation checkpoint was evaluated on the separate 4,050-image test split:

| Test metric | Result |
| --- | ---: |
| Accuracy | **93.63%** |
| Loss | **0.175938** |
| Macro precision | **93.43%** |
| Macro recall | **93.42%** |
| Macro F1 | **93.40%** |
| Evaluation time for entire test pass | **3.79 s** |

The reported pass time includes image loading, device transfers, loss calculation, and prediction collection, with CUDA synchronization. It should **not** be interpreted as pure model inference time or edge-device performance.

### Reproducibility identifiers

- Best weights SHA-256: `8e5f2b0da974d9b719039e44527a3e6eb09e1b46cc9b20ded25a95d177e2e470`
- Validation split SHA-256: `a88a3e0c7d45eb44079a74245d9503457713aa8af4657ffec2f8fd9cfc32bf41`
- Test split SHA-256: `6902f9b300a12d30bd6aeffd2432f9dee36efb7083f7237c78db35121a460928`

## 7. Comparison with existing custom CNNs

The comparison below uses **test accuracy** for every model. Model A and Model C figures come from the team's main README; Model D figures come from the completed experiment.

| Metric | Model A (standard) | Model C (selected lightweight Model B) | Model D (residual lightweight) |
| --- | ---: | ---: | ---: |
| Test accuracy | **93.90%** | **94.37%** | **93.63%** |
| Trainable parameters | 94,762 | 46,373 | **15,442** |
| Saved weights (MB) | 0.387218 | 0.196521 | **0.082062** |
| Conv/linear MACs per image | 41,288,960 | 18,561,536 | **4,728,768** |

Compared with **Model C**, Model D offers:

- **66.7% fewer trainable parameters**.
- **74.5% fewer convolution/linear MACs per image**.
- **58.2% smaller saved weights file**.
- **0.74 percentage points lower test accuracy** (93.63% vs. 94.37%).

Compared with **Model A**, Model D has 0.27 percentage points lower test accuracy while requiring substantially fewer parameters and MACs.

**Interpretation:** The experiment supports Model D as a compact alternative when memory and estimated convolution/linear computation matter more than obtaining the highest accuracy. It does **not** show that residual shortcuts specifically caused an improvement, because there is no otherwise-identical non-residual ablation here. Nor do the MAC counts establish real-world latency or power consumption.

## 8. Limitations and next steps

- These are results from a **single recorded seed/run**, not a multi-seed robustness study.
- A matched ablation without residual shortcuts would help isolate the effect of the shortcut design.
- Actual latency, peak RAM, and energy should be measured on intended edge hardware before making deployment claims.
- Class-specific errors should be discussed using the saved per-class report and confusion matrix; these details have not been summarized in this document.
- Further hyperparameter or architecture selection should use **validation data**, without repeatedly optimizing against the final test set.

## 9. Commands and artifact locations

Run from the repository root with the project environment activated and EuroSAT RGB images at the configured data root:

```powershell
python -m cnn_assignment train --model model_d --optimizer adam --lr 0.001 --epochs 30 --device auto --run-name model_d_residual_adam_v1
python -m cnn_assignment validate --run outputs/custom_cnn/model_d_residual_adam_v1 --compare-history
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/model_d_residual_adam_v1
python -m cnn_assignment plot --run outputs/custom_cnn/model_d_residual_adam_v1
```

**Existing local artifacts:**

- Run history/config/checkpoints: `outputs/custom_cnn/model_d_residual_adam_v1/`
- Validation report: `outputs/reports/validate_20261015T130850849632Z/`
- Test report: `outputs/reports/evaluate-test_20261015T130916906896Z/`

These artifact paths are from the experiment machine. Raw images and model checkpoints are intentionally not committed in the group's Git repository; preserve them locally or share them through the team's agreed artifact process.

## 10. Summary

Model D achieved **93.63% test accuracy** and **93.40% macro F1** on EuroSAT RGB with **15,442 trainable parameters**, an **82 KB saved state dictionary**, and **4.73 M convolution/linear MACs per image**. Relative to the group's selected lightweight Model C, this substantially reduces estimated model resource requirements at a modest cost in test accuracy. The result strengthens the assignment's discussion of accuracy–memory–computation trade-offs in resource-constrained image classification.
