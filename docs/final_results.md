# My final assignment results

I consolidate the four existing final test reports and the saved histories. I did not rerun models, train, or evaluate the test set during this consolidation. These are EuroSAT test results, separate from the validation results used for selection and the published ImageNet results in [my pretrained selection notes](pretrained_model_selection.md).

In this report, **Model A is `model_a`** and **final lightweight Model B is `model_c`**. Existing `model_b` remains my initial lightweight design baseline. Its 12,965 parameters and 5,141,760 MACs provide design context; it is not one of the four final test reports. I keep every code identifier and saved path unchanged. Historical notebook sections use the original A/B/C labels.

I selected `model_c` using validation results. Its 46,373 trainable parameters satisfy the 100,000 cap. Each final run has one seed (42) and 30 completed epochs. I selected each checkpoint by minimum validation loss, so its validation accuracy need not be its maximum-accuracy epoch. I keep the selected configurations fixed after viewing the test results.

## Final test metrics

All four reports use the same 4,050 test images, class order and 64 x 64 RGB inputs. Macro averages give each class equal weight. Percentages below are computed from the saved full-precision values; display rounding does not drive calculations.

| Model | Loss | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- | --- |
| Model A (standard) | 0.173574 | 93.901235 | 93.900989 | 93.702222 | 93.749068 |
| Model B (final lightweight) | 0.179913 | 94.370370 | 94.225456 | 94.251111 | 94.182223 |
| MobileNetV2 | 0.055913 | 98.123457 | 98.112113 | 97.948889 | 98.024607 |
| ShuffleNetV2 x0.5 | 0.100200 | 96.370370 | 96.211199 | 96.231111 | 96.213641 |

## Parameters, storage and computation

| Model | Total / trainable parameters | Actual weights bytes | KB | MB | MiB | 64 x 64 conv/linear MACs |
| --- | --- | --- | --- | --- | --- | --- |
| Model A (standard) | 94,762 / 94,762 | 387,218 | 387.218 | 0.387218 | 0.369280 | 41,288,960 |
| Model B (final lightweight) | 46,373 / 46,373 | 196,521 | 196.521 | 0.196521 | 0.187417 | 18,561,536 |
| MobileNetV2 | 2,236,682 / 2,236,682 | 9,180,270 | 9180.270 | 9.180270 | 8.754988 | 24,461,312 |
| ShuffleNetV2 x0.5 | 352,042 / 352,042 | 1,542,078 | 1542.078 | 1.542078 | 1.470640 | 3,230,848 |

KB = 1,000 bytes, MB = 1,000,000 bytes, KiB = 1,024 bytes and MiB = 1,048,576 bytes. These are the actual `best_weights.pt` sizes, including BatchNorm buffers and serialization overhead, confirmed against file sizes and SHA-256 hashes. They exclude full resume checkpoints, source ImageNet weights and optimizer state. Estimated FP32 parameter-only storage for Model A / B is 379.048 / 185.492 KB (four bytes per parameter); it excludes those buffers and overhead.

MACs count convolution and linear multiply-accumulates for one 64 x 64 image. They exclude normalization, activations, pooling, channel shuffling and data loading. MACs are computation estimates, not measured inference speed or total memory use.

## Selected runs and training time

| Model | Selected epoch | Validation loss | Validation accuracy (%) | Mean train s/epoch | Mean excluding first (s) | Median (s) | Total training (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model A (standard) | 30 | 0.172110070 | 94.123457 | 26.776831297 | 25.266525355 | 25.246450300 | 13.388415648 |
| Model B (final lightweight) | 28 | 0.177608554 | 94.000000 | 30.532371087 | 25.623894797 | 23.736000600 | 15.266185543 |
| MobileNetV2 | 27 | 0.057098039 | 98.024691 | 42.157938063 | 36.505410586 | 32.867405200 | 21.078969032 |
| ShuffleNetV2 x0.5 | 29 | 0.112947836 | 96.197531 | 37.263938253 | 31.275347793 | 25.879357800 | 18.631969127 |

I use Adam 0.001 for both final custom runs and Adam 0.0001 for both pretrained runs, with weight decay 0.0001, batch size 64 and seed 42. The pretrained backbones were initialized from MobileNetV2 `IMAGENET1K_V2` / ShuffleNetV2 x0.5 `IMAGENET1K_V1`, then all layers were fine-tuned after replacing the classifiers. Custom normalization uses the saved training-set statistics; pretrained normalization uses ImageNet statistics. All runs use the saved splits, class order and geometric augmentation.

Training times come from every saved `train_seconds` value, including epochs after the selected checkpoint. They include training data loading and processing but exclude validation; total training minutes are the sum divided by 60. Excluding the first epoch removes only that epoch, not every timing outlier. These laptop runs were made at different times, so they do not form a controlled speed benchmark. The saved hardware record is a GTX 1650 Max-Q with approximately 4 GiB of GPU memory; software is PyTorch 2.5.1 / Torchvision 0.20.1. The [synthetic verification](pretrained_verification.json) also records the hardware.

I compared Adam against plain SGD and SGD with momentum 0.9, including SGD learning rates 0.01 and 0.003. Adam produced the lowest validation loss for the final Model A and B. Momentum retains a decaying history of gradients, which can smooth updates and speed progress along consistent directions; it can also overshoot. In my runs, momentum SGD improved selected-checkpoint accuracy over plain SGD 0.01 for all three custom designs, but it did not beat Adam's validation loss for the two final designs. The initial `model_b` baseline selected momentum SGD. One seed and a 30-epoch budget do not establish a generally optimal optimizer. The complete [12-run comparison](custom_cnn_results.md) retains the underlying evidence.

## Per-class test results

I show precision / recall / F1 as percentages for every class. Support is the number of true-class images, identical across models. The complete unrounded CSVs are linked below.

| Class | Support | Model A (standard) P / R / F1 (%) | Model B (final lightweight) P / R / F1 (%) | MobileNetV2 P / R / F1 (%) | ShuffleNetV2 x0.5 P / R / F1 (%) |
| --- | --- | --- | --- | --- | --- |
| AnnualCrop | 450 | 95.9135 / 88.6667 / 92.1478 | 92.4612 / 92.6667 / 92.5638 | 96.9231 / 98.0000 / 97.4586 | 95.1002 / 94.8889 / 94.9944 |
| Forest | 450 | 97.1800 / 99.5556 / 98.3535 | 98.2340 / 98.8889 / 98.5604 | 98.6813 / 99.7778 / 99.2265 | 97.8118 / 99.3333 / 98.5667 |
| HerbaceousVegetation | 450 | 88.8430 / 95.5556 / 92.0771 | 90.8696 / 92.8889 / 91.8681 | 97.5824 / 98.6667 / 98.1215 | 96.0000 / 96.0000 / 96.0000 |
| Highway | 375 | 95.4155 / 88.8000 / 91.9890 | 95.5621 / 86.1333 / 90.6031 | 98.1183 / 97.3333 / 97.7242 | 92.8760 / 93.8667 / 93.3687 |
| Industrial | 375 | 97.5000 / 93.6000 / 95.5102 | 96.3351 / 98.1333 / 97.2259 | 98.9160 / 97.3333 / 98.1183 | 98.1333 / 98.1333 / 98.1333 |
| Pasture | 300 | 94.3894 / 95.3333 / 94.8590 | 91.2773 / 97.6667 / 94.3639 | 97.2696 / 95.0000 / 96.1214 | 93.8511 / 96.6667 / 95.2381 |
| PermanentCrop | 375 | 85.6410 / 89.0667 / 87.3203 | 92.0000 / 85.8667 / 88.8276 | 97.8082 / 95.2000 / 96.4865 | 93.2249 / 91.7333 / 92.4731 |
| Residential | 450 | 94.6809 / 98.8889 / 96.7391 | 98.6577 / 98.0000 / 98.3278 | 97.8214 / 99.7778 / 98.7899 | 99.1131 / 99.3333 / 99.2231 |
| River | 375 | 90.7859 / 89.3333 / 90.0538 | 88.1910 / 93.6000 / 90.8150 | 98.6631 / 98.4000 / 98.5314 | 96.6667 / 92.8000 / 94.6939 |
| SeaLake | 450 | 98.6607 / 98.2222 / 98.4410 | 98.6667 / 98.6667 / 98.6667 | 99.3377 / 100.0000 / 99.6678 | 99.3348 / 99.5556 / 99.4451 |

PermanentCrop has the lowest F1 for both custom models: 87.32% for A and 88.83% for B. MobileNetV2 reaches 96.49% and ShuffleNetV2 92.47% for this class. Model B's Highway F1 falls from A's 91.99% to 90.60%, despite its higher overall accuracy. MobileNetV2 has the highest F1 in eight classes; ShuffleNetV2 leads for Industrial and Residential. These are observations from one fixed split and single-seed runs.

## Largest confusion directions

Rows of each saved confusion matrix are true classes; columns are predicted classes. I list the five largest off-diagonal directions, breaking ties alphabetically. Percentages divide by the true-class support, not the total test set. These are directional errors, not merged class pairs.

| Model | True class -> predicted class | Count | True-class support | True-class error (%) |
| --- | --- | --- | --- | --- |
| Model A (standard) | AnnualCrop -> PermanentCrop | 30 | 450 | 6.6667 |
| Model A (standard) | PermanentCrop -> HerbaceousVegetation | 28 | 375 | 7.4667 |
| Model A (standard) | Industrial -> Residential | 16 | 375 | 4.2667 |
| Model A (standard) | Highway -> PermanentCrop | 12 | 375 | 3.2000 |
| Model A (standard) | Highway -> River | 10 | 375 | 2.6667 |
| Model B (final lightweight) | PermanentCrop -> HerbaceousVegetation | 25 | 375 | 6.6667 |
| Model B (final lightweight) | Highway -> River | 20 | 375 | 5.3333 |
| Model B (final lightweight) | PermanentCrop -> AnnualCrop | 12 | 375 | 3.2000 |
| Model B (final lightweight) | AnnualCrop -> PermanentCrop | 10 | 450 | 2.2222 |
| Model B (final lightweight) | PermanentCrop -> River | 9 | 375 | 2.4000 |
| MobileNetV2 | Industrial -> Residential | 7 | 375 | 1.8667 |
| MobileNetV2 | Pasture -> AnnualCrop | 7 | 300 | 2.3333 |
| MobileNetV2 | PermanentCrop -> AnnualCrop | 6 | 375 | 1.6000 |
| MobileNetV2 | PermanentCrop -> HerbaceousVegetation | 5 | 375 | 1.3333 |
| MobileNetV2 | AnnualCrop -> Pasture | 4 | 450 | 0.8889 |
| ShuffleNetV2 x0.5 | River -> Highway | 15 | 375 | 4.0000 |
| ShuffleNetV2 x0.5 | PermanentCrop -> HerbaceousVegetation | 13 | 375 | 3.4667 |
| ShuffleNetV2 x0.5 | AnnualCrop -> PermanentCrop | 12 | 450 | 2.6667 |
| ShuffleNetV2 x0.5 | Highway -> PermanentCrop | 9 | 375 | 2.4000 |
| ShuffleNetV2 x0.5 | AnnualCrop -> Pasture | 8 | 450 | 1.7778 |

Model A predicts PermanentCrop for 30 AnnualCrop images; Model B reduces this direction to 10. Conversely, Highway -> River increases from 10 to 20. PermanentCrop -> HerbaceousVegetation appears in all four top-five lists (28 / 25 / 5 / 13 errors). MobileNetV2's largest directions each contain seven errors. ShuffleNetV2's largest is River -> Highway (15, or 4.00% of River images). Similar vegetation textures and narrow linear features are possible explanations; I have not inspected the errors to confirm them.

## My accuracy, storage and computation discussion

My final lightweight Model B reaches 94.370370% test accuracy, 0.469136 percentage points above Model A. It uses 51.06% fewer parameters, 49.25% less saved-file storage and 55.04% fewer conv/linear MACs. This supports the wider depthwise-separable design as my compact final model, although the small accuracy difference is not evidence of a repeatable advantage across seeds.

MobileNetV2 improves over Model B by 3.753086 percentage points. Its saved file is 46.71 times as large and its MAC count is 1.32 times as high. It is my highest-accuracy candidate in this comparison, but its 9.180270 MB weights and roughly 2.24 million parameters are costly for very small devices.

ShuffleNetV2 improves over Model B by 2.000000 percentage points. It needs 7.85 times the saved-file storage, yet uses 82.59% fewer MACs. This shows why parameter count, storage and computation must be compared separately. ShuffleNetV2 is attractive when arithmetic cost matters and 1.542078 MB weights are acceptable; Model B is attractive under the assignment's strict sub-100k parameter cap and a small storage budget. Both pretrained candidates exceed that custom-model cap.

Pretraining supplies learned features, while my custom Model B starts from scratch and gives me control over its architecture and footprint. This comparison combines architecture, initialization and different learning rates, so I cannot attribute the accuracy gains to pretraining alone. I did not test quantization, activation-memory use on an edge board, energy use or target-device latency. Depthwise operations and channel shuffling may have hardware-dependent efficiency; smaller MAC counts do not guarantee faster execution.

Saved test-pass times are 42.740828 / 3.463148 / 4.113200 / 3.711242 seconds for A / B / MobileNetV2 / ShuffleNetV2. They include loading, transfers, loss and prediction collection. Model A's much longer pass illustrates uncontrolled run conditions; I do not use these numbers as isolated inference latency or as a device-speed ranking.

## Custom architectures and hardware-aware activation choice

Model A uses three 3 x 3 standard convolutions with channels 3 -> 32 -> 64 -> 128, stride 1 and padding 1. Each convolution is followed by BatchNorm, ReLU and 2 x 2 max-pooling. Global average pooling produces 128 values and a 128 -> 10 linear layer produces raw logits. Convolution biases are disabled; BatchNorm has trainable scale/bias. Block parameter counts are 928, 18,560 and 73,984, plus 1,290 classifier parameters: 94,762 total.

Final Model B (`model_c`) uses 3 x 3 depthwise convolutions (one filter per input channel), then 1 x 1 pointwise convolutions, with channels 3 -> 64 -> 128 -> 256. Each depthwise/pointwise pair is followed by one BatchNorm/ReLU/2 x 2 max-pooling sequence. There is no extra activation between the two convolutions. Global average pooling feeds a 256 -> 10 linear layer. Its blocks have 347, 9,024 and 34,432 parameters, plus 2,570 classifier parameters: 46,373 total. The initial `model_b` uses narrower 32 / 64 / 128 channels and a 128 -> 10 classifier, totaling 12,965 parameters.

For a standard convolution, parameters are input channels x output channels x kernel area. A depthwise/pointwise pair uses input channels x kernel area + input channels x output channels. I add two affine BatchNorm parameters per output channel and classifier weights plus ten biases. Pooling and ReLU have no trainable parameters.

I choose ReLU for my custom models because max(0, x) needs a simple comparison/clamp and no exponentials or division. This suits limited compute hardware and vectorized CPU/GPU operations. The implementation uses in-place ReLU, which can reduce extra activation storage; it does not eliminate feature-map memory. I retain the pretrained architectures' existing activations, including MobileNetV2's ReLU6. I use raw classifier logits with cross-entropy instead of a separate training-time softmax. These are hardware-aware design reasons, not measured edge-device speed or power claims.

## Source artifacts and reproducibility

I verified saved predictions against the test CSV in its exact order, reconstructed confusion counts from those predictions, recalculated accuracy and per-class/macro precision, recall and F1, checked normalized matrices and confirmed checkpoint hashes/file sizes. I read histories to determine selected epochs and training times. None of these checks performs inference. Test loss is copied from the report because saved predictions do not contain every class logit needed to recompute cross-entropy.

The saved test-split SHA-256 is `6902f9b300a12d30bd6aeffd2432f9dee36efb7083f7237c78db35121a460928`. Source hashes and full-precision derived summaries are in [the saved-file audit](final_results_audit.json). Shared links below point to allowlisted files. Weight and resume paths remain local-only and ignored.

- **Model A (standard)**, `model_a_adam_lr0.001`: [config](../outputs/custom_cnn/model_a_adam_lr0.001/config.json), [history](../outputs/custom_cnn/model_a_adam_lr0.001/history.csv), [test metrics](../outputs/reports/evaluate-test_20261009T051335669280Z/metrics.json), [per-class CSV](../outputs/reports/evaluate-test_20261009T051335669280Z/per_class_metrics.csv), [confusion counts](../outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix.csv), [normalized CSV](../outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.csv), [count figure](../outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix.png), [normalized figure](../outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.png), [predictions](../outputs/reports/evaluate-test_20261009T051335669280Z/predictions.csv). Local-only checkpoint: `outputs/custom_cnn/model_a_adam_lr0.001/best_weights.pt`.

- **Model B (final lightweight)**, `model_c_adam_lr0.001`: [config](../outputs/custom_cnn/model_c_adam_lr0.001/config.json), [history](../outputs/custom_cnn/model_c_adam_lr0.001/history.csv), [test metrics](../outputs/reports/evaluate-test_20261009T051430041328Z/metrics.json), [per-class CSV](../outputs/reports/evaluate-test_20261009T051430041328Z/per_class_metrics.csv), [confusion counts](../outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix.csv), [normalized CSV](../outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.csv), [count figure](../outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix.png), [normalized figure](../outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.png), [predictions](../outputs/reports/evaluate-test_20261009T051430041328Z/predictions.csv). Local-only checkpoint: `outputs/custom_cnn/model_c_adam_lr0.001/best_weights.pt`.

- **MobileNetV2**, `mobilenet_v2_adam_lr0.0001_initial1`: [config](../outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/config.json), [history](../outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/history.csv), [test metrics](../outputs/reports/evaluate-test_20261009T051445171381Z/metrics.json), [per-class CSV](../outputs/reports/evaluate-test_20261009T051445171381Z/per_class_metrics.csv), [confusion counts](../outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix.csv), [normalized CSV](../outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.csv), [count figure](../outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix.png), [normalized figure](../outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.png), [predictions](../outputs/reports/evaluate-test_20261009T051445171381Z/predictions.csv). Local-only checkpoint: `outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/best_weights.pt`.

- **ShuffleNetV2 x0.5**, `shufflenet_v2_x0_5_adam_lr0.0001_initial1`: [config](../outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/config.json), [history](../outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/history.csv), [test metrics](../outputs/reports/evaluate-test_20261009T051458714369Z/metrics.json), [per-class CSV](../outputs/reports/evaluate-test_20261009T051458714369Z/per_class_metrics.csv), [confusion counts](../outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix.csv), [normalized CSV](../outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.csv), [count figure](../outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix.png), [normalized figure](../outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.png), [predictions](../outputs/reports/evaluate-test_20261009T051458714369Z/predictions.csv). Local-only checkpoint: `outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/best_weights.pt`.

## Assignment requirement audit and remaining gaps

I checked [the supplied assignment](../EN3150_Assignment_03_In23.pdf), including its submission instructions.

| Requirement | Evidence / status |
| --- | --- |
| Dataset other than CIFAR-10, at most 64 x 64, 70/15/15 split | Complete: EuroSAT RGB, 27,000 images; 18,900 / 4,050 / 4,050 saved splits. Class order and split fingerprints are checked. Notebook sections 2-5 document preparation and data checks. |
| Enter dataset information in the assignment sheet | **Pending confirmation:** I found no saved confirmation, completed sheet copy or submission receipt in the notebook or project documentation. Dataset preparation alone does not prove the sheet was updated. |
| Standard and depthwise-separable custom networks; lightweight <=100k | Complete: Model A `model_a`, final Model B `model_c` with 46,373 parameters. Kernels, filters, classifier and parameter formulas are detailed above and in notebook diagrams. |
| Hardware-aware activation justification | Complete: ReLU comparison/clamp, absence of exponentials/division and activation-storage limits are explained above and in the notebook audit addition. |
| Optimizer/rate choice; SGD and momentum comparison | Complete: 12 saved runs, four settings per design, momentum 0.9; validation selection and interpretation are documented here and in custom results. |
| Train custom models >=20 epochs and plot train/validation loss | Complete: both final custom runs have 30 epochs. Notebook sections 7.25 and 7.26 read histories and show curves/timing; shared histories and saved custom plots remain indexed. |
| Test accuracy, confusion, precision, recall; A/B size/time/accuracy table | Complete: four saved test reports, the comparison/timing tables above and per-class/confusion links. Actual and estimated KB are distinguished. |
| Two lightweight pretrained networks, same splits, fine-tuning and test metrics | Complete: 30-epoch MobileNetV2 / ShuffleNetV2 runs, all-layer fine-tuning, saved configs/history/model summaries and final test reports. Notebook section 9.6 shows saved learning curves. |
| Final custom versus pretrained accuracy, memory and compute discussion | Complete above; distinguishes disk storage, parameters, MACs and unmeasured target-device speed. |
| GitHub/SVN profile link and evidence of work over time | Repository/profile links are available: [project](https://github.com/DPHeshanRanasinghe/EN3150-Assignment03-CNN), [profile](https://github.com/DPHeshanRanasinghe). Include them in the final PDF. I did not make a commit or assess whether the existing commit history satisfies the duration requirement. |
| PDF report used for grading; names and index numbers; one group submission | **Pending:** this Markdown consolidation is not the required final PDF. Confirm group number, every member name/index number and include them in that PDF. No finished submission PDF or confirmation was found. |
| Separate report and code uploads to Moodle; filename Yourgroupno_A03_EN3150 | **Pending:** produce the correctly named report/code deliverables and upload them. I did not access Moodle or confirm submission. |
| Complete documented runnable code | Shared CLI and guarded analysis checks are available. Raw data and weights remain local-only; a teammate must obtain them as documented. A clean-machine reproduction and final packaged submission remain to be confirmed. |
| Original interpretation, attribution, timely submission | Sources and uncertainty are stated. Verify the final report's attribution, group authorship and deadline before submission; I cannot certify submission timing or authorship from artifacts. |

The dataset-sheet confirmation and final submission package are the substantive remaining gaps. Repeated seeds and edge-board latency/energy/quantization measurements would strengthen conclusions, but the assignment does not require them. I do not claim that single-seed differences are statistically established.

## Notebook analysis audit

I checked the new custom-comparison and final-test analysis cells against the saved files. Selection uses minimum validation loss, accuracy uses that epoch, units use decimal KB/MB consistently, timing excludes validation and confusion percentages use true-class support. The numerical observations about error directions, per-class F1, selected epochs and timing agree with the saved values after display rounding.

I made the pretrained-history cell define its own column helper so it works after setup without section 7.22. Section 9 still runs in its stated order after setup. Fresh-kernel checks execute copies with model construction, checkpoint loading, training and evaluation forbidden. Source notebook outputs, execution counts, IDs and metadata are preserved. Existing historical statements that no test had been run remain as earlier experiment records; the new audit note identifies the completed final stage without rewriting those records.

The older preservation fixture differed from 43 cells already present in the supplied notebook before this task. I refreshed it from the pre-edit snapshot and recorded the old and supplied hashes in [the preservation audit](../tests/fixtures/notebook_preservation_audit.md). All 113 supplied cells retain their outputs, execution counts, IDs and metadata; I repaired one cell's source and appended one Markdown audit cell. All 93 experiment/split/final-report artifact hashes remain unchanged.

Verification passed six saved-file/fresh-kernel checks and the existing notebook output-preservation/source-compilation check. `git diff --check` also passed. Selected reports and summaries pass `git check-ignore`; weights, resume files, archives, raw datasets, personal notes and incomplete reports remain ignored. No model forward pass, training or validation/test evaluation was performed by these checks.

Run the saved-file and guarded analysis checks with:

```powershell
conda run -n ml_env_fixed python -m unittest discover -s tests -p test_final_results.py -v
```

These checks read reports/histories and render saved-analysis figures in a disposable kernel. They do not execute training or validation/test inference.
