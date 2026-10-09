# Shared CNN assignment results

I share the saved configs, histories, selected validation and final test reports, and plots listed here. The [custom CNN discussion](../docs/custom_cnn_results.md) gives validation metrics, training-time means, parameter counts and MACs. The [final results](../docs/final_results.md) separately report test metrics. Checkpoints were selected using minimum validation loss.

For the final assignment report, required standard Model A is `model_a`, and
required final lightweight Model B is `model_c` (46,373 parameters, below the
100,000 cap). Existing `model_b` is the initial lightweight baseline. Historical
A/B/C labels remain unchanged. I describe the two new candidates in
[my pretrained selection notes](../docs/pretrained_model_selection.md), with
[synthetic GPU verification evidence](../docs/pretrained_verification.json).
Both selected pretrained runs are complete. I share their configs, histories and
model summaries below; their weights and resume checkpoints remain local-only.
Original ImageNet source files in `pretrained_weights/` remain ignored.

## Final test results and selected pretrained runs

I consolidate the four saved test reports in [my final results](../docs/final_results.md), with a [full-precision saved-file audit](../docs/final_results_audit.json). No model was rerun during consolidation. Dataset-sheet completion and the final PDF/code submission remain pending confirmation. All seven files in each selected test report are allowlisted for sharing.

- **Model A (standard)**: [metrics.json](reports/evaluate-test_20261009T051335669280Z/metrics.json), [per_class_metrics.csv](reports/evaluate-test_20261009T051335669280Z/per_class_metrics.csv), [confusion_matrix.csv](reports/evaluate-test_20261009T051335669280Z/confusion_matrix.csv), [confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.csv), [confusion_matrix.png](reports/evaluate-test_20261009T051335669280Z/confusion_matrix.png), [confusion_matrix_normalized.png](reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.png), [predictions.csv](reports/evaluate-test_20261009T051335669280Z/predictions.csv).

- **Model B (final lightweight)**: [metrics.json](reports/evaluate-test_20261009T051430041328Z/metrics.json), [per_class_metrics.csv](reports/evaluate-test_20261009T051430041328Z/per_class_metrics.csv), [confusion_matrix.csv](reports/evaluate-test_20261009T051430041328Z/confusion_matrix.csv), [confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.csv), [confusion_matrix.png](reports/evaluate-test_20261009T051430041328Z/confusion_matrix.png), [confusion_matrix_normalized.png](reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.png), [predictions.csv](reports/evaluate-test_20261009T051430041328Z/predictions.csv).

- **MobileNetV2**: [metrics.json](reports/evaluate-test_20261009T051445171381Z/metrics.json), [per_class_metrics.csv](reports/evaluate-test_20261009T051445171381Z/per_class_metrics.csv), [confusion_matrix.csv](reports/evaluate-test_20261009T051445171381Z/confusion_matrix.csv), [confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.csv), [confusion_matrix.png](reports/evaluate-test_20261009T051445171381Z/confusion_matrix.png), [confusion_matrix_normalized.png](reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.png), [predictions.csv](reports/evaluate-test_20261009T051445171381Z/predictions.csv).

  Shared run sources: [config.json](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/config.json), [history.csv](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/history.csv), [model_summary.json](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/model_summary.json). Local-only weights: `pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/best_weights.pt`; local-only resume: `pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/last_checkpoint.pt`.

- **ShuffleNetV2 x0.5**: [metrics.json](reports/evaluate-test_20261009T051458714369Z/metrics.json), [per_class_metrics.csv](reports/evaluate-test_20261009T051458714369Z/per_class_metrics.csv), [confusion_matrix.csv](reports/evaluate-test_20261009T051458714369Z/confusion_matrix.csv), [confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.csv), [confusion_matrix.png](reports/evaluate-test_20261009T051458714369Z/confusion_matrix.png), [confusion_matrix_normalized.png](reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.png), [predictions.csv](reports/evaluate-test_20261009T051458714369Z/predictions.csv).

  Shared run sources: [config.json](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/config.json), [history.csv](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/history.csv), [model_summary.json](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/model_summary.json). Local-only weights: `pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/best_weights.pt`; local-only resume: `pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/last_checkpoint.pt`.

## Start here

- [Comparison plot for all 12 experiments](reports/plot_20261009T015733299178Z/optimizer_comparison.png).
- [Comparison CSV for all 12 experiments](reports/plot_20261009T015733299178Z/optimizer_comparison.csv).
- [Latest complete validation reports](#latest-complete-validation-reports).
- [Proposed shared files and sizes](#proposed-shared-files-and-sizes).

The `.gitignore` allowlist makes these existing files eligible for Git. They have not been staged, committed or pushed by this update. Teammates receive them after they are added and committed. New reports remain ignored until deliberately selected.

## Experiment sources and local-only checkpoints

All 12 experiment folders stay at their original paths. I share every `config.json` and `history.csv`. Model weights and full resume checkpoints remain local-only and ignored. The checkpoint paths below are plain text because they will not exist in a teammate's checkout unless supplied separately. The six original runs have no full resume checkpoint.

| Run | Shared config | Shared history | Local-only model weights | Local-only resume checkpoint |
| --- | --- | --- | --- | --- |
| `model_a_adam_lr0.001` | [config.json](custom_cnn/model_a_adam_lr0.001/config.json) | [history.csv](custom_cnn/model_a_adam_lr0.001/history.csv) | `custom_cnn/model_a_adam_lr0.001/best_weights.pt` | Not saved |
| `model_a_sgd_lr0.003` | [config.json](custom_cnn/model_a_sgd_lr0.003/config.json) | [history.csv](custom_cnn/model_a_sgd_lr0.003/history.csv) | `custom_cnn/model_a_sgd_lr0.003/best_weights.pt` | `custom_cnn/model_a_sgd_lr0.003/last_checkpoint.pt` |
| `model_a_sgd_lr0.01` | [config.json](custom_cnn/model_a_sgd_lr0.01/config.json) | [history.csv](custom_cnn/model_a_sgd_lr0.01/history.csv) | `custom_cnn/model_a_sgd_lr0.01/best_weights.pt` | Not saved |
| `model_a_sgd_momentum_lr0.01` | [config.json](custom_cnn/model_a_sgd_momentum_lr0.01/config.json) | [history.csv](custom_cnn/model_a_sgd_momentum_lr0.01/history.csv) | `custom_cnn/model_a_sgd_momentum_lr0.01/best_weights.pt` | Not saved |
| `model_b_adam_lr0.001` | [config.json](custom_cnn/model_b_adam_lr0.001/config.json) | [history.csv](custom_cnn/model_b_adam_lr0.001/history.csv) | `custom_cnn/model_b_adam_lr0.001/best_weights.pt` | Not saved |
| `model_b_sgd_lr0.003` | [config.json](custom_cnn/model_b_sgd_lr0.003/config.json) | [history.csv](custom_cnn/model_b_sgd_lr0.003/history.csv) | `custom_cnn/model_b_sgd_lr0.003/best_weights.pt` | `custom_cnn/model_b_sgd_lr0.003/last_checkpoint.pt` |
| `model_b_sgd_lr0.01` | [config.json](custom_cnn/model_b_sgd_lr0.01/config.json) | [history.csv](custom_cnn/model_b_sgd_lr0.01/history.csv) | `custom_cnn/model_b_sgd_lr0.01/best_weights.pt` | Not saved |
| `model_b_sgd_momentum_lr0.01` | [config.json](custom_cnn/model_b_sgd_momentum_lr0.01/config.json) | [history.csv](custom_cnn/model_b_sgd_momentum_lr0.01/history.csv) | `custom_cnn/model_b_sgd_momentum_lr0.01/best_weights.pt` | Not saved |
| `model_c_adam_lr0.001` | [config.json](custom_cnn/model_c_adam_lr0.001/config.json) | [history.csv](custom_cnn/model_c_adam_lr0.001/history.csv) | `custom_cnn/model_c_adam_lr0.001/best_weights.pt` | `custom_cnn/model_c_adam_lr0.001/last_checkpoint.pt` |
| `model_c_sgd_lr0.003` | [config.json](custom_cnn/model_c_sgd_lr0.003/config.json) | [history.csv](custom_cnn/model_c_sgd_lr0.003/history.csv) | `custom_cnn/model_c_sgd_lr0.003/best_weights.pt` | `custom_cnn/model_c_sgd_lr0.003/last_checkpoint.pt` |
| `model_c_sgd_lr0.01` | [config.json](custom_cnn/model_c_sgd_lr0.01/config.json) | [history.csv](custom_cnn/model_c_sgd_lr0.01/history.csv) | `custom_cnn/model_c_sgd_lr0.01/best_weights.pt` | `custom_cnn/model_c_sgd_lr0.01/last_checkpoint.pt` |
| `model_c_sgd_momentum_lr0.01` | [config.json](custom_cnn/model_c_sgd_momentum_lr0.01/config.json) | [history.csv](custom_cnn/model_c_sgd_momentum_lr0.01/history.csv) | `custom_cnn/model_c_sgd_momentum_lr0.01/best_weights.pt` | `custom_cnn/model_c_sgd_momentum_lr0.01/last_checkpoint.pt` |

## Latest complete validation reports

I share metrics JSON, per-class metrics, and raw/normalized confusion matrices as CSV and PNG for the latest complete report of each available run and split. Per-image `predictions.csv` files stay local-only. Each local report remains complete with all seven original artifacts; the shared subset contains six of them.

I previously checked all 32 saved validation reports against their predictions, confusion matrices, class metrics, selected histories and checkpoint/split hashes. Each uses 4,050 validation images and matches the current saved checkpoint and split. I did not rerun validation or evaluate the test set.

| Run | Split | Summary | Per-class metrics | Confusion tables | Confusion plots |
| --- | --- | --- | --- | --- | --- |
| `model_a_sgd_lr0.003` | validation | [metrics.json](reports/validate_20261008T094415349696Z/metrics.json) | [CSV](reports/validate_20261008T094415349696Z/per_class_metrics.csv) | [Raw](reports/validate_20261008T094415349696Z/confusion_matrix.csv) / [Normalized](reports/validate_20261008T094415349696Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261008T094415349696Z/confusion_matrix.png) / [Normalized](reports/validate_20261008T094415349696Z/confusion_matrix_normalized.png) |
| `model_a_sgd_lr0.01` | validation | [metrics.json](reports/notebook_validation_np5be7f3/validation_check/metrics.json) | [CSV](reports/notebook_validation_np5be7f3/validation_check/per_class_metrics.csv) | [Raw](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.csv) / [Normalized](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.csv) | [Raw](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.png) / [Normalized](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.png) |
| `model_b_sgd_lr0.003` | validation | [metrics.json](reports/validate_20261008T163011272132Z/metrics.json) | [CSV](reports/validate_20261008T163011272132Z/per_class_metrics.csv) | [Raw](reports/validate_20261008T163011272132Z/confusion_matrix.csv) / [Normalized](reports/validate_20261008T163011272132Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261008T163011272132Z/confusion_matrix.png) / [Normalized](reports/validate_20261008T163011272132Z/confusion_matrix_normalized.png) |
| `model_c_adam_lr0.001` | validation | [metrics.json](reports/validate_20261009T004611267049Z/metrics.json) | [CSV](reports/validate_20261009T004611267049Z/per_class_metrics.csv) | [Raw](reports/validate_20261009T004611267049Z/confusion_matrix.csv) / [Normalized](reports/validate_20261009T004611267049Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261009T004611267049Z/confusion_matrix.png) / [Normalized](reports/validate_20261009T004611267049Z/confusion_matrix_normalized.png) |
| `model_c_sgd_lr0.003` | validation | [metrics.json](reports/validate_20261009T015643680837Z/metrics.json) | [CSV](reports/validate_20261009T015643680837Z/per_class_metrics.csv) | [Raw](reports/validate_20261009T015643680837Z/confusion_matrix.csv) / [Normalized](reports/validate_20261009T015643680837Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261009T015643680837Z/confusion_matrix.png) / [Normalized](reports/validate_20261009T015643680837Z/confusion_matrix_normalized.png) |
| `model_c_sgd_lr0.01` | validation | [metrics.json](reports/validate_20261009T010913208275Z/metrics.json) | [CSV](reports/validate_20261009T010913208275Z/per_class_metrics.csv) | [Raw](reports/validate_20261009T010913208275Z/confusion_matrix.csv) / [Normalized](reports/validate_20261009T010913208275Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261009T010913208275Z/confusion_matrix.png) / [Normalized](reports/validate_20261009T010913208275Z/confusion_matrix_normalized.png) |
| `model_c_sgd_momentum_lr0.01` | validation | [metrics.json](reports/validate_20261009T013446396158Z/metrics.json) | [CSV](reports/validate_20261009T013446396158Z/per_class_metrics.csv) | [Raw](reports/validate_20261009T013446396158Z/confusion_matrix.csv) / [Normalized](reports/validate_20261009T013446396158Z/confusion_matrix_normalized.csv) | [Raw](reports/validate_20261009T013446396158Z/confusion_matrix.png) / [Normalized](reports/validate_20261009T013446396158Z/confusion_matrix_normalized.png) |

No separate saved validation report was found for these five runs. Their shared histories provide the saved validation metrics; I did not regenerate missing reports:

- [model_a_adam_lr0.001](custom_cnn/model_a_adam_lr0.001/history.csv).
- [model_a_sgd_momentum_lr0.01](custom_cnn/model_a_sgd_momentum_lr0.01/history.csv).
- [model_b_adam_lr0.001](custom_cnn/model_b_adam_lr0.001/history.csv).
- [model_b_sgd_lr0.01](custom_cnn/model_b_sgd_lr0.01/history.csv).
- [model_b_sgd_momentum_lr0.01](custom_cnn/model_b_sgd_momentum_lr0.01/history.csv).

For Model A SGD 0.01, the latest report is `reports/notebook_validation_np5be7f3/validation_check/`. The older CLI summary linked by the discussion is also shared. No saved test report was found in this collection.

## Selected saved plots

I share one latest copy of each distinct plot set: the all-12 comparison, older comparisons with different coverage, and six single-run learning curves. Older comparisons describe the runs available when they were saved, so the Adam-only plot covers A and B. Historical byte-identical copies remain at their original local paths and are ignored.

| Coverage | Shared files |
| --- | --- |
| All 12 experiments | [optimizer_comparison.csv](reports/plot_20261009T015733299178Z/optimizer_comparison.csv) / [optimizer_comparison.png](reports/plot_20261009T015733299178Z/optimizer_comparison.png) |
| 2 runs: `model_a_adam_lr0.001`, `model_b_adam_lr0.001` | [optimizer_comparison.csv](reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.csv) / [optimizer_comparison.png](reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.png) |
| 6 runs: `model_a_adam_lr0.001`, `model_a_sgd_lr0.01`, `model_a_sgd_momentum_lr0.01`, `model_b_adam_lr0.001`, `model_b_sgd_lr0.01`, `model_b_sgd_momentum_lr0.01` | [optimizer_comparison.csv](reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.csv) / [optimizer_comparison.png](reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.png) |
| 8 runs: `model_a_adam_lr0.001`, `model_a_sgd_lr0.003`, `model_a_sgd_lr0.01`, `model_a_sgd_momentum_lr0.01`, `model_b_adam_lr0.001`, `model_b_sgd_lr0.003`, `model_b_sgd_lr0.01`, `model_b_sgd_momentum_lr0.01` | [optimizer_comparison.csv](reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.csv) / [optimizer_comparison.png](reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.png) |
| 7 runs: `model_a_adam_lr0.001`, `model_a_sgd_lr0.003`, `model_a_sgd_lr0.01`, `model_a_sgd_momentum_lr0.01`, `model_b_adam_lr0.001`, `model_b_sgd_lr0.01`, `model_b_sgd_momentum_lr0.01` | [optimizer_comparison.csv](reports/plot_20261008T094425745583Z/optimizer_comparison.csv) / [optimizer_comparison.png](reports/plot_20261008T094425745583Z/optimizer_comparison.png) |
| `model_a_adam_lr0.001` | [learning_curves.png](reports/notebook_model_a_adam_lr0.001_o_de8yz8/learning_curves.png) |
| `model_a_sgd_lr0.01` | [learning_curves.png](reports/notebook_model_a_sgd_lr0.01_dds9vu60/learning_curves.png) |
| `model_a_sgd_momentum_lr0.01` | [learning_curves.png](reports/notebook_model_a_sgd_momentum_lr0.01_ht64ho8p/learning_curves.png) |
| `model_b_adam_lr0.001` | [learning_curves.png](reports/notebook_model_b_adam_lr0.001_cq_llo7n/learning_curves.png) |
| `model_b_sgd_lr0.01` | [learning_curves.png](reports/notebook_model_b_sgd_lr0.01_p0hv2658/learning_curves.png) |
| `model_b_sgd_momentum_lr0.01` | [learning_curves.png](reports/notebook_model_b_sgd_momentum_lr0.01_92g3ehi3/learning_curves.png) |

## Older validation summaries with unique timing

I share these small metrics JSON files because their measured full-pass timing is unique content. Their other artifacts remain local-only. Timing includes loading, transfers, loss and prediction collection; it is not pure model latency. Evaluation-time batch size, worker count and data-root overrides were not recorded, so I did not assume those settings were equal.

| Older shared summary | Run | Recorded full-pass seconds |
| --- | --- | ---: |
| [notebook_validation_1ajjxkt3](reports/notebook_validation_1ajjxkt3/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.719083700008923 |
| [notebook_validation_1djte80s](reports/notebook_validation_1djte80s/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 5.615811000010581 |
| [notebook_validation_32o2rse1](reports/notebook_validation_32o2rse1/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.971622600001865 |
| [notebook_validation_387lz32a](reports/notebook_validation_387lz32a/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.9492169999939506 |
| [notebook_validation_42e8tmyk](reports/notebook_validation_42e8tmyk/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 3.8161175999994157 |
| [notebook_validation_4m9bzils](reports/notebook_validation_4m9bzils/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 36.80526170000667 |
| [notebook_validation_5_b713pk](reports/notebook_validation_5_b713pk/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 3.715805799991358 |
| [notebook_validation_7sh5re66](reports/notebook_validation_7sh5re66/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 4.357682699999714 |
| [notebook_validation_bgij0_pe](reports/notebook_validation_bgij0_pe/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 5.048284500000591 |
| [notebook_validation_dbsn1rvs](reports/notebook_validation_dbsn1rvs/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 3.694769300003827 |
| [notebook_validation_ebqujor6](reports/notebook_validation_ebqujor6/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 21.840540299999702 |
| [notebook_validation_f685kc_k](reports/notebook_validation_f685kc_k/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 4.1758163000049535 |
| [notebook_validation_fk74rfbn](reports/notebook_validation_fk74rfbn/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 5.385460499994224 |
| [notebook_validation_hj60y4kr](reports/notebook_validation_hj60y4kr/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 4.5704412999984925 |
| [notebook_validation_l2rgjhmd](reports/notebook_validation_l2rgjhmd/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 6.628228999994462 |
| [notebook_validation_m7o4zsbo](reports/notebook_validation_m7o4zsbo/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 9.917565600000671 |
| [notebook_validation_mp22epc7](reports/notebook_validation_mp22epc7/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 6.918638699993608 |
| [notebook_validation_nqiq36c8](reports/notebook_validation_nqiq36c8/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 5.915024000001722 |
| [notebook_validation_o3ekd1ua](reports/notebook_validation_o3ekd1ua/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 4.991830300001311 |
| [notebook_validation_r6hd0lhs](reports/notebook_validation_r6hd0lhs/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 5.5439220999978716 |
| [notebook_validation_swcvrk3a](reports/notebook_validation_swcvrk3a/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.800786899999366 |
| [notebook_validation_yjmy5tjj](reports/notebook_validation_yjmy5tjj/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.9390631999995094 |
| [notebook_validation_zj2251yj](reports/notebook_validation_zj2251yj/validation_check/metrics.json) | `model_a_sgd_lr0.01` | 2.9887037000007695 |
| [validate_20261008T025314239180Z](reports/validate_20261008T025314239180Z/metrics.json) | `model_a_sgd_lr0.01` | 27.37926080000034 |
| [validate_20261008T040003803031Z](reports/validate_20261008T040003803031Z/metrics.json) | `model_a_sgd_lr0.01` | 3.4791551000016625 |

Other shared source records: [original Model A SGD validation summary](refactor_checks/validation_model_a_sgd/metrics.json) and [notebook verification record](reports/notebook_verification.json).

## Local-only files preserved

| Local path (relative to outputs/) | Status |
| --- | --- |
| `custom_cnn/<run>/best_weights.pt` | Model weights; ignored. Exact paths are listed above. |
| `custom_cnn/<run>/last_checkpoint.pt` | Full resume checkpoints for the six newer runs; ignored. |
| `reports/**/predictions.csv` | Per-image predictions; ignored. |
| `custom_cnn/adam_comparison.*`, `custom_cnn/optimizer_comparison.*` and original run PNGs | Original training artifacts remain unchanged; the selected report plots are shared instead. |
| `archive/reports/` | 128 archived redundant plot folders; ignored. |
| `archive/manifest.json` | Local-only move log, hashes and retained/uncertain inventory; ignored. |
| `refactor_checks/` except the shared validation metrics JSON above | Original refactor and preservation artifacts; ignored. |
| `reports/notebook_20261007T025636510579Z/` | Empty/incomplete report, left in place and ignored. |
| `reports/notebook_20261008T024505890992Z/` | Empty/incomplete report, left in place and ignored. |

The remaining historical report copies are local-only unless one of their files is explicitly listed in the shared inventory below. I kept all 25 report folders referenced by the notebook, documentation or fixtures at their original local paths. Historical notebook outputs and embedded images are unchanged; some paths printed in old output text are local-only and require the original local results collection. No experiment folder or artifact was moved or rewritten by this Git-sharing update.

The archive operation retained 32 complete validation reports and 26 plot folders locally, plus two empty folders. It moved only 128 complete, unreferenced, byte-identical plot folders and checked SHA-256 hashes before and after each move. The original paths and details remain in the ignored local manifest.

The CLI reads runs from `outputs/custom_cnn/` and creates fresh timestamped reports under `outputs/reports/`. The notebook reads the same experiment sources and creates fresh report folders there. I did not change code, regenerate results, train, or evaluate the test set. Raw datasets, personal `notes.md`, temporary files, weights, resume checkpoints and archives stay ignored.

## Proposed shared files and sizes

This inventory lists the 149 selected result/documentation files eligible for Git, including the final reports, pretrained histories and saved-file audit. Sizes are saved-file bytes, not Git compression. Project-level `README.md`, `.gitignore`, source, tests and the notebook are outside this result inventory.

**Total: 149 files, 8,303,835 bytes (7.919 MiB).**

| Shared file (relative to the project root) | Bytes |
| --- | ---: |
| [docs/custom_cnn_results.md](../docs/custom_cnn_results.md) | 15,029 |
| [docs/final_results.md](../docs/final_results.md) | 25,581 |
| [docs/final_results_audit.json](../docs/final_results_audit.json) | 30,199 |
| [docs/pretrained_model_selection.md](../docs/pretrained_model_selection.md) | 13,067 |
| [docs/pretrained_verification.json](../docs/pretrained_verification.json) | 6,145 |
| [outputs/README.md](README.md) | 46,076 |
| [outputs/custom_cnn/model_a_adam_lr0.001/config.json](custom_cnn/model_a_adam_lr0.001/config.json) | 630 |
| [outputs/custom_cnn/model_a_adam_lr0.001/history.csv](custom_cnn/model_a_adam_lr0.001/history.csv) | 3,793 |
| [outputs/custom_cnn/model_a_sgd_lr0.003/config.json](custom_cnn/model_a_sgd_lr0.003/config.json) | 1,395 |
| [outputs/custom_cnn/model_a_sgd_lr0.003/history.csv](custom_cnn/model_a_sgd_lr0.003/history.csv) | 3,760 |
| [outputs/custom_cnn/model_a_sgd_lr0.01/config.json](custom_cnn/model_a_sgd_lr0.01/config.json) | 628 |
| [outputs/custom_cnn/model_a_sgd_lr0.01/history.csv](custom_cnn/model_a_sgd_lr0.01/history.csv) | 3,736 |
| [outputs/custom_cnn/model_a_sgd_momentum_lr0.01/config.json](custom_cnn/model_a_sgd_momentum_lr0.01/config.json) | 659 |
| [outputs/custom_cnn/model_a_sgd_momentum_lr0.01/history.csv](custom_cnn/model_a_sgd_momentum_lr0.01/history.csv) | 3,756 |
| [outputs/custom_cnn/model_b_adam_lr0.001/config.json](custom_cnn/model_b_adam_lr0.001/config.json) | 630 |
| [outputs/custom_cnn/model_b_adam_lr0.001/history.csv](custom_cnn/model_b_adam_lr0.001/history.csv) | 3,786 |
| [outputs/custom_cnn/model_b_sgd_lr0.003/config.json](custom_cnn/model_b_sgd_lr0.003/config.json) | 1,395 |
| [outputs/custom_cnn/model_b_sgd_lr0.003/history.csv](custom_cnn/model_b_sgd_lr0.003/history.csv) | 3,751 |
| [outputs/custom_cnn/model_b_sgd_lr0.01/config.json](custom_cnn/model_b_sgd_lr0.01/config.json) | 628 |
| [outputs/custom_cnn/model_b_sgd_lr0.01/history.csv](custom_cnn/model_b_sgd_lr0.01/history.csv) | 3,724 |
| [outputs/custom_cnn/model_b_sgd_momentum_lr0.01/config.json](custom_cnn/model_b_sgd_momentum_lr0.01/config.json) | 659 |
| [outputs/custom_cnn/model_b_sgd_momentum_lr0.01/history.csv](custom_cnn/model_b_sgd_momentum_lr0.01/history.csv) | 3,742 |
| [outputs/custom_cnn/model_c_adam_lr0.001/config.json](custom_cnn/model_c_adam_lr0.001/config.json) | 1,396 |
| [outputs/custom_cnn/model_c_adam_lr0.001/history.csv](custom_cnn/model_c_adam_lr0.001/history.csv) | 3,792 |
| [outputs/custom_cnn/model_c_sgd_lr0.003/config.json](custom_cnn/model_c_sgd_lr0.003/config.json) | 1,395 |
| [outputs/custom_cnn/model_c_sgd_lr0.003/history.csv](custom_cnn/model_c_sgd_lr0.003/history.csv) | 3,756 |
| [outputs/custom_cnn/model_c_sgd_lr0.01/config.json](custom_cnn/model_c_sgd_lr0.01/config.json) | 1,394 |
| [outputs/custom_cnn/model_c_sgd_lr0.01/history.csv](custom_cnn/model_c_sgd_lr0.01/history.csv) | 3,710 |
| [outputs/custom_cnn/model_c_sgd_momentum_lr0.01/config.json](custom_cnn/model_c_sgd_momentum_lr0.01/config.json) | 1,425 |
| [outputs/custom_cnn/model_c_sgd_momentum_lr0.01/history.csv](custom_cnn/model_c_sgd_momentum_lr0.01/history.csv) | 3,773 |
| [outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/config.json](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/config.json) | 2,406 |
| [outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/history.csv](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/history.csv) | 3,812 |
| [outputs/pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/model_summary.json](pretrained_cnn/mobilenet_v2_adam_lr0.0001_initial1/model_summary.json) | 1,080 |
| [outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/config.json](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/config.json) | 2,428 |
| [outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/history.csv](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/history.csv) | 3,814 |
| [outputs/pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/model_summary.json](pretrained_cnn/shufflenet_v2_x0_5_adam_lr0.0001_initial1/model_summary.json) | 1,094 |
| [outputs/refactor_checks/validation_model_a_sgd/metrics.json](refactor_checks/validation_model_a_sgd/metrics.json) | 1,348 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix.csv](reports/evaluate-test_20261009T051335669280Z/confusion_matrix.csv) | 460 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix.png](reports/evaluate-test_20261009T051335669280Z/confusion_matrix.png) | 115,580 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.csv) | 1,570 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.png](reports/evaluate-test_20261009T051335669280Z/confusion_matrix_normalized.png) | 125,439 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/metrics.json](reports/evaluate-test_20261009T051335669280Z/metrics.json) | 1,765 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/per_class_metrics.csv](reports/evaluate-test_20261009T051335669280Z/per_class_metrics.csv) | 725 |
| [outputs/reports/evaluate-test_20261009T051335669280Z/predictions.csv](reports/evaluate-test_20261009T051335669280Z/predictions.csv) | 326,811 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix.csv](reports/evaluate-test_20261009T051430041328Z/confusion_matrix.csv) | 458 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix.png](reports/evaluate-test_20261009T051430041328Z/confusion_matrix.png) | 112,685 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.csv) | 1,480 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.png](reports/evaluate-test_20261009T051430041328Z/confusion_matrix_normalized.png) | 122,990 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/metrics.json](reports/evaluate-test_20261009T051430041328Z/metrics.json) | 1,766 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/per_class_metrics.csv](reports/evaluate-test_20261009T051430041328Z/per_class_metrics.csv) | 722 |
| [outputs/reports/evaluate-test_20261009T051430041328Z/predictions.csv](reports/evaluate-test_20261009T051430041328Z/predictions.csv) | 326,317 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix.csv](reports/evaluate-test_20261009T051445171381Z/confusion_matrix.csv) | 454 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix.png](reports/evaluate-test_20261009T051445171381Z/confusion_matrix.png) | 109,513 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.csv) | 1,231 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.png](reports/evaluate-test_20261009T051445171381Z/confusion_matrix_normalized.png) | 116,366 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/metrics.json](reports/evaluate-test_20261009T051445171381Z/metrics.json) | 1,987 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/per_class_metrics.csv](reports/evaluate-test_20261009T051445171381Z/per_class_metrics.csv) | 696 |
| [outputs/reports/evaluate-test_20261009T051445171381Z/predictions.csv](reports/evaluate-test_20261009T051445171381Z/predictions.csv) | 325,749 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix.csv](reports/evaluate-test_20261009T051458714369Z/confusion_matrix.csv) | 457 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix.png](reports/evaluate-test_20261009T051458714369Z/confusion_matrix.png) | 112,122 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.csv](reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.csv) | 1,384 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.png](reports/evaluate-test_20261009T051458714369Z/confusion_matrix_normalized.png) | 121,083 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/metrics.json](reports/evaluate-test_20261009T051458714369Z/metrics.json) | 2,006 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/per_class_metrics.csv](reports/evaluate-test_20261009T051458714369Z/per_class_metrics.csv) | 708 |
| [outputs/reports/evaluate-test_20261009T051458714369Z/predictions.csv](reports/evaluate-test_20261009T051458714369Z/predictions.csv) | 326,201 |
| [outputs/reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.csv](reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.csv) | 395 |
| [outputs/reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.png](reports/notebook_adam_comparison_24vnjhbr/optimizer_comparison.png) | 252,367 |
| [outputs/reports/notebook_model_a_adam_lr0.001_o_de8yz8/learning_curves.png](reports/notebook_model_a_adam_lr0.001_o_de8yz8/learning_curves.png) | 157,577 |
| [outputs/reports/notebook_model_a_sgd_lr0.01_dds9vu60/learning_curves.png](reports/notebook_model_a_sgd_lr0.01_dds9vu60/learning_curves.png) | 185,001 |
| [outputs/reports/notebook_model_a_sgd_momentum_lr0.01_ht64ho8p/learning_curves.png](reports/notebook_model_a_sgd_momentum_lr0.01_ht64ho8p/learning_curves.png) | 161,223 |
| [outputs/reports/notebook_model_b_adam_lr0.001_cq_llo7n/learning_curves.png](reports/notebook_model_b_adam_lr0.001_cq_llo7n/learning_curves.png) | 129,730 |
| [outputs/reports/notebook_model_b_sgd_lr0.01_p0hv2658/learning_curves.png](reports/notebook_model_b_sgd_lr0.01_p0hv2658/learning_curves.png) | 161,110 |
| [outputs/reports/notebook_model_b_sgd_momentum_lr0.01_92g3ehi3/learning_curves.png](reports/notebook_model_b_sgd_momentum_lr0.01_92g3ehi3/learning_curves.png) | 159,641 |
| [outputs/reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.csv](reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.csv) | 887 |
| [outputs/reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.png](reports/notebook_optimizer_comparison_4ggf6q3x/optimizer_comparison.png) | 553,395 |
| [outputs/reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.csv](reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.csv) | 1,119 |
| [outputs/reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.png](reports/notebook_optimizer_comparison_vulm7o7i/optimizer_comparison.png) | 661,107 |
| [outputs/reports/notebook_validation_1ajjxkt3/validation_check/metrics.json](reports/notebook_validation_1ajjxkt3/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_1djte80s/validation_check/metrics.json](reports/notebook_validation_1djte80s/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_32o2rse1/validation_check/metrics.json](reports/notebook_validation_32o2rse1/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_387lz32a/validation_check/metrics.json](reports/notebook_validation_387lz32a/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_42e8tmyk/validation_check/metrics.json](reports/notebook_validation_42e8tmyk/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_4m9bzils/validation_check/metrics.json](reports/notebook_validation_4m9bzils/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_5_b713pk/validation_check/metrics.json](reports/notebook_validation_5_b713pk/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_7sh5re66/validation_check/metrics.json](reports/notebook_validation_7sh5re66/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_bgij0_pe/validation_check/metrics.json](reports/notebook_validation_bgij0_pe/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_dbsn1rvs/validation_check/metrics.json](reports/notebook_validation_dbsn1rvs/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_ebqujor6/validation_check/metrics.json](reports/notebook_validation_ebqujor6/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_f685kc_k/validation_check/metrics.json](reports/notebook_validation_f685kc_k/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_fk74rfbn/validation_check/metrics.json](reports/notebook_validation_fk74rfbn/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_hj60y4kr/validation_check/metrics.json](reports/notebook_validation_hj60y4kr/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_l2rgjhmd/validation_check/metrics.json](reports/notebook_validation_l2rgjhmd/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_m7o4zsbo/validation_check/metrics.json](reports/notebook_validation_m7o4zsbo/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_mp22epc7/validation_check/metrics.json](reports/notebook_validation_mp22epc7/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.csv](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.csv) | 470 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.png](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix.png) | 119,339 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.csv](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.csv) | 1,364 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.png](reports/notebook_validation_np5be7f3/validation_check/confusion_matrix_normalized.png) | 131,852 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/metrics.json](reports/notebook_validation_np5be7f3/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_np5be7f3/validation_check/per_class_metrics.csv](reports/notebook_validation_np5be7f3/validation_check/per_class_metrics.csv) | 711 |
| [outputs/reports/notebook_validation_nqiq36c8/validation_check/metrics.json](reports/notebook_validation_nqiq36c8/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_o3ekd1ua/validation_check/metrics.json](reports/notebook_validation_o3ekd1ua/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_r6hd0lhs/validation_check/metrics.json](reports/notebook_validation_r6hd0lhs/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_swcvrk3a/validation_check/metrics.json](reports/notebook_validation_swcvrk3a/validation_check/metrics.json) | 1,348 |
| [outputs/reports/notebook_validation_yjmy5tjj/validation_check/metrics.json](reports/notebook_validation_yjmy5tjj/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_validation_zj2251yj/validation_check/metrics.json](reports/notebook_validation_zj2251yj/validation_check/metrics.json) | 1,349 |
| [outputs/reports/notebook_verification.json](reports/notebook_verification.json) | 1,146 |
| [outputs/reports/plot_20261008T094425745583Z/optimizer_comparison.csv](reports/plot_20261008T094425745583Z/optimizer_comparison.csv) | 1,004 |
| [outputs/reports/plot_20261008T094425745583Z/optimizer_comparison.png](reports/plot_20261008T094425745583Z/optimizer_comparison.png) | 622,817 |
| [outputs/reports/plot_20261009T015733299178Z/optimizer_comparison.csv](reports/plot_20261009T015733299178Z/optimizer_comparison.csv) | 1,584 |
| [outputs/reports/plot_20261009T015733299178Z/optimizer_comparison.png](reports/plot_20261009T015733299178Z/optimizer_comparison.png) | 938,923 |
| [outputs/reports/validate_20261008T025314239180Z/metrics.json](reports/validate_20261008T025314239180Z/metrics.json) | 1,348 |
| [outputs/reports/validate_20261008T040003803031Z/metrics.json](reports/validate_20261008T040003803031Z/metrics.json) | 1,349 |
| [outputs/reports/validate_20261008T094415349696Z/confusion_matrix.csv](reports/validate_20261008T094415349696Z/confusion_matrix.csv) | 478 |
| [outputs/reports/validate_20261008T094415349696Z/confusion_matrix.png](reports/validate_20261008T094415349696Z/confusion_matrix.png) | 120,850 |
| [outputs/reports/validate_20261008T094415349696Z/confusion_matrix_normalized.csv](reports/validate_20261008T094415349696Z/confusion_matrix_normalized.csv) | 1,650 |
| [outputs/reports/validate_20261008T094415349696Z/confusion_matrix_normalized.png](reports/validate_20261008T094415349696Z/confusion_matrix_normalized.png) | 138,359 |
| [outputs/reports/validate_20261008T094415349696Z/metrics.json](reports/validate_20261008T094415349696Z/metrics.json) | 1,369 |
| [outputs/reports/validate_20261008T094415349696Z/per_class_metrics.csv](reports/validate_20261008T094415349696Z/per_class_metrics.csv) | 736 |
| [outputs/reports/validate_20261008T163011272132Z/confusion_matrix.csv](reports/validate_20261008T163011272132Z/confusion_matrix.csv) | 488 |
| [outputs/reports/validate_20261008T163011272132Z/confusion_matrix.png](reports/validate_20261008T163011272132Z/confusion_matrix.png) | 124,499 |
| [outputs/reports/validate_20261008T163011272132Z/confusion_matrix_normalized.csv](reports/validate_20261008T163011272132Z/confusion_matrix_normalized.csv) | 1,659 |
| [outputs/reports/validate_20261008T163011272132Z/confusion_matrix_normalized.png](reports/validate_20261008T163011272132Z/confusion_matrix_normalized.png) | 140,639 |
| [outputs/reports/validate_20261008T163011272132Z/metrics.json](reports/validate_20261008T163011272132Z/metrics.json) | 1,340 |
| [outputs/reports/validate_20261008T163011272132Z/per_class_metrics.csv](reports/validate_20261008T163011272132Z/per_class_metrics.csv) | 722 |
| [outputs/reports/validate_20261009T004611267049Z/confusion_matrix.csv](reports/validate_20261009T004611267049Z/confusion_matrix.csv) | 461 |
| [outputs/reports/validate_20261009T004611267049Z/confusion_matrix.png](reports/validate_20261009T004611267049Z/confusion_matrix.png) | 114,793 |
| [outputs/reports/validate_20261009T004611267049Z/confusion_matrix_normalized.csv](reports/validate_20261009T004611267049Z/confusion_matrix_normalized.csv) | 1,556 |
| [outputs/reports/validate_20261009T004611267049Z/confusion_matrix_normalized.png](reports/validate_20261009T004611267049Z/confusion_matrix_normalized.png) | 126,437 |
| [outputs/reports/validate_20261009T004611267049Z/metrics.json](reports/validate_20261009T004611267049Z/metrics.json) | 1,343 |
| [outputs/reports/validate_20261009T004611267049Z/per_class_metrics.csv](reports/validate_20261009T004611267049Z/per_class_metrics.csv) | 736 |
| [outputs/reports/validate_20261009T010913208275Z/confusion_matrix.csv](reports/validate_20261009T010913208275Z/confusion_matrix.csv) | 475 |
| [outputs/reports/validate_20261009T010913208275Z/confusion_matrix.png](reports/validate_20261009T010913208275Z/confusion_matrix.png) | 121,559 |
| [outputs/reports/validate_20261009T010913208275Z/confusion_matrix_normalized.csv](reports/validate_20261009T010913208275Z/confusion_matrix_normalized.csv) | 1,646 |
| [outputs/reports/validate_20261009T010913208275Z/confusion_matrix_normalized.png](reports/validate_20261009T010913208275Z/confusion_matrix_normalized.png) | 136,161 |
| [outputs/reports/validate_20261009T010913208275Z/metrics.json](reports/validate_20261009T010913208275Z/metrics.json) | 1,346 |
| [outputs/reports/validate_20261009T010913208275Z/per_class_metrics.csv](reports/validate_20261009T010913208275Z/per_class_metrics.csv) | 736 |
| [outputs/reports/validate_20261009T013446396158Z/confusion_matrix.csv](reports/validate_20261009T013446396158Z/confusion_matrix.csv) | 465 |
| [outputs/reports/validate_20261009T013446396158Z/confusion_matrix.png](reports/validate_20261009T013446396158Z/confusion_matrix.png) | 116,177 |
| [outputs/reports/validate_20261009T013446396158Z/confusion_matrix_normalized.csv](reports/validate_20261009T013446396158Z/confusion_matrix_normalized.csv) | 1,485 |
| [outputs/reports/validate_20261009T013446396158Z/confusion_matrix_normalized.png](reports/validate_20261009T013446396158Z/confusion_matrix_normalized.png) | 129,644 |
| [outputs/reports/validate_20261009T013446396158Z/metrics.json](reports/validate_20261009T013446396158Z/metrics.json) | 1,376 |
| [outputs/reports/validate_20261009T013446396158Z/per_class_metrics.csv](reports/validate_20261009T013446396158Z/per_class_metrics.csv) | 724 |
| [outputs/reports/validate_20261009T015643680837Z/confusion_matrix.csv](reports/validate_20261009T015643680837Z/confusion_matrix.csv) | 482 |
| [outputs/reports/validate_20261009T015643680837Z/confusion_matrix.png](reports/validate_20261009T015643680837Z/confusion_matrix.png) | 123,910 |
| [outputs/reports/validate_20261009T015643680837Z/confusion_matrix_normalized.csv](reports/validate_20261009T015643680837Z/confusion_matrix_normalized.csv) | 1,719 |
| [outputs/reports/validate_20261009T015643680837Z/confusion_matrix_normalized.png](reports/validate_20261009T015643680837Z/confusion_matrix_normalized.png) | 139,765 |
| [outputs/reports/validate_20261009T015643680837Z/metrics.json](reports/validate_20261009T015643680837Z/metrics.json) | 1,348 |
| [outputs/reports/validate_20261009T015643680837Z/per_class_metrics.csv](reports/validate_20261009T015643680837Z/per_class_metrics.csv) | 766 |
