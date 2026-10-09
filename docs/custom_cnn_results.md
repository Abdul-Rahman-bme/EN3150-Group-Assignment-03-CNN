# My custom CNN validation results

For the final assignment report, required standard Model A is `model_a`, and
required final lightweight Model B is `model_c` (46,373 parameters, below the
100,000 cap). Existing `model_b` is my initial lightweight baseline. I selected
`model_c` using validation results. The tables below retain the historical A/B/C
labels and saved identifiers; no run or artifact is renamed. See
[my naming and pretrained selection notes](pretrained_model_selection.md).

I completed 12 custom CNN experiments: four configurations each for Model A (StandardCNN), Model B (LightweightCNN) and Model C (WideLightweightCNN). I read each saved `config.json` and `history.csv` under `outputs/custom_cnn/` and checked the saved validation reports where available. I calculated the values below from these files, rather than from rounded comparison values.

**These are validation results, not test results.** I used validation loss to select checkpoints and configurations. I did not run training or evaluate the test set to prepare this document.

## How I selected and compared runs

I selected the epoch with the minimum `val_loss` in each history. If losses tie, I use the first matching epoch, as the training code only replaces a checkpoint when loss decreases. I report `val_accuracy` from that same epoch, not necessarily the epoch with maximum accuracy. Epoch numbers start at 1.

Each configuration has one seed (42) and a 30-epoch budget. All 12 histories contain 30 completed epochs. The saved configs use batch size 64, weight decay 0.0001 and CUDA. I have no repeated-seed results, so these runs do not show how much results vary between seeds.

## Saved configurations

Each run link opens its saved folder, which contains the source `config.json` and `history.csv`. I include trainable parameters and convolution/linear MACs per 64 x 64 RGB image for every run, using the saved validation-report counts for its architecture. Momentum is 0 for plain SGD (the config omits it and the optimizer code defaults to 0). Momentum is not applicable to Adam; the saved Adam configs do not specify an SGD momentum value.

| Run | Model | Optimizer | Learning rate | Momentum | Seed | Batch size | Weight decay | Budget / completed epochs | Trainable parameters | Conv/linear MACs per image |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [model_a_adam_lr0.001](../outputs/custom_cnn/model_a_adam_lr0.001/) | A | Adam | 0.001 | N/A | 42 | 64 | 0.0001 | 30 / 30 | 94,762 | 41,288,960 |
| [model_a_sgd_lr0.003](../outputs/custom_cnn/model_a_sgd_lr0.003/) | A | SGD | 0.003 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 94,762 | 41,288,960 |
| [model_a_sgd_lr0.01](../outputs/custom_cnn/model_a_sgd_lr0.01/) | A | SGD | 0.01 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 94,762 | 41,288,960 |
| [model_a_sgd_momentum_lr0.01](../outputs/custom_cnn/model_a_sgd_momentum_lr0.01/) | A | SGD with momentum | 0.01 | 0.9 | 42 | 64 | 0.0001 | 30 / 30 | 94,762 | 41,288,960 |
| [model_b_adam_lr0.001](../outputs/custom_cnn/model_b_adam_lr0.001/) | B | Adam | 0.001 | N/A | 42 | 64 | 0.0001 | 30 / 30 | 12,965 | 5,141,760 |
| [model_b_sgd_lr0.003](../outputs/custom_cnn/model_b_sgd_lr0.003/) | B | SGD | 0.003 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 12,965 | 5,141,760 |
| [model_b_sgd_lr0.01](../outputs/custom_cnn/model_b_sgd_lr0.01/) | B | SGD | 0.01 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 12,965 | 5,141,760 |
| [model_b_sgd_momentum_lr0.01](../outputs/custom_cnn/model_b_sgd_momentum_lr0.01/) | B | SGD with momentum | 0.01 | 0.9 | 42 | 64 | 0.0001 | 30 / 30 | 12,965 | 5,141,760 |
| [model_c_adam_lr0.001](../outputs/custom_cnn/model_c_adam_lr0.001/) | C | Adam | 0.001 | N/A | 42 | 64 | 0.0001 | 30 / 30 | 46,373 | 18,561,536 |
| [model_c_sgd_lr0.003](../outputs/custom_cnn/model_c_sgd_lr0.003/) | C | SGD | 0.003 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 46,373 | 18,561,536 |
| [model_c_sgd_lr0.01](../outputs/custom_cnn/model_c_sgd_lr0.01/) | C | SGD | 0.01 | 0 | 42 | 64 | 0.0001 | 30 / 30 | 46,373 | 18,561,536 |
| [model_c_sgd_momentum_lr0.01](../outputs/custom_cnn/model_c_sgd_momentum_lr0.01/) | C | SGD with momentum | 0.01 | 0.9 | 42 | 64 | 0.0001 | 30 / 30 | 46,373 | 18,561,536 |

## Source artifact paths

I link directly to the saved config and history for each run below. All paths are relative to this document in `docs/`. The saved validation reports I checked are linked in the validation-report table.

| Run | Saved configuration | Saved epoch history |
| --- | --- | --- |
| `model_a_adam_lr0.001` | [../outputs/custom_cnn/model_a_adam_lr0.001/config.json](../outputs/custom_cnn/model_a_adam_lr0.001/config.json) | [../outputs/custom_cnn/model_a_adam_lr0.001/history.csv](../outputs/custom_cnn/model_a_adam_lr0.001/history.csv) |
| `model_a_sgd_lr0.003` | [../outputs/custom_cnn/model_a_sgd_lr0.003/config.json](../outputs/custom_cnn/model_a_sgd_lr0.003/config.json) | [../outputs/custom_cnn/model_a_sgd_lr0.003/history.csv](../outputs/custom_cnn/model_a_sgd_lr0.003/history.csv) |
| `model_a_sgd_lr0.01` | [../outputs/custom_cnn/model_a_sgd_lr0.01/config.json](../outputs/custom_cnn/model_a_sgd_lr0.01/config.json) | [../outputs/custom_cnn/model_a_sgd_lr0.01/history.csv](../outputs/custom_cnn/model_a_sgd_lr0.01/history.csv) |
| `model_a_sgd_momentum_lr0.01` | [../outputs/custom_cnn/model_a_sgd_momentum_lr0.01/config.json](../outputs/custom_cnn/model_a_sgd_momentum_lr0.01/config.json) | [../outputs/custom_cnn/model_a_sgd_momentum_lr0.01/history.csv](../outputs/custom_cnn/model_a_sgd_momentum_lr0.01/history.csv) |
| `model_b_adam_lr0.001` | [../outputs/custom_cnn/model_b_adam_lr0.001/config.json](../outputs/custom_cnn/model_b_adam_lr0.001/config.json) | [../outputs/custom_cnn/model_b_adam_lr0.001/history.csv](../outputs/custom_cnn/model_b_adam_lr0.001/history.csv) |
| `model_b_sgd_lr0.003` | [../outputs/custom_cnn/model_b_sgd_lr0.003/config.json](../outputs/custom_cnn/model_b_sgd_lr0.003/config.json) | [../outputs/custom_cnn/model_b_sgd_lr0.003/history.csv](../outputs/custom_cnn/model_b_sgd_lr0.003/history.csv) |
| `model_b_sgd_lr0.01` | [../outputs/custom_cnn/model_b_sgd_lr0.01/config.json](../outputs/custom_cnn/model_b_sgd_lr0.01/config.json) | [../outputs/custom_cnn/model_b_sgd_lr0.01/history.csv](../outputs/custom_cnn/model_b_sgd_lr0.01/history.csv) |
| `model_b_sgd_momentum_lr0.01` | [../outputs/custom_cnn/model_b_sgd_momentum_lr0.01/config.json](../outputs/custom_cnn/model_b_sgd_momentum_lr0.01/config.json) | [../outputs/custom_cnn/model_b_sgd_momentum_lr0.01/history.csv](../outputs/custom_cnn/model_b_sgd_momentum_lr0.01/history.csv) |
| `model_c_adam_lr0.001` | [../outputs/custom_cnn/model_c_adam_lr0.001/config.json](../outputs/custom_cnn/model_c_adam_lr0.001/config.json) | [../outputs/custom_cnn/model_c_adam_lr0.001/history.csv](../outputs/custom_cnn/model_c_adam_lr0.001/history.csv) |
| `model_c_sgd_lr0.003` | [../outputs/custom_cnn/model_c_sgd_lr0.003/config.json](../outputs/custom_cnn/model_c_sgd_lr0.003/config.json) | [../outputs/custom_cnn/model_c_sgd_lr0.003/history.csv](../outputs/custom_cnn/model_c_sgd_lr0.003/history.csv) |
| `model_c_sgd_lr0.01` | [../outputs/custom_cnn/model_c_sgd_lr0.01/config.json](../outputs/custom_cnn/model_c_sgd_lr0.01/config.json) | [../outputs/custom_cnn/model_c_sgd_lr0.01/history.csv](../outputs/custom_cnn/model_c_sgd_lr0.01/history.csv) |
| `model_c_sgd_momentum_lr0.01` | [../outputs/custom_cnn/model_c_sgd_momentum_lr0.01/config.json](../outputs/custom_cnn/model_c_sgd_momentum_lr0.01/config.json) | [../outputs/custom_cnn/model_c_sgd_momentum_lr0.01/history.csv](../outputs/custom_cnn/model_c_sgd_momentum_lr0.01/history.csv) |

## Results at the minimum-validation-loss epoch

I retain the saved decimal values for loss and accuracy (a fraction from 0 to 1). I calculate accuracy percentages from the saved fractions and show six decimal places. Training means use all saved `train_seconds` values and are shown to nine decimal places.

| Run | Selected epoch | Validation loss | Validation accuracy (fraction) | Validation accuracy (%) | Mean training seconds, epochs 1-30 | Mean training seconds, epochs 2-30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_a_adam_lr0.001` | 30 | 0.17211006972156925 | 0.9412345679012346 | 94.123457 | 26.776831297 | 25.266525355 |
| `model_a_sgd_lr0.003` | 29 | 0.45945246942249346 | 0.8362962962962963 | 83.629630 | 28.104214260 | 24.185851866 |
| `model_a_sgd_lr0.01` | 27 | 0.3795960882857994 | 0.8720987654320987 | 87.209877 | 26.005597597 | 26.032462355 |
| `model_a_sgd_momentum_lr0.01` | 30 | 0.1745064631435606 | 0.9419753086419753 | 94.197531 | 23.936241177 | 23.952596724 |
| `model_b_adam_lr0.001` | 27 | 0.24448609859119227 | 0.9155555555555556 | 91.555556 | 27.785075647 | 24.949462348 |
| `model_b_sgd_lr0.003` | 28 | 0.65117317031931 | 0.782962962962963 | 78.296296 | 27.930627453 | 23.396881048 |
| `model_b_sgd_lr0.01` | 28 | 0.483141946454107 | 0.8353086419753086 | 83.530864 | 25.385775127 | 25.460868924 |
| `model_b_sgd_momentum_lr0.01` | 25 | 0.23470281453780187 | 0.9182716049382716 | 91.827160 | 26.102648627 | 26.212977455 |
| `model_c_adam_lr0.001` | 28 | 0.17760855402475523 | 0.94 | 94.000000 | 30.532371087 | 25.623894797 |
| `model_c_sgd_lr0.003` | 29 | 0.5881116829095063 | 0.7960493827160494 | 79.604938 | 24.334127470 | 24.399790600 |
| `model_c_sgd_lr0.01` | 29 | 0.416601156776334 | 0.8501234567901235 | 85.012346 | 34.293015390 | 29.173842307 |
| `model_c_sgd_momentum_lr0.01` | 29 | 0.21827401213807823 | 0.9232098765432099 | 92.320988 | 25.919956440 | 26.003364821 |

I calculate the first mean as `sum(train_seconds for epochs 1-30) / 30`. I calculate the second as `sum(train_seconds for epochs 2-30) / 29`. These are training-pass times per epoch; they exclude `val_seconds`. The second mean removes the first epoch, which can include startup overhead. Hardware and data loading affect these timings, so I do not treat them as inference speed.

## Saved validation-report checks

I checked the following existing `metrics.json` reports against the selected history rows. Each report uses the validation split with 4,050 images and records `history_comparison.matches: true`. All reported accuracies match the saved fractions, and the largest loss difference is about 5.6e-17, consistent with decimal parsing. I did not rerun validation.

| Run | Selected epoch | Saved validation report |
| --- | ---: | --- |
| `model_a_sgd_lr0.003` | 29 | [metrics.json](../outputs/reports/validate_20261008T094415349696Z/metrics.json) |
| `model_a_sgd_lr0.01` | 27 | [metrics.json](../outputs/reports/validate_20261008T040003803031Z/metrics.json) |
| `model_b_sgd_lr0.003` | 28 | [metrics.json](../outputs/reports/validate_20261008T163011272132Z/metrics.json) |
| `model_c_adam_lr0.001` | 28 | [metrics.json](../outputs/reports/validate_20261009T004611267049Z/metrics.json) |
| `model_c_sgd_lr0.003` | 29 | [metrics.json](../outputs/reports/validate_20261009T015643680837Z/metrics.json) |
| `model_c_sgd_lr0.01` | 29 | [metrics.json](../outputs/reports/validate_20261009T010913208275Z/metrics.json) |
| `model_c_sgd_momentum_lr0.01` | 29 | [metrics.json](../outputs/reports/validate_20261009T013446396158Z/metrics.json) |

The original [Model A SGD validation report](../outputs/refactor_checks/validation_model_a_sgd/metrics.json) also matches epoch 27. I found readable checkpoint-validation reports for the seven runs listed above. For the other five runs, I use the saved histories without claiming a separate checkpoint-validation check. Some older notebook report folders were inaccessible, so I do not claim a complete audit of those folders.

## Compact accuracy comparison

I compare validation accuracy (%) at each run's minimum-validation-loss checkpoint below. I calculate these percentages from the saved histories and round only the display to six decimal places. These are validation results, not test results or maximum-epoch accuracies.

| Optimizer | Learning rate | Model A (%) | Model B (%) | Model C (%) |
| --- | ---: | ---: | ---: | ---: |
| Adam | 0.001 | 94.123457 | 91.555556 | 94.000000 |
| SGD | 0.01 | 87.209877 | 83.530864 | 85.012346 |
| SGD with momentum 0.9 | 0.01 | 94.197531 | 91.827160 | 92.320988 |
| SGD | 0.003 | 83.629630 | 78.296296 | 79.604938 |

## Configurations I selected

I compared the minimum validation loss of the four configurations within each model. My selections are:

| Model | Selected configuration | Selected epoch | Validation loss | Validation accuracy (%) |
| --- | --- | ---: | ---: | ---: |
| A | Adam 0.001 | 30 | 0.17211006972156925 | 94.123457 |
| B | SGD with momentum 0.01 (momentum 0.9) | 25 | 0.23470281453780187 | 91.827160 |
| C | Adam 0.001 | 28 | 0.17760855402475523 | 94.000000 |

I selected Adam 0.001 for Model A even though SGD with momentum 0.01 has slightly higher selected-checkpoint accuracy (94.197531% versus 94.123457%). Adam has the lower validation loss (0.17211006972156925 versus 0.1745064631435606), which is my selection rule.

## Model size and computation

I use the parameter and MAC counts from the saved validation reports linked above. These counts are the same across optimizer configurations for each architecture.

| Model | Trainable parameters | Estimated convolution/linear MACs per 64 x 64 RGB image |
| --- | ---: | ---: |
| A | 94,762 | 41,288,960 |
| B | 12,965 | 5,141,760 |
| C | 46,373 | 18,561,536 |

MACs are multiply-accumulate computation estimates, not measured inference speed. These counts cover convolution and linear layers only; they exclude pooling, normalization, activations and data loading. A lower MAC count does not by itself prove faster inference.

## My discussion

I found that Model C improved over Model B in all four matched optimizer and learning-rate configurations: it had lower selected validation loss and higher accuracy in each case. With Adam 0.001, C reached 94.000000%, compared with B at 91.555556% and A at 94.123457%. C nearly matched A, with a gap of 0.123457 percentage points, while using 46,373 rather than 94,762 parameters and 18,561,536 rather than 41,288,960 MACs per image. This makes C a useful balance between accuracy and estimated computation in these runs.

I also see a cost for widening the lightweight model: C uses more parameters and MACs than B. B is the smallest model, and its best configuration is SGD with momentum 0.01 at 91.827160% validation accuracy. For C, Adam gives both lower validation loss and higher accuracy than momentum SGD (92.320988%). Plain SGD at 0.003 gives lower accuracy than plain SGD at 0.01 for all three models within the same 30-epoch budget.

I base these observations on one seed per configuration and validation-based selection. I cannot conclude that the same ranking will hold across seeds or on the test set. I preserved the saved experiment files and notebook outputs when writing this report.
