# Test evidence — 16 October 2026

Command: `python -m unittest discover -s tests -v` in the activated Python 3.11
CPU project environment. Completed: **34 test methods**, **24 passed**,
**2 failed methods** (5 failure records including model subtests), **8 errored
methods** (9 error records including model subtests), no skipped tests.
Runner summary: `FAILED (failures=5, errors=9)`, exit status 1.

Passing checks include both new CompactEuroSAT tests, all six Model C tests,
all eight pretrained tests (including genuine official weight initialization),
fresh-kernel final saved analysis, final pretrained timing, static notebook source
compilation/output preservation, metric calculations and RNG restoration.

## Failed methods

- `test_predictions_confusions_and_metrics_agree`: four model subtests reach the
  literal split hash assertion. Saved metrics expect the CRLF serialization;
  committed CSV bytes use LF. Predictions, labels, confusion and per-class metric
  comparisons preceding that assertion pass. Independent report-generator checks
  also verify every model and recognize the CRLF serialization in memory.
- `test_source_hashes_and_all_historical_notebook_outputs`: first historical source
  CSV has the same LF/CRLF serialization difference. No expected hash was changed.

## Errored methods

- Final results: selected epoch/file-size check needs missing `best_weights.pt`.
- Independent notebook-analysis kernels need missing trained Model A SGD weights.
- Root notebook-analysis kernel needs the missing original `EuroSAT.zip` archive.
- CLI prepare test needs missing raw images.
- Completed-run skip test needs trained weights to classify the historical run as complete.
- All-checkpoint shape/cost test reaches a missing trained checkpoint.
- Simulated resume test checks real image existence before simulated epochs; both
  Model A and C subtests therefore error for missing images.
- Immutable-artifact check needs missing `outputs/refactor_checks/source_manifest.json`.

No unexpected model-forward, gradient, optimizer, pretrained-loading or plotting
implementation failure was found. No existing test was modified, skipped or
disabled to obtain a pass. The full suite is **not green**.

## Separate report checks

`python report/generate.py` completed with all assertions passing. It verifies
split integrity, four saved prediction/metric sets, formula-vs-tensor layer costs,
candidate costs and optimizer selection, and writes CSV/LaTeX/vector outputs.
`bash report/build.sh` compiles the report using pdfLaTeX/BibTeX; final build evidence
and output page count are recorded in `pdf_validation.json` after validation.

Full runner output is retained in the cloud workspace at
`/workspace/.runtime/cnn/final-full-suite.log`; repository audit files summarize
the relevant causes without embedding machine-specific temporary paths in the report.
