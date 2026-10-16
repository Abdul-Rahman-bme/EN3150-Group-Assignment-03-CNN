# Group 54 report and reproducibility audit

`54_A03_EN3150.pdf` is an evidence-audited **draft**, not a claim that all requested
experiments or submission tasks are complete. Group number 54 was provided by the
user. Names match both the original and current team README; the group should
confirm submission spelling. The assignment PDF actually present is
`../EN3150_Assignment_03_In23.pdf`; its contents were read as the grading specification.

## Rebuild

From the repository root, activate the project Python environment (in this cloud
machine, `source /workspace/.runtime/cnn/activate.sh`), then:

```sh
python report/generate.py
bash report/build.sh
python report/validate_pdf.py
```

Python dependencies are the project environment plus scikit-learn and matplotlib.
The generator uses Git to inspect `origin/nipun`'s architecture and the audited
remote refs, without merging Model D. Fetch the requested branches before running
it in a fresh checkout. The LaTeX build requires TeX Live with pdfLaTeX, BibTeX,
TikZ, booktabs, natbib, xurl, lmodern and microtype. PDF validation uses Poppler's
`pdftotext` and `pdfinfo`. No shell escape is used.
The generator asserts consistency before writing comparison results; it performs
no training and never changes the historical data/notebooks.

`main.tex` and `references.bib` are editable. `figures/` contains vector figures
generated from saved artifacts. Architecture diagrams are editable TikZ in
`main.tex`. `tables/` contains CSV evidence and generated LaTeX tables. `audit/`
contains branch, split, provenance, candidate-cost and test evidence.

## Evidence levels and limitations

- All four final prediction files match the canonical test memberships and order.
  Accuracy, macro precision/recall/F1, per-class metrics and confusion matrices
  were recalculated and agree with the saved reports.
- Parameter and conv/linear MAC counts were recalculated from executed source.
- Training curves, validation selection and epoch timings come from saved histories.
- Historical serialized weight sizes are recorded in metrics; trained checkpoints
  are missing and have **not** been remeasured or used for new inference here.
- The compact notebook's split algorithm reconstructs the canonical memberships,
  but its batch size, scheduler, augmentation and selection rule differ. Its
  printed 94.72% is historical evidence, not a new controlled result. The exact
  Kaggle manifests/checkpoint are absent.
- Model D's architecture costs were independently checked, but its results are
  Markdown-only claims without committed primary run artifacts. It is excluded
  from the verified master comparison.
- Dataset downloads from Zenodo, the deprecated official host and the user-supplied
  Hugging Face URL returned proxy 403. Required dataset/API domains were saved in
  the environment draft for review/save/publication; that does not apply runtime
  access or prove a successful download.
- The full suite has known absent-artifact and LF/CRLF hash failures, listed in
  `audit/test_results.md`. No tests or assertions were disabled.
- The dataset assignment-sheet update and separate single-group Moodle submission
  require group action. The report does not claim either was performed.

## Controlled CompactEuroSAT run (not performed)

After the original RGB images are available, first validate **all** saved pixel
hashes using `python -m cnn_assignment prepare --check-images`. Do not generate
new splits. Then use a fresh run name:

```sh
python -m cnn_assignment train --model compact_eurosat --optimizer adam --lr 0.001 --epochs 30 --batch-size 64 --seed 42 --weight-decay 0.0001 --run-name compact_common_seed42_v1 --device auto
python -m cnn_assignment validate --run outputs/custom_cnn/compact_common_seed42_v1 --compare-history --device auto
```

Select the architecture/configuration using validation and resource costs, record
the decision, then explicitly run test evaluation once:

```sh
python -m cnn_assignment evaluate-test --run outputs/custom_cnn/compact_common_seed42_v1 --device auto
```

Do not overwrite historical results or treat synthetic optimizer tests as dataset
training. The shared pipeline keeps fixed LR, FP32, no early stopping, canonical
normalization/geometric augmentation, and minimum-validation-loss selection.
Existing `model_a`, `model_b`, `model_c` identifiers remain unchanged.

The original evaluation CLI uses literal split-file hashes; replay of historical
CRLF-fingerprinted runs on LF checkouts needs a separately reviewed portability
change or matching CSV serialization outside the tracked checkout. The audit
compares CRLF hashes in memory and leaves all original bytes/expected hashes alone.
