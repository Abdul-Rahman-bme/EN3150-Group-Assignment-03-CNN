# Assignment and reproducibility checklist

Authoritative specification: `EN3150_Assignment_03_In23.pdf` (read, four pages).
The requested `(1)` filename was not present; the supplied repository PDF covers
the requirements. Report section numbers refer to `main.tex`.

| Item | Evidence / status |
|---|---|
| Dataset source | Official EuroSAT repository linked in report bibliography; raw RGB archive absent |
| Not CIFAR-10 | EuroSAT filenames, classes and preparation code |
| RGB, 64×64 | Loader checks and saved configs; fresh pixel verification pending images |
| 70/15/15 split | 18,900 / 4,050 / 4,050 rows; stratification independently checked |
| Saved filenames, labels and class order | `data/splits/*.csv`, `audit/split_audit.json` |
| No overlap | Unique paths and saved pixel hashes across manifests; images not reread |
| Normalization / augmentation | Run configs, `transforms.py`, report §2 |
| Seed, optimizer, LR, batch, epochs | All final runs seed 42, batch 64, 30 epochs; custom Adam .001, pretrained Adam .0001; weight decay .0001 |
| Model A architecture | Standard conv/BN/ReLU/max-pool; formula tables and source checked |
| Model B architecture/cap | Selected `model_c`, separable blocks, 46,373 parameters <100,000 |
| Layer details and calculations (20 marks) | Report §3 and appendix; `tables/*_layers.csv` |
| Hardware-aware activation | ReLU discussion; no unmeasured speed claim |
| Optimizer comparison (15 marks) | 12 histories; Adam, SGD .003/.01, SGD momentum .9/.01; report §4 |
| Minimum 20 epochs (25 marks) | All saved final histories contain 30 epochs; new training not claimed |
| Checkpoint rule | Minimum validation loss in shared source; selected history epochs verified |
| Training/validation curves | Four vector figures generated from histories |
| Custom test metrics/CM | Recomputed from saved predictions, not new checkpoint inference |
| Parameter count/model size/time table | Source costs, clearly labelled recorded sizes/timings |
| Genuine pretrained models (20 marks) | Official hashes/state initialization tested, historical configs inspected; trained files absent |
| Same split pretrained/custom | Canonical predictions and saved fingerprints agree, accounting explicitly for LF/CRLF |
| Final trade-off analysis (20 marks) | Report §7; accuracy, stored size and computation, no universal-best claim |
| Compact controlled run | Pipeline prepared; NOT RUN because images absent and hosts return proxy 403 |
| Model D additional verified run | Config, 30-epoch history, predictions, per-class metrics and confusion matrices checked; included in primary comparison |
| MAC definition | Conv/linear only, excludes BN/pooling/activation/memory; no latency equivalence |
| Git history/profile | Preserved history since 10 September; profile and repository links in PDF |
| Group identification | Group 54 supplied by user; four names/indices agree with original/current README |
| Dataset assignment sheet | Not verifiable from repo; group must update/confirm |
| Report build | pdfLaTeX/BibTeX; `audit/pdf_validation.json` contains final evidence |
| Code runnable | CLI, synthetic forward/update/checkpoint tests and report generator pass; dataset workflow pending artifacts |
| Full suite | Completed, non-green; see `audit/test_results.md` |
| Separate code/report submission | Not submitted; one group submission remains required |
| Peak RAM / edge latency / energy / quantization / multiple seeds | Not measured; limitations stated |

No historical test result was used to choose a replacement Model B. The current
selected lightweight model remains `model_c`; CompactEuroSAT needs controlled
validation before a new selection decision.
