The CompactEuroSAT notebook was developed against an older main branch and used
a different optimization protocol from the shared CLI. This branch merges current
main without discarding either history and adds the same compact architecture as
`compact_eurosat` in the common training pipeline.

The candidate has independently verified costs of 11,466 parameters and 3,882,784
conv/linear MACs per 64×64 RGB input. The proposed controlled run reuses the saved
18,900/4,050/4,050 splits, seed 42, batch size 64, 30 epochs, Adam 0.001, weight
decay 0.0001 and minimum-validation-loss checkpoint selection. Existing model
identifiers and historical artifacts remain unchanged.

The historical notebook split algorithm reconstructs the same memberships, but its
batch size 128, cosine schedule, colour jitter, maximum 60 epochs and validation-F1
selection differ. Its printed 94.72% test result is not used to replace Model B.
The new controlled experiment has **not run**: images are absent and official/user
download routes return proxy 403. Test evaluation is pending a validation-based
decision. Selected final Model B remains `model_c`; no new comparative metric is
invented. Model D is an optional architecture audit because its primary result
artifacts are absent.

The group-54 editable LaTeX report, compiled PDF, vector figures and CSV evidence
recompute the four existing final metrics from saved predictions and model costs
from source. File-size measurements and training timings are explicitly historical.

Validation: all 14 existing Model C/pretrained tests and both new candidate tests
pass; report generation consistency assertions pass. The full suite ran 34 methods
and remains non-green due to absent artifacts and historical LF/CRLF hash checks;
see `report/audit/test_results.md`. No tests were disabled. Report build and PDF
validation are documented under `report/`.

**Recommendation:** keep this PR in draft. Preserve `model_c` as selected Model B;
do not merge based on CompactEuroSAT's historical test accuracy. Complete the
controlled experiment and review its validation/resource trade-off first.
