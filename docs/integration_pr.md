# Draft integration PR: browser handoff

The integration branch is pushed to the fork. The GitHub connector's draft-PR
request returned `403 Resource not accessible by integration`; no PR was created.
Neither main branch was pushed or merged. Browser authentication may permit the
requested action even though the connector lacks access.

## Exact browser steps

1. Sign in as `DPHeshanRanasinghe` and open
   [the upstream comparison](https://github.com/Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN/compare/main...DPHeshanRanasinghe:integration/heshan-main-20261009?expand=1).
2. If needed, click **compare across forks**. Confirm **base repository** =
   `Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN`, **base branch** = `main`,
   **head fork** = `DPHeshanRanasinghe/EN3150-Assignment03-CNN`, and
   **compare branch** = `integration/heshan-main-20261009`.
3. Enter the title below and paste only the description under “PR description”.
   Use the button dropdown to choose **Create Draft Pull Request**, then submit
   the draft. Do not merge it. These steps follow
   [GitHub's fork-PR instructions](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork).

## PR title

Integrate completed CNN workflows and final results with upstream baseline

## PR description

Integrates the completed custom/pretrained workflows and final results into `main` while preserving Abdul's existing preparation and standard-CNN work.

- Includes all 12 custom experiment configs/histories, the two completed pretrained runs, four saved final test reports, shared CLI workflows, notebook analysis and results/setup documentation. Model identifiers, saved paths and CSV splits remain unchanged. Weights, images, resume checkpoints and archives remain ignored.
- Retains Abdul's two notebooks, historical outputs, existing documentation and source byte-for-byte, and restores the upstream team details in the shared README. His 2,193,226-parameter flattened/dropout baseline remains distinct from the 94,762-parameter BatchNorm/global-pooling CLI `model_a`; neither implementation nor its results were substituted for the other.
- Resolves the sole merge conflict in `requirements.txt` using the tested torch 2.5.1 / torchvision 0.20.1 pair and the union of notebook/CLI dependencies. Preserves upstream's original pins in `requirements-upstream.txt`. Original upstream ignore rules are retained alongside the selected-report allowlist.

Validation: all 30 targeted workflow/saved-result tests passed; three notebooks compiled; original notebook/result files remained intact; Abdul's exact model definition passed synthetic shape/finite checks for batches 1/20/64; the seed-42 split procedure matched all three committed split memberships; diff/ignore checks passed. Two tests initially encountered missing ignored raw images, then passed on a focused rerun after a local-only dataset link. The original notebook suite's real validation passes were not repeated. No dataset training or test-set evaluation was run. A new clean-machine dependency install was not verified.

This remains a draft pending group review of the report convention: standard Model A = `model_a`, final lightweight Model B = `model_c`, and `model_b` = initial lightweight baseline. Abdul's historical 92.27% result is retained separately from the four full-precision final reports.

Remaining gaps: six indexed validation reports were never committed due to local read permissions; their allowlist/index entries are preserved. Confirm dataset-sheet completion, finalize the PDF with group identifiers/repository links, package code and complete Moodle submission. Repeated seeds and edge-device latency/energy measurements remain limitations rather than assignment requirements.

Detailed decisions and evidence: `docs/integration.md`, `docs/integration_checks.json` and `docs/final_results.md`.
