# Branch audit and Git decisions

Read-only inspection preceded source edits: status, all branch refs, remote,
full reachable graph, commit stats, branch-specific diffs and notebook/model
contents. The initial checkout was clean on `work`, at main commit `8b81b38`.
Initial fetch configuration tracked only main; an explicit fetch of all head refs
made the expected branches inspectable. No remote was rewritten.

The exact audited full SHAs are in `baseline_refs.json` and `branch_audit.csv`.
Ahead/behind here describe those original remote refs relative to `8b81b38`,
before the new local synchronization and candidate/report work.

| Branch | Ahead / behind | Unique useful work | Status / action |
|---|---:|---|---|
| main | 0 / 0 | Canonical integrated source, splits, 12 custom histories and four final prediction/report sets | Preserve unchanged |
| heshan | 0 / 0 | Same tip as main | Already integrated; no merge needed |
| integration/heshan-main-20261009 | 0 / 1 | Historical integration and PR handoff | Ancestor of main; preserve |
| data/eurosat-pipeline | 0 / 22 | Original preparation notebook and dataset documentation | Already merged via PR #4; preserve |
| setup/project-structure | 0 / 28 | Original shared structure | Already merged; preserve |
| model/standard-cnn | 0 / 20 | Original standard-CNN notebook | Already merged via PR #5; preserve |
| simple_model | 18 / 19 | `notebooks/03_model_B_compact_cnn.ipynb`, `lightweight_CNN`; CompactEuroSAT architecture and historical printed results | Useful candidate; synchronized normally with main, no conflicts |
| nipun | 3 / 0 | `ResidualDepthwiseCNN`, registration and `docs/model_d_results.md` | Preserve separate optional experiment; primary run artifacts absent |

Already-integrated branch tips contain no unmerged commits; historical work is
retained by main's ancestry. They were not merged again or deleted. The two
CompactEuroSAT notebook files remain byte-for-byte as on the pre-sync branch.
Nipun changes were inspected as source; only its two architecture class definitions
were executed for cost/shape analysis. Its Markdown result is not promoted into
the primary comparison.

## Actions performed

1. Fetched all remote refs without resetting a checkout.
2. Created `backup/work-before-final-audit` at the initial main-equivalent state.
3. Created local `simple_model` from the audited remote tip. The clone's main-only
   refspec prevented `--track`, so explicit `--no-track` checkout was used; this
   did not change source or remote configuration.
4. Created `backup/simple_model-before-sync` at `7d81c51`.
5. Merged `origin/main`, producing `79a3bcb`, with no conflicts.
6. Inspected status and diff checks. Pushed the synchronization as a normal
   fast-forward update to remote `simple_model`; command succeeded.
7. Committed shared-pipeline candidate implementation and meaningful tests as
   `49d74a0` (`Add CompactEuroSAT candidate to the common training pipeline`).
8. Added the report, generator, evidence tables/figures and audit documentation;
   the final commit and push status are recorded in the task's final response.

No force push, hard reset, branch deletion, main merge, result replacement or
historical notebook rewrite was performed. Main remains the canonical selected
model state. A draft PR is appropriate; merging is not justified until controlled
candidate validation is available.

Git read and push authentication work through the platform proxy. The raw
`gh` token status is unusable and `api.github.com` returned proxy 403; these do
not imply missing Git access. The needed API and dataset hosts were saved to an
environment network draft, which still needs user review/save/publication.
