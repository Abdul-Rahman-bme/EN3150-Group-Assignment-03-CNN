# Draft pull request

Created successfully:
https://github.com/Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN/pull/7

Title: **Evaluate CompactEuroSAT as a candidate lightweight Model B**

Base: `main`; head: `simple_model`. Kept in draft because the controlled
CompactEuroSAT experiment is blocked by absent images/download-route access.
Do not merge based on the historical test result. The existing selected Model B
remains `model_c`.

The successful creation command was:

```sh
gh pr create --draft --repo Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN --base main --head simple_model --title 'Evaluate CompactEuroSAT as a candidate lightweight Model B' --body-file report/audit/PR_DESCRIPTION.md
```

For review:

```sh
gh pr view 7 --repo Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN
gh pr diff 7 --repo Abdul-Rahman-bme/EN3150-Group-Assignment-03-CNN
```

Do not infer API failure from `gh auth status` or direct curl alone: those checks
looked unusable in this environment, but the actual supported PR operation worked.
No force push or main merge was performed.
