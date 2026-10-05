## Summary

<!-- What does this change and why? Link the issue. -->

## Type

- [ ] Data pipeline
- [ ] Model / training / evaluation
- [ ] Inference / API / jobs
- [ ] Frontend
- [ ] Deployment / CI
- [ ] Documentation

## Checklist

- [ ] Tests added or updated, and `pytest` passes locally
- [ ] `ruff check`, `ruff format --check`, and `mypy src/` pass
- [ ] Preprocessing is still shared between training and serving
- [ ] The locked test split was not used for tuning or inspection
- [ ] Any reported numbers come from a tracked run (config, dataset version, commit)
- [ ] Docs updated (README, model card, dataset card) where behavior changed
