# Model releases

Release bundles live here as immutable directories, one per version. They are
**not committed to git**: model binaries are published separately, for example
as GitHub release assets.

```
model_releases/
├── active.json              # version new audits are pinned to, plus activation history
└── <version>/
    ├── manifest.json        # model kind and SHA-256, preprocessing version, label map,
    │                        # temperature, review policy, provenance, limitations, licenses
    ├── model.pt             # or model/ for the random forest
    └── evaluation.json      # report the release was promoted on (optional)
```

Build and activate a release:

```bash
uv run python -m beltwatch.release.bundle --kind unet \
    --model models/unet-resnet18-ce/best.pt --version beltwatch-0.1.0 \
    --evaluation reports/unet_resnet18-val/report.json \
    --note "first U-Net release" --activate
```

Roll back by activating a previous version. This restores its model,
preprocessing, calibration, and review policy together:

```python
from pathlib import Path
from beltwatch.release.bundle import activate

activate(Path("model_releases"), "beltwatch-0.0.9", reason="rollback: <why>")
```

Audits keep the version they were created with. The worker refuses to process
an audit whose pinned release is missing, rather than silently using another
model.
