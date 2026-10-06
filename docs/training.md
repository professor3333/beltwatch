# Training Runbook

How to run the full pipeline and train models on a GPU machine: Google Colab,
Kaggle, or a rented NVIDIA instance. Laptops are used only for development
against small samples.

Budget (planning estimates, to be checked with a timed run): about 25 GB of
disk for the dataset and extraction, 2 to 8 GPU-hours per U-Net run, and a 20
to 50 GPU-hour development allowance.

## 1. Set up

```bash
git clone https://github.com/professor3333/beltwatch.git && cd beltwatch
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is not installed
uv sync --extra cu126                                  # CUDA 12.6 PyTorch build
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Build the dataset

```bash
uv run dvc repro            # download → validate → duplicates → splits
```

This downloads and verifies the pinned 7.5 GB archive (resumable), then
writes the git-tracked outputs to `data/manifests/`, `data/splits/`, and
`dvc.lock`. Check them before training:

- `data/manifests/zerowaste-f-validation.json`: valid and quarantined counts,
  the category mapping, and the warning counts.
- `data/manifests/zerowaste-f-duplicates.json`: exact-duplicate groups and
  cross-split pairs.
- `data/splits/zerowaste-f-splits.json`: split sizes, exclusion reasons, and
  the split manifest ID.

Commit these outputs and `dvc.lock` to a branch and open a PR. They record
the actual dataset version that every later result refers to.

## 3. Classical baselines

```bash
uv run python -m beltwatch.baselines.random_forest --out models/random-forest
uv run python scripts/evaluate.py --model all-background --split val
uv run python scripts/evaluate.py --model random-forest --model-dir models/random-forest --split val
```

Tune the forest on **val** only, for example `--pixels-per-class`,
`--max-depth`, and `--long-side`. Keep the best configuration's report.

## 4. U-Net

Time the schedule first:

```bash
uv run python scripts/train.py --config configs/unet.yaml --max-steps 300 --max-val-images 50
```

`time/epoch_seconds` in MLflow (and the log) gives the cost per epoch.
Multiply by `optimization.epochs` to check the budget. Then run in full:

```bash
uv run python scripts/train.py --config configs/unet.yaml
uv run python scripts/evaluate.py --model unet --model-dir models/unet-resnet18-ce --split val
```

If a session ends early, resume:

```bash
uv run python scripts/train.py --config configs/unet.yaml --resume models/unet-resnet18-ce/latest.pt
```

`latest.pt` is written at the end of every epoch, so a resume loses at most
the epoch in progress. Checkpoints are written to a temporary file and then
renamed, so a session that dies mid-write leaves the previous checkpoint
intact.

On Colab, the notebook keeps the archive, `output_dir`, reports, `mlflow.db`
and a copy of the dataset-version files on Google Drive (about 9 GB). After a
disconnect, reconnect and run all cells from the top: the archive is verified
rather than downloaded again, a trained forest is reused, and training resumes
from `latest.pt`. The 300-step timing run writes to its own directory, so
running it again does not overwrite the full run's checkpoint. Each resume
opens a new MLflow run. The interrupted run stays marked as running.

## 5. SegFormer-B0 (challenger)

```bash
uv run python scripts/train.py --config configs/segformer_b0.yaml --max-steps 300 --max-val-images 50
uv run python scripts/train.py --config configs/segformer_b0.yaml
uv run python scripts/evaluate.py --model segformer --model-dir models/segformer-b0-ce --split val
```

Its configuration matches the U-Net's in data, preprocessing, augmentation,
schedule, and loss, and a test enforces this, so the comparison isolates the
architecture. Choose the deployed model on validation foreground macro IoU,
minority-class IoU, memory, and measured CPU latency. A well-tuned U-Net is a
legitimate winner. The MiT-B0 weights are for noncommercial research and
evaluation only.

## 6. Inspect runs

```bash
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each run records its configuration, metrics, and lineage tags: git commit,
split manifest ID, dataset MD5, preprocessing version, encoder weights, and
the best checkpoint's SHA-256.

## Rules

- Model selection and error analysis use **val**. The **test** split is
  evaluated only for a declared release
  (`scripts/evaluate.py --split test --declared-release <name>`).
- Report every class, including weak ones, and report a missed target as a
  miss.
- Keep `models/`, `mlflow.db`, and `reports/` out of git. Release bundles are
  published separately.
