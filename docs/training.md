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

Tune the forest on **val** only. The declared grid is
`configs/random_forest_tuning.yaml`:

```bash
uv run python -m beltwatch.baselines.tune_random_forest
```

The tuner fits each forest setting once on train, then scores each
class-frequency correction (`prior_power`) on val through the same
evaluation runner as every other model. Class-stratified pixel sampling makes
the forest believe foreground is far more common than it is, and the
correction scales each class's probability by `(training pixel share / sampled
share) ** prior_power`. The tuner keeps the candidate with the highest val
foreground macro IoU in `models/random-forest` (with `tuning.json`, the record
of every candidate) and its report in `reports/random-forest-val`. Every
candidate's report stays under `models/random-forest-tuning/reports`. A rerun
reuses forests and reports that are already saved. Only the selected forest's
weights are kept unless `--keep-all` is passed.

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
and a copy of the dataset-version files on Google Drive (about 9 GB). The
archive downloads to the Colab disk and is copied to Drive only after it is
verified. Streaming it straight into Drive ran at about 1 MB/s and Zenodo
dropped the connection. The setup cell pulls the latest code into a reused
runtime. After a disconnect, reconnect and run all cells from the top: the
archive is copied back from Drive and verified rather than downloaded again, a trained forest is reused, and training resumes
from `latest.pt`. The 300-step timing run writes to its own directory, so
running it again does not overwrite the full run's checkpoint. Each resume
opens a new MLflow run. The interrupted run stays marked as running.

On Kaggle, `notebooks/train_kaggle.ipynb` trains one model per run (set
`MODEL` to `unet` or `segformer`), so each run takes about 5 to 6 hours and
fits Kaggle's 12-hour session limit. Choose the GPU T4 x2 accelerator and turn
Internet on, then use **Save Version → Save & Run All (Commit)**: the run
continues without an open browser tab. The dataset is built on local disk, and
`/kaggle/working/beltwatch-runs` (checkpoints, reports, `mlflow.db`, and the
dataset-version files) is saved as the version's output. To resume a run that
stopped early, attach that version's output with **Add Input** and commit
again: the notebook copies it back and training resumes from `latest.pt`.

`MODEL = "baselines"` runs on a CPU session (Accelerator: None), so it uses
no GPU quota. Upload the trained runs as a private Kaggle Dataset, one folder
per run holding `best.pt` and `run.json`, and attach it. The notebook then
does the following on val:

1. Re-scores each checkpoint, so its report carries `confusions.npz`.
2. Compares SegFormer with the U-Net.
3. Evaluates the all-background baseline.
4. Tunes the random forest.
5. Compares each neural model with the tuned forest.

`MODEL = "analysis"` also runs on a CPU session and uses the same attached
dataset. For each checkpoint it does the following:

1. Fits temperatures on the calibration split: on all pixels, on foreground
   pixels, and one per predicted class (`--fit-scope all|foreground|per-class`).
2. Re-scores val with each fit (`reports/calibrated-<scope>/`).
3. Writes the error-analysis report on val (`error-reports/<run>/`).
4. Benchmarks the model with 4 threads on 20 val frames (`benchmarks/<run>.json`).
   It uses an unactivated bundle built only for this purpose, not a release
   candidate.

Each benchmark records the session's hardware. Kaggle's CPU session has 4
cores but more than 8 GB of RAM, so memory is measured as peak RSS, not
enforced. The error reports contain ZeroWaste frames, so keep them out of git.

After `dvc repro`, the notebook checks the rebuilt dataset-version files
against the commit. If only the download time in
`zerowaste-f-download.json` differs (and the `dvc.lock` hash that follows
from it), it restores the committed copies, so runs are recorded against a
clean commit rather than `-dirty`. Any other difference is reported and left
in place. The SegFormer run of 2026-10-12 is tagged `-dirty` for exactly this
reason; its other dataset-version files are byte-identical to the commit.

## 5. SegFormer-B0 (challenger)

```bash
uv run python scripts/train.py --config configs/segformer_b0.yaml --max-steps 300 --max-val-images 50
uv run python scripts/train.py --config configs/segformer_b0.yaml
uv run python scripts/evaluate.py --model segformer --model-dir models/segformer-b0-ce --split val
```

Its configuration matches the U-Net's in data, preprocessing, augmentation,
schedule, and loss, and a test enforces this, so the comparison isolates the
architecture. Compare the two val reports with
`scripts/compare.py reports/segformer_b0-val reports/unet_resnet18-val`,
which gives the difference a paired-bootstrap interval. Choose the deployed
model on validation foreground macro IoU, minority-class IoU, memory, and
measured CPU latency. A well-tuned U-Net is a legitimate winner. The MiT-B0
weights are for noncommercial research and evaluation only.

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
