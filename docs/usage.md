# Usage

Command reference for everything beyond the [README quickstart](../README.md#quickstart).
All commands run from the repository root after `uv sync --extra cpu`.

## Getting the data

The data pipeline is defined in [`dvc.yaml`](../dvc.yaml) and run with
[DVC](https://dvc.org/):

```bash
uv run dvc repro            # download → validate → duplicates → splits
```

The `download` stage (also runnable directly with
`uv run python -m beltwatch.data.download`) downloads the pinned ZeroWaste-f
release (about 7.5 GB, resumable), verifies its size and MD5 against
[`configs/data.yaml`](../configs/data.yaml), extracts it to
`data/raw/zerowaste-f/`, and writes the measured image counts to
`data/manifests/zerowaste-f-download.json`. Reserve about 20 GB of free disk
for the archive plus the extracted files. Use `--skip-extract` to download and
verify only.

## Baselines

```bash
uv run python -m beltwatch.baselines.random_forest --out models/random-forest
uv run python scripts/evaluate.py --model all-background --split val
uv run python scripts/evaluate.py --model random-forest --model-dir models/random-forest --split val
uv run python -m beltwatch.baselines.tune_random_forest   # val-only grid in configs/random_forest_tuning.yaml
```

Reports go to `reports/<model>-<split>/` (`report.json`, `per_image.csv`, and
the per-image confusion matrices in `confusions.npz`). To compare two models
on the same images with a paired bootstrap over recording groups:

```bash
uv run python scripts/compare.py reports/unet_resnet18-val reports/random-forest-val
```

## Training

```bash
uv run python scripts/train.py --config configs/unet.yaml
uv run python scripts/evaluate.py --model unet --model-dir models/unet-resnet18-ce --split val
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Full training needs an NVIDIA GPU. See the [training runbook](training.md) and
the [Colab](../notebooks/train_colab.ipynb) or
[Kaggle](../notebooks/train_kaggle.ipynb) notebook.

## Calibration, error analysis and benchmark

Calibrate on the calibration split, check the result on val, and include it in
a release:

```bash
uv run python -m beltwatch.evaluation.calibration --model unet --model-dir models/unet-resnet18-ce --out reports/calibration-unet
uv run python scripts/evaluate.py --model unet --model-dir models/unet-resnet18-ce --split val --temperature <T>   # or one per class: T0,T1,T2,T3,T4
uv run python -m beltwatch.release.bundle --kind unet --model models/unet-resnet18-ce/best.pt \
    --version <v> --calibration reports/calibration-unet/calibration.json
```

Analyse errors on val, and benchmark a release on the target CPU:

```bash
uv run python scripts/error_report.py --model unet --model-dir models/unet-resnet18-ce --split val --out reports/errors-unet
uv run python scripts/benchmark.py --release model_releases/<v> --threads 4 --out reports/benchmark.json
```

## Audit service

Build and activate a release. Trained weights are not in the repository; to
try the service without them, use the all-background sanity-check release (it
never predicts a material):

```bash
uv run python -m beltwatch.release.bundle --kind all-background --version demo \
    --note "service check; not a trained model" --activate
# or, with a trained SegFormer-B0 checkpoint and its calibration
# (the way beltwatch-0.1.0 was built):
uv run python -m beltwatch.release.bundle --kind segformer --version <v> \
    --model models/segformer-b0-ce/best.pt \
    --calibration reports/calibration-segformer/calibration.json --activate
```

Start the API and the inference worker (two terminals), then submit an audit
and follow it:

```bash
uv run uvicorn beltwatch.api.app:create_app --factory --port 8000
uv run python -m beltwatch.jobs.worker

curl -F "files=@frame.png" \
     -F 'options={"camera_id":"line-1","inspection_region":[[0.05,0.1],[0.95,0.1],[0.95,0.9],[0.05,0.9]]}' \
     http://localhost:8000/v1/audits
curl http://localhost:8000/v1/audits/<audit_id>
```

The review UI is at `http://localhost:8000/`: upload snapshots, draw the
inspection region, follow progress, work through the priority-ordered review
queue (overlay, original, and uncertainty views with estimated visible
coverage), record decisions, and export the audit record. Interactive API docs
are at `http://localhost:8000/docs`. Endpoints, failure handling and the data
model are in [architecture.md](architecture.md).

## Feedback export

Reviewer feedback reaches training data only through annotation review, and
only from audits created with `"training_consent": true`:

```bash
uv run python -m beltwatch.data.feedback export --data-dir var --out data/feedback/<batch>
uv run python -m beltwatch.data.feedback review data/feedback/<batch> <image_id> \
    --decision accept --annotator <name> --reason "<why>"
uv run python -m beltwatch.data.feedback eligible data/feedback/<batch>
```

Only accepted candidates with a corrected pixel mask are eligible for a dataset
version; a confirmation is an image-level label and does not validate the
predicted pixels.

## Deployment and rollback

```bash
docker compose build
docker compose run --rm --no-deps --user root api \
    python -m beltwatch.release.activate <version> --reason "initial" --releases-dir /releases
docker compose up -d --wait            # HTTPS proxy, API, worker
python3 scripts/deploy_smoke.py --base-url https://localhost --insecure --expect-version <version>
```

Rollback, backup, monitoring and the reliability soak test
(`scripts/soak_test.py`) are in [operations.md](operations.md). The same code
also runs as a one-container demo (`python -m beltwatch.demo`, packaged for a
Hugging Face Docker Space) that fetches the pinned release, refuses training
consent and deletes uploads after 24 h. It is verified locally but not
published ([operations.md](operations.md#hosted-demo)).

## Usability study

The protocol, consent script and declared analysis are in
[usability_study.md](usability_study.md):

```bash
uv run python -m beltwatch.study.prepare --out data/study/study-1          # matched val sets
uv run python -m beltwatch.study.session data/study/study-1  # API + worker, submit, plans
# per participant: open http://127.0.0.1:8000/study.html and load plans/P0n.json; Ctrl-C after
uv run python -m beltwatch.study.analyze data/study/study-1 --results results-P*.json --out reports/study-1
```

## README figures

The per-class chart is drawn from the committed test records:

```bash
python3 scripts/results_chart.py docs/results/beltwatch-0.1.0/test docs/assets/test-iou.svg
```

The demo GIF (`docs/assets/review-demo.gif`) is a headless-browser recording
of the review UI running `beltwatch-0.1.0` on procedurally generated images,
because ZeroWaste-f frames are not redistributed in this repository.
