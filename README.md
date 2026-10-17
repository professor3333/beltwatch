<div align="center">

# BeltWatch

**Visual contamination auditing for paper-recycling conveyor lines**

Semantic segmentation · visible-coverage estimation · uncertainty-aware review queue · auditable human feedback

[![ci](https://github.com/professor3333/beltwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/beltwatch/actions/workflows/ci.yml)
[![container](https://github.com/professor3333/beltwatch/actions/workflows/container.yml/badge.svg)](https://github.com/professor3333/beltwatch/actions/workflows/container.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Dataset: ZeroWaste-f](https://img.shields.io/badge/dataset-ZeroWaste--f-green.svg)](https://ai.bu.edu/zerowaste/)
![Status: in development](https://img.shields.io/badge/status-in%20development-orange.svg)

</div>

> [!NOTE]
> **Project status: first release evaluated.** Release `beltwatch-0.1.0`
> (SegFormer-B0) was declared on val and scored once on the locked test
> split: foreground macro IoU **0.336**, 18.4 points above the tuned random
> forest. It **misses** the proposed 0.40 target and the review-workload
> target. See [Results](#results). The pipeline, the audit service and the
> deployment are built and tested (311 tests).

## Architecture

```mermaid
flowchart LR
    subgraph Training ["Offline training (GPU)"]
        A[ZeroWaste-f<br/>pinned release] --> B[Validate · dedupe ·<br/>split manifests]
        B --> C[DVC dataset<br/>version]
        C --> D[PyTorch training<br/>+ MLflow]
        D --> E[Evaluation ·<br/>calibration]
        E --> F[Versioned<br/>release bundle]
    end

    subgraph Serving ["Serving (CPU, Docker Compose)"]
        G[Browser review UI] --> H[FastAPI]
        H --> I[(SQLite<br/>audit jobs)]
        I --> J[Inference worker]
        J --> K[Masks · coverage ·<br/>review flags]
        K --> H
    end

    F --> J
    G --> L[Human corrections]
    L --> M[Annotation review]
    M --> C
```

## The problem

Paper-recycling streams contain materials that operators want removed:
**cardboard, soft plastic film, rigid plastic, and metal**. These are hard to
spot. Waste arrives crushed, folded, dirty, overlapping, and sometimes
translucent. A plastic bag can be almost invisible against paper, and printed
cardboard can look like the surrounding stream.

BeltWatch helps a quality technician decide **which inspection images deserve
attention** and shows the suspected material clearly. For each image it
answers:

- **What** material might be present?
- **Where** is it visible?
- **How much** of the inspected region does it occupy?
- **Does it need review?** This combines coverage, model uncertainty, and
  image quality.
- **What did the reviewer decide?** Each image keeps an append-only,
  versioned audit record.

### What BeltWatch measures, and what it does not

BeltWatch reports **estimated visible coverage**: the fraction of valid pixels
inside an operator-defined inspection region that the model assigns to each
material.

$$
\text{coverage}_c = \frac{\text{pixels predicted as material } c \text{ inside the region}}{\text{valid pixels inside the region}}
$$

It does **not** estimate contamination by weight, certify bale purity, or
prove that an image is clean. "Background" means "outside the four target
labels", not "verified clean paper".

## Workflow

1. Upload a batch of conveyor snapshots and draw the inspection region.
2. An asynchronous job validates the images and runs segmentation on CPU.
3. Per-material visible coverage and uncertainty are computed for each image.
4. A review queue ranks images. It prioritizes high coverage, reserves a share
   for uncertain cases, and adds a random sample of low-risk images to catch
   missed material.
5. The reviewer inspects overlays and confirms or corrects each finding.
6. BeltWatch exports an audit report with the model version and provenance.
7. Reviewed corrections become candidates for the next dataset version.

## Approach

| Role | Model |
|---|---|
| Sanity check | All-background prediction |
| Classical baseline | Random forest on color, gradient, and multiscale texture features |
| Deep-learning baseline | U-Net with an ImageNet-pretrained ResNet-18 encoder |
| Challenger | SegFormer-B0 with a five-class head |

The **primary metric is foreground macro IoU** over the four material
classes, so abundant background pixels can't inflate the score. Supporting
metrics:

- per-class IoU, precision, and recall
- visible-coverage MAE in percentage points
- recall versus review workload
- calibration error
- CPU latency and memory

Confidence intervals use a paired bootstrap over recording groups. A neural
model must clearly beat the *tuned* classical baseline to be adopted, and a
well-tuned U-Net is an acceptable winner.

Details: [design §9–§16](docs/design.md#9-model-strategy) and the
[model card](docs/model_card.md).

## Results

Release `beltwatch-0.1.0`: SegFormer-B0, temperature 1.04, chosen on val and
scored **once** on the locked test split (929 images) on 2026-10-15. The raw
outputs are in [`docs/results/beltwatch-0.1.0/`](docs/results/beltwatch-0.1.0/README.md),
and the full tables, including validation, are in the
[model card](docs/model_card.md#evaluation).

The official test split shares recordings with training (only sequence 08 is
unseen), so these results hold for **held-out time windows of the same
recordings**, not for new recordings or facilities. Intervals are 95% paired
bootstrap over the 5 test recording sequences.

| Model (test) | Foreground macro IoU | Cardboard | Soft plastic | Rigid plastic | Metal | Coverage MAE (bias), pp |
|---|---|---|---|---|---|---|
| All-background | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.97 (−17.97) |
| Random forest (tuned on val) | 0.152 (0.125–0.161) | 0.425 | 0.165 | 0.005 | 0.013 | 16.81 (+16.43) |
| **SegFormer-B0 (release)** | **0.336** (0.285–0.368) | 0.536 | 0.509 | 0.109 | 0.191 | **4.98** (−0.65) |

- **Release − random forest:** +0.184 foreground macro IoU (+0.158 to
  +0.207), with every class's interval above zero.
- **Unseen recording (sequence 08, 215 images):** 0.321. Rigid plastic
  falls to 0.038 there.
- **Per class:** precision and recall are in the
  [model card](docs/model_card.md#locked-test-split-929-images-declared-release-scored-once).
  The release finds 73% of cardboard pixels and 60% of soft plastic, but
  only 13% of rigid plastic and 28% of metal.
- **Calibration (T 1.04):** test NLL 0.292 → 0.290, ECE 0.019 → 0.015,
  foreground ECE 0.242 → 0.234. Foreground confidence remains poorly
  calibrated.
- **Review workload:** 89.7% of test images are audit-positive (more than
  5% true coverage). The release's ordering scores an area of 0.542, random
  0.503 and a perfect ordering 0.552. At this prevalence the curve cannot
  show a useful ordering.
- **CPU benchmark:** full pipeline p50 1.33 s, p95 1.59 s, 0.73 images/s,
  958 MB peak. Measured on a Kaggle CPU session with 4 cores, **not** the
  documented 4 vCPU / 8 GB host.

| Proposed target | Result |
|---|---|
| ≥ 5 points over the tuned random forest | **Met:** +18.4 |
| Foreground macro IoU ≥ 0.40 | **Missed:** 0.336 |
| Coverage MAE ≤ 5 points | **Met, narrowly:** 4.98 (one val seed scored 5.38) |
| Recall vs workload clearly better than random | **Missed:** not demonstrable at 89.7% prevalence |
| Seed spread ≤ 2 points | **Met:** 0.29 points over 3 seeds (val) |
| p95 ≤ 3 s, ≥ 0.5 images/s, ≤ 2 GB | **Met** on Kaggle; not yet measured on the documented host |
| ≥ 99% of 200 jobs succeed, no duplicate completions | **Met:** 200 of 200, with the worker killed 13 times ([soak test](docs/operations.md#reliability-soak-test); local M1 Mac, not the documented host) |

## Tech stack

| Area | Tools |
|---|---|
| Modeling | PyTorch, Hugging Face Transformers, scikit-learn, scikit-image |
| Data and experiments | DVC, MLflow, Pillow/OpenCV, NumPy |
| Serving | FastAPI, Pydantic, SQLite, a small HTML/JS frontend |
| Operations | Docker Compose, Prometheus, GitHub Actions |
| Optional optimization | ONNX Runtime, adopted only if benchmarks justify it |

## Status

| Component | Status |
|---|---|
| Pinned, checksummed, resumable download; validation and quarantine | Built and tested; checked on real archive data |
| Duplicate and leakage audit; official-repaired split policy | Built and tested; checked on real data |
| Shared preprocessing; mask conversion; augmentation | Built and tested; overlays checked on real frames |
| All-background and random-forest baselines | Done: full data; forest tuned on val (rf-tuning-1) |
| U-Net (ResNet-18) and SegFormer-B0 training with MLflow | Done: trained on Kaggle GPUs; SegFormer-B0 selected |
| Temperature-scaling calibration (overall and foreground ECE) | Done: fitted on the calibration split; T 1.04 for the release |
| Evaluation suite: foreground macro IoU, coverage MAE, review workload, group bootstrap | Built and tested |
| Error-analysis report and CPU benchmark | Done on val ([findings](docs/error_analysis.md)); benchmark on Kaggle CPU, not yet on the documented host |
| Audit service: durable jobs, leases, retries, idempotency, append-only feedback, reports | Built and tested; exercised live; 200-job soak test with worker kills |
| Feedback → annotation review → training candidates | Built and tested; no consented feedback exists yet, so no dataset version uses it |
| Monitoring: Prometheus metrics with active-release provenance, runbook | Built and tested |
| Review UI | Built; exercised in a headless browser |
| Versioned release bundles, activation, and rollback | Built; switch and rollback verified in CI |
| Docker Compose deployment (HTTPS proxy, API, worker, Prometheus) | Built; deployed and smoke-tested in CI |
| Fresh-checkout reproduction | Checked 2026-10-17: clone → `uv sync` → tests → service with the all-background and the `beltwatch-0.1.0` bundles |
| Ablations and 3-seed finalists | Done for loss, crop sampling and class weights (none adopted); 3 seeds of the finalist. Inference resolution not run |
| Locked test-set evaluation of a declared release | Done once, for `beltwatch-0.1.0` ([results](#results)) |
| Usability-study kit (protocol, matched val sets, study page, declared analysis) | Built, tested, dry-run with scripted sessions; **no participant sessions yet** ([protocol](docs/usability_study.md)) |
| Staged robustness dataset, hosted demo, ONNX/INT8 | Not started |

The full plan, targets, and Definition of Done are in
[`docs/design.md`](docs/design.md). The GPU steps are in the
[training runbook](docs/training.md).

## Data

BeltWatch uses **ZeroWaste-f** from Boston University and collaborators:
RGB frames from an operating paper-recycling conveyor with polygon labels for
cardboard, soft plastic, rigid plastic, and metal.

- Project: <https://ai.bu.edu/zerowaste/>
- Release: [Zenodo record 6412647](https://zenodo.org/records/6412647) (`zerowaste-f-final.zip`, about 7.5 GB)
- Paper: Bashkirova et al., [PMLR vol. 220](https://proceedings.mlr.press/v220/bashkirova23a/bashkirova23a.pdf)
- Measured: 4,503 valid 1920 × 1080 images (release splits train 3,002 ·
  val 572 · test 929), 0 quarantined. After the
  [official-repaired split policy](docs/dataset_card.md#split-policy-official-repaired):
  train 2,888 · val 370 · calibration 200 · test 929 (locked) · excluded 116,
  split manifest `c8285117…` in `data/splits/`.

> [!IMPORTANT]
> **License discrepancy.** The Zenodo record shows CC BY 4.0, while the
> project website and [repository](https://github.com/dbash/zerowaste) state
> CC BY-NC 4.0. BeltWatch treats the dataset as **attributed and
> noncommercial**. The dataset is downloaded by the user and is never
> redistributed in this repository. See
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Known limitations (by design)

- The frames come from related video footage, and the dataset's official
  train/val/test splits share recording sequences
  ([details](docs/dataset_card.md#splits-and-leakage-findings)), so results on
  the official test split may be optimistic. BeltWatch keeps the official
  test set but repairs leakage: it removes training frames near evaluation
  frames and training duplicates of evaluation images, and reports an
  unseen-recording slice separately
  ([split policy](docs/dataset_card.md#split-policy-official-repaired)).
- The data covers one facility and a narrow range of operating conditions.
  BeltWatch makes no claim about other facilities or cameras without local
  validation.
- It recognizes only four target materials. Rigid plastic (test IoU 0.109)
  and metal (0.191) are weak: rigid plastic is mostly confused with soft
  plastic, and metal is mostly missed as background.
- The test split has now been read. Any further tuning needs a new,
  untouched evaluation set before a fresh unbiased claim.
- Visible coverage is not contamination by weight.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra cpu           # create .venv with locked deps + CPU PyTorch
                              # (on an NVIDIA training machine: --extra cu126)
uv run pytest -m "not slow"   # tests (no network or dataset needed)
uv run ruff check             # lint
uv run ruff format --check    # formatting
uv run mypy                   # strict type checking of src/
```

### Baselines

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

### Training models

```bash
uv run python scripts/train.py --config configs/unet.yaml
uv run python scripts/evaluate.py --model unet --model-dir models/unet-resnet18-ce --split val
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Calibrate on the calibration split, check the result on val, and include it
in a release:

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

Full training needs an NVIDIA GPU. See the [training runbook](docs/training.md)
and the [Colab](notebooks/train_colab.ipynb) or [Kaggle](notebooks/train_kaggle.ipynb) notebook.

### Running the audit service

```bash
# 1. Build and activate a model release. Trained weights are not in the
#    repository; to try the service without them, use the all-background
#    sanity-check release (it never predicts a material):
uv run python -m beltwatch.release.bundle --kind all-background --version demo \
    --note "service check; not a trained model" --activate
#    or, with a trained SegFormer-B0 checkpoint and its calibration
#    (the way beltwatch-0.1.0 was built):
uv run python -m beltwatch.release.bundle --kind segformer --version <v> \
    --model models/segformer-b0-ce/best.pt \
    --calibration reports/calibration-segformer/calibration.json --activate

# 2. Start the API and the inference worker (two terminals)
uv run uvicorn beltwatch.api.app:create_app --factory --port 8000
uv run python -m beltwatch.jobs.worker

# 3. Submit an audit and follow it
curl -F "files=@frame.png" \
     -F 'options={"camera_id":"line-1","inspection_region":[[0.05,0.1],[0.95,0.1],[0.95,0.9],[0.05,0.9]]}' \
     http://localhost:8000/v1/audits
curl http://localhost:8000/v1/audits/<audit_id>
```

The review UI is at `http://localhost:8000/`: upload snapshots, draw the
inspection region, follow progress, work through the priority-ordered review
queue (overlay, original, and uncertainty views with estimated visible
coverage), record decisions, and export the audit record. Interactive API
docs are at `http://localhost:8000/docs`. The endpoints, failure handling, and
data model are described in [`docs/architecture.md`](docs/architecture.md).

Reviewer feedback reaches training data only through annotation review, and
only from audits created with `"training_consent": true`:

```bash
uv run python -m beltwatch.data.feedback export --data-dir var --out data/feedback/<batch>
uv run python -m beltwatch.data.feedback review data/feedback/<batch> <image_id> \
    --decision accept --annotator <name> --reason "<why>"
uv run python -m beltwatch.data.feedback eligible data/feedback/<batch>
```

Only accepted candidates with a corrected pixel mask are eligible for a
dataset version; a confirmation is an image-level label and does not validate
the predicted pixels.

### Deploying with Docker Compose

```bash
docker compose build
docker compose run --rm --no-deps --user root api \
    python -m beltwatch.release.activate <version> --reason "initial" --releases-dir /releases
docker compose up -d --wait            # HTTPS proxy, API, worker
python3 scripts/deploy_smoke.py --base-url https://localhost --insecure --expect-version <version>
```

The release, rollback, backup, and monitoring procedures are in
[`docs/operations.md`](docs/operations.md), together with a reliability soak
test (`scripts/soak_test.py`) that submits many audits while killing and
restarting the worker. CI builds the image and performs a
real deployment, release switch, and rollback on every change to the serving
code.

### Running the usability study

The protocol, consent script and declared analysis are in
[`docs/usability_study.md`](docs/usability_study.md). In short:

```bash
uv run python -m beltwatch.study.prepare --out data/study/study-1          # matched val sets
uv run python -m beltwatch.study.submit data/study/study-1 --participants 8  # through the API
# per participant: open http://localhost:8000/study.html and load plans/P0n.json
uv run python -m beltwatch.study.analyze data/study/study-1 --results results-P*.json --out reports/study-1
```

### Getting the data

The data pipeline is defined in [`dvc.yaml`](dvc.yaml) and run with
[DVC](https://dvc.org/):

```bash
uv run dvc repro            # download → validate → duplicates → splits
```

The `download` stage (also runnable directly with
`uv run python -m beltwatch.data.download`) downloads the pinned ZeroWaste-f release (about 7.5 GB, resumable),
verifies its size and MD5 against [`configs/data.yaml`](configs/data.yaml),
extracts it to `data/raw/zerowaste-f/`, and writes the measured image counts
to `data/manifests/zerowaste-f-download.json`. Reserve about 20 GB of free
disk for the archive plus the extracted files. Use `--skip-extract` to
download and verify only.

## Repository layout

```
configs/data.yaml        Pinned dataset release (record, version, size, MD5) and data paths
dvc.yaml                 Data pipeline stages
src/beltwatch/labels.py  Single source of truth for class IDs, names, and the ignore index
src/beltwatch/data/      Data configuration, verified downloader, validation, duplicate audit, splits,
                         mask conversion, augmentation, and the PyTorch dataset
src/beltwatch/inference/ Shared preprocessing and coverage computation used by training and serving
src/beltwatch/evaluation/ Segmentation metrics, coverage error, review workload, group bootstrap,
                         and the evaluation runner used for every model
src/beltwatch/baselines/ All-background sanity check, the random-forest baseline, and its val-only tuner
configs/random_forest_tuning.yaml   Declared random-forest tuning grid
src/beltwatch/models/    U-Net (ResNet-18 encoder) and SegFormer-B0 (MiT-B0 encoder)
src/beltwatch/training/  Training configuration, losses, and the tracked, resumable training loop
configs/unet.yaml, configs/segformer_b0.yaml   Matched training configurations
notebooks/               Colab and Kaggle notebooks: GPU training, CPU evaluation runs
scripts/evaluate.py      Evaluate a model on a split (the test split requires a declared release)
scripts/compare.py       Paired-bootstrap difference between two reports on the same images
scripts/train.py         Train a model from a YAML config
scripts/error_report.py  Error-analysis report (val only)
scripts/benchmark.py     CPU latency, throughput, and memory of a release
src/beltwatch/api/       FastAPI application and settings
src/beltwatch/jobs/      SQLite job store, per-image pipeline, inference worker
src/beltwatch/review/    Review policy (routing, priority, random audits)
src/beltwatch/study/     Usability study: set selection, submission, declared analysis
src/beltwatch/release/   Versioned, checksummed release bundles, declarations, and runtime parity
frontend/                Review UI and usability-study page (static HTML/CSS/JS, no external dependencies)
Dockerfile, compose.yaml, docker/   Serving image, Compose stack, Caddy and Prometheus config
scripts/deploy_smoke.py  Deployment smoke test (standard library only)
scripts/soak_test.py     Reliability soak test: many audits, worker kills, duplicate checks
configs/review_policy.yaml  Versioned review thresholds
configs/releases/        Release declarations, committed before the test split is read
docs/results/            Recorded outputs of each declared release's session
data/manifests/, data/splits/  Git-tracked dataset version (pipeline outputs)
tests/                   Unit and data tests (no network needed)
docs/design.md           Full design: problem, data, models, evaluation, system, Definition of Done
docs/dataset_card.md     Dataset provenance, licensing, split policy, and known limitations
docs/model_card.md       Intended use, models, results, failure modes
docs/evaluation.md       Evaluation protocol: metrics, coverage, review workload, confidence intervals
docs/training.md         GPU training runbook
docs/architecture.md     Service architecture, failure handling, data model
docs/operations.md       Deploy, release, rollback, backup, monitoring
docs/error_analysis.md   Error report contents and findings log
docs/usability_study.md  Usability-study protocol and declared analysis
THIRD_PARTY_NOTICES.md   Dataset and model licensing
```

See [design §29](docs/design.md#29-repository-structure) for the intended
structure.

## License

The code is released under the [MIT License](LICENSE). The ZeroWaste-f
dataset and the pretrained SegFormer weights have their own, more restrictive
terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

This project builds on the ZeroWaste dataset by Dina Bashkirova et al. and on
SegFormer by NVIDIA Research.
