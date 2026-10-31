# Operations

How to deploy, release, roll back, back up, and monitor BeltWatch on a single
CPU host with Docker Compose.

## Deploy

Requirements: a Linux VM with Docker (4 vCPU / 8 GB RAM is the planning
target), a DNS name pointing at it, and ports 80 and 443 open.

```bash
git clone https://github.com/professor3333/beltwatch.git && cd beltwatch
export BELTWATCH_DOMAIN=beltwatch.example.org       # "localhost" uses a local CA
export BELTWATCH_METRICS_TOKEN=$(openssl rand -hex 16)
docker compose build
# Copy a release bundle into ./model_releases/<version>/ (see below), then:
docker compose run --rm --no-deps --user root api \
    python -m beltwatch.release.activate <version> --reason "initial deployment" --releases-dir /releases
docker compose up -d --wait
python3 scripts/deploy_smoke.py --base-url https://$BELTWATCH_DOMAIN --expect-version <version>
```

Services: `proxy` (Caddy: HTTPS, upload size limit, `/metrics` blocked
publicly), `api`, `worker` (memory-limited to 3 GB), and optionally
`prometheus` (`docker compose --profile monitoring up -d`, bound to
localhost:9090). Training never runs on this host.

## Release procedure

1. **Build an immutable image.** Tag it, e.g. with
   `BELTWATCH_IMAGE_TAG=$(git rev-parse --short HEAD)`.
2. **Verify the model artifact.** The release bundle's SHA-256 is checked
   every time it is loaded, and a mismatched preprocessing version or label
   map is refused.
3. **Start the new release** alongside the previous one. Bundles are
   immutable directories, so both remain on disk.
4. **Run readiness and sample-inference checks** with
   `scripts/deploy_smoke.py --expect-version <new>`.
5. **Switch traffic** by activating the new version. New audits pin it
   immediately, and running audits keep their own version.
6. **Keep the previous image and bundle** for rollback.

**Release validation in CI.** The `release` workflow
(`.github/workflows/release.yml`) runs on a version tag (`v0.1.0` validates
`beltwatch-0.1.0`), or by hand with
`gh workflow run release -f release=<version>`. It fetches the declared
weights from the private Kaggle Dataset named by the repository variable
`KAGGLE_CHECKPOINTS_DATASET`, using the `KAGGLE_API_TOKEN` secret. It also
fetches the release's val folder from the pinned archive and checks every
file against the committed manifest. Then it checks:

1. The weights match the declared SHA-256.
2. A temperature refitted on the calibration split reproduces the declared
   one.
3. Val, re-scored at that temperature, reproduces the declared foreground
   macro IoU.
4. The bundle built from these outputs matches the declaration.
5. The bundle, loaded as the worker loads it, predicts what the evaluated
   checkpoint predicts (runtime parity).

It then runs the host benchmark. It never fetches or scores the test split:
a release is scored on test once, in the Kaggle `release` session. Only JSON
reports are uploaded.

**Result for `beltwatch-0.1.0`** (2026-10-19, run 39202671139 at commit
`5e2f5a5`; [`ci-validation/`](results/beltwatch-0.1.0/ci-validation/)). Every
check passed:

- refitted temperature: 1.04 on 200 calibration images
- val foreground macro IoU: 0.3185609 against the declared 0.3185608
- parity: maximum probability difference 0 on 5 val frames

The chained benchmark is run 3 above.

Promote a model only with an evaluation report that passes the release gates
in [evaluation.md](evaluation.md), and record why it was promoted
(`--reason`) and which version it replaces (written to
`model_releases/active.json`).

## Roll back

```bash
docker compose run --rm --no-deps --user root api \
    python -m beltwatch.release.activate <previous-version> --reason "rollback: <what went wrong>" --releases-dir /releases
python3 scripts/deploy_smoke.py --base-url https://$BELTWATCH_DOMAIN --expect-version <previous-version>
```

This restores the complete bundle: weights, preprocessing version, label map,
temperature, and review policy. To roll back the application code as well,
start the previous image (`BELTWATCH_IMAGE_TAG=<previous> docker compose up -d`).
CI performs a real switch and rollback on every change to the serving code
(`.github/workflows/container.yml`).

## Back up

```bash
# Consistent SQLite snapshot while the service runs
docker compose exec api python -c "import sqlite3; sqlite3.connect('/data/beltwatch.sqlite3').backup(sqlite3.connect('/data/backup.sqlite3'))"
# Database snapshot plus uploads and artifacts
docker run --rm -v beltwatch_data:/data -v "$PWD":/out alpine tar czf /out/beltwatch-data-$(date +%F).tgz -C /data .
# Release bundles
tar czf model_releases-$(date +%F).tgz model_releases
```

Uploads expire after `BELTWATCH_RETENTION_HOURS` (24 by default), so backups
mostly preserve audit records and feedback.

## Monitor

`/metrics` (Prometheus format, bearer token when `BELTWATCH_METRICS_TOKEN` is
set) exposes request counts and latency per route, rejected uploads by reason,
queue length, the age of the oldest pending audit, and the number of live
workers. `beltwatch_model_release_info` (value 1) carries the provenance of the
active release in its labels: version, model kind, weights SHA-256,
preprocessing version and review-policy version.
`beltwatch_live_workers_by_release` counts workers by the release they have
loaded, so a worker still serving an old bundle after a switch shows up. Each audit records per-image processing time, quality flags,
uncertainty, and routing, which the JSON report and the database expose for
model monitoring by camera and model version.

| Signal | First thing to check |
|---|---|
| Oldest pending audit keeps growing | Worker capacity, worker logs, `beltwatch_live_workers` |
| `/health/ready` is 503 | The `checks` field: database, model release, workers |
| `beltwatch_live_workers_by_release` shows a version other than `beltwatch_model_release_info` | Restart the worker so it loads the active bundle; jobs keep the release they were pinned to |
| Rising `retake_image` routing | Camera focus, lighting, lens cleanliness |
| Shift in predicted plastic coverage | Both the actual material mix and model behaviour; review a random sample |
| More reviewer disagreement | Collect a representative labelled sample and re-evaluate |

Drift signals are prompts to investigate, not proof that accuracy has fallen.
Retrain only after a verified data or performance problem. Random audits of
low-risk images keep monitoring from seeing only flagged cases.

## Reliability soak test

`scripts/soak_test.py` (standard library only) submits many single-image
audits to a running service. Each audit gets its own inspection region, so
none is served from the result cache, and at most `--in-flight` are
unfinished at a time. With `--worker-command` it runs the worker itself and
kills its whole process group with SIGKILL every `--kill-every` seconds,
restarting it at once. Each kill leaves a claimed audit behind, and that
audit must be picked up again after its lease expires. Every tenth request
is sent twice with the same Idempotency-Key and must return the same audit.

When all audits have finished, the script checks each one: final status,
exactly one image result, a retrievable overlay, and a report with one row.
With `--database` it also checks the job store's rows (read-only): one image
per audit, no pending image in a finished audit, no duplicate artifacts, and
all three artifacts for every complete image. It exits non-zero unless at
least 99% of audits complete and no duplicate is found.

```bash
export BELTWATCH_DATA_DIR=var BELTWATCH_RELEASES_DIR=model_releases BELTWATCH_LEASE_SECONDS=20
uv run uvicorn beltwatch.api.app:create_app --factory --port 8000 &
python3 scripts/soak_test.py --audits 200 --images frames/*.png \
    --worker-command "uv run python -m beltwatch.jobs.worker" --kill-every 15 \
    --database var/beltwatch.sqlite3 --out soak.json
```

**Result for `beltwatch-0.1.0`** (2026-10-16, 4 val frames, Apple M1 with 8
cores and 8 GB, lease 20 s, 10 audits in flight;
[`soak.json`](results/beltwatch-0.1.0/soak.json)):

- all 200 audits complete, with no image errors
- 13 worker kills; 12 audits were claimed again after a lost lease, and each
  completed once
- no duplicate results or artifacts in the API or the database
- 20 of 20 repeated requests returned the same audit
- submission to completion took 9.7 s at p50 and 24.9 s at p95, which
  includes the 20 s lease wait after each kill (1.02 audits/s overall)

This is a development machine, not the 4 vCPU / 8 GB target, and the run
used a lease far shorter than the 120 s default to make recovery quick.

## Benchmark on the planning host

The `benchmark` workflow (`.github/workflows/benchmark.yml`) measures a
declared release on the planning host of 4 vCPU and 8 GB. It runs on a
GitHub-hosted runner (4 vCPU and 16 GB for a public repository) and holds the
serving image to 4 CPUs and 8 GB, and the compose stack to 5.5 GB in total
(`docker/benchmark.compose.yaml`). A runner is a shared cloud VM, not
dedicated hardware. The record names the CPU model, so a reader can judge it.

Steps in the workflow:

1. Build the serving image at the commit.
2. Fetch the declared weights from a private Kaggle Dataset (see below).
3. Fetch the first 20 frames of the val folder from the pinned archive by
   range request (`python -m beltwatch.data.sample`). The test split is
   refused.
4. Rebuild the bundle from the committed calibration and val records in
   `docs/results/<version>/`, and check it against the declaration.
5. Benchmark the bundle inside the image, as the Kaggle session did (20
   frames, 4 threads). **The engineering targets are judged on this step.**
6. Bring up the compose stack and smoke-test it.
7. Time audits end to end through the HTTPS proxy: 40 audits one at a time,
   then 60 with four in flight. These times include the upload, the queue
   and status polling every 0.5 s. They are reported alongside the targets,
   not judged against them.
8. Record each container's limits and its cgroup memory peak. The peak
   includes page cache, so it reads higher than the resident memory.

The only artifact is the JSON record (`benchmark-host.json`, plus the two
end-to-end summaries). Weights and frames are never uploaded or printed.

**Setup (once):**

1. On Kaggle, open Settings → API and create a token.
2. Add it as a repository secret named `KAGGLE_API_TOKEN`:
   `gh secret set KAGGLE_API_TOKEN` (paste the token when prompted).
   Repository secrets are not passed to workflows triggered from forks.
3. Keep the checkpoints in the private Kaggle Dataset used by the notebook,
   one folder per run (`segformer-b0-ce/best.pt`). The SegFormer weights are
   noncommercial.

**Run:**

```bash
gh workflow run benchmark -f kaggle_dataset=<owner>/<dataset> -f release=beltwatch-0.1.0
gh run watch
gh run download <run-id> -n benchmark-host-beltwatch-0.1.0 -D benchmark-host/
```

Copy `benchmark-host.json` into `docs/results/<version>/` before quoting any
of its numbers.

**Results for `beltwatch-0.1.0`** (2026-10-19, three runs;
[`host-benchmark/`](results/beltwatch-0.1.0/host-benchmark/)). The bundle
matched the declaration in every run. Run 3 ran inside the release workflow.

| | Run 1 (AMD EPYC 7763) | Run 2 (AMD EPYC 9V45) | Run 3 (AMD EPYC 9V74) |
|---|---|---|---|
| Full pipeline p50 / p95 | 0.76 / 0.79 s | 0.46 / 0.48 s | 0.60 / 0.62 s |
| Throughput | 1.32 images/s | 2.17 images/s | 1.65 images/s |
| Peak RSS (benchmark) | 1071 MB | 1057 MB | 1015 MB |
| End to end, 1 in flight, p50 / p95 | 1.61 / 2.12 s (40 of 40) | 1.08 / 1.59 s (40 of 40) | 1.11 / 2.10 s (40 of 40) |
| End to end, 4 in flight | invalid (cache hits) | 1.63 audits/s, p95 1.93 s (60 of 60) | 1.30 audits/s, p95 2.98 s (60 of 60) |
| cgroup memory peak: worker / api / proxy | 1008 / 376 / 28 MB | 1041 / 375 / 28 MB | 1067 / 381 / 26 MB |

All three engineering targets are met in every run. Most of the difference
between the runs comes from the CPU GitHub assigned. In run 1, the
four-in-flight run used the same seed as the sequential run. Its first 40
audits repeated the sequential run's regions and were served from the result
cache, so it is not reported. Since then the workflow uses a second seed, and
the record refuses any run with cache hits.

## Hosted demo

The public demo is packaged for a Hugging Face Docker Space (CPU basic:
2 shared vCPUs, temporary disk). **Status (2026-10-31): not published.**
Docker Spaces on CPU now require a paid Hugging Face plan, and no paid host
has been chosen. Everything below has been verified locally (fetch from the
private repository, `deploy_smoke`, the UI in a headless browser). The
container needs no Hugging Face-specific feature apart from HTTPS in front of
port 7860, so it can run on any container host. It differs from the Compose
deployment in these ways:

- **One container.** `python -m beltwatch.demo` fetches the release, then
  runs the API and the worker as child processes. It exits as soon as either
  one stops, and the Space restarts it. The Space provides HTTPS; there is
  no Caddy and no Prometheus.
- **Public-demo mode** (`BELTWATCH_PUBLIC_DEMO=true`). The API refuses
  `training_consent: true` (`422 training_consent_unavailable`), and the UI
  hides the consent box and shows a notice: uploads are deleted after 24 h
  and never used for training, and estimates are not meaningful on images
  unlike ZeroWaste-f. `/metrics` gets a random bearer token at each start,
  so it is effectively closed. At most 10 audits wait in the queue, and the
  worker uses 2 threads.
- **Weights stay private.** Release bundles live in a private model
  repository (`professor3333/beltwatch-releases`, one directory per version).
  At startup, `beltwatch.release.fetch` downloads the version pinned in the
  Space's Dockerfile at a fixed commit of that repository. It checks the
  weights digest against the release declaration, then activates the bundle.
  A moved branch or a different file cannot be served silently. The Space
  reads the repository with the secret `HF_TOKEN`, a fine-grained read-only
  token for that one repository. The API and the worker never see the token.
- **No bundled samples.** The demo does not redistribute dataset images.
  Visitors upload their own images, and every upload goes through the real
  inference path.
- The disk is temporary: a restart (and the Space's sleep after inactivity)
  deletes every audit.

Publish or update it from a clean checkout of `main`:

```bash
# Once per release: upload the bundle; prints the model repository's commit.
uv run python scripts/hosted_demo.py upload-release \
    --repo professor3333/beltwatch-releases --version beltwatch-0.1.0

# Assemble the Space from HEAD (committed files only) and push it.
uv run python scripts/hosted_demo.py build-space --out /tmp/beltwatch-space \
    --release-repo professor3333/beltwatch-releases --release-revision <commit> \
    --version beltwatch-0.1.0 --push professor3333/beltwatch
```

Then check the deployment like any other:
`python3 scripts/deploy_smoke.py --base-url https://professor3333-beltwatch.hf.space
--expect-version beltwatch-0.1.0`. To roll back, run `build-space` again
from the previous commit or for the previous release. The Space then holds
exactly that build (the push replaces every file).

## Known trade-offs

- A single host means downtime during host maintenance. There is no high
  availability.
- SQLite with one worker suits demonstration scale. If concurrency grows,
  revisit the database, the queue, and the worker count; the model interface
  does not change.
- Uploaded images never appear in logs. Request logs carry request IDs,
  routes, status codes, and timings.
