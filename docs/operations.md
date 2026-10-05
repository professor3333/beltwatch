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
workers. Each audit records per-image processing time, quality flags,
uncertainty, and routing, which the JSON report and the database expose for
model monitoring by camera and model version.

| Signal | First thing to check |
|---|---|
| Oldest pending audit keeps growing | Worker capacity, worker logs, `beltwatch_live_workers` |
| `/health/ready` is 503 | The `checks` field: database, model release, workers |
| Rising `retake_image` routing | Camera focus, lighting, lens cleanliness |
| Shift in predicted plastic coverage | Both the actual material mix and model behaviour; review a random sample |
| More reviewer disagreement | Collect a representative labelled sample and re-evaluate |

Drift signals are prompts to investigate, not proof that accuracy has fallen.
Retrain only after a verified data or performance problem. Random audits of
low-risk images keep monitoring from seeing only flagged cases.

## Known trade-offs

- A single host means downtime during host maintenance. There is no high
  availability.
- SQLite with one worker suits demonstration scale. If concurrency grows,
  revisit the database, the queue, and the worker count; the model interface
  does not change.
- Uploaded images never appear in logs. Request logs carry request IDs,
  routes, status codes, and timings.
