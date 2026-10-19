# Results record: beltwatch-0.1.0

The JSON outputs of the release session for `beltwatch-0.1.0`, as declared in
[`configs/releases/beltwatch-0.1.0.yaml`](../../../configs/releases/beltwatch-0.1.0.yaml).
They were produced by the Kaggle notebook's `release` mode on a CPU session
(2026-10-15) at commit `1c2e080`, without local modifications, and are copied
here unchanged. `soak.json` and `host-benchmark/` are the exceptions: they
were produced later against the same bundle, the soak test on a local
machine and the host benchmark on GitHub-hosted runners (2026-10-19). The numbers in the README and the [model card](../../model_card.md)
come from these files. `tests/unit/test_release_declaration.py` checks them
against the declaration.

| File | Contents |
|---|---|
| `calibration.json` | Temperature refit on the calibration split (200 images), ECE before and after |
| `val/release.json` | The release re-scored on val at the declared temperature |
| `val/random-forest.json` | The refitted reference forest on val (reproduced its declared score) |
| `bundle/manifest.json` | Manifest of the built release bundle (weights digest, temperature, review policy) |
| `benchmark.json` | CPU latency, throughput and memory of the bundle (Kaggle, 4 cores) |
| `test/release.json` | The release on the locked test split (929 images), with the sequence-08 slice |
| `test/release-uncalibrated.json` | The same weights at temperature 1 (calibration before/after) |
| `test/all-background.json`, `test/random-forest.json` | Reference baselines on test |
| `test/compare-release-vs-*.json` | Paired bootstrap differences over recording groups |
| `soak.json` | Reliability soak test of the bundle: 200 audits with worker kills (Apple M1, local; [operations.md](../../operations.md#reliability-soak-test)) |
| `host-benchmark/run-<id>.json` | The bundle benchmarked on the 4 vCPU / 8 GB profile by the `benchmark` workflow (GitHub-hosted runner, image limited to 4 CPUs / 8 GB), one file per run; run 39196619485's four-in-flight figure is invalid (cache hits; [operations.md](../../operations.md#benchmark-on-the-planning-host)) |
| `test/complete.json` | Marks the test run complete; the test split is not scored again for this release |

Per-image tables, confusion matrices, weights and the forest are not
committed. The weights are identified by their SHA-256
(`7be82fde…`), which matches the digest MLflow logged when they were trained.
