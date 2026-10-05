# Dataset Card: ZeroWaste-f (as used by BeltWatch)

> This card is filled in as the data pipeline is built. Values marked
> *pending* have not been measured yet.

## Source

| Field | Value |
|---|---|
| Dataset | ZeroWaste-f, from *ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes* |
| Creators | Dina Bashkirova and collaborators, Boston University |
| Project page | <https://ai.bu.edu/zerowaste/> |
| Release | Zenodo record [6412647](https://zenodo.org/records/6412647), version 1.2.1 |
| DOI | [10.5281/zenodo.6412647](https://doi.org/10.5281/zenodo.6412647) |
| File | `zerowaste-f-final.zip` |
| Size | 7,518,242,799 bytes |
| MD5 | `e26e31a58080bca6782dca0e56074c5d` |
| Pinned in | [`configs/data.yaml`](../configs/data.yaml) |

The values above were read from the Zenodo API. The downloader refuses any
file that does not match the pinned size and MD5.

## License

**Unresolved discrepancy:**

- The Zenodo record's metadata lists **CC BY 4.0** (`cc-by-4.0`).
- The project website and the [GitHub repository](https://github.com/dbash/zerowaste)
  state **CC BY-NC 4.0**.

BeltWatch follows the more restrictive reading and uses the data only for an
attributed, noncommercial demonstration. Anyone planning commercial reuse
should clarify the terms with the dataset authors first. This repository does
not redistribute any dataset files.

## Contents

| Field | Value |
|---|---|
| Modality | RGB frames from an operating paper-recycling conveyor |
| Labels | Polygon segmentation: cardboard, soft plastic, rigid plastic, metal |
| Background | Everything else, including paper, the belt, and other objects. **Not** a "clean paper" label |
| Images listed on the project website | 4,503 |
| Images actually extracted | *pending*: recorded in `data/manifests/zerowaste-f-download.json` after the first download |
| Source category IDs → BeltWatch IDs | *pending*: derived by name from the annotation files via `beltwatch.labels.build_source_remap` |
| Overlapping-polygon policy | *pending* |
| Released splits used | *pending* |

## Known limitations

- Many frames come from related video footage, so split independence cannot
  be assumed. A duplicate and near-duplicate audit is *pending*.
- The images come from a single facility under a narrow range of operating
  conditions.
- Rare materials have relatively few examples.
- Overlapping and translucent material produces ambiguous boundaries.
- The released frames carry no continuous-video event annotations.
