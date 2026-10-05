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
| Images in the release archive | 4,503: train 3,002 · val 572 · test 929 (counted from the archive's zip index; confirmed by `validate` after a full download) |
| Archive layout | `splits_final_deblurred/{train,val,test}/{data,sem_seg,labels.json}` |
| Image format | 1920 × 1080 RGB PNG |
| Annotations | COCO `labels.json` per split, polygon segmentations (`iscrowd: 0`) |
| Pixel masks | Official `sem_seg/` PNGs: 8-bit single-channel, `0` = background, `1–4` = source category IDs |
| Source categories | `1 rigid_plastic · 2 cardboard · 3 metal · 4 soft_plastic` |
| Source IDs → BeltWatch IDs | `1→3 · 2→1 · 3→4 · 4→2` (`0→0`), derived by name with `beltwatch.labels.build_source_remap` |
| Overlapping polygons | The official `sem_seg` masks are the authoritative pixel labels; they encode the authors' resolution of overlaps. Polygons are validated for consistency only |
| Released splits used | The release's `train` / `val` / `test` as the starting point; see the leakage findings below |

## Validation

`uv run dvc repro validate` checks every image/mask pair and writes three
git-tracked files to `data/manifests/`:

- `zerowaste-f-images.csv`: one row per usable image, with sequence ID,
  frame index, size, SHA-256 hashes, and pixel counts per BeltWatch class.
- `zerowaste-f-quarantine.csv`: excluded images with explicit reasons
  (decode failures, missing or mismatched masks, unexpected mask values,
  unknown categories, files missing from or absent in the annotations). Raw
  files are never moved or modified.
- `zerowaste-f-validation.json`: counts per split, error and warning counts,
  class pixel fractions, and images per sequence and split.

Polygon problems (degenerate, zero-area, or out-of-bounds polygons) and
disagreements between a mask and its annotations are recorded as warnings
and do not exclude the image.

## Splits and leakage findings

File names have the form `<sequence>_frame_<index>.PNG`, for example
`09_frame_003000.PNG`. The sequence prefix is treated as the recording group.
Measured from the archive index:

| Sequence | train | val | test |
|---|---|---|---|
| 01 | 38 | 157 | 262 |
| 02 | 101 | 100 | |
| 03 | 201 | 71 | 251 |
| 04 | 352 | 101 | |
| 05 | 352 | | 100 |
| 06 | 351 | | |
| 07 | 351 | | |
| 08 | | | 215 |
| 09 | 453 | | 101 |
| 10 | 502 | 100 | |
| 11 | 100 | | |
| 12 | 201 | 43 | |

- **The released splits share recording sequences.** Only sequence 08 is
  exclusive to test. Sequences 01 and 03 appear in all three splits.
- Within a sequence, splits mostly occupy separate contiguous frame ranges,
  often thousands of frames apart. Some are much closer:
  - **Sequence 09:** test frames 3000–4000 fall inside the train range
    1000–49000, and frame 3000 exists in both train and test (the files
    differ byte-for-byte).
  - **Sequence 02:** val ends at frame 991 and train starts at frame 1000.
- Two train files carry a `-2` suffix (`09_frame_002000-2`,
  `10_frame_030000-2`) next to a file with the same base name. The bytes
  differ, but in a sampled pair the class pixel counts are within a few
  percent of each other, so they are likely near-duplicates.

### Duplicate audit

`uv run dvc repro duplicates` computes 64-bit perceptual hashes (pHash and
dHash) for every valid image. It reports byte-identical groups, pairs within
the configured pHash Hamming distance (`duplicates.phash_max_hamming`,
default 6), and, for each sequence present in several splits, the smallest
frame gap between splits. Outputs: `zerowaste-f-hashes.csv`,
`zerowaste-f-duplicates.csv`, and `zerowaste-f-duplicates.json`.

Threshold calibration on a 22-image real sample (train and val):

| Pair | pHash distance |
|---|---|
| `09_frame_002000` vs `09_frame_002000-2` | 0 (near-duplicate) |
| Consecutive frames 10 apart (sequence 04) | 14–22 |
| Consecutive train frames (sequence 01) | 16–18 |
| Median over all pairs | 28 |

The default threshold separates true near-duplicates from neighbouring
frames. Neighbouring frames are still strongly correlated without being
hash-level duplicates, so perceptual hashing alone cannot establish
independence. Sequence-level grouping is required. Full-dataset counts are
*pending* the download.

Consequences for evaluation: results on the released test split may be
optimistic because of shared sequences and temporally adjacent frames. The
split policy (*pending*) must address this using the duplicate audit and
sequence IDs, and every reported result must state which split policy it
uses.

## Known limitations

- Many frames come from related video footage, and the released splits share
  sequences (see above). A duplicate and near-duplicate audit is *pending*.
- The images come from a single facility under a narrow range of operating
  conditions.
- Rare materials have relatively few examples.
- Overlapping and translucent material produces ambiguous boundaries.
- The released frames carry no continuous-video event annotations.
