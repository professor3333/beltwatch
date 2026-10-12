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
| Images in the release archive | 4,503: train 3,002 · val 572 · test 929 (counted from the zip index, then confirmed by `validate` on the full download on 2026-10-12; 0 quarantined) |
| Archive layout | `splits_final_deblurred/{train,val,test}/{data,sem_seg,labels.json}` |
| Image format | 1920 × 1080 RGB PNG |
| Annotations | COCO `labels.json` per split, polygon segmentations (`iscrowd: 0`) |
| Pixel masks | Official `sem_seg/` PNGs: 8-bit single-channel, `0` = background, `1–4` = source category IDs |
| Source categories | `1 rigid_plastic · 2 cardboard · 3 metal · 4 soft_plastic` |
| Source IDs → BeltWatch IDs | `1→3 · 2→1 · 3→4 · 4→2` (`0→0`), derived by name with `beltwatch.labels.build_source_remap` |
| Overlapping polygons | The official `sem_seg` masks are the authoritative pixel labels; they encode the authors' resolution of overlaps. Polygons are validated for consistency only |
| Released splits used | The release's `train` / `val` / `test`, repaired: see *Split policy* below |

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

Measured on the full download (2026-10-12, `zerowaste-f-validation.json`):
4,503 valid images, 0 quarantined. Warnings: 86 images without annotations,
73 zero-area polygons, and 3 categories annotated but absent from the mask.

| Class | Share of all pixels | Images containing it |
|---|---|---|
| background | 83.08% | 4,503 |
| cardboard | 10.75% | 4,208 |
| soft plastic | 5.35% | 3,241 |
| rigid plastic | 0.72% | 1,357 |
| metal | 0.10% | 346 |

Rigid plastic and metal together cover under 1% of pixels, so they are the
hardest classes to learn and the noisiest to evaluate.

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
independence. Sequence-level grouping is required.

Full-dataset result (2026-10-12, `zerowaste-f-duplicates.json`): no
byte-identical groups and 41 near-duplicate pairs at pHash distance ≤ 6.
Thirty-eight pairs fall within one split (36 test–test, 2 train–train, the
`-2` files). Three cross splits, each at a frame gap of 0, meaning the release
holds the same frame number in two splits:

| Pair | Handled by |
|---|---|
| train `09_frame_003000` – test `09_frame_003000` | train copy excluded |
| val `01_frame_049000` – test `01_frame_049000` | val copy excluded |
| val `03_frame_049000` – test `03_frame_049000` | val copy excluded |

### Split policy: `official-repaired`

Decided on 2026-10-05, before any model was trained, and configured in
[`configs/data.yaml`](../configs/data.yaml) (`splits`). Implemented by
`uv run dvc repro splits`.

1. **Start from the release's train/val/test**, so results stay comparable
   with published ZeroWaste work.
2. **Calibration holdout:** whole val sequences `02` and `10` move to a
   grouped calibration split, used only for temperature scaling.
3. **Temporal buffer:** a train frame within **500 frames** of any val,
   calibration, or test frame in the same sequence is excluded. The buffer
   size is a project convention.
4. **Duplicate repair:** a train image that is an exact or near-duplicate of
   any evaluation image is excluded, and so is a val or calibration image
   that duplicates a test image.
5. **The test set stays locked.** It contains exactly the release's valid
   test images; the stage refuses to change test membership.
6. **Unseen-recording slice:** test images whose sequence has no remaining
   train images are marked `unseen_sequence` and reported separately.

Measured sizes on the full dataset (2026-10-12), split manifest ID
`c82851178745739c24daa50338ff5f65cb38f8a172420ec5d39a5fdd28d73ead`:

| Split | Images | Sequences |
|---|---|---|
| train | 2,888 | 01–07, 09–12 |
| val | 370 | 01, 03, 04, 12 |
| calibration | 200 | 02, 10 |
| test (locked) | 929 | 01, 03, 05, 08, 09 |
| excluded | 116 | 01 (14), 02 (50), 03 (1), 09 (51) |
| unseen-recording test slice | 215 | 08 |

Exclusions: 113 train frames within the 500-frame buffer, 1 train duplicate of
a test frame, and 2 val duplicates of test frames. Val is 2 images smaller
than the 372 projected from file names before the audit because of the two
val–test duplicates above. The same manifest ID was produced independently on
Colab and on Kaggle.

**What results on these splits support:** performance on held-out time
windows of the same recordings, plus a single-sequence unseen-recording slice.
They do **not** support claims about other facilities, cameras, or operating
conditions. With only one unseen sequence, the unseen-recording slice has no
meaningful between-group confidence interval, so it is reported as a
qualitative check. Every reported result names this split policy and the
split manifest ID from `data/splits/zerowaste-f-splits.json`.

## Known limitations

- Many frames come from related video footage, and the released splits share
  sequences (see above). The duplicate audit removed the three cross-split
  near-duplicates it found, but neighbouring frames outside the buffer remain
  correlated, so independence between splits is not guaranteed.
- The images come from a single facility under a narrow range of operating
  conditions.
- Rare materials have relatively few examples.
- Overlapping and translucent material produces ambiguous boundaries.
- The released frames carry no continuous-video event annotations.
