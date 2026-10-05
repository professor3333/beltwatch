# Third-Party Notices

BeltWatch's own source code is licensed under the [MIT License](LICENSE).
The third-party data and model weights it relies on have their own terms,
listed below. **These terms are more restrictive than MIT**, and anyone using
BeltWatch with them must follow them.

## Datasets

### ZeroWaste-f

- **Authors:** Dina Bashkirova and collaborators, Boston University
- **Project:** <https://ai.bu.edu/zerowaste/>
- **Release used:** Zenodo record [6412647](https://zenodo.org/records/6412647), version 1.2.1,
  DOI [10.5281/zenodo.6412647](https://doi.org/10.5281/zenodo.6412647), file `zerowaste-f-final.zip`
  (7,518,242,799 bytes, MD5 `e26e31a58080bca6782dca0e56074c5d`)
- **Code repository:** <https://github.com/dbash/zerowaste>
- **Paper:** <https://proceedings.mlr.press/v220/bashkirova23a/bashkirova23a.pdf>
- **License: unresolved discrepancy.**
  - The Zenodo record displays **CC BY 4.0**.
  - The project website and GitHub repository state **CC BY-NC 4.0**.

  BeltWatch follows the more restrictive reading, so its use of the dataset is
  **attributed and noncommercial**. Clarify the terms with the dataset authors
  before any commercial reuse.
- **Redistribution:** this repository does **not** include the dataset or any
  of its images. Users download it themselves from the official release.

## Pretrained models

### SegFormer (NVIDIA)

- **Source:** <https://github.com/NVlabs/SegFormer>
- **License:** the original SegFormer release is limited to **noncommercial
  research and evaluation** use. See
  <https://github.com/NVlabs/SegFormer#license>.
- **Usage in BeltWatch:** a candidate challenger model (SegFormer-B0). Each
  model release built from SegFormer weights will record the exact checkpoint
  and its license.

### ResNet-18 ImageNet weights

- **Usage in BeltWatch:** encoder initialization for the U-Net baseline.
- **License:** recorded with the exact weight source when the baseline is
  implemented.

## Software dependencies

Python package licenses will be listed here once the dependency set is
locked.
