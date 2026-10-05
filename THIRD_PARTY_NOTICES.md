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

### SegFormer / MiT-B0 encoder (NVIDIA)

- **Source:** <https://github.com/NVlabs/SegFormer>. Weights:
  [`nvidia/mit-b0`](https://huggingface.co/nvidia/mit-b0) on Hugging Face,
  where the license is listed as `other`.
- **License:** NVIDIA's license for the original SegFormer release limits use
  to **noncommercial research and evaluation**. See
  <https://github.com/NVlabs/SegFormer#license>.
- **Usage in BeltWatch:** encoder initialization for the SegFormer-B0
  challenger (`configs/segformer_b0.yaml`). Training runs and checkpoints
  record the weight source in the `encoder_weights` lineage tag, and any
  release built from them inherits this restriction.

### ResNet-18 ImageNet weights (torchvision)

- **Source:** `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`.
- **License:** torchvision is BSD-3-Clause. The weights were trained on
  ImageNet-1k, whose terms restrict use to noncommercial research.
- **Usage in BeltWatch:** encoder initialization for the U-Net baseline.

## Software dependencies

Key libraries: PyTorch and torchvision (BSD-3-Clause), Hugging Face
Transformers (Apache-2.0), scikit-learn and scikit-image (BSD-3-Clause),
FastAPI (MIT), MLflow (Apache-2.0), DVC (Apache-2.0), NumPy and Pillow
(BSD-style / HPND). The full, pinned dependency set is in `uv.lock`.
