---
title: BeltWatch
emoji: ♻️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Visible-coverage review of recycling-belt images
---

# BeltWatch (public demo)

BeltWatch segments four unwanted materials (cardboard, soft plastic, rigid
plastic, metal) in images of a paper-recycling conveyor belt. It estimates
their **visible coverage** inside an inspection region you draw, and ranks the
images for human review. Source, documentation and results:
<https://github.com/professor3333/beltwatch>.

This Space runs code commit `{git_commit}` with model release
`{release_version}` (weights SHA-256 `{release_sha256}`). Every upload goes
through the real inference path: the same API, worker and release bundle as
the self-hosted deployment, on 2 shared CPUs, so an audit takes a few seconds
per image and a queue can form.

## Read before using

- **Estimated visible coverage** is the share of the inspection region's
  pixels that the model labels as a material. It is not contamination by
  weight or mass, and it is not bale purity.
- **Background means "none of the four target materials".** It includes
  paper, the belt and anything else. A clean-looking result does not
  certify clean paper.
- The model was trained and evaluated on one facility's recordings
  (ZeroWaste-f). On other cameras, lighting or scenes its estimates are not
  meaningful. On ZeroWaste-f's locked test split it reached a foreground
  macro IoU of 0.336. Rigid plastic (IoU 0.11) and metal (IoU 0.19) are
  weak. See the model card in the repository.
- Results are suggestions for a person to confirm or correct.
- No sample images are bundled: the demo does not redistribute dataset
  images. ZeroWaste-f is available from its authors (below) if you want
  conveyor frames to try.

## Your uploads

Uploads, results and review decisions are deleted after **24 hours**, or
when you press "Delete audit". They are never used for training: this demo
refuses training consent. The Space's storage is temporary, so a restart
removes everything earlier.

## Attribution and licences

- Code: MIT.
- Data: ZeroWaste-f (Bashkirova et al., Boston University and
  collaborators), <https://zenodo.org/records/6412647>. Zenodo lists CC BY
  4.0, while the project website and GitHub list CC BY-NC 4.0. BeltWatch
  treats it as attributed and noncommercial.
- Model: SegFormer-B0 architecture with the `nvidia/mit-b0` encoder. The
  original SegFormer release is limited to noncommercial research and
  evaluation. The weights are not distributed from this Space.
